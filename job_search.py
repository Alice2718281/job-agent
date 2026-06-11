"""Job search module using SerpAPI Google Jobs API."""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("job_preferences.json")


class JobSearcher:
    """Search for jobs using SerpAPI Google Jobs API."""

    TARGET_TITLES = [
        "Data Scientist",
        "Data Analyst",
        "Product Analyst",
        "Product Manager",
    ]

    SKILL_QUERIES = [
        "SQL Python Power BI analytics",
        "SQL Python product analytics",
        "Data Analyst SQL Python Power BI",
        "Data Scientist SQL Python machine learning",
        "Product Analyst SQL Python experimentation",
        "Product Manager data analytics SQL",
    ]

    def __init__(
        self,
        api_key: str,
        max_api_calls: int = 18,
        results_per_query: int = 10,
        recent_hours: int = 48,
        request_timeout: int = 25,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ):
        """Initialize job searcher with SerpAPI settings."""
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"
        self.max_api_calls = max_api_calls
        self.results_per_query = results_per_query
        self.recent_hours = recent_hours
        self.request_timeout = request_timeout
        self.config = self._load_config(config_path)

    def search_jobs(self) -> List[Dict]:
        """Search for jobs matching target criteria."""
        all_jobs = []
        search_queries = self._build_search_queries()

        for query, location in search_queries[: self.max_api_calls]:
            logger.info("Searching for: %s near %s", query, location)
            jobs = self._fetch_jobs(query, location)
            all_jobs.extend(jobs)
            logger.info("Found %s recent jobs for query: %s", len(jobs), query)

        deduplicated = self._deduplicate_jobs(all_jobs)
        logger.info("Total recent jobs after deduplication: %s", len(deduplicated))
        return deduplicated

    def _build_search_queries(self) -> List[Tuple[str, str]]:
        """Build prioritized searches from editable preferences."""
        queries: List[Tuple[str, str]] = []
        search_config = self.config.get("search", {})
        titles = search_config.get("title_keywords") or self.TARGET_TITLES
        skill_queries = search_config.get("skill_queries") or self.SKILL_QUERIES
        locations = search_config.get("locations") or ["New York, NY"]

        for location in locations:
            for title in titles:
                queries.append((f'"{title}"', location))

        primary_location = locations[0]
        for query in skill_queries:
            queries.append((query, primary_location))

        return queries

    @staticmethod
    def _load_config(config_path: Path) -> Dict:
        try:
            with Path(config_path).open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Could not load %s: %s. Using built-in search defaults.", config_path, error)
            return {}

    def _fetch_jobs(self, query: str, location: str) -> List[Dict]:
        """Fetch jobs from SerpAPI for a specific query."""
        params = {
            "engine": "google_jobs",
            "q": query,
            "location": location,
            "api_key": self.api_key,
            "num": self.results_per_query,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()

            jobs = []
            for job in data.get("jobs_results", []):
                parsed = self._parse_job(job)
                if self._is_recent_posting(parsed.get("posted_at", "")):
                    parsed["search_query"] = query
                    parsed["search_location"] = location
                    parsed["raw_detected_extensions"] = job.get("detected_extensions", {})
                    jobs.append(parsed)
            return jobs
        except requests.exceptions.RequestException as e:
            logger.error(
                "Error fetching jobs for query '%s' near '%s': %s",
                query,
                location,
                self._safe_request_error(e),
            )
            return []

    def _parse_job(self, job: Dict) -> Dict:
        """Normalize a SerpAPI Google Jobs result."""
        apply_options = job.get("apply_options") or []
        apply_link = job.get("link") or job.get("share_link") or ""
        if apply_options:
            apply_link = apply_options[0].get("link") or apply_link

        detected_extensions = job.get("detected_extensions", {})
        posted_at = (
            job.get("posted_at")
            or detected_extensions.get("posted_at")
            or detected_extensions.get("date_posted")
            or ""
        )

        salary = detected_extensions.get("salary") or job.get("salary") or ""
        schedule_type = (
            detected_extensions.get("schedule_type")
            or detected_extensions.get("work_from_home")
            or ""
        )

        return {
            "title": job.get("title", ""),
            "company": job.get("company_name", ""),
            "location": job.get("location", ""),
            "apply_link": apply_link,
            "description": job.get("description", ""),
            "posted_at": posted_at,
            "job_id": job.get("job_id", ""),
            "salary": salary,
            "schedule_type": schedule_type,
            "via": job.get("via", ""),
        }

    def _deduplicate_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Remove duplicate jobs by company, title, and location."""
        seen = set()
        unique_jobs = []

        for job in jobs:
            key = (
                self._normalize_key(job.get("company", "")),
                self._normalize_key(job.get("title", "")),
                self._normalize_key(job.get("location", "")),
            )

            if key not in seen and all(key):
                seen.add(key)
                unique_jobs.append(job)

        return unique_jobs

    def _is_recent_posting(self, posted_at: str) -> bool:
        """Check if job was posted within the configured recent-hours window."""
        if not posted_at:
            return True

        hours_old = self._posted_at_to_hours(posted_at)
        if hours_old is None:
            return True
        return hours_old <= self.recent_hours

    @staticmethod
    def _posted_at_to_hours(posted_at: str) -> Optional[int]:
        """Parse common SerpAPI relative posting dates into approximate hours."""
        value = posted_at.lower().strip()
        if any(token in value for token in ("just posted", "today", "new")):
            return 0

        match = re.search(r"(\d+)\s+(minute|minutes|hour|hours|day|days|week|weeks|month|months)", value)
        if not match:
            return None

        amount = int(match.group(1))
        unit = match.group(2)

        if unit.startswith("minute"):
            return 0
        if unit.startswith("hour"):
            return amount
        if unit.startswith("day"):
            return amount * 24
        if unit.startswith("week"):
            return amount * 24 * 7
        if unit.startswith("month"):
            return amount * 24 * 30
        return None

    @staticmethod
    def _normalize_key(value: str) -> str:
        """Normalize text for stable deduplication."""
        return re.sub(r"\s+", " ", value.lower()).strip()

    def _safe_request_error(self, error: requests.exceptions.RequestException) -> str:
        """Return request error text without API keys or full request URLs."""
        response = getattr(error, "response", None)
        if response is not None:
            return f"HTTP {response.status_code}"
        return error.__class__.__name__
