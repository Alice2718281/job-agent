"""Official job page enrichment for searched jobs."""
import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

from posted_date_utils import PostedDateParser

logger = logging.getLogger(__name__)


class OfficialLinkEnricher:
    """Find and validate official/company-controlled apply links."""

    PREFERRED_DOMAINS = (
        "greenhouse.io",
        "lever.co",
        "myworkdayjobs.com",
        "workdayjobs.com",
        "ashbyhq.com",
        "smartrecruiters.com",
        "bamboohr.com",
        "jobvite.com",
        "icims.com",
    )

    THIRD_PARTY_DOMAINS = (
        "dice.com",
        "indeed.com",
        "ziprecruiter.com",
        "talent.com",
        "jooble.org",
        "monster.com",
        "simplyhired.com",
        "linkedin.com/jobs/search",
    )

    def __init__(
        self,
        serpapi_key: str,
        max_enrichment_searches: int = 15,
        request_timeout: int = 20,
        run_date=None,
    ):
        self.serpapi_key = serpapi_key
        self.max_enrichment_searches = max_enrichment_searches
        self.request_timeout = request_timeout
        self.base_url = "https://serpapi.com/search"
        self.searches_used = 0
        self.date_parser = PostedDateParser(run_date=run_date)

    def enrich_job(self, job: Dict) -> Dict:
        """Find possible official job page, parse it, and choose link status."""
        enriched = {
            **job,
            "original_apply_link": job.get("apply_link", ""),
            "official_apply_link": "",
            "possible_official_link": "",
            "link_status": "Third-party link only",
            "link_status_reason": "official link not found",
            "official_match_confidence": 0.0,
            "field_sources": {},
            "is_active": True,
            "inactive_reason": "",
        }

        original_html = self._fetch_html(job.get("apply_link", ""))
        self._apply_page_metadata(enriched, original_html, source_label="third-party page")

        candidates = self._search_official_candidates(job)
        logger.info(
            "Official link candidates for %s at %s: %s",
            job.get("title"),
            job.get("company"),
            [candidate.get("link") for candidate in candidates[:5]],
        )

        best_candidate = None
        for candidate in candidates:
            candidate_html = self._fetch_html(candidate.get("link", ""))
            candidate_text = self.date_parser.html_to_text(candidate_html.get("html", ""))
            confidence, reason = self._score_candidate(job, candidate, candidate_text)
            logger.info(
                "Official candidate confidence %.2f for %s: %s",
                confidence,
                candidate.get("link"),
                reason,
            )
            candidate["confidence"] = confidence
            candidate["reason"] = reason
            candidate["html_response"] = candidate_html
            candidate["text"] = candidate_text
            if best_candidate is None or confidence > best_candidate["confidence"]:
                best_candidate = candidate

        if best_candidate is None:
            self._apply_google_posted_date(enriched)
            return enriched

        confidence = best_candidate["confidence"]
        enriched["official_match_confidence"] = round(confidence, 2)
        enriched["link_status_reason"] = best_candidate["reason"]

        if confidence >= 0.72:
            enriched["official_apply_link"] = best_candidate["link"]
            enriched["apply_link"] = best_candidate["link"]
            enriched["link_status"] = "Official link found"
            logger.info("Replaced original link with official link: %s", best_candidate["link"])
            self._apply_page_metadata(enriched, best_candidate["html_response"], source_label="official page")
        elif confidence >= 0.50:
            enriched["possible_official_link"] = best_candidate["link"]
            enriched["link_status"] = "Possible official match"
            logger.info("Kept original link; possible official match: %s", best_candidate["link"])
            self._apply_page_metadata(enriched, best_candidate["html_response"], source_label="possible official page")
        else:
            logger.info("Kept original link; official match confidence too low")

        self._apply_google_posted_date(enriched)
        return enriched

    def _search_official_candidates(self, job: Dict) -> List[Dict]:
        if self.searches_used >= self.max_enrichment_searches:
            logger.info("Skipping official search; enrichment search budget exhausted")
            return []

        query = self._build_search_query(job)
        logger.info("Official link search query: %s", query)
        self.searches_used += 1

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "num": 8,
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Official link search failed: %s", e.__class__.__name__)
            return []

        candidates = []
        for result in data.get("organic_results", []):
            link = result.get("link", "")
            if not link or not self._looks_official(job.get("company", ""), link):
                continue
            candidates.append({
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "link": link,
            })
        return candidates

    def _build_search_query(self, job: Dict) -> str:
        company = job.get("company", "")
        title = job.get("title", "")
        location = job.get("location", "")
        return (
            f'"{company}" "{title}" "{location}" '
            '(careers OR jobs OR greenhouse OR lever OR ashby OR workday OR smartrecruiters)'
        )

    def _score_candidate(self, job: Dict, candidate: Dict, candidate_text: str) -> tuple:
        title = self._normalize(job.get("title", ""))
        candidate_title = self._normalize(f"{candidate.get('title', '')} {candidate_text[:500]}")
        title_similarity = SequenceMatcher(None, title, candidate_title).ratio()

        company_score = 1.0 if self._looks_official(job.get("company", ""), candidate.get("link", "")) else 0.0
        location_score = self._location_overlap(job.get("location", ""), candidate_text)
        keyword_score = self._keyword_overlap(job.get("description", ""), candidate_text)
        req_score = self._requisition_overlap(job.get("description", ""), candidate_text)

        confidence = (
            company_score * 0.35
            + title_similarity * 0.30
            + location_score * 0.15
            + keyword_score * 0.15
            + req_score * 0.05
        )
        reason = (
            f"company={company_score:.2f}, title={title_similarity:.2f}, "
            f"location={location_score:.2f}, keywords={keyword_score:.2f}, req={req_score:.2f}"
        )
        return confidence, reason

    def _apply_page_metadata(self, job: Dict, response: Dict, source_label: str):
        html = response.get("html", "")
        if not html:
            return

        active, reason = self.date_parser.is_active_page(html, response.get("status_code"))
        if not active:
            job["is_active"] = False
            job["inactive_reason"] = reason
            logger.info("Filtered inactive page candidate: %s", reason)

        posted_date, posted_source, posted_age_days = self.date_parser.parse_from_html(html)
        if posted_date != "Unknown":
            job["posted_date"] = posted_date
            job["posted_age_days"] = posted_age_days
            job["field_sources"]["posted_date"] = source_label if posted_source == "page text" else posted_source
            logger.info("Posted date source: %s -> %s", job["field_sources"]["posted_date"], posted_date)

        text = self.date_parser.html_to_text(html)
        if source_label in {"official page", "possible official page"} and text:
            job["description"] = text[:12000]
            job["field_sources"]["description"] = source_label

    def _apply_google_posted_date(self, job: Dict):
        if job.get("posted_date") and job.get("posted_date") != "Unknown":
            return
        posted_date, source, age_days = self.date_parser.parse_google_posted_at(job.get("posted_at", ""))
        job["posted_date"] = posted_date
        job["posted_age_days"] = age_days
        job["field_sources"]["posted_date"] = source
        logger.info("Posted date source: %s -> %s", source, posted_date)

    def _fetch_html(self, url: str) -> Dict:
        if not url:
            return {"html": "", "status_code": None}
        try:
            response = requests.get(
                url,
                timeout=self.request_timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            return {"html": response.text or "", "status_code": response.status_code}
        except requests.exceptions.RequestException:
            return {"html": "", "status_code": None}

    def _looks_official(self, company: str, url: str) -> bool:
        domain = urlparse(url).netloc.lower()
        company_tokens = [
            token for token in re.split(r"[^a-z0-9]+", company.lower())
            if len(token) >= 3 and token not in {"inc", "llc", "corp", "corporation", "company"}
        ]
        if any(preferred in domain for preferred in self.PREFERRED_DOMAINS):
            return True
        return any(token in domain for token in company_tokens)

    @staticmethod
    def _location_overlap(location: str, text: str) -> float:
        tokens = [token for token in re.split(r"[^a-z0-9]+", location.lower()) if len(token) >= 2]
        if not tokens:
            return 0.0
        lowered = text.lower()
        matches = sum(1 for token in tokens if token in lowered)
        return min(1.0, matches / max(1, len(tokens)))

    @staticmethod
    def _keyword_overlap(original_description: str, candidate_text: str) -> float:
        important = [
            "sql", "python", "product", "analytics", "experiment", "machine learning",
            "ai", "dashboard", "metrics", "data science",
        ]
        original = original_description.lower()
        candidate = candidate_text.lower()
        relevant = [word for word in important if word in original]
        if not relevant:
            relevant = important
        matches = sum(1 for word in relevant if word in candidate)
        return min(1.0, matches / max(1, len(relevant)))

    @staticmethod
    def _requisition_overlap(original_description: str, candidate_text: str) -> float:
        pattern = re.compile(r"\b(?:req|requisition|job id)[:#\s-]*([a-z0-9-]{4,})", re.IGNORECASE)
        original_ids = set(pattern.findall(original_description or ""))
        candidate_ids = set(pattern.findall(candidate_text or ""))
        if not original_ids or not candidate_ids:
            return 0.0
        return 1.0 if original_ids & candidate_ids else 0.0

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value.lower()).strip()
