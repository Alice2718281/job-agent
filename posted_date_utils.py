"""Utilities for posted-date parsing and job page status checks."""
import json
import logging
import re
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PostedDateParser:
    """Parse and validate job posted dates from structured and unstructured text."""

    DATE_PATTERNS = [
        r"\b(?:post|posted|post date|date posted|posting date)\s*:?\s*([A-Z][a-z]{2,9}\s+\d{1,2},\s+\d{4})",
        r"\b(?:post|posted|post date|date posted|posting date)\s*:?\s*(\d{1,2}/\d{1,2}/\d{2,4})",
        r"\b(?:post|posted|post date|date posted|posting date)\s*:?\s*(\d{4}-\d{2}-\d{2})",
    ]

    RELATIVE_PATTERN = re.compile(
        r"\b(\d+)\s+(minute|minutes|hour|hours|day|days|week|weeks|month|months)\s+ago\b",
        re.IGNORECASE,
    )

    INACTIVE_TERMS = (
        "job is no longer available",
        "position is no longer available",
        "this job is no longer accepting applications",
        "no longer accepting applications",
        "job has expired",
        "posting has expired",
        "position has been filled",
        "this position has closed",
        "job closed",
        "job unavailable",
        "page not found",
        "404",
    )

    def __init__(self, run_date: Optional[date] = None, stale_days: int = 183):
        self.run_date = run_date or date.today()
        self.stale_days = stale_days

    def parse_from_html(self, html: str) -> Tuple[str, str, Optional[int]]:
        """Parse posted date from HTML using visible text first, then JSON-LD."""
        text = self.html_to_text(html)
        posted_date, source, age_days = self.parse_from_text(text, source="page text")
        if posted_date != "Unknown":
            return posted_date, source, age_days

        posted_date = self._parse_json_ld_date(html)
        if posted_date:
            age_days = (self.run_date - posted_date).days
            return posted_date.isoformat(), "structured data", age_days

        return "Unknown", "unknown", None

    def parse_from_text(self, text: str, source: str) -> Tuple[str, str, Optional[int]]:
        """Parse absolute or relative posted date from plain text."""
        if not text:
            return "Unknown", "unknown", None

        relative = self._parse_relative_date(text)
        if relative:
            age_days = (self.run_date - relative).days
            return relative.isoformat(), source, age_days

        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            parsed = self._parse_absolute_date(match.group(1))
            if parsed:
                age_days = (self.run_date - parsed).days
                return parsed.isoformat(), source, age_days

        return "Unknown", "unknown", None

    def parse_google_posted_at(self, posted_at: str) -> Tuple[str, str, Optional[int]]:
        """Parse SerpAPI/Google Jobs relative posted date."""
        posted_date, _, age_days = self.parse_from_text(posted_at, source="SerpAPI")
        if posted_date == "Unknown" and posted_at:
            lowered = posted_at.lower()
            if "today" in lowered or "just posted" in lowered or "new" == lowered.strip():
                return self.run_date.isoformat(), "SerpAPI", 0
            if "yesterday" in lowered:
                parsed = self.run_date - timedelta(days=1)
                return parsed.isoformat(), "SerpAPI", 1
        return posted_date, "SerpAPI" if posted_date != "Unknown" else "unknown", age_days

    def is_stale(self, posted_age_days: Optional[int]) -> bool:
        """Return true when a known posted date is older than stale_days."""
        return posted_age_days is not None and posted_age_days > self.stale_days

    def is_active_page(self, html: str, status_code: Optional[int] = None) -> Tuple[bool, str]:
        """Detect obvious closed, expired, removed, or unavailable job pages."""
        if status_code is not None and status_code >= 400:
            return False, f"HTTP {status_code}"

        text = self.html_to_text(html).lower()
        for term in self.INACTIVE_TERMS:
            if term in text:
                return False, f"inactive term: {term}"

        return True, "active or unknown"

    @staticmethod
    def html_to_text(html: str) -> str:
        """Convert HTML to compact visible text."""
        soup = BeautifulSoup(html or "", "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()

    def _parse_relative_date(self, text: str) -> Optional[date]:
        match = self.RELATIVE_PATTERN.search(text)
        if not match:
            return None

        amount = int(match.group(1))
        unit = match.group(2).lower()

        if unit.startswith("minute") or unit.startswith("hour"):
            return self.run_date
        if unit.startswith("day"):
            return self.run_date - timedelta(days=amount)
        if unit.startswith("week"):
            return self.run_date - timedelta(days=amount * 7)
        if unit.startswith("month"):
            return self.run_date - timedelta(days=amount * 30)
        return None

    @staticmethod
    def _parse_absolute_date(value: str) -> Optional[date]:
        formats = ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d")
        for fmt in formats:
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
        return None

    def _parse_json_ld_date(self, html: str) -> Optional[date]:
        soup = BeautifulSoup(html or "", "html.parser")
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                data = json.loads(script.string or "")
            except json.JSONDecodeError:
                continue
            parsed = self._extract_date_posted(data)
            if parsed:
                return parsed
        return None

    def _extract_date_posted(self, data) -> Optional[date]:
        if isinstance(data, list):
            for item in data:
                parsed = self._extract_date_posted(item)
                if parsed:
                    return parsed
        if not isinstance(data, dict):
            return None
        if "datePosted" in data:
            return self._parse_absolute_date(str(data["datePosted"])[:10])
        for value in data.values():
            parsed = self._extract_date_posted(value)
            if parsed:
                return parsed
        return None
