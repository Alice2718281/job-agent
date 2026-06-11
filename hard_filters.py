"""Single preference gate for jobs before AI scoring and Slack output."""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("job_preferences.json")


class JobPreferenceFilter:
    """Validate jobs against editable user requirements."""

    REMOTE_PATTERN = re.compile(
        r"\b(remote|anywhere|work from home|wfh)\b",
        re.IGNORECASE,
    )

    YOE_PATTERNS = [
        re.compile(r"\b(\d+)\s*[-–]\s*(\d+)\s*(?:years?|yrs?)", re.IGNORECASE),
        re.compile(r"\b(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+)?experience", re.IGNORECASE),
        re.compile(r"\b(?:at least|minimum|min\.?)\s+(\d+)\s*(?:years?|yrs?)", re.IGNORECASE),
        re.compile(r"\b(\d+)\s*(?:years?|yrs?)\s+of\s+experience\s+required", re.IGNORECASE),
    ]

    NO_SPONSORSHIP_PATTERNS = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"not able to (?:consider|sponsor)",
            r"unable to (?:consider|sponsor|provide)",
            r"cannot (?:consider|sponsor|provide)",
            r"will not (?:sponsor|provide)",
            r"do not (?:sponsor|provide)",
            r"does not (?:sponsor|provide)",
            r"no visa sponsorship",
            r"without sponsorship",
            r"must be authorized to work.*without.*sponsorship",
            r"must not require.*sponsorship",
            r"(?:now|currently) or in the future.*(?:sponsor|sponsorship|visa)",
        )
    ]

    THIRD_PARTY_DOMAINS = (
        "bebee.com",
        "builtinnyc.com",
        "dice.com",
        "indeed.com",
        "jobilize.com",
        "jooble.org",
        "linkedin.com",
        "monster.com",
        "simplyhired.com",
        "talent.com",
        "therundown.ai",
        "whatjobs.com",
        "ziprecruiter.com",
    )

    OFFICIAL_ATS_DOMAINS = (
        "ashbyhq.com",
        "bamboohr.com",
        "greenhouse.io",
        "icims.com",
        "jobvite.com",
        "lever.co",
        "myworkdayjobs.com",
        "smartrecruiters.com",
        "workdayjobs.com",
    )

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = json.load(file)

        requirements = self.config.get("requirements", {})
        self.allowed_location_terms = self._normalize_terms(
            requirements.get("allowed_location_terms", [])
        )
        self.remote_us_terms = self._normalize_terms(
            requirements.get("remote_us_terms", [])
        )
        self.banned_seniority_terms = self._normalize_terms(
            requirements.get("banned_seniority_terms", [])
        )
        self.max_years_experience = int(requirements.get("max_years_experience", 4))
        self.baseline_salary = int(requirements.get("baseline_salary", X)) # replace by real baseline salary
        self.remote_requires_us = bool(requirements.get("remote_requires_us", True))
        self.fail_on_no_sponsorship = bool(requirements.get("fail_on_no_sponsorship", True))
        self.require_official_or_company_link = bool(
            requirements.get("require_official_or_company_link", True)
        )

    def passes(self, job: Dict) -> Tuple[bool, str, str]:
        """Return whether a job passes requirements, plus reason/evidence."""
        if not job.get("is_active", True):
            return False, "inactive_job", str(job.get("inactive_reason") or "inactive/expired")

        posted_age_days = job.get("posted_age_days")
        if posted_age_days is not None:
            try:
                age = int(posted_age_days)
                if age > 183:
                    return False, "posted_date_older_than_6_months", f"{age} days"
            except (TypeError, ValueError):
                pass

        location = str(job.get("location") or "Unknown")
        if not self._is_allowed_location(location, job):
            return False, "location_outside_target_area", location

        title = str(job.get("title") or "")
        seniority = self._find_banned_seniority(title)
        if seniority:
            return False, "banned_seniority_title", seniority

        text = self._best_available_text(job)
        yoe = self._extract_required_yoe(text)
        if yoe is None:
            logger.info(
                "YOE unknown for %s - %s; not excluding for YOE",
                job.get("company"),
                job.get("title"),
            )
        elif yoe > self.max_years_experience:
            return False, "yoe_above_limit", f"{yoe:g}+ years"

        salary_values = self._extract_salary_values(text)
        if salary_values and max(salary_values) < self.baseline_salary:
            return False, "salary_below_baseline", f"highest detected salary ${max(salary_values):,}"

        if self.fail_on_no_sponsorship:
            no_sponsorship = self._find_no_sponsorship(text)
            if no_sponsorship:
                return False, "no_sponsorship", no_sponsorship

        if self.require_official_or_company_link and not self._has_acceptable_apply_link(job):
            return False, "third_party_link_only", str(job.get("apply_link") or "")

        return True, "passed", ""

    def _is_allowed_location(self, location: str, job: Dict) -> bool:
        location_text = location.lower()
        combined_text = self._best_available_text(job).lower()

        if any(term in location_text for term in self.allowed_location_terms):
            return True

        if self.REMOTE_PATTERN.search(location_text):
            if not self.remote_requires_us:
                return True
            return any(term in location_text or term in combined_text for term in self.remote_us_terms)

        return False

    def _find_banned_seniority(self, title: str) -> str:
        lowered = title.lower()
        for term in self.banned_seniority_terms:
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                return term
        return ""

    def _find_no_sponsorship(self, text: str) -> str:
        for pattern in self.NO_SPONSORSHIP_PATTERNS:
            match = pattern.search(text)
            if match:
                return self._trim_text(match.group(0), 180)
        return ""

    def _has_acceptable_apply_link(self, job: Dict) -> bool:
        link_status = str(job.get("link_status") or "").lower()
        if link_status == "official link found":
            return True

        apply_link = str(job.get("apply_link") or "")
        if not apply_link:
            return False

        domain = urlparse(apply_link).netloc.lower()
        if any(ats_domain in domain for ats_domain in self.OFFICIAL_ATS_DOMAINS):
            return True
        if any(third_party in domain for third_party in self.THIRD_PARTY_DOMAINS):
            return False

        company_tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", str(job.get("company") or "").lower())
            if len(token) >= 3 and token not in {"inc", "llc", "corp", "corporation", "company"}
        ]
        return any(token in domain for token in company_tokens)

    @classmethod
    def _extract_salary_values(cls, text: str) -> List[int]:
        values = []
        patterns = (
            r"\$\s*(\d{2,3}(?:,\d{3})+)",
            r"\$\s*(\d{2,3}(?:\.\d+)?)\s*k\b",
        )

        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                if not cls._has_salary_context(text, match.start(), match.end()):
                    continue
                raw = match.group(1).replace(",", "")
                value = float(raw)
                if value < 1000:
                    value *= 1000
                value = int(value)
                if 50000 <= value <= 400000:
                    values.append(value)
        return values

    @staticmethod
    def _has_salary_context(text: str, start: int, end: int) -> bool:
        before = max(0, start - 100)
        after = min(len(text), end + 100)
        context = text[before:after].lower()
        positive_terms = (
            "annual", "base", "base pay", "base salary", "compensation",
            "expected pay", "pay range", "salary", "salary range", "wage",
        )
        negative_terms = (
            "valuation", "valued at", "funding", "raised", "revenue",
            "market cap", "assets under management", "aum", "worth",
        )
        return any(term in context for term in positive_terms) and not any(
            term in context for term in negative_terms
        )

    @staticmethod
    def _normalize_terms(terms: List[str]) -> List[str]:
        return [str(term).lower().strip() for term in terms if str(term).strip()]

    @staticmethod
    def _trim_text(text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0].strip() + "..."

    # Backward-compatible method name for the small validation script.
    def passes_final_hard_filters(self, job: Dict) -> Tuple[bool, str, str]:
        return self.passes(job)

    def _extract_required_yoe(self, text: str) -> Optional[int]:
        if not text:
            return None

        for pattern in self.YOE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            values = [int(value) for value in match.groups() if value and value.isdigit()]
            if values:
                return max(values) if len(values) > 1 else values[0]

        return None

    @staticmethod
    def _best_available_text(job: Dict) -> str:
        return " ".join(
            str(job.get(field) or "")
            for field in (
                "description",
                "title",
                "location",
                "salary",
                "schedule_type",
                "link_status_reason",
            )
        )
