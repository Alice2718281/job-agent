"""LLM-assisted normalization for display fields."""
import json
import logging
from typing import Dict

from openai import OpenAI

logger = logging.getLogger(__name__)


class JobNormalizer:
    """Normalize noisy job fields into standard English display values."""

    PROMPT_TEMPLATE = """
Normalize the following job fields for display. Do not invent missing data.
Use English. If a field cannot be confidently normalized, keep the original value
but add "(original)" at the end. Use "Unknown" if empty.

Rules:
- salary format example: "$140K-$210K/year"; use "Unknown" if no salary is stated.
- work_style must be one of: Full-time, Hybrid, Remote, On-site, Unknown.
- location should be clean English when possible.
- posted_date should remain as provided; do not invent a date.

Input:
Title: {title}
Company: {company}
Location: {location}
Salary: {salary}
Work style / schedule: {schedule_type}
Posted date: {posted_date}
Link status: {link_status}
Job text excerpt:
{description_excerpt}

Return only JSON:
{{
  "title": "...",
  "company": "...",
  "location": "...",
  "salary": "...",
  "work_style": "...",
  "posted_date": "..."
}}
"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def normalize_job(self, job: Dict) -> Dict:
        """Normalize display fields using a cheap LLM model with safe fallback."""
        prompt = self.PROMPT_TEMPLATE.format(
            title=job.get("title", ""),
            company=job.get("company", ""),
            location=job.get("location", ""),
            salary=job.get("salary", ""),
            schedule_type=job.get("schedule_type", ""),
            posted_date=job.get("posted_date", "") or job.get("posted_at", ""),
            link_status=job.get("link_status", ""),
            description_excerpt=str(job.get("description", ""))[:3000],
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You normalize job fields and return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=250,
            )
            result = json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            logger.error("Field normalization failed for %s at %s: %s", job.get("title"), job.get("company"), e)
            result = {}

        normalized = {**job}
        normalized.setdefault("field_sources", {})
        normalized["title"] = self._value(result.get("title"), job.get("title"))
        normalized["company"] = self._value(result.get("company"), job.get("company"))
        normalized["location"] = self._value(result.get("location"), job.get("location"))
        normalized["salary"] = self._value(result.get("salary"), job.get("salary") or "Unknown")
        normalized["schedule_type"] = self._normalize_work_style(
            result.get("work_style"),
            job.get("schedule_type"),
        )
        normalized["posted_date"] = self._value(
            result.get("posted_date"),
            job.get("posted_date") or job.get("posted_at") or "Unknown",
        )
        source = "LLM normalization" if result else "fallback normalization"
        for field in ("title", "company", "location", "salary", "schedule_type", "posted_date"):
            normalized["field_sources"].setdefault(field, source)
        normalized["field_sources"]["normalization"] = source
        logger.info(
            "Normalized fields for %s at %s: location=%s, salary=%s, work_style=%s, source=%s",
            normalized.get("title"),
            normalized.get("company"),
            normalized.get("location"),
            normalized.get("salary"),
            normalized.get("schedule_type"),
            source,
        )
        return normalized

    @staticmethod
    def _value(value, fallback) -> str:
        value = str(value or "").strip()
        if value:
            return value
        fallback = str(fallback or "").strip()
        return fallback or "Unknown"

    @staticmethod
    def _normalize_work_style(value, fallback) -> str:
        allowed = {"Full-time", "Hybrid", "Remote", "On-site", "Unknown"}
        text = str(value or "").strip()
        if text in allowed:
            return text
        fallback_text = str(fallback or "").lower()
        if "hybrid" in fallback_text:
            return "Hybrid"
        if "remote" in fallback_text:
            return "Remote"
        if "on-site" in fallback_text or "onsite" in fallback_text or "in office" in fallback_text:
            return "On-site"
        if "full" in fallback_text:
            return "Full-time"
        return "Unknown"
