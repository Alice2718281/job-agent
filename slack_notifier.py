"""Slack notification module for sending daily job reports."""
import logging
from datetime import datetime
from typing import Dict, List
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)


class SlackNotifier:
    """Send job report messages through a Slack incoming webhook."""

    def __init__(self, webhook_url: str, request_timeout: int = 20):
        """Initialize Slack notifier with an incoming webhook URL."""
        self.webhook_url = webhook_url
        self.request_timeout = request_timeout

    def send_job_report(self, jobs: List[Dict]) -> bool:
        """Send a Slack report with top job opportunities."""
        if not jobs:
            logger.warning("No jobs to send in Slack report")
            return False

        payload = self._build_payload(jobs)

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            logger.info("Slack report sent successfully with %s jobs", len(jobs))
            return True
        except requests.exceptions.RequestException as e:
            logger.error("Error sending Slack report: %s", self._safe_request_error(e))
            return False

    def _build_payload(self, jobs: List[Dict]) -> Dict:
        """Build a Slack Block Kit payload."""
        sent_at = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p %Z")
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Daily Job Opportunities Report",
                    "emoji": False,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Found:* {len(jobs)} matching job opportunities\n"
                        f"*Sent at:* {sent_at}"
                    ),
                },
            },
            {"type": "divider"},
        ]

        for index, job in enumerate(jobs, 1):
            blocks.extend(self._build_job_blocks(index, job))

        return {
            "text": f"Daily Job Opportunities Report: {len(jobs)} jobs",
            "blocks": blocks[:50],
        }

    def _build_job_blocks(self, index: int, job: Dict) -> List[Dict]:
        """Build Slack blocks for one job."""
        title = self._escape(job.get("title", "N/A"))
        company = self._escape(job.get("company", "N/A"))
        location = self._escape(job.get("location", "N/A"))
        apply_link = job.get("apply_link") or ""
        score = self._escape(str(job.get("score", "N/A")))
        match_summary = self._escape(job.get("match_summary", "N/A"))
        main_gap = self._escape(job.get("main_gap", "None or unknown"))
        recommended_resume = self._escape(job.get("recommended_resume", "DS"))
        sponsor_risk = self._escape(job.get("sponsor_risk", "Unknown"))
        sponsorship_evidence = self._escape(job.get("sponsorship_evidence") or "")
        salary = self._escape(job.get("salary") or "Not listed")
        schedule_type = self._escape(job.get("schedule_type") or "Not listed")
        posted_date = self._escape(job.get("posted_date") or job.get("posted_at") or "Unknown")
        posted_age = self._format_posted_age(job.get("posted_age_days"))
        link_status = self._escape(job.get("link_status") or "Third-party link only")
        possible_link = job.get("possible_official_link") or ""
        original_link = job.get("original_apply_link") or ""

        title_line = f"*{index}. {title} at {company}*"
        if apply_link:
            title_line = f"*{index}. <{apply_link}|{title} at {company}>*"

        sponsorship_line = f"*Sponsor risk:* {sponsor_risk}"
        if sponsorship_evidence and sponsorship_evidence.lower() != "no mention":
            sponsorship_line += f" | *Sponsor evidence:* {sponsorship_evidence[:220]}"

        link_line = f"*Link status:* [{link_status}]"
        if possible_link:
            link_line += f" <{possible_link}|Possible official>"
        if possible_link and original_link:
            link_line += f" | <{original_link}|Original source>"

        return [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{title_line}\n"
                        f"*Location:* {location}\n"
                        f"*Score:* {score}/10 | *Resume:* {recommended_resume} | "
                        f"{sponsorship_line}\n"
                        f"*Posted:* {posted_date} | *Age:* {posted_age} | *Salary:* {salary} | "
                        f"*Work style:* {schedule_type}\n"
                        f"{link_line}\n"
                        f"*Why it matches:* {match_summary}\n"
                        f"*Main gap:* {main_gap}"
                    ),
                },
            },
            {"type": "divider"},
        ]

    @staticmethod
    def _escape(value: object) -> str:
        """Escape Slack mrkdwn control characters."""
        text = str(value)
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    @staticmethod
    def _format_posted_age(posted_age_days: object) -> str:
        if posted_age_days is None or posted_age_days == "":
            return "Unknown"
        try:
            days = int(posted_age_days)
        except (TypeError, ValueError):
            return "Unknown"
        if days == 0:
            return "Today"
        if days == 1:
            return "1 day"
        return f"{days} days"

    @staticmethod
    def _safe_request_error(error: requests.exceptions.RequestException) -> str:
        """Return request error text without exposing webhook details."""
        response = getattr(error, "response", None)
        if response is not None:
            return f"HTTP {response.status_code}: {response.text[:120]}"
        return error.__class__.__name__
