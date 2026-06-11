"""Job scoring module using OpenAI API."""
import json
import logging
import re
from typing import Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)


class JobScorer:
    """Score jobs using OpenAI API based on user criteria."""

    SCORING_PROMPT_TEMPLATE = """
You are a career advisor evaluating job opportunities for a candidate.
Score the job from 1 to 10 based on fit. Be selective: only strong opportunities
that are plausibly better than the candidate's current role should
score 7+. The job has already passed hard requirements for location, seniority,
years of experience, visible salary floor, sponsorship wording, active status, and
apply-link quality.

CANDIDATE PROFILE:
- Describe yourself here

JOB DETAILS:
Title: {title}
Company: {company}
Location: {location}
Posted: {posted_at}
Salary: {salary}
Schedule / flexibility: {schedule_type}
Apply source: {via}
Relevant job evidence:
{job_evidence}
Detected experience requirement: {experience_requirement}
Detected sponsorship evidence: {sponsorship_evidence}
Fit penalty signals:
{fit_penalty_signals}

EVALUATION CRITERIA:
- List Criteria here

Return only valid JSON in this exact shape:
{{
  "score": <integer 1-10>,
  "match_summary": "<why it matches in 1-2 sentences>",
  "main_gap": "<main reason this is not perfect>",
  "recommended_resume": "<DS or PM>",
  "sponsor_risk": "<low, medium, high, or no mention>",
  "reasoning": "<concise explanation of the score>"
}}
"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """Initialize scorer with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def score_job(self, job: Dict) -> Dict:
        """Score a job and return scoring details."""
        prompt = self.SCORING_PROMPT_TEMPLATE.format(
            title=job.get("title", ""),
            company=job.get("company", ""),
            location=job.get("location", ""),
            posted_at=job.get("posted_at", ""),
            salary=job.get("salary", ""),
            schedule_type=job.get("schedule_type", ""),
            via=job.get("via", ""),
            job_evidence=self._build_job_evidence(job),
            experience_requirement=self._extract_experience_requirement(
                job.get("description", "")
            ),
            sponsorship_evidence=self._extract_sponsorship_evidence(
                job.get("description", "")
            ),
            fit_penalty_signals=self._build_fit_penalty_signals(job),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You evaluate jobs and respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=450,
            )

            response_text = response.choices[0].message.content.strip()
            result = json.loads(response_text)
            return self._normalize_result(result, job)

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON response for %s at %s: %s",
                job.get("title"),
                job.get("company"),
                e,
            )
            return self._default_score()
        except Exception as e:
            logger.error(
                "Error scoring job '%s' at '%s': %s",
                job.get("title"),
                job.get("company"),
                e,
            )
            return self._default_score()

    def _normalize_result(self, result: Dict, job: Dict = None) -> Dict:
        """Validate and normalize the scorer response."""
        score = result.get("score", 5)
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 5
        score = max(1, min(10, score))

        recommended_resume = str(result.get("recommended_resume", "DS")).upper()
        if recommended_resume not in {"DS", "PM"}:
            recommended_resume = "DS"

        sponsor_risk = str(result.get("sponsor_risk", "no mention")).lower()
        if sponsor_risk not in {"low", "medium", "high", "no mention"}:
            sponsor_risk = "no mention"

        sponsorship_evidence = ""
        if job is not None:
            sponsorship_evidence = self._extract_sponsorship_evidence(
                job.get("description", "")
            )
            deterministic_risk = self._determine_sponsor_risk(sponsorship_evidence)
            if deterministic_risk in {"high", "no mention"}:
                sponsor_risk = deterministic_risk

        return {
            "score": score,
            "match_summary": str(result.get("match_summary", "")),
            "main_gap": str(result.get("main_gap", "")),
            "recommended_resume": recommended_resume,
            "sponsor_risk": sponsor_risk,
            "sponsorship_evidence": sponsorship_evidence,
            "reasoning": str(result.get("reasoning", "")),
        }

    def _build_job_evidence(self, job: Dict, max_chars: int = 2400) -> str:
        """Extract the highest-signal parts of a posting to reduce prompt tokens."""
        description = self._normalize_text(job.get("description", ""))
        if not description:
            return "No detailed description available."

        sections = {
            "Responsibilities": self._extract_section(
                description,
                [
                    "responsibilities",
                    "what you will do",
                    "what you'll do",
                    "role overview",
                    "the role",
                ],
            ),
            "Qualifications": self._extract_section(
                description,
                [
                    "qualifications",
                    "requirements",
                    "what we are looking for",
                    "what we're looking for",
                    "skills",
                    "experience",
                    "years of experience",
                    "minimum qualifications",
                    "basic qualifications",
                    "preferred qualifications",
                ],
            ),
            "Experience requirement": self._extract_experience_requirement(description),
            "Company / team": self._extract_section(
                description,
                [
                    "about",
                    "about us",
                    "about the company",
                    "who we are",
                    "our team",
                ],
            ),
            "Compensation / work style": self._keyword_snippets(
                description,
                [
                    "salary",
                    "base",
                    "compensation",
                    "hybrid",
                    "remote",
                    "visa",
                    "sponsor",
                    "sponsorship",
                    "h1b",
                    "entry level",
                    "years of experience",
                ],
            ),
            "Sponsorship": self._extract_sponsorship_evidence(description),
        }

        evidence_parts: List[str] = []
        for label, text in sections.items():
            if text:
                evidence_parts.append(f"{label}: {self._trim_text(text, 650)}")

        if not evidence_parts:
            evidence_parts.append(f"Summary: {self._trim_text(description, max_chars)}")

        evidence = "\n".join(evidence_parts)
        return self._trim_text(evidence, max_chars)

    def _extract_experience_requirement(self, text: str, max_chars: int = 700) -> str:
        """Extract explicit years-of-experience requirements for fit scoring."""
        description = self._normalize_text(text)
        if not description:
            return "Not found"

        patterns = [
            r"[^.\n]*(?:\d+\+?|\d+\s*-\s*\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+)?experience[^.\n]*",
            r"[^.\n]*(?:experience\s+of|experience with|experienced in)[^.\n]*(?:\d+\+?|\d+\s*-\s*\d+)\s*(?:years?|yrs?)[^.\n]*",
            r"[^.\n]*(?:entry[- ]level|new grad|early career|junior)[^.\n]*",
            r"[^.\n]*(?:minimum qualifications|basic qualifications|preferred qualifications)[^.\n]*(?:\d+\+?|\d+\s*-\s*\d+)\s*(?:years?|yrs?)[^.\n]*",
        ]

        matches: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, description, flags=re.IGNORECASE):
                snippet = match.group(0).strip(" -:;")
                if snippet and snippet not in matches:
                    matches.append(snippet)
                if len(" ".join(matches)) >= max_chars:
                    break
            if len(" ".join(matches)) >= max_chars:
                break

        if not matches:
            fallback = self._keyword_snippets(
                description,
                ["years of experience", "year of experience", "yrs", "entry level", "junior"],
                max_chars=max_chars,
            )
            return fallback or "Not found"

        return self._trim_text(" ".join(matches), max_chars)

    def _extract_sponsorship_evidence(self, text: str, max_chars: int = 900) -> str:
        """Extract exact visa/sponsorship wording from a job description."""
        description = self._normalize_text(text)
        if not description:
            return "None found"

        evidence = self._keyword_snippets(
            description,
            [
                "sponsor",
                "sponsorship",
                "visa",
                "h-1b",
                "h1b",
                "opt",
                "cpt",
                "e-verify",
                "everify",
                "work authorization",
                "employment authorization",
                "authorized to work",
                "require authorization",
                "requires authorization",
                "require visa",
                "require sponsorship",
                "now or in the future",
                "future sponsorship",
            ],
            max_chars=max_chars,
        )
        if not evidence:
            return "None found"

        immigration_terms = (
            "sponsor", "sponsorship", "visa", "h-1b", "h1b", "opt", "cpt",
            "e-verify", "everify", "work authorization", "employment authorization",
            "authorized to work", "require authorization", "require visa",
            "require sponsorship", "now or in the future", "future sponsorship",
        )
        if not any(term in evidence.lower() for term in immigration_terms):
            return "None found"
        return evidence

    @staticmethod
    def _determine_sponsor_risk(sponsorship_evidence: str) -> str:
        """Deterministically handle obvious no-sponsorship or no-mention cases."""
        evidence = sponsorship_evidence.lower().strip()
        if not evidence or evidence in {"no mention", "none found"}:
            return "no mention"

        no_sponsor_patterns = [
            r"not able to (?:consider|sponsor)",
            r"unable to (?:consider|sponsor|provide)",
            r"cannot (?:consider|sponsor|provide)",
            r"will not (?:sponsor|provide)",
            r"do not (?:sponsor|provide)",
            r"does not (?:sponsor|provide)",
            r"no visa sponsorship",
            r"without sponsorship",
            r"now or in the future.*(?:sponsor|sponsorship|visa)",
            r"currently or in the future.*(?:sponsor|sponsorship|visa)",
            r"must be authorized to work.*without.*sponsorship",
            r"must not require.*sponsorship",
        ]
        if any(re.search(pattern, evidence) for pattern in no_sponsor_patterns):
            return "high"

        return "mentioned"

    def _extract_section(self, text: str, headings: List[str], max_chars: int = 1200) -> str:
        """Extract a section by heading, falling back to keyword snippets."""
        heading_pattern = "|".join(
            re.escape(heading)
            for heading in sorted(headings, key=len, reverse=True)
        )
        next_heading_pattern = (
            r"responsibilities|what you will do|what you'll do|role overview|the role|"
            r"qualifications|requirements|what we are looking for|what we're looking for|"
            r"skills|experience|about|about us|about the company|who we are|our team|"
            r"benefits|compensation|salary|equal opportunity"
        )
        pattern = re.compile(
            rf"(?:^|\n)\s*(?:{heading_pattern})\s*:?\s*(.*?)(?=\n\s*(?:{next_heading_pattern})\s*:?\s|\Z)",
            re.IGNORECASE | re.DOTALL,
        )

        match = pattern.search(text)
        if match:
            return self._trim_text(match.group(1), max_chars)

        return self._keyword_snippets(text, headings, max_chars=max_chars)

    def _keyword_snippets(self, text: str, keywords: List[str], max_chars: int = 900) -> str:
        """Collect compact sentences containing important keywords."""
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        snippets = []
        lowered_keywords = [keyword.lower() for keyword in keywords]

        for sentence in sentences:
            normalized = sentence.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if any(keyword in lowered for keyword in lowered_keywords):
                snippets.append(normalized)
            if len(" ".join(snippets)) >= max_chars:
                break

        return self._trim_text(" ".join(snippets), max_chars)

    def _build_fit_penalty_signals(self, job: Dict) -> str:
        """Build deterministic negative-fit notes for scoring, without filtering the job."""
        signals = []
        location = str(job.get("location") or "")
        title = str(job.get("title") or "")
        description = self._normalize_text(str(job.get("description") or ""))

        location_signal = self._location_penalty_signal(location, description)
        if location_signal:
            signals.append(location_signal)

        seniority_signal = self._seniority_penalty_signal(title)
        if seniority_signal:
            signals.append(seniority_signal)

        yoe_signal = self._yoe_penalty_signal(description)
        if yoe_signal:
            signals.append(yoe_signal)

        if not signals:
            return "None detected."
        return "\n".join(f"- {signal}" for signal in signals)

    @staticmethod
    def _location_penalty_signal(location: str, description: str) -> str:
        location_text = f"{location} {description[:1200]}".lower()
        location_only = location.lower()

        nyc_terms = (
            "new york", "nyc", "new york city", "manhattan", "brooklyn",
            "queens", "bronx", "staten island", "jersey city", "hoboken",
            "newark", "ny metro", "nyc metro", "new york metro",
        )
        if any(term in location_text for term in nyc_terms):
            return ""

        is_remote = "remote" in location_text
        remote_us_terms = (
            "remote us", "remote, us", "remote united states", "remote, united states",
            "remote within the united states", "remote anywhere in the united states",
            "us remote", "u.s. remote", "united states remote",
        )
        if is_remote and any(term in location_text for term in remote_us_terms):
            return ""

        non_target_locations = (
            "mclean", "washington", "dc", "boston", "philadelphia",
            "san francisco", "california", "ca", "virginia", "va",
            "massachusetts", "ma", "pennsylvania", "pa",
        )
        if any(term in location_only for term in non_target_locations):
            return f"Location penalty: outside NYC metro / NJ / Remote US ({location or 'Unknown'})."

        if location and location.strip().lower() not in {"unknown", "not found"}:
            return f"Location review: not clearly NYC metro / NJ / Remote US ({location})."
        return ""

    @staticmethod
    def _seniority_penalty_signal(title: str) -> str:
        pattern = (
            r"\b(staff|principal|lead|director|head|vp|vice president|"
            r"executive)\b"
        )
        match = re.search(pattern, title or "", flags=re.IGNORECASE)
        if not match:
            return ""
        return f"Seniority penalty: title contains '{match.group(0)}'."

    @staticmethod
    def _yoe_penalty_signal(description: str) -> str:
        if not description:
            return ""

        patterns = [
            r"[^.\n]*(?:at least|minimum|min\.?|requires?|required|must have|need)\s+(\d{1,2})\+?\s*(?:years?|yrs?)[^.\n]*",
            r"[^.\n]*(\d{1,2})\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+)?experience\s+(?:required|minimum|needed|in|with)[^.\n]*",
            r"[^.\n]*(\d{1,2})\+\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+)?experience[^.\n]*",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, description, flags=re.IGNORECASE):
                years = int(match.group(1))
                if years >= 5:
                    evidence = match.group(0).strip(" -:;")
                    return f"YOE penalty: requires {years}+ years. Evidence: {evidence}"
        return ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace while preserving loose paragraph boundaries."""
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _trim_text(text: str, max_chars: int) -> str:
        """Trim text to a character budget without cutting mid-word when possible."""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        trimmed = text[:max_chars].rsplit(" ", 1)[0].strip()
        return trimmed + "..."

    @staticmethod
    def _default_score() -> Dict:
        """Return a default score when scoring fails."""
        return {
            "score": 5,
            "match_summary": "Unable to score automatically. Review manually before applying.",
            "main_gap": "Scoring failed.",
            "recommended_resume": "DS",
            "sponsor_risk": "no mention",
            "sponsorship_evidence": "",
            "reasoning": "OpenAI scoring error.",
        }
