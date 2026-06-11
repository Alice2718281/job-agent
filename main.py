"""Main module orchestrating the daily job agent workflow."""
import os
import sys
import logging
from typing import List
from dotenv import load_dotenv, find_dotenv

from job_search import JobSearcher
from normalizer import JobNormalizer
from official_link_enricher import OfficialLinkEnricher
from hard_filters import JobPreferenceFilter
from scorer import JobScorer
from slack_notifier import SlackNotifier
from database import JobDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DailyJobAgent:
    """Main agent that orchestrates job search, scoring, and reporting."""

    def __init__(self):
        """Initialize the agent with API credentials."""
        # Load environment variables (search parent directories if needed)
        load_dotenv(find_dotenv())
        
        # Get API keys and credentials
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.db_path = os.getenv("DATABASE_PATH", "./job_results.db")
        self.max_serpapi_searches = self._get_int_env("MAX_SERPAPI_SEARCHES_PER_RUN", 18)
        self.results_per_query = self._get_int_env("SERPAPI_RESULTS_PER_QUERY", 10)
        self.recent_hours = self._get_int_env("RECENT_POSTING_HOURS", 48)
        self.jobs_to_score_limit = self._get_int_env("JOBS_TO_SCORE_LIMIT", 15)
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.normalizer_model = os.getenv("NORMALIZER_MODEL", "gpt-4o-mini")
        self.max_enrichment_searches = self._get_int_env("MAX_ENRICHMENT_SEARCHES_PER_RUN", 15)
        self.preference_filter = JobPreferenceFilter()
        default_min_report_score = int(
            self.preference_filter.config.get("scoring", {}).get("minimum_report_score", 6)
        )
        self.min_report_score = self._get_int_env("MIN_REPORT_SCORE", default_min_report_score)
        
        # Validate credentials
        self._validate_credentials()
        
        # Initialize components
        self.job_searcher = JobSearcher(
            self.serpapi_key,
            max_api_calls=self.max_serpapi_searches,
            results_per_query=self.results_per_query,
            recent_hours=self.recent_hours,
        )
        self.enricher = OfficialLinkEnricher(
            self.serpapi_key,
            max_enrichment_searches=self.max_enrichment_searches,
        )
        self.normalizer = JobNormalizer(self.openai_key, model=self.normalizer_model)
        self.job_scorer = JobScorer(self.openai_key, model=self.openai_model)
        self.slack_notifier = SlackNotifier(self.slack_webhook_url)
        self.database = JobDatabase(self.db_path)

    def _validate_credentials(self):
        """Validate that all required credentials are provided."""
        missing = []
        
        if not self.serpapi_key:
            missing.append("SERPAPI_API_KEY")
        if not self.openai_key:
            missing.append("OPENAI_API_KEY")
        if not self.slack_webhook_url:
            missing.append("SLACK_WEBHOOK_URL")
        
        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            logger.error("Please create a .env file based on .env.example")
            sys.exit(1)

    @staticmethod
    def _get_int_env(name: str, default: int) -> int:
        """Read an integer environment variable with a safe fallback."""
        value = os.getenv(name)
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning("Invalid %s=%r. Using default %s.", name, value, default)
            return default

    def run(self, num_jobs_to_send: int = 10, dry_run: bool = False):
        """Run the complete job search, scoring, and Slack notification workflow."""
        logger.info("=" * 60)
        logger.info("Starting Daily Job Agent")
        logger.info("=" * 60)
        
        try:
            # Step 1: Search for jobs
            logger.info("Step 1: Searching for jobs...")
            jobs = self.job_searcher.search_jobs()
            logger.info(f"Found {len(jobs)} potential jobs")
            
            if not jobs:
                logger.warning("No jobs found in search")
                return False
            
            # Step 2: Filter new jobs (not in database)
            logger.info("Step 2: Filtering new jobs...")
            new_jobs = self._filter_new_jobs(jobs)
            logger.info(f"Found {len(new_jobs)} new jobs (not previously sent)")
            
            if not new_jobs:
                logger.info("No new jobs to process")
                return False

            # Step 3: Enrich, normalize, and apply one preference gate before AI scoring
            logger.info("Step 3: Enriching jobs and applying preference requirements...")
            eligible_jobs = self._prepare_eligible_jobs(new_jobs)
            logger.info("Kept %s jobs after preference requirements", len(eligible_jobs))

            if not eligible_jobs:
                logger.warning("No jobs passed preference requirements")
                return False

            jobs_for_ai = eligible_jobs[: self.jobs_to_score_limit]

            # Step 4: Score jobs
            logger.info("Step 4: Scoring eligible jobs with AI...")
            scored_jobs = self._score_jobs(jobs_for_ai, store=not dry_run)
            logger.info(f"Scored {len(scored_jobs)} jobs")
            
            # Step 5: Get top jobs to send
            logger.info(f"Step 5: Selecting top {num_jobs_to_send} jobs...")
            if dry_run:
                top_jobs = [
                    job for job in scored_jobs
                    if int(job.get("score", 0)) >= self.min_report_score
                ]
                top_jobs = sorted(top_jobs, key=lambda job: job.get("score", 0), reverse=True)
                top_jobs = top_jobs[:num_jobs_to_send]
            else:
                unsent_candidates = self.database.get_unsent_scored_jobs(
                    limit=num_jobs_to_send * 5,
                    min_score=self.min_report_score,
                )
                top_jobs = self._filter_report_candidates(unsent_candidates)[:num_jobs_to_send]
            logger.info(f"Selected {len(top_jobs)} top jobs to send")
            
            if not top_jobs:
                logger.warning("No top scored jobs available")
                return False
            
            # Step 6: Send Slack report
            if dry_run:
                logger.info("DRY RUN: Would send Slack report with following jobs:")
                for job in top_jobs:
                    logger.info(f"  - {job['title']} at {job['company']} ({job['score']}/10)")
            else:
                logger.info("Step 6: Sending Slack report...")
                success = self.slack_notifier.send_job_report(top_jobs)
                
                if success:
                    # Mark jobs as sent
                    for job in top_jobs:
                        self.database.mark_job_sent(job['id'])
                    logger.info(f"Successfully sent Slack report with {len(top_jobs)} jobs")
                    return True
                else:
                    logger.error("Failed to send Slack report")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during agent run: {e}", exc_info=True)
            return False
        finally:
            self.database.close()

    def _filter_new_jobs(self, jobs: List[dict]) -> List[dict]:
        """Filter out jobs that already exist in the database."""
        new_jobs = []
        existing_jobs = []
        
        for job in jobs:
            if not self.database.job_exists(
                job.get("company", ""),
                job.get("title", ""),
                job.get("location", "")
            ):
                new_jobs.append(job)
            else:
                existing_jobs.append(job)

        if existing_jobs:
            logger.info(
                "Skipped %s jobs already in database; examples: %s",
                len(existing_jobs),
                ", ".join(
                    f"{job.get('company', '')} - {job.get('title', '')}"
                    for job in existing_jobs[:5]
                ),
            )
        
        return new_jobs

    def _prepare_eligible_jobs(self, jobs: List[dict]) -> List[dict]:
        """Find official links, normalize fields, and keep only jobs that pass preferences."""
        kept_jobs = []

        for i, job in enumerate(jobs, 1):
            logger.info(
                "  Enriching job %s/%s: %s at %s",
                i,
                len(jobs),
                job.get("title"),
                job.get("company"),
            )
            enriched = self.enricher.enrich_job(job)
            normalized = self.normalizer.normalize_job(enriched)

            passes, reason, evidence = self.preference_filter.passes(normalized)
            if not passes:
                logger.info(
                    "Filtered out job: %s at %s (%s: %s)",
                    normalized.get("title"),
                    normalized.get("company"),
                    reason,
                    evidence,
                )
                continue

            kept_jobs.append(normalized)

        return kept_jobs

    def _filter_report_candidates(self, jobs: List[dict]) -> List[dict]:
        """Apply current preferences to stored scored jobs before Slack output."""
        kept_jobs = []
        for job in jobs:
            passes, reason, evidence = self.preference_filter.passes(job)
            if passes:
                kept_jobs.append(job)
            else:
                logger.info(
                    "Skipped stored scored job before report: %s at %s (%s: %s)",
                    job.get("title"),
                    job.get("company"),
                    reason,
                    evidence,
                )
        return kept_jobs

    def _score_jobs(self, jobs: List[dict], store: bool = True) -> List[dict]:
        """Score jobs and optionally store them in the database."""
        scored_jobs = []
        
        for i, job in enumerate(jobs, 1):
            logger.info(f"  Scoring job {i}/{len(jobs)}: {job.get('title')} at {job.get('company')}")

            job_id = None
            if store:
                job_id = self.database.add_job(
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    location=job.get("location", ""),
                    apply_link=job.get("apply_link", ""),
                    description=job.get("description", ""),
                    posted_date=job.get("posted_date") or job.get("posted_at", ""),
                    salary=job.get("salary", ""),
                    schedule_type=job.get("schedule_type", ""),
                    original_apply_link=job.get("original_apply_link", ""),
                    official_apply_link=job.get("official_apply_link", ""),
                    possible_official_link=job.get("possible_official_link", ""),
                    link_status=job.get("link_status", "Third-party link only"),
                    posted_age_days=job.get("posted_age_days"),
                    field_sources=job.get("field_sources", {})
                )

            # Score the job
            score_result = self.job_scorer.score_job(job)

            if store and job_id is not None:
                self.database.add_scored_job(
                    job_id=job_id,
                    score=score_result.get("score", 5),
                    reasoning=score_result.get("reasoning", ""),
                    match_summary=score_result.get("match_summary", ""),
                    main_gap=score_result.get("main_gap", ""),
                    recommended_resume=score_result.get("recommended_resume", "DS"),
                    sponsor_risk=score_result.get("sponsor_risk", "no mention"),
                    sponsorship_evidence=score_result.get("sponsorship_evidence", "")
                )
            
            scored_jobs.append({
                "id": job_id,
                **job,
                **score_result
            })
        
        return scored_jobs

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily Job Agent")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending Slack report"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=10,
        help="Number of jobs to send in Slack report (default: 10)"
    )
    
    args = parser.parse_args()
    
    agent = DailyJobAgent()
    success = agent.run(num_jobs_to_send=args.jobs, dry_run=args.dry_run)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
