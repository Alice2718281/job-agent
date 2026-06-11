# Daily Job Agent

Daily Job Agent searches Google Jobs through SerpAPI, keeps only jobs that pass your editable requirements, scores the eligible jobs with OpenAI, and sends the best unsent matches to Slack.

## Current Shape

```text
1. Search Google Jobs from job_preferences.json
2. Deduplicate by company + title + location
3. Skip jobs already stored in SQLite
4. Enrich each new job with an official/company apply link when possible
5. Normalize display fields
6. Apply one preference gate
7. Score only eligible jobs with OpenAI
8. Send top unsent jobs that still pass the current preference gate
```

The important rule: hard requirements are handled before AI scoring. OpenAI ranks only jobs that already passed your requirements.

## Editable Preferences

Update [job_preferences.json](job_preferences.json) to change your search and filters.

### Search

`search.title_keywords` controls exact title searches. 

`search.skill_queries` catches roles whose title is less direct but whose text matches your stack.

`search.locations` controls the Google Jobs search locations. 

### Requirements

`requirements.allowed_location_terms` defines the Area terms that pass the location gate. Add locations here when you want to expand the allowed area.

`requirements.max_years_experience`: If the job text clearly requires more than 4 years, it fails.

`requirements.baseline_salary`: If a salary range is visible, the job passes when the high end is at least $X amount. Unknown salary does not fail automatically.

`requirements.fail_on_no_sponsorship` is currently `true`. Jobs fail if the posting says the company will not sponsor, cannot sponsor, requires work authorization without sponsorship, or similar wording.

`requirements.require_official_or_company_link` is currently `true`. Jobs fail if the only apply link is a known third-party board or reposting site. Official ATS links and company-domain links pass.

`requirements.banned_seniority_terms` blocks roles that are likely too senior, such as Staff, Principal, Senior, Sr, Lead, Director, Head, VP, Vice President, and Executive.

## Setup

Install dependencies:

```bash
cd daily-job-agent
pip install -r requirements.txt
```

Create `.env` with:

```text
SERPAPI_API_KEY=your_serpapi_key
OPENAI_API_KEY=your_openai_key
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
DATABASE_PATH=./job_results.db
OPENAI_MODEL=gpt-4o-mini
NORMALIZER_MODEL=gpt-4o-mini
MAX_SERPAPI_SEARCHES_PER_RUN=18
MAX_ENRICHMENT_SEARCHES_PER_RUN=15
SERPAPI_RESULTS_PER_QUERY=10
RECENT_POSTING_HOURS=48
JOBS_TO_SCORE_LIMIT=15
MIN_REPORT_SCORE=6
```

`MIN_REPORT_SCORE` can also be set in `job_preferences.json` under `scoring.minimum_report_score`. The environment variable wins if both are present.

## Run

Dry run without Slack:

```bash
python main.py --dry-run
```

Send Slack report:

```bash
python main.py
```

Send fewer jobs:

```bash
python main.py --jobs 5
```

## Main Files

```text
daily-job-agent/
├── job_preferences.json   # Editable search and filter requirements
├── main.py                # Pipeline orchestration
├── job_search.py          # SerpAPI Google Jobs search
├── hard_filters.py        # Single preference gate
├── official_link_enricher.py
├── normalizer.py
├── scorer.py              # OpenAI ranking for eligible jobs
├── slack_notifier.py
├── database.py
└── job_results.db
```

## Notes

- Google Jobs may return aggregator links. The enricher tries to find an official/company link. If it cannot, the preference gate rejects known third-party-only results by default.
- Existing scored jobs in SQLite are checked against the current preference gate before Slack output, so changing `job_preferences.json` affects stored unsent jobs too.
- Unknown salary and unknown YOE are allowed through unless the posting provides clear evidence that violates your requirements.
