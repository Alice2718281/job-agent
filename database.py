"""Database module for storing and managing job postings."""
import json
import sqlite3
from typing import List, Dict


class JobDatabase:
    """SQLite database for job storage and deduplication."""

    def __init__(self, db_path: str = "job_results.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Jobs table to store all discovered jobs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                location TEXT NOT NULL,
                apply_link TEXT UNIQUE NOT NULL,
                description TEXT,
                posted_date TEXT,
                salary TEXT,
                schedule_type TEXT,
                original_apply_link TEXT,
                official_apply_link TEXT,
                possible_official_link TEXT,
                link_status TEXT,
                posted_age_days INTEGER,
                field_sources TEXT,
                discovered_date TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, title, location)
            )
        """)
        
        # Scored jobs table to store scoring results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scored_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL UNIQUE,
                score INTEGER NOT NULL,
                reasoning TEXT,
                match_summary TEXT,
                main_gap TEXT,
                recommended_resume TEXT,
                sponsor_risk TEXT,
                sponsorship_evidence TEXT,
                scored_date TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            )
        """)
        
        # Sent report tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sent_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                sent_date TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            )
        """)

        self._add_column_if_missing("jobs", "salary", "TEXT")
        self._add_column_if_missing("jobs", "schedule_type", "TEXT")
        self._add_column_if_missing("jobs", "original_apply_link", "TEXT")
        self._add_column_if_missing("jobs", "official_apply_link", "TEXT")
        self._add_column_if_missing("jobs", "possible_official_link", "TEXT")
        self._add_column_if_missing("jobs", "link_status", "TEXT")
        self._add_column_if_missing("jobs", "posted_age_days", "INTEGER")
        self._add_column_if_missing("jobs", "field_sources", "TEXT")
        self._add_column_if_missing("scored_jobs", "sponsorship_evidence", "TEXT")
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sent_reports_job_id
            ON sent_reports(job_id)
        """)
        
        self.conn.commit()

    def _add_column_if_missing(self, table_name: str, column_name: str, column_type: str):
        """Add a column to existing SQLite databases created by older versions."""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def job_exists(self, company: str, title: str, location: str) -> bool:
        """Check if a job with the same company, title, and location exists."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id FROM jobs 
            WHERE LOWER(company) = LOWER(?) 
            AND LOWER(title) = LOWER(?) 
            AND LOWER(location) = LOWER(?)
        """, (company, title, location))
        return cursor.fetchone() is not None

    def add_job(
        self, 
        title: str, 
        company: str, 
        location: str,
        apply_link: str,
        description: str = None,
        posted_date: str = None,
        salary: str = None,
        schedule_type: str = None,
        original_apply_link: str = None,
        official_apply_link: str = None,
        possible_official_link: str = None,
        link_status: str = None,
        posted_age_days: int = None,
        field_sources: Dict = None
    ) -> int:
        """Add a new job to the database. Returns job ID."""
        cursor = self.conn.cursor()
        safe_apply_link = apply_link or f"missing-link:{company}:{title}:{location}"
        try:
            cursor.execute("""
                INSERT INTO jobs (
                    title, company, location, apply_link, description,
                    posted_date, salary, schedule_type, original_apply_link,
                    official_apply_link, possible_official_link, link_status,
                    posted_age_days, field_sources
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                title, company, location, safe_apply_link, description,
                posted_date, salary, schedule_type, original_apply_link,
                official_apply_link, possible_official_link, link_status,
                posted_age_days, json.dumps(field_sources or {})
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception:
            # If insert fails due to unique constraint on apply_link or UNIQUE(company,title,location),
            # return the existing job id instead of raising.
            cursor.execute("""
                SELECT id FROM jobs WHERE apply_link = ? OR (LOWER(company)=LOWER(?) AND LOWER(title)=LOWER(?) AND LOWER(location)=LOWER(?))
            """, (safe_apply_link, company, title, location))
            row = cursor.fetchone()
            if row:
                return row[0]
            raise

    def add_scored_job(
        self,
        job_id: int,
        score: int,
        reasoning: str,
        match_summary: str,
        main_gap: str,
        recommended_resume: str,
        sponsor_risk: str,
        sponsorship_evidence: str = ""
    ):
        """Add scoring results for a job."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO scored_jobs 
                (
                    job_id, score, reasoning, match_summary, main_gap,
                    recommended_resume, sponsor_risk, sponsorship_evidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, score, reasoning, match_summary, main_gap,
                recommended_resume, sponsor_risk, sponsorship_evidence
            ))
            self.conn.commit()
        except Exception:
            # If a scored record already exists for this job_id, update it instead
            cursor.execute("""
                UPDATE scored_jobs SET
                    score = ?,
                    reasoning = ?,
                    match_summary = ?,
                    main_gap = ?,
                    recommended_resume = ?,
                    sponsor_risk = ?,
                    sponsorship_evidence = ?,
                    scored_date = CURRENT_TIMESTAMP
                WHERE job_id = ?
            """, (
                score, reasoning, match_summary, main_gap, recommended_resume,
                sponsor_risk, sponsorship_evidence, job_id
            ))
            self.conn.commit()

    def get_unsent_scored_jobs(self, limit: int = 10, min_score: int = 1) -> List[Dict]:
        """Get top-scored unsent jobs sorted by score."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                j.id,
                j.title,
                j.company,
                j.location,
                j.apply_link,
                j.description,
                j.posted_date,
                j.salary,
                j.schedule_type,
                j.original_apply_link,
                j.official_apply_link,
                j.possible_official_link,
                j.link_status,
                j.posted_age_days,
                j.field_sources,
                s.score,
                s.reasoning,
                s.match_summary,
                s.main_gap,
                s.recommended_resume,
                s.sponsor_risk,
                s.sponsorship_evidence
            FROM jobs j
            JOIN scored_jobs s ON j.id = s.job_id
            LEFT JOIN sent_reports sr ON j.id = sr.job_id
            WHERE sr.id IS NULL
            AND s.score >= ?
            AND (j.posted_age_days IS NULL OR j.posted_age_days <= 183)
            ORDER BY s.score DESC, j.discovered_date DESC
            LIMIT ?
        """, (min_score, limit))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def mark_job_sent(self, job_id: int):
        """Mark a job as sent in a Slack report."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO sent_reports (job_id)
            VALUES (?)
        """, (job_id,))
        self.conn.commit()

    def get_job_count(self) -> int:
        """Get total number of jobs in database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs")
        return cursor.fetchone()[0]

    def close(self):
        """Close database connection."""
        self.conn.close()
