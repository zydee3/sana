"""Corpus database: the topic queue and work records, one SQLite file.

The DB is the source of truth for everything except article text (flat files under
texts/). WAL mode so the backend can enqueue topics while the crawler holds the write
lock briefly. Statuses:
  topics: pending | active | done
  works:  candidate | rejected | retracted | kept_miss | kept_text
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .models import Candidate

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  query TEXT NOT NULL,
  openalex_id TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  added_by TEXT NOT NULL DEFAULT 'manual',
  last_crawled_at TEXT,
  watermark TEXT
);
CREATE TABLE IF NOT EXISTS works (
  work_id TEXT PRIMARY KEY,
  openalex_id TEXT, doi TEXT, pmcid TEXT,
  title TEXT NOT NULL,
  year INTEGER, authors TEXT, license TEXT,
  topic_id INTEGER REFERENCES topics(id),
  discovered_via TEXT NOT NULL,
  status TEXT NOT NULL,
  reject_reason TEXT,
  study_type TEXT,
  evidence_grade INTEGER,
  triage_confidence REAL,
  text_path TEXT,
  text_source TEXT,
  fetched_at TEXT,
  expanded_at TEXT
);
"""


@dataclass(frozen=True)
class Topic:
    id: int
    name: str
    query: str
    openalex_id: str | None
    watermark: str | None


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def add_topic(
    conn: sqlite3.Connection,
    name: str,
    query: str,
    openalex_id: str | None = None,
    added_by: str = "manual",
) -> None:
    """Enqueue a topic; re-adding an existing name is a no-op (anyone may enqueue)."""
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO topics (name, query, openalex_id, added_by) VALUES (?, ?, ?, ?)",
            (name, query, openalex_id, added_by),
        )


def claim_next_topic(
    conn: sqlite3.Connection, recrawl_days: int, now: datetime | None = None
) -> Topic | None:
    """Oldest pending topic, else the stalest done topic past the re-crawl interval."""
    cutoff = _now_iso((now or datetime.now(UTC)) - timedelta(days=recrawl_days))
    row = conn.execute(
        """SELECT id, name, query, openalex_id, watermark FROM topics
           WHERE status = 'pending' OR (status = 'done' AND last_crawled_at < ?)
           ORDER BY status = 'pending' DESC, last_crawled_at, id LIMIT 1""",
        (cutoff,),
    ).fetchone()
    if row is None:
        return None
    with conn:
        conn.execute("UPDATE topics SET status = 'active' WHERE id = ?", (row["id"],))
    return Topic(row["id"], row["name"], row["query"], row["openalex_id"], row["watermark"])


def finish_topic(
    conn: sqlite3.Connection,
    topic_id: int,
    watermark: str | None,
    now: datetime | None = None,
) -> None:
    with conn:
        conn.execute(
            "UPDATE topics SET status = 'done', last_crawled_at = ?, "
            "watermark = COALESCE(?, watermark) WHERE id = ?",
            (_now_iso(now), watermark, topic_id),
        )


def seen(conn: sqlite3.Connection, c: Candidate) -> bool:
    """Known under any identifier — sources name the same paper differently."""
    row = conn.execute(
        """SELECT 1 FROM works WHERE work_id = ?
           OR (doi IS NOT NULL AND doi = ?) OR (pmcid IS NOT NULL AND pmcid = ?) LIMIT 1""",
        (c.work_id, c.doi, c.pmcid),
    ).fetchone()
    return row is not None


def record_work(
    conn: sqlite3.Connection,
    c: Candidate,
    topic_id: int | None,
    status: str,
    reject_reason: str | None = None,
    study_type: str | None = None,
    evidence_grade: int | None = None,
    triage_confidence: float | None = None,
) -> None:
    with conn:
        conn.execute(
            """INSERT OR IGNORE INTO works
               (work_id, openalex_id, doi, pmcid, title, year, authors, license, topic_id,
                discovered_via, status, reject_reason, study_type, evidence_grade,
                triage_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                c.work_id,
                c.openalex_id,
                c.doi,
                c.pmcid,
                c.title,
                c.year,
                c.authors,
                c.license,
                topic_id,
                c.discovered_via,
                status,
                reject_reason,
                study_type,
                evidence_grade,
                triage_confidence,
            ),
        )


def set_fetched(
    conn: sqlite3.Connection,
    work_id: str,
    text_path: str,
    text_source: str,
    now: datetime | None = None,
) -> None:
    with conn:
        conn.execute(
            "UPDATE works SET status = 'kept_text', text_path = ?, text_source = ?, "
            "fetched_at = ? WHERE work_id = ?",
            (text_path, text_source, _now_iso(now), work_id),
        )


def set_pmcid(conn: sqlite3.Connection, work_id: str, pmcid: str) -> None:
    with conn:
        conn.execute("UPDATE works SET pmcid = ? WHERE work_id = ?", (pmcid, work_id))


def unexpanded_kept(conn: sqlite3.Connection, topic_id: int, limit: int) -> list[str]:
    """OpenAlex ids of this topic's kept works the citation walk hasn't visited."""
    rows = conn.execute(
        """SELECT openalex_id FROM works
           WHERE topic_id = ? AND status IN ('kept_text', 'kept_miss')
           AND expanded_at IS NULL AND openalex_id IS NOT NULL LIMIT ?""",
        (topic_id, limit),
    ).fetchall()
    return [str(r["openalex_id"]) for r in rows]


def set_expanded(conn: sqlite3.Connection, openalex_id: str, now: datetime | None = None) -> None:
    with conn:
        conn.execute(
            "UPDATE works SET expanded_at = ? WHERE openalex_id = ?",
            (_now_iso(now), openalex_id),
        )
