"""corpus.db access for the processing side, with the migrations this project owns.

The scraper created the file and owns `topics`/`works`; processing adds judgment
columns on `works` and an `abstracts` table (abstracts are not stored by the crawler —
they are rehydrated from Europe PMC when a work needs judging). Migrations are additive
and idempotent so either project can open the DB at any version.

label_source distinguishes where a study_type came from: NULL/'publisher' for the
34.5k rows the crawler filled from publisher pub types, 'claude-<model>' for inferred.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(os.environ.get("SANA_CORPUS_DB", "/sana-data/corpus/corpus.db"))

# Added to works: judgment output. relevance 0-10, domain one of labels.DOMAINS.
WORK_COLUMNS = {
    "relevance": "INTEGER",
    "domain": "TEXT",
    "label_source": "TEXT",
    "label_confidence": "REAL",
    # Set when clean+chunk has processed the work, including when it produced nothing,
    # so the chunking runner is a no-op on restart.
    "chunked_at": "TEXT",
    # The distilled head's output for works no model provider ever judged: P(relevance>=5)
    # and a predicted domain. Kept apart from relevance/domain because it is a ranker at
    # ~0.77 precision, not a judgment, and a threshold on it must stay re-choosable.
    "gate_p5": "REAL",
    "gate_domain": "TEXT",
    # Set when findings extraction has read the work, including when it produced no
    # findings, so the extraction runner is a no-op on restart.
    "extracted_at": "TEXT",
    # The single 0-1 scalar the client bundle filters on, composed from relevance /
    # gate_p5 / evidence_grade by quality.py, plus which judgment it came from.
    # Derived, not authoritative: `corpus quality` rewrites both from scratch.
    "quality": "REAL",
    "quality_source": "TEXT",
    # The journal the client's card renders, rehydrated by venue.py. venue_source is
    # 'openalex' | 'epmc' | 'missing' — 'missing' is a closed-out row, not a retry.
    "venue": "TEXT",
    "venue_source": "TEXT",
    # The author names the client's card renders. The crawler stored the publisher's own
    # string for EPMC-discovered works and nothing at all for OpenAlex/citation ones, so
    # authors.py rehydrates the empty ones as a JSON array. authors_source is
    # 'openalex' | 'epmc' | 'missing' — 'missing' is a closed-out row, not a retry.
    "authors_source": "TEXT",
    # When retraction.py last asked OpenAlex and Europe PMC whether this work was
    # retracted. Stamped whether or not it was, so the re-check is a resumable no-op;
    # a retracted work also has its status flipped, which is what keeps it out of bundles.
    "retraction_checked_at": "TEXT",
}

ABSTRACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS abstracts (
  work_id TEXT PRIMARY KEY,
  abstract TEXT,
  source TEXT NOT NULL,
  fetched_at TEXT NOT NULL
);
"""

# Chunk metadata (relevance, domain, study_type, year) is joined from works rather
# than copied, so a re-judged work never leaves stale labels behind on its chunks.
CHUNKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  work_id TEXT NOT NULL REFERENCES works(work_id),
  idx INTEGER NOT NULL,
  section TEXT NOT NULL,
  heading TEXT,
  text TEXT NOT NULL,
  n_words INTEGER NOT NULL,
  UNIQUE (work_id, idx)
);
CREATE INDEX IF NOT EXISTS chunks_work ON chunks(work_id);
"""

# The citable unit of the client bundle: a claim, its mandatory caveats, and an anchor
# into the chunk that supports it. finding_id is a content hash of (work_id, claim) —
# never derived from chunk_id, because the client persists it across re-chunking runs.
FINDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
  finding_id TEXT PRIMARY KEY,
  work_id TEXT NOT NULL REFERENCES works(work_id),
  claim TEXT NOT NULL,
  caveats TEXT NOT NULL,
  anchor_chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id),
  char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL,
  quote TEXT NOT NULL,
  extracted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS findings_work ON findings(work_id);
"""


# What the client has already been given. A row that was shipped and is no longer
# shippable (a retraction, most of all) must leave as a tombstone, because the client
# applies bundles by primary key — a row that merely stops appearing never gets removed.
SHIPPED_SCHEMA = """
CREATE TABLE IF NOT EXISTS shipped (
  kind TEXT NOT NULL,
  row_id TEXT NOT NULL,
  first_shipped TEXT NOT NULL,
  PRIMARY KEY (kind, row_id)
);
"""


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Apply the additive processing migrations. Returns what it changed."""
    applied: list[str] = []
    existing = _columns(conn, "works")
    for name, decl in WORK_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE works ADD COLUMN {name} {decl}")
            applied.append(f"works.{name}")
    if not _columns(conn, "abstracts"):
        conn.executescript(ABSTRACTS_SCHEMA)
        applied.append("abstracts")
    if not _columns(conn, "chunks"):
        conn.executescript(CHUNKS_SCHEMA)
        applied.append("chunks")
    if not _columns(conn, "findings"):
        conn.executescript(FINDINGS_SCHEMA)
        applied.append("findings")
    if not _columns(conn, "shipped"):
        conn.executescript(SHIPPED_SCHEMA)
        applied.append("shipped")
    conn.commit()
    return applied


def connect(path: Path = DEFAULT_DB, *, read_only: bool = False) -> sqlite3.Connection:
    """Open corpus.db; migrates unless opened read-only."""
    if read_only:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn = sqlite3.connect(path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    migrate(conn)
    return conn
