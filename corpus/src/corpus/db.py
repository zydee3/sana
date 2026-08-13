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
}

ABSTRACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS abstracts (
  work_id TEXT PRIMARY KEY,
  abstract TEXT,
  source TEXT NOT NULL,
  fetched_at TEXT NOT NULL
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
