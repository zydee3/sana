"""BM25 retrieval over chunk text, and rank fusion with the dense scan.

golden-queries.md asserts that lexical overlap between a user message and the
evidence that answers it is low, which is the case for using a dense encoder at
all. That claim has never been measured — this module is the arm that measures
it. FTS5 ships inside the SQLite the corpus already lives in, so the lexical side
costs one table and no model.

The FTS table is external-content over `chunks`: the text is not duplicated, only
the inverted index, and a rebuild after re-chunking is `build(rebuild=True)`.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Sequence

Log = Callable[[str], None]

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, content='chunks', content_rowid='rowid', tokenize='porter unicode61'
);
"""

# A user message is prose, not FTS5 syntax: "i can't fall asleep" is a syntax error
# and "don't" tokenises to two terms anyway. Tokens under three characters ("i", "my",
# "is", "do") carry no retrieval signal and only widen the OR, so they are dropped.
_TOKEN = re.compile(r"[a-z0-9]+")
_MIN_TOKEN = 3


def to_match(query: str) -> str:
    """Turn a natural-language message into an OR of quoted FTS5 terms."""
    terms = [t for t in _TOKEN.findall(query.lower()) if len(t) >= _MIN_TOKEN]
    return " OR ".join(f'"{t}"' for t in terms)


def indexed(conn: sqlite3.Connection) -> int:
    """Rows in the FTS index, or 0 when it does not exist yet."""
    try:
        return int(conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])
    except sqlite3.OperationalError:
        return 0


def build(conn: sqlite3.Connection, log: Log, *, rebuild: bool = False) -> int:
    """Create and populate the FTS index over `chunks`. No-op when already current."""
    chunks = int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])
    if rebuild:
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.commit()
    elif indexed(conn) == chunks:
        log(f"chunks_fts already covers {chunks} chunks")
        return chunks
    conn.executescript(FTS_SCHEMA)
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('delete-all')")
    conn.execute("INSERT INTO chunks_fts(rowid, text) SELECT rowid, text FROM chunks")
    conn.commit()
    log(f"indexed {chunks} chunks")
    return chunks


def search(conn: sqlite3.Connection, query: str, k: int) -> list[str]:
    """Top-k chunk ids by BM25. FTS5's bm25() is negative, best first."""
    match = to_match(query)
    if not match:
        return []
    rows = conn.execute(
        "SELECT c.chunk_id FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (match, k),
    ).fetchall()
    return [str(r[0]) for r in rows]


def rrf(rankings: Sequence[Sequence[str]], k: int, *, c: int = 60) -> list[str]:
    """Reciprocal rank fusion — the standard dense+lexical combiner. Scores are
    1/(c+rank) summed across arms, so no arm's score scale has to be calibrated
    against the other's (cosine and BM25 are not comparable numbers)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (c + rank)
    return sorted(scores, key=lambda cid: -scores[cid])[:k]
