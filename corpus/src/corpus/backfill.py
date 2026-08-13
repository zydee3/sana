"""Abstract backfill for the whole kept corpus — the input judging needs.

Two sources, cheapest first. The stored full texts already contain the abstract: the
PMC-to-text converter delimits it with a `\x9f=…=\x9f` rule (present in 1992/2000 sampled
files), so the 245k kept_text rows need no network at all. Only rows without a local
text go to Europe PMC, 25 ids per request through a small thread pool.

Resumable: work is a kept work with no `abstracts` row, so a restart re-does nothing.
Papers EPMC does not know get an explicit source='missing' row (abstract NULL) so the
runner terminates instead of retrying them forever — judging falls back to title-only.
The one exception is the text pass, which also reconsiders 'missing' rows: a local text
can rescue a paper EPMC never had. That rescan is bounded by the markerless files
(~0.4%), so a rerun still costs about a second.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from . import epmc, sample
from .models import Paper

TEXTS_DIR = Path(os.environ.get("SANA_TEXTS_DIR", "/sana-data/corpus/texts"))

HEAD_BYTES = 40_000
ABSTRACT_CHARS = 2_000
MIN_CHARS = 100
COMMIT_ROWS = 5_000
ROUND_BATCHES = 4  # EPMC batches per worker between commits

ABSTRACT_MARK = re.compile("\x9f=+\x9f")
_WS = re.compile(r"\s+")

Log = Callable[[str], None]
Row = tuple[str, str | None, str]

UPSERT = (
    "INSERT INTO abstracts (work_id, abstract, source, fetched_at) VALUES (?,?,?,?)"
    " ON CONFLICT(work_id) DO UPDATE SET abstract=excluded.abstract,"
    " source=excluded.source, fetched_at=excluded.fetched_at"
)

# Pending = kept work the judging step still has no text for.
_PENDING = " FROM works LEFT JOIN abstracts ON abstracts.work_id = works.work_id WHERE"
_UNJUDGEABLE = " works.status LIKE 'kept%' AND abstracts.work_id IS NULL"
_NO_TEXT = (
    " works.status LIKE 'kept%' AND (abstracts.work_id IS NULL OR abstracts.abstract IS NULL)"
)


def store(conn: sqlite3.Connection, rows: Sequence[Row]) -> int:
    """Write abstracts (or explicit misses) and commit. Empty is a no-op."""
    if not rows:
        return 0
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    conn.executemany(UPSERT, [(w, a, s, stamp) for w, a, s in rows])
    conn.commit()
    return len(rows)


def local_path(text_path: str) -> Path:
    """text_path is the crawler pod's container path; the same file sits in TEXTS_DIR."""
    return TEXTS_DIR / Path(text_path).name


def head_abstract(raw: str) -> str | None:
    """The abstract block that follows the converter's rule, or None if absent/too short."""
    m = ABSTRACT_MARK.search(raw)
    if not m:
        return None
    body = _WS.sub(" ", raw[m.end() : m.end() + ABSTRACT_CHARS]).strip()
    return body if len(body) >= MIN_CHARS else None


def read_head(path: Path) -> str | None:
    try:
        with path.open(errors="replace") as f:
            return head_abstract(f.read(HEAD_BYTES))
    except OSError:
        return None


def pending_texts(conn: sqlite3.Connection, limit: int | None = None) -> list[tuple[str, str]]:
    sql = f"SELECT works.work_id, works.text_path{_PENDING}{_NO_TEXT}"
    rows = conn.execute(f"{sql} AND works.text_path IS NOT NULL").fetchall()
    return [(str(r[0]), str(r[1])) for r in rows[:limit]]


def pending_papers(
    conn: sqlite3.Connection, *, require_ids: bool, limit: int | None = None
) -> list[Paper]:
    cols = ", ".join(f"works.{c}" for c in sample.ROW_COLUMNS.split(", "))
    where = " AND (works.pmcid IS NOT NULL OR works.doi IS NOT NULL)" if require_ids else ""
    rows = conn.execute(f"SELECT {cols}{_PENDING}{_UNJUDGEABLE}{where}").fetchall()
    return [
        Paper(
            work_id=work_id,
            title=title,
            year=year,
            doi=doi,
            pmcid=pmcid,
            discovered_via=via,
            status=status,
            study_type=study_type,
            stratum=sample.stratum_of(via, status, year),
        )
        for work_id, title, year, doi, pmcid, via, status, study_type in rows[:limit]
    ]


def text_pass(conn: sqlite3.Connection, log: Log, limit: int | None = None) -> tuple[int, int]:
    """Extract abstracts from the stored texts. Files with no marker are left for EPMC."""
    todo = pending_texts(conn, limit)
    log(f"text pass: {len(todo)} kept works with a local text and no abstract")
    found = 0
    buf: list[Row] = []
    for i, (work_id, text_path) in enumerate(todo, 1):
        body = read_head(local_path(text_path))
        if body:
            buf.append((work_id, body, "text"))
            found += 1
        if len(buf) >= COMMIT_ROWS:
            store(conn, buf)
            buf = []
            log(f"  {i}/{len(todo)} scanned, {found} abstracts")
    store(conn, buf)
    log(f"text pass done: {found}/{len(todo)} abstracts from local text")
    return len(todo), found


def _safe_fetch(batch: Sequence[Paper]) -> dict[str, str] | None:
    """None means the request failed outright — no rows written, so a rerun retries it."""
    try:
        return epmc.fetch_batch(batch)
    except Exception:  # noqa: BLE001 (any network/parse failure is retried by the next run)
        return None


def epmc_pass(
    conn: sqlite3.Connection, log: Log, workers: int = 4, limit: int | None = None
) -> tuple[int, int]:
    todo = pending_papers(conn, require_ids=True, limit=limit)
    batches = [todo[i : i + epmc.BATCH] for i in range(0, len(todo), epmc.BATCH)]
    log(f"epmc pass: {len(todo)} works, {len(batches)} requests, {workers} workers")
    found = failed = 0
    per_round = max(1, workers * ROUND_BATCHES)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(batches), per_round):
            group = batches[start : start + per_round]
            rows: list[Row] = []
            for batch, got in zip(group, pool.map(_safe_fetch, group), strict=True):
                if got is None:
                    failed += len(batch)
                    continue
                rows += [
                    (p.work_id, got.get(p.work_id), "epmc" if p.work_id in got else "missing")
                    for p in batch
                ]
                found += len(got)
            store(conn, rows)
            done = min((start + per_round) * epmc.BATCH, len(todo))
            log(f"  {done}/{len(todo)} requested, {found} abstracts, {failed} deferred")
    log(f"epmc pass done: {found}/{len(todo)} abstracts, {failed} deferred to the next run")
    return len(todo), found


def mark_missing(conn: sqlite3.Connection, log: Log) -> int:
    """Close out works neither source can serve, so the runner is a no-op next time."""
    leftover = pending_papers(conn, require_ids=False)
    store(conn, [(p.work_id, None, "missing") for p in leftover])
    log(f"marked {len(leftover)} works title-only (no local text, no EPMC id)")
    return len(leftover)
