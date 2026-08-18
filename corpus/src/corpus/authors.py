"""Author rehydration for the client bundle's card format.

The crawler only stored authors for Europe PMC discoveries: its OpenAlex intake path
never read `authorships`, so every openalex- and citation-discovered work carries a NULL
authors column — 30% of the works the first bundles shipped had an empty `authors` array
(measured 2026-08-18 over 537 shipped works: 163 empty, all of them W-scheme ids).

Two sources, same priority as venue.py:

- **OpenAlex** by DOI, 50 per filtered page, `authorships[].author.display_name`. Full
  names ("Evelyn J. Bromet"), which is what the card wants to render.
- **Europe PMC** by PMCID, fallback for the pool works with no DOI. Its `authorList`
  is the NLM initials form ("Murray JK") — the same shape the crawler stored, so a
  bundle mixes both forms; the array is canonical, the name style is the publisher's.

Rehydrated rows are written as a JSON array, which bundle.authors_array passes through
unchanged; the crawler's legacy comma-joined strings keep being split there.

Resumable: work is a pool row with no authors and `authors_source IS NULL`, and a work
neither source can name gets `authors_source='missing'` (authors NULL) so a rerun does
not retry it forever.
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.parse
from collections.abc import Callable, Sequence
from typing import Any

from . import epmc
from .http import get_json
from .models import Paper

OA_BASE = "https://api.openalex.org/works"
OA_BATCH = 50
Log = Callable[[str], None]
Fetch = Callable[[str], Any]

# Rows the crawler did fill keep their string; only empty ones are rehydrated.
PENDING_SQL = """
SELECT work_id, title, year, doi, pmcid FROM works
WHERE status = 'kept_text' AND relevance >= 7 AND authors_source IS NULL
  AND (authors IS NULL OR trim(authors) IN ('', '[]'))
ORDER BY work_id
"""

UPDATE = "UPDATE works SET authors = ?, authors_source = ? WHERE work_id = ?"


def pending(conn: sqlite3.Connection, limit: int | None = None) -> list[Paper]:
    rows = conn.execute(PENDING_SQL).fetchall()
    return [
        Paper(
            work_id=work_id,
            title=title,
            year=year,
            doi=doi,
            pmcid=pmcid,
            discovered_via="",
            status="kept_text",
            study_type=None,
            stratum="",
        )
        for work_id, title, year, doi, pmcid in rows[:limit]
    ]


def store(conn: sqlite3.Connection, rows: Sequence[tuple[str, list[str] | None, str]]) -> int:
    """Write (work_id, names, source) triples as JSON arrays. Empty is a no-op."""
    if not rows:
        return 0
    conn.executemany(
        UPDATE,
        [
            (json.dumps(names) if names else None, source, work_id)
            for work_id, names, source in rows
        ],
    )
    conn.commit()
    return len(rows)


def _oa_url(dois: Sequence[str]) -> str:
    params = {
        "filter": "doi:" + "|".join(dois),
        "select": "doi,authorships",
        "per-page": str(len(dois)),
    }
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    return f"{OA_BASE}?{urllib.parse.urlencode(params)}"


def _norm_doi(doi: str) -> str:
    return doi.lower().removeprefix("https://doi.org/")


def fetch_openalex(papers: Sequence[Paper], fetch: Fetch = get_json) -> dict[str, list[str]]:
    """Author names for one batch of papers with DOIs, keyed by work_id. Misses are absent."""
    by_doi = {_norm_doi(p.doi): p.work_id for p in papers if p.doi}
    if not by_doi:
        return {}
    out: dict[str, list[str]] = {}
    for r in fetch(_oa_url(list(by_doi))).get("results") or []:
        names = [
            str((a.get("author") or {}).get("display_name") or "").strip()
            for a in (r.get("authorships") or [])
        ]
        names = [n for n in names if n]
        work_id = by_doi.get(_norm_doi(str(r.get("doi") or "")))
        if work_id and names:
            out[work_id] = names
    return out


def resolve(
    papers: Sequence[Paper], fetch: Fetch = get_json
) -> list[tuple[str, list[str] | None, str]]:
    """One batch through both sources, OpenAlex first. Every paper gets a row."""
    found: dict[str, tuple[list[str] | None, str]] = {
        w: (names, "openalex") for w, names in fetch_openalex(papers, fetch).items()
    }
    leftover = [p for p in papers if p.work_id not in found]
    if leftover:
        found |= {w: (names, "epmc") for w, names in epmc.fetch_authors(leftover, fetch).items()}
    return [(p.work_id, *found.get(p.work_id, (None, "missing"))) for p in papers]


def run(
    conn: sqlite3.Connection,
    log: Log,
    limit: int | None = None,
    fetch: Fetch = get_json,
) -> tuple[int, int]:
    """Backfill authors for the pending pool. Returns (attempted, resolved)."""
    todo = pending(conn, limit)
    batches = (len(todo) + OA_BATCH - 1) // OA_BATCH
    log(f"authors: {len(todo)} pool works pending, {batches} batches")
    resolved = failed = 0
    for start in range(0, len(todo), OA_BATCH):
        batch = todo[start : start + OA_BATCH]
        try:
            rows = resolve(batch, fetch)
        except Exception as e:  # noqa: BLE001 (network/parse failure: leave rows for a rerun)
            failed += len(batch)
            log(f"  batch at {start} deferred: {e}")
            continue
        store(conn, rows)
        resolved += sum(1 for _, names, _ in rows if names)
        log(f"  {start + len(batch)}/{len(todo)} requested, {resolved} named, {failed} deferred")
    log(f"authors done: {resolved}/{len(todo)} resolved, {failed} deferred to the next run")
    return len(todo), resolved
