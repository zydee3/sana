"""Venue rehydration for the client bundle's card format.

The crawler stored title/year/authors but never the journal, and the contract's work
record renders one. Two sources, in a measured priority (100 pool works, 2026-08-17):

- **OpenAlex** by DOI, 50 per filtered page (1 credit/page). `primary_location.source
  .display_name` is display-cased — "BMC Medicine", "The Canadian Journal of Psychiatry".
- **Europe PMC** by PMCID, fallback. `journalInfo.journal.title` is the NLM title:
  sentence case ("BMC medicine") and sometimes the full legal name with its subtitle
  ("Prevention science : the official journal of ..."). Both sources covered 100/100,
  and 72 of the 94 differences were casing alone — so OpenAlex leads on rendering, and
  EPMC exists for the 63 pool works with no DOI.

Resumable: work is a pool row with `venue_source IS NULL`, and a work neither source
knows gets `venue_source='missing'` (venue NULL) so a rerun does not retry it forever.

Pool = kept_text with relevance >= 7: exactly the works findings extraction reads, and
therefore every work that can ever reach a bundle. Scoring the other ~345k works would
buy nothing the client can render.
"""

from __future__ import annotations

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

PENDING_SQL = """
SELECT work_id, title, year, doi, pmcid FROM works
WHERE status = 'kept_text' AND relevance >= 7 AND venue_source IS NULL
ORDER BY work_id
"""

UPDATE = "UPDATE works SET venue = ?, venue_source = ? WHERE work_id = ?"


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


def store(conn: sqlite3.Connection, rows: Sequence[tuple[str, str | None, str]]) -> int:
    """Write (work_id, venue, source) triples. Empty is a no-op."""
    if not rows:
        return 0
    conn.executemany(UPDATE, [(venue, source, work_id) for work_id, venue, source in rows])
    conn.commit()
    return len(rows)


def _oa_url(dois: Sequence[str]) -> str:
    params = {
        "filter": "doi:" + "|".join(dois),
        "select": "doi,primary_location",
        "per-page": str(len(dois)),
    }
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    return f"{OA_BASE}?{urllib.parse.urlencode(params)}"


def _norm_doi(doi: str) -> str:
    return doi.lower().removeprefix("https://doi.org/")


def fetch_openalex(papers: Sequence[Paper], fetch: Fetch = get_json) -> dict[str, str]:
    """Venues for one batch of papers with DOIs, keyed by work_id. Misses are absent."""
    by_doi = {_norm_doi(p.doi): p.work_id for p in papers if p.doi}
    if not by_doi:
        return {}
    out: dict[str, str] = {}
    for r in fetch(_oa_url(list(by_doi))).get("results") or []:
        source = (r.get("primary_location") or {}).get("source") or {}
        name = str(source.get("display_name") or "").strip()
        work_id = by_doi.get(_norm_doi(str(r.get("doi") or "")))
        if work_id and name:
            out[work_id] = name
    return out


def resolve(papers: Sequence[Paper], fetch: Fetch = get_json) -> list[tuple[str, str | None, str]]:
    """One batch through both sources, OpenAlex first. Every paper gets a row."""
    found = {w: (v, "openalex") for w, v in fetch_openalex(papers, fetch).items()}
    leftover = [p for p in papers if p.work_id not in found]
    if leftover:
        found |= {w: (v, "epmc") for w, v in epmc.fetch_venues(leftover, fetch).items()}
    return [(p.work_id, *found.get(p.work_id, (None, "missing"))) for p in papers]


def run(
    conn: sqlite3.Connection,
    log: Log,
    limit: int | None = None,
    fetch: Fetch = get_json,
) -> tuple[int, int]:
    """Backfill venues for the pending pool. Returns (attempted, resolved)."""
    todo = pending(conn, limit)
    log(f"venue: {len(todo)} pool works pending, {(len(todo) + OA_BATCH - 1) // OA_BATCH} batches")
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
        resolved += sum(1 for _, venue, _ in rows if venue)
        log(f"  {start + len(batch)}/{len(todo)} requested, {resolved} venues, {failed} deferred")
    log(f"venue done: {resolved}/{len(todo)} resolved, {failed} deferred to the next run")
    return len(todo), resolved
