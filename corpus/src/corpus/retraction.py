"""Retraction re-check for the shippable pool.

The contract's Gap 3 says a retracted work must never appear in a client bundle, and
the bundle enforces that by shipping only `status='kept_text'`. But `status='retracted'`
is only ever set at discovery time, from OpenAlex's `is_retracted` — and `from_epmc`
never sets that flag, so the 73% of the pool discovered through Europe PMC was never
checked at all. This closes the gap for the works that can actually ship.

Both sources are queried for every work rather than one as a fallback, because
retraction is an OR over evidence and the sources lag each other: OpenAlex carries
`is_retracted` (from Crossref/PubMed notices), Europe PMC carries `pubTypeList` with
"Retracted Publication". Venue rehydration could stop at the first answer; this cannot.

Resumable: work is a pool row with `retraction_checked_at IS NULL`, stamped whether or
not the work turned out retracted, so a rerun only sees works that arrived since.
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
WHERE status = 'kept_text' AND relevance >= 7 AND retraction_checked_at IS NULL
ORDER BY work_id
"""

MARK = "UPDATE works SET status = 'retracted', retraction_checked_at = ? WHERE work_id = ?"
STAMP = "UPDATE works SET retraction_checked_at = ? WHERE work_id = ?"


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


def _norm_doi(doi: str) -> str:
    return doi.lower().removeprefix("https://doi.org/")


def _oa_url(dois: Sequence[str]) -> str:
    params = {
        "filter": "doi:" + "|".join(dois),
        "select": "doi,is_retracted",
        "per-page": str(len(dois)),
    }
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    return f"{OA_BASE}?{urllib.parse.urlencode(params)}"


def fetch_openalex(papers: Sequence[Paper], fetch: Fetch = get_json) -> set[str]:
    """work_ids OpenAlex reports retracted. Papers it does not know are simply absent."""
    by_doi = {_norm_doi(p.doi): p.work_id for p in papers if p.doi}
    if not by_doi:
        return set()
    out: set[str] = set()
    for r in fetch(_oa_url(list(by_doi))).get("results") or []:
        work_id = by_doi.get(_norm_doi(str(r.get("doi") or "")))
        if work_id and r.get("is_retracted"):
            out.add(work_id)
    return out


def check(papers: Sequence[Paper], fetch: Fetch = get_json) -> tuple[set[str], set[str]]:
    """(openalex hits, epmc hits) for one slice — both sources, every paper."""
    return fetch_openalex(papers, fetch), epmc.fetch_retracted(papers, fetch)


def store(conn: sqlite3.Connection, papers: Sequence[Paper], retracted: set[str], now: str) -> None:
    """Mark the retracted, stamp the rest — one transaction per slice."""
    for p in papers:
        conn.execute(MARK if p.work_id in retracted else STAMP, (now, p.work_id))
    conn.commit()


def run(
    conn: sqlite3.Connection,
    log: Log,
    now: str,
    limit: int | None = None,
    fetch: Fetch = get_json,
) -> dict[str, int]:
    """Re-check the pending pool. Returns counts, including per-source hits."""
    todo = pending(conn, limit)
    slices = (len(todo) + OA_BATCH - 1) // OA_BATCH
    log(f"retraction: {len(todo)} pool works pending, {slices} slices")
    stats = {"checked": 0, "retracted": 0, "openalex": 0, "epmc": 0, "both": 0, "deferred": 0}
    for start in range(0, len(todo), OA_BATCH):
        slice_ = todo[start : start + OA_BATCH]
        try:
            oa, epmc = check(slice_, fetch)
        except Exception as e:  # noqa: BLE001 (network/parse failure: leave rows for a rerun)
            stats["deferred"] += len(slice_)
            log(f"  slice at {start} deferred: {e}")
            continue
        store(conn, slice_, oa | epmc, now)
        stats["checked"] += len(slice_)
        stats["retracted"] += len(oa | epmc)
        stats["openalex"] += len(oa - epmc)
        stats["epmc"] += len(epmc - oa)
        stats["both"] += len(oa & epmc)
        if oa | epmc:
            log(f"  slice at {start}: retracted {sorted(oa | epmc)}")
    log(
        f"retraction done: {stats['checked']} checked, {stats['retracted']} retracted "
        f"(openalex-only {stats['openalex']}, epmc-only {stats['epmc']}, both {stats['both']}), "
        f"{stats['deferred']} deferred"
    )
    return stats
