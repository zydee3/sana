"""Discovery via OpenAlex: topic-filtered listing, free entity lookups, citation edges.

The API is metered (verified 2026-08-07): filtered lists cost 1 credit/page, text search
10 credits/page, entity lookups 0 — so discovery filters on curated topic ids and never
uses `search=`. `ids.pmcid` is no longer populated; PMCIDs come from the Europe PMC join
(europepmc.pmcids_for_dois). Anonymous tier is 1,000 credits/day; a free API key raises
it to 10,000 ($1/day).
"""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Callable
from typing import Any

from .http import get_json
from .models import Candidate, canonical_id

BASE = "https://api.openalex.org"
SELECT = "id,doi,display_name,publication_year,is_retracted,open_access,abstract_inverted_index"

Fetch = Callable[[str], Any]


def _short_id(url_or_id: str | None) -> str | None:
    """'https://openalex.org/W123' -> 'W123' (ids arrive as URLs)."""
    if not url_or_id:
        return None
    return url_or_id.rsplit("/", 1)[-1]


def _short_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.removeprefix("https://doi.org/")


def rebuild_abstract(inv: dict[str, list[int]] | None) -> str | None:
    """OpenAlex ships abstracts as word -> positions; sort positions to restore the text."""
    if not inv:
        return None
    positions = [(p, word) for word, places in inv.items() for p in places]
    return " ".join(word for _, word in sorted(positions))


def from_openalex(r: dict[str, Any], discovered_via: str = "openalex") -> Candidate:
    openalex_id = _short_id(r.get("id"))
    doi = _short_doi(r.get("doi"))
    return Candidate(
        work_id=canonical_id(openalex_id, doi, None),
        title=(r.get("display_name") or "").strip(),
        discovered_via=discovered_via,
        openalex_id=openalex_id,
        doi=doi,
        year=r.get("publication_year"),
        abstract=rebuild_abstract(r.get("abstract_inverted_index")),
        is_retracted=bool(r.get("is_retracted")),
        is_oa=bool((r.get("open_access") or {}).get("is_oa")),
    )


def _url(path: str, params: dict[str, str]) -> str:
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        params = {**params, "api_key": key}
    return f"{BASE}{path}?{urllib.parse.urlencode(params)}"


def works_by_topic(
    topic_id: str,
    since: str | None,
    fetch: Fetch = get_json,
    per_page: int = 200,
    max_pages: int = 5,
) -> tuple[list[Candidate], bool]:
    """One credit per page. Returns (candidates, truncated-by-page-cap)."""
    flt = f"primary_topic.id:{topic_id},is_oa:true"
    if since:
        flt += f",from_publication_date:{since}"
    cursor = "*"
    out: list[Candidate] = []
    for _ in range(max_pages):
        url = _url(
            "/works",
            {"filter": flt, "select": SELECT, "per-page": str(per_page), "cursor": cursor},
        )
        payload = fetch(url)
        results = payload.get("results", [])
        out.extend(from_openalex(r) for r in results if r.get("id"))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor or not results:
            return out, False
    return out, True


def citers(openalex_id: str, fetch: Fetch = get_json, per_page: int = 50) -> list[Candidate]:
    """One page of works citing this one (1 credit) — the outward citation walk."""
    url = _url(
        "/works",
        {"filter": f"cites:{openalex_id},is_oa:true", "select": SELECT, "per-page": str(per_page)},
    )
    payload = fetch(url)
    return [from_openalex(r, "citation") for r in payload.get("results", []) if r.get("id")]


def referenced_ids(openalex_id: str, fetch: Fetch = get_json) -> list[str]:
    """Ids this work cites, via a free entity lookup."""
    payload = fetch(_url(f"/works/{openalex_id}", {"select": "referenced_works"}))
    refs = payload.get("referenced_works") or []
    return [sid for ref in refs if (sid := _short_id(ref))]


def work_by_id(openalex_id: str, fetch: Fetch = get_json) -> Candidate | None:
    """Free entity lookup for a single work (used when expanding references)."""
    payload = fetch(_url(f"/works/{openalex_id}", {"select": SELECT}))
    if not payload or not payload.get("id"):
        return None
    return from_openalex(payload, "citation")
