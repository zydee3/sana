"""Europe PMC: biomedical discovery, DOI->PMCID join, and JATS full-text fallback.

Free API, no key. Discovery uses FIRST_IDATE index windows as the incremental
mechanism (OpenAlex's from_updated_date filter is paywalled). resultType=core
returns abstract, license, and publisher-declared pub types alongside the ids.
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from typing import Any

from .http import get_bytes, get_json
from .models import Candidate, canonical_id

REST_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
SEARCH_URL = f"{REST_URL}/search"

Fetch = Callable[[str], Any]
FetchBytes = Callable[[str], bytes]


def from_epmc(r: dict[str, Any]) -> Candidate:
    """Map a `resultType=core` record to the cross-source Candidate shape."""
    pub_types = tuple((r.get("pubTypeList") or {}).get("pubType", []))
    year = r.get("pubYear")
    return Candidate(
        work_id=canonical_id(None, r.get("doi"), r.get("pmcid")),
        title=(r.get("title") or "").strip(),
        discovered_via="europepmc",
        doi=r.get("doi"),
        pmcid=r.get("pmcid"),
        year=int(year) if year else None,
        authors=r.get("authorString"),
        license=r.get("license"),
        abstract=r.get("abstractText"),
        is_oa=r.get("isOpenAccess") == "Y",
        pub_types=pub_types,
    )


def search_window(
    query: str,
    since: str | None,
    fetch: Fetch = get_json,
    page_size: int = 100,
    max_pages: int = 5,
) -> tuple[list[Candidate], bool]:
    """Open-access hits for a topic query, incremental via FIRST_IDATE windows.

    Returns (candidates, truncated-by-page-cap). cursorMark pages through deep
    result sets (verified up to pageSize=1000).
    """
    q = f"({query}) AND IN_EPMC:Y AND OPEN_ACCESS:Y"
    if since:
        q += f" AND FIRST_IDATE:[{since} TO *]"
    cursor = "*"
    out: list[Candidate] = []
    for _ in range(max_pages):
        params = urllib.parse.urlencode(
            {
                "query": q,
                "format": "json",
                "pageSize": page_size,
                "resultType": "core",
                "cursorMark": cursor,
            }
        )
        payload = fetch(f"{SEARCH_URL}?{params}")
        results = payload.get("resultList", {}).get("result", [])
        out.extend(from_epmc(r) for r in results if r.get("pmcid") or r.get("doi"))
        next_cursor = payload.get("nextCursorMark")
        if not results or not next_cursor or next_cursor == cursor:
            return out, False
        cursor = next_cursor
    return out, True


def pmcids_for_dois(dois: Iterable[str], fetch: Fetch = get_json) -> dict[str, str]:
    """Batch-join DOIs to PMCIDs with one OR-query (~20 DOIs per call).

    Recently published papers often have no PMCID yet (PMC deposition lags);
    they are simply absent from the result.
    """
    wanted = [d for d in dois if d]
    if not wanted:
        return {}
    q = " OR ".join(f'DOI:"{d}"' for d in wanted)
    params = urllib.parse.urlencode(
        {"query": q, "format": "json", "resultType": "lite", "pageSize": len(wanted) + 5}
    )
    payload = fetch(f"{SEARCH_URL}?{params}")
    results = payload.get("resultList", {}).get("result", [])
    return {r["doi"].lower(): r["pmcid"] for r in results if r.get("doi") and r.get("pmcid")}


def full_text(pmcid: str, fetch_bytes: FetchBytes = get_bytes) -> str | None:
    """Plain text from the fullTextXML endpoint (JATS), or None if unavailable.

    Fallback fetch route when the PMC OA bucket has no .txt for the paper.
    """
    try:
        xml = fetch_bytes(f"{REST_URL}/{pmcid}/fullTextXML")
    except OSError:
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    body = root.find(".//body")
    node = body if body is not None else root
    text = " ".join(chunk.strip() for chunk in node.itertext() if chunk.strip())
    return text or None
