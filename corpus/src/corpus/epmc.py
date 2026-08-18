"""Abstract rehydration from Europe PMC.

The crawler never stored abstracts (it kept ids + full text), but judging needs
title+abstract. EPMC search takes OR-joined id queries, so abstracts come back in
batches of BATCH ids per request instead of one request per paper. Abstracts arrive with
JATS-ish inline markup (<h4>Background</h4>…) — stripped to plain text here.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Callable, Sequence
from typing import Any

from .http import get_json
from .models import Paper

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
BATCH = 25
RETRACTED_PUB_TYPE = "retracted publication"

Fetch = Callable[[str], Any]

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def plain(text: str) -> str:
    """Markup out, whitespace collapsed. Section headers keep a colon so they read."""
    text = re.sub(r"</h[0-9]>", ": ", text)
    return _WS.sub(" ", _TAG.sub(" ", text)).strip()


def _term(p: Paper) -> str | None:
    if p.pmcid:
        return f"PMCID:{p.pmcid}"
    if p.doi:
        return f'DOI:"{p.doi}"'
    return None


def _url(terms: Sequence[str]) -> str:
    params = {
        "query": " OR ".join(terms),
        "format": "json",
        "resultType": "core",
        "pageSize": str(len(terms)),
    }
    return f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"


def _search(
    papers: Sequence[Paper], value: Callable[[dict[str, Any]], str], fetch: Fetch
) -> dict[str, str]:
    """One search request, results mapped back onto work_ids by pmcid then doi.

    `value` pulls the field wanted out of a result; empty values are dropped so the
    caller can tell "EPMC has no such field for this paper" from a stored blank.
    """
    terms = [t for t in (_term(p) for p in papers) if t]
    if not terms:
        return {}
    results = (fetch(_url(terms)).get("resultList") or {}).get("result") or []
    by_pmcid = {p.pmcid.upper(): p.work_id for p in papers if p.pmcid}
    by_doi = {p.doi.lower(): p.work_id for p in papers if p.doi}
    out: dict[str, str] = {}
    for r in results:
        got = value(r).strip()
        if not got:
            continue
        pmcid = str(r.get("pmcid") or "").upper()
        doi = str(r.get("doi") or "").lower()
        work_id = by_pmcid.get(pmcid) or by_doi.get(doi)
        if work_id:
            out[work_id] = got
    return out


def fetch_batch(papers: Sequence[Paper], fetch: Fetch = get_json) -> dict[str, str]:
    """Abstracts for one batch, keyed by work_id. Missing papers are simply absent."""
    return _search(papers, lambda r: plain(str(r.get("abstractText") or "")), fetch)


def fetch_venues(papers: Sequence[Paper], fetch: Fetch = get_json) -> dict[str, str]:
    """Journal titles, batched. These are NLM titles — sentence case, subtitle included."""
    out: dict[str, str] = {}
    for start in range(0, len(papers), BATCH):
        out.update(_search(papers[start : start + BATCH], _journal, fetch))
    return out


def _journal(r: dict[str, Any]) -> str:
    return str(((r.get("journalInfo") or {}).get("journal") or {}).get("title") or "")


def fetch_retracted(papers: Sequence[Paper], fetch: Fetch = get_json) -> set[str]:
    """work_ids EPMC tags "Retracted Publication", batched. The other source is OpenAlex."""
    out: set[str] = set()
    for start in range(0, len(papers), BATCH):
        out |= set(_search(papers[start : start + BATCH], _retracted, fetch))
    return out


def _retracted(r: dict[str, Any]) -> str:
    types = [str(t).lower() for t in (r.get("pubTypeList") or {}).get("pubType", [])]
    return RETRACTED_PUB_TYPE if RETRACTED_PUB_TYPE in types else ""


def fetch_abstracts(papers: Sequence[Paper], fetch: Fetch = get_json) -> dict[str, str]:
    """Abstracts for all papers, batched. Callers record misses themselves."""
    out: dict[str, str] = {}
    for start in range(0, len(papers), BATCH):
        out.update(fetch_batch(papers[start : start + BATCH], fetch))
    return out
