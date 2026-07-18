"""Discovery: find open-access papers via the Europe PMC search API.

Europe PMC is used only to locate papers and their metadata; the PDF itself comes from
the PMC open-access bucket (see pmc_oa.py). We filter to full-text open-access records so
that a downloadable PDF is likely to exist.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from .http import get_json
from .models import Paper

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _parse_search(payload: dict[str, Any]) -> list[Paper]:
    results = payload.get("resultList", {}).get("result", [])
    return [Paper.from_epmc(r) for r in results if r.get("pmcid")]


def search(query: str, limit: int = 1) -> list[Paper]:
    q = f"({query}) AND IN_EPMC:Y AND OPEN_ACCESS:Y"
    params = urllib.parse.urlencode(
        {
            "query": q,
            "format": "json",
            "pageSize": max(1, limit),
            "resultType": "core",
        }
    )
    return _parse_search(get_json(f"{SEARCH_URL}?{params}"))[:limit]
