"""The corpus record. This schema is the contract the backend reads against."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Paper:
    pmcid: str
    title: str
    doi: str | None
    authors: str | None
    year: str | None
    license: str | None

    @classmethod
    def from_epmc(cls, r: dict[str, Any]) -> Paper:
        """Build from a Europe PMC `resultType=core` search result."""
        # firstPublicationDate is "YYYY-MM-DD"; fall back to pubYear.
        year = (r.get("firstPublicationDate") or "")[:4] or r.get("pubYear")
        return cls(
            pmcid=r["pmcid"],
            title=(r.get("title") or "").strip(),
            doi=r.get("doi"),
            authors=r.get("authorString"),
            year=year or None,
            license=r.get("license"),
        )
