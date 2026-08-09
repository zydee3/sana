"""Corpus records. These schemas are the contract the backend reads against."""

from __future__ import annotations

from dataclasses import dataclass, field

# The evidence grade retrieval weights (1 strongest .. 5 weakest) is a pure lookup on
# study type — never a drop reason. NULL grade means the study type isn't known yet.
GRADE_BY_STUDY_TYPE = {
    "meta_analysis": 1,
    "systematic_review": 1,
    "rct": 2,
    "cohort": 3,
    "case_control": 3,
    "cross_sectional": 4,
    "observational": 4,
    "case_report": 5,
    "opinion": 5,
    "qualitative": 5,
    "other": 5,
}


@dataclass(frozen=True)
class Candidate:
    """A discovered paper, normalized across sources, before the quality gate."""

    work_id: str  # canonical: OpenAlex W-id, else doi:<doi>, else pmcid:<id>
    title: str
    discovered_via: str  # openalex | europepmc | citation
    openalex_id: str | None = None
    doi: str | None = None
    pmcid: str | None = None
    year: int | None = None
    authors: str | None = None
    license: str | None = None
    abstract: str | None = None
    is_retracted: bool = False
    is_oa: bool = True
    pub_types: tuple[str, ...] = field(default_factory=tuple)


def canonical_id(openalex_id: str | None, doi: str | None, pmcid: str | None) -> str:
    """One key per paper regardless of which source named it. Raises on no id at all."""
    if openalex_id:
        return openalex_id
    if doi:
        return f"doi:{doi}"
    if pmcid:
        return f"pmcid:{pmcid}"
    raise ValueError("paper has no usable identifier")
