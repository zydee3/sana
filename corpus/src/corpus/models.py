"""Records and label vocabularies for the processing side."""

from __future__ import annotations

from dataclasses import dataclass

# Mirrors scraper/models.py GRADE_BY_STUDY_TYPE — the ladder is deterministic from
# study_type, so a work labeled here can be graded here (quality.backfill_grades).
# The two copies exist because scraper/ is stdlib-only and the projects do not import
# each other; test_quality asserts this one is complete over STUDY_TYPES.
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

STUDY_TYPES = tuple(GRADE_BY_STUDY_TYPE)

# Coarse topical bucket, chosen so relevance thresholds can be picked per domain later.
DOMAINS = (
    "mental-health",
    "sleep",
    "stress",
    "pain",
    "lifestyle",
    "other-medical",
    "off-topic",
)


@dataclass(frozen=True)
class Paper:
    """A work as the judging step sees it."""

    work_id: str
    title: str
    year: int | None
    doi: str | None
    pmcid: str | None
    discovered_via: str
    status: str
    study_type: str | None
    stratum: str
    abstract: str | None = None


@dataclass(frozen=True)
class Verdict:
    """One model's judgment of one paper."""

    work_id: str
    relevance: int
    domain: str
    study_type: str
    confidence: float
