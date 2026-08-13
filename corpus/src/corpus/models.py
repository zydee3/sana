"""Records and label vocabularies for the processing side."""

from __future__ import annotations

from dataclasses import dataclass

# Mirrors scraper/models.py GRADE_BY_STUDY_TYPE keys — that lookup stays the single
# source of truth for evidence_grade; processing only writes study_type.
STUDY_TYPES = (
    "meta_analysis",
    "systematic_review",
    "rct",
    "cohort",
    "case_control",
    "cross_sectional",
    "observational",
    "case_report",
    "opinion",
    "qualitative",
    "other",
)

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
