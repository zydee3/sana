"""Agreement metrics between two judging runs over the same papers.

Relevance is graded, so exact-match agreement understates usefulness: what matters for
retrieval is whether two models would put a paper on the same side of a threshold, and
how far apart they are when they differ. Reported: mean absolute difference, agreement
within +-1, keep/drop agreement at a threshold, and exact agreement on domain and
study_type.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .models import Verdict

KEEP_THRESHOLD = 7


@dataclass(frozen=True)
class Agreement:
    n: int
    mean_abs_diff: float
    within_1: float
    keep_agree: float
    domain_agree: float
    study_type_agree: float


def _pairs(a: Sequence[Verdict], b: Sequence[Verdict]) -> list[tuple[Verdict, Verdict]]:
    by_id = {v.work_id: v for v in b}
    return [(x, by_id[x.work_id]) for x in a if x.work_id in by_id]


def agreement(
    a: Sequence[Verdict], b: Sequence[Verdict], threshold: int = KEEP_THRESHOLD
) -> Agreement:
    pairs = _pairs(a, b)
    n = len(pairs)
    if n == 0:
        return Agreement(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    diffs = [abs(x.relevance - y.relevance) for x, y in pairs]
    return Agreement(
        n=n,
        mean_abs_diff=sum(diffs) / n,
        within_1=sum(d <= 1 for d in diffs) / n,
        keep_agree=sum((x.relevance >= threshold) == (y.relevance >= threshold) for x, y in pairs)
        / n,
        domain_agree=sum(x.domain == y.domain for x, y in pairs) / n,
        study_type_agree=sum(x.study_type == y.study_type for x, y in pairs) / n,
    )


def disagreements(
    a: Sequence[Verdict], b: Sequence[Verdict], *, relevance_gap: int = 2
) -> list[str]:
    """work_ids worth adjudicating: relevance far apart, or a different domain/type."""
    out = []
    for x, y in _pairs(a, b):
        if (
            abs(x.relevance - y.relevance) >= relevance_gap
            or x.domain != y.domain
            or x.study_type != y.study_type
        ):
            out.append(x.work_id)
    return out
