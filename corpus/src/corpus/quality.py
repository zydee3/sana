"""The one composed `quality` 0-1 per work that the client bundle filters on.

The client's `respond` takes a single opaque scalar and a floor (tau_q); composing it
from the corpus' partial signals is ingestion's job. Three signals exist and they are
not interchangeable:

- `relevance` 0-10 is a real sonnet judgment, the gold field, but sparse (26.4k works).
- `gate_p5` 0-1 is the distilled head's keep score: dense, but a ranker at 0.69
  measured precision, not a judgment.
- `evidence_grade` 1 (meta-analysis) .. 5 (opinion) is deterministic from study_type
  and NULL for ~78% of the judged-relevant pool.

So the base comes from whichever judgment exists, and evidence_grade only ever
discounts it. gate_p5 is scaled by the measured 0.69 precision because that is what
the number is worth relative to a real judgment. A NULL grade means "no evidence this
is weak", not "weak": with 78% of the pool unlabeled a pessimistic default would gut
the bundle, so unknown reads as grade 1. The grade ladder itself is a stated prior,
not a measurement — it is the one part of this formula nothing has validated yet.

Semantics must stay stable across bundles because tau_q is tuned against them:
changing any constant here bumps the bundle `schema_version`.
"""

from __future__ import annotations

import sqlite3

# Measured precision of the distilled gate at the standard threshold (distill.py's
# held-out validation). A gate-sourced work can never read as well as the judgment it
# is standing in for.
GATE_PRECISION = 0.69

# Discount per evidence_grade step below 1, linear in the grade. Spans 0.24 across the
# ladder: a case report keeps 76% of the quality its relevance alone would give it.
GRADE_PENALTY = 0.06

SONNET = "sonnet"
GATE = "gate"

RECOMPUTE = """
UPDATE works SET
  quality_source = CASE WHEN relevance IS NOT NULL THEN ?
                        WHEN gate_p5 IS NOT NULL THEN ? END,
  quality = ROUND(
    MAX(0.0, MIN(1.0,
      (CASE WHEN relevance IS NOT NULL THEN relevance / 10.0 ELSE gate_p5 * ? END)
      * (1.0 - ? * (COALESCE(evidence_grade, 1) - 1))
    )), 4)
WHERE relevance IS NOT NULL OR gate_p5 IS NOT NULL
"""

TOTALS = """
SELECT quality_source, COUNT(*), AVG(quality), MIN(quality), MAX(quality)
FROM works WHERE quality IS NOT NULL GROUP BY quality_source ORDER BY 2 DESC
"""

# What the client bundle would actually see: works with at least one citable finding.
SHIPPABLE = """
SELECT COUNT(*), AVG(w.quality), MIN(w.quality), MAX(w.quality)
FROM works w WHERE w.quality IS NOT NULL
  AND EXISTS (SELECT 1 FROM findings f WHERE f.work_id = w.work_id)
"""


def compose(
    relevance: int | None, gate_p5: float | None, evidence_grade: int | None
) -> tuple[float, str] | None:
    """The formula, in Python, for tests and for anything that needs it row-wise.

    Must stay identical to RECOMPUTE — the SQL is the bulk path, this is the spec.
    """
    if relevance is not None:
        base, source = relevance / 10.0, SONNET
    elif gate_p5 is not None:
        base, source = gate_p5 * GATE_PRECISION, GATE
    else:
        return None
    grade = evidence_grade if evidence_grade is not None else 1
    scaled = base * (1.0 - GRADE_PENALTY * (grade - 1))
    return round(max(0.0, min(1.0, scaled)), 4), source


def recompute(conn: sqlite3.Connection) -> int:
    """Rewrite quality/quality_source for every work with a signal. Idempotent.

    A full rewrite rather than a resumable pass: quality is a pure function of columns
    that keep changing underneath it (judging and labeling are both still running), so
    a cached value is only trustworthy if refreshing it is cheap and total.
    """
    cur = conn.execute(RECOMPUTE, (SONNET, GATE, GATE_PRECISION, GRADE_PENALTY))
    conn.commit()
    return cur.rowcount
