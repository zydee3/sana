"""Retrieval quality on the golden set — the number an index decision is made on.

index-bench scores approximate backends by recall@10 against the exact scan, which
asks "does it return the same chunks" rather than "does it return useful ones". A
backend that drops an irrelevant chunk loses recall and costs nothing; one that drops
the only chunk naming an effect size loses the answer. So the choice between exact and
approximate search is scored here instead, against the hand-judged (query, chunk)
fixture: P@k, P@3, and how many queries get a relevant chunk into the top 3.

Unjudged pairs count as not relevant, and `unjudged` is reported alongside every
metric — a row with unjudged > 0 is a floor, not a measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import index
from .index import Floats, Log

Judgments = dict[tuple[int, str], int]


def load_judgments(path: Path) -> Judgments:
    """Read the (query_idx, chunk_id, relevant) fixture. Rows from a retired chunking
    carry a `<target>:` prefix on chunk_id and simply never match a live retrieval."""
    out: Judgments = {}
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        query_idx, chunk_id, relevant = line.split("\t")[:3]
        out[(int(query_idx), chunk_id)] = int(relevant)
    return out


@dataclass(frozen=True)
class EvalRow:
    backend: str
    params: str
    n: int
    queries: int
    p_at_k: float
    p_at_3: float
    hit_at_3: int
    unjudged: int

    def line(self) -> str:
        return (
            f"{self.backend:<12} {self.params:<28} n={self.n:<8} "
            f"P@10 {self.p_at_k:.3f}  P@3 {self.p_at_3:.3f}  "
            f"hit@3 {self.hit_at_3}/{self.queries}  unjudged {self.unjudged}"
        )


def score(
    hits: Sequence[Sequence[str]], judgments: Judgments, *, k: int
) -> tuple[dict[str, float], list[tuple[int, str]]]:
    """Metrics for one backend. `hits[i]` is the ranked chunk ids for query i+1."""
    precisions, at3, hit3 = [], [], 0
    unjudged: list[tuple[int, str]] = []
    for i, ranked in enumerate(hits):
        query_idx = i + 1
        verdicts = []
        for chunk_id in ranked[:k]:
            verdict = judgments.get((query_idx, chunk_id))
            if verdict is None:
                unjudged.append((query_idx, chunk_id))
            verdicts.append(verdict or 0)
        precisions.append(sum(verdicts) / k)
        at3.append(sum(verdicts[:3]) / 3)
        hit3 += 1 if any(verdicts[:3]) else 0
    return (
        {
            "queries": len(hits),
            "p_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "p_at_3": float(np.mean(at3)) if at3 else 0.0,
            "hit_at_3": hit3,
            "unjudged": len(unjudged),
        },
        unjudged,
    )


def evaluate(
    vecs: Floats,
    ids: Sequence[str],
    queries: Floats,
    judgments: Judgments,
    log: Log,
    *,
    backends: Sequence[str],
    k: int = 10,
    out_dir: Path | None = None,
) -> tuple[list[EvalRow], list[tuple[int, str]]]:
    """Build each backend over `vecs` and score its top-k against the fixture."""
    out_dir = out_dir or index.INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[EvalRow] = []
    unjudged: dict[tuple[int, str], None] = {}
    for name in backends:
        for built in index.BUILDERS[name](vecs, out_dir):
            hits = [[ids[i] for i in built.search(q, k)] for q in queries]
            metrics, missing = score(hits, judgments, k=k)
            unjudged.update(dict.fromkeys(missing))
            row = EvalRow(
                backend=built.backend,
                params=built.params,
                n=len(vecs),
                queries=int(metrics["queries"]),
                p_at_k=metrics["p_at_k"],
                p_at_3=metrics["p_at_3"],
                hit_at_3=int(metrics["hit_at_3"]),
                unjudged=int(metrics["unjudged"]),
            )
            rows.append(row)
            log(row.line())
    return rows, list(unjudged)
