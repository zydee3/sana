"""Client-profile retrieval: rank the published claims for the golden queries.

`evaluate` scores the *server* profile — chunk vectors through an index backend. The
client pulls findings and embeds claim text locally (bundle contract Gap 4), so its
retrieval unit is a one-sentence claim over a few thousand rows, not a ~178-word passage
over hundreds of thousands. The chunk fixture does not carry across that gap: only 15 of
its 293 judged-relevant chunks are an anchor in the current bundle, so the client profile
needs its own finding-keyed judgments, in the same three-column shape.

Ranking is exact cosine over every live claim — at bundle scale there is no index
decision to make, which is why nothing here touches `index`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import bundle, embed
from .evaluate import Judgments


@dataclass(frozen=True)
class Card:
    """One finding as the client holds it: the claim it embeds plus what it renders."""

    finding_id: str
    work_id: str
    claim: str
    caveats: str
    quality: float
    title: str


def load_cards(out_dir: Path) -> tuple[str, list[Card]]:
    """(bundle_id, live cards) read from the published bundle the way a client reads it."""
    manifest, payloads = bundle.read_published(out_dir)
    rows = {k: [json.loads(line) for line in v.splitlines()] for k, v in payloads.items()}
    works, _ = bundle.split(rows["works"])
    findings, _ = bundle.split(rows["findings"])
    by_id = {w["work_id"]: w for w in works}
    cards = [
        Card(
            finding_id=f["finding_id"],
            work_id=f["work_id"],
            claim=f["claim"],
            caveats=f["caveats"],
            quality=float(by_id[f["work_id"]]["quality"]),
            title=str(by_id[f["work_id"]]["title"]),
        )
        for f in findings
    ]
    return str(manifest["bundle_id"]), cards


def load_judgments(path: Path) -> Judgments:
    """Read the (query_idx, finding_id, relevant) fixture, skipping rows still unjudged.

    A dumped row carries `?` until someone judges it; skipping those keeps them in the
    `unjudged` count instead of silently scoring them as irrelevant."""
    out: Judgments = {}
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        query_idx, finding_id, relevant = line.split("\t")[:3]
        if relevant not in ("0", "1"):
            continue
        out[(int(query_idx), finding_id)] = int(relevant)
    return out


def rank(
    cards: Sequence[Card],
    queries: Sequence[str],
    spec: embed.ModelSpec,
    *,
    k: int,
    workers: int,
    threads: int = 1,
) -> list[list[tuple[str, float]]]:
    """Top-k (finding_id, score) per query, embedding claims with the bundle's encoder."""
    vecs = embed.encode_parallel(spec, [c.claim for c in cards], workers=workers, threads=threads)
    # One query per encode call, for the reason cmd_eval gives: int8 activation scales are
    # taken over the whole input tensor, so a batched query is not the vector the client's
    # one-query-at-a-time serving path produces.
    embedder = embed.Embedder(spec, threads=threads)
    qv = np.vstack([embedder.encode([q], is_query=True) for q in queries])
    ids = [c.finding_id for c in cards]
    return [embed.search(q, vecs, ids, k) for q in qv]


DUMP_HEADER = "query_idx\tfinding_id\trelevant\trank\tscore\twork_id\tquality\tclaim\tcaveats"


def dump(
    hits: Sequence[Sequence[tuple[str, float]]],
    cards: Sequence[Card],
    path: Path,
    *,
    judgments: Judgments | None = None,
) -> int:
    """Write the hits as a judgeable fixture; known verdicts are carried, the rest are `?`."""
    by_id = {c.finding_id: c for c in cards}
    lines = [DUMP_HEADER]
    for i, ranked in enumerate(hits):
        query_idx = i + 1
        for pos, (finding_id, score) in enumerate(ranked, start=1):
            c = by_id[finding_id]
            verdict = (judgments or {}).get((query_idx, finding_id))
            lines.append(
                f"{query_idx}\t{finding_id}\t{'?' if verdict is None else verdict}\t{pos}\t"
                f"{score:.3f}\t{c.work_id}\t{c.quality:.3f}\t{c.claim}\t{c.caveats}"
            )
    path.write_text("\n".join(lines) + "\n")
    return len(lines) - 1
