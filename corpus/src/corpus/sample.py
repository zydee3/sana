"""Stratified sampling of kept works, for calibration and for measuring the corpus.

Strata are discovered_via x status x year bucket: the three dimensions that plausibly
move relevance (EPMC deep-page keyword hits are the suspect population, kept_miss rows
have no text, and old papers skew study type). Sampling is seeded so a run is
reproducible from (n, seed) alone.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from .models import Paper

ROW_COLUMNS = "work_id, title, year, doi, pmcid, discovered_via, status, study_type"


def year_bucket(year: int | None) -> str:
    if year is None:
        return "unknown"
    if year < 2000:
        return "pre2000"
    if year < 2010:
        return "2000s"
    if year < 2020:
        return "2010s"
    return "2020s"


def stratum_of(discovered_via: str, status: str, year: int | None) -> str:
    return f"{discovered_via}/{status}/{year_bucket(year)}"


def _ids_by_stratum(conn: sqlite3.Connection) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    rows = conn.execute(
        "SELECT work_id, discovered_via, status, year FROM works WHERE status LIKE 'kept%'"
        " ORDER BY work_id"
    )
    for work_id, via, status, year in rows:
        groups[stratum_of(via, status, year)].append(work_id)
    return dict(groups)


def allocate(sizes: dict[str, int], n: int, floor: int) -> dict[str, int]:
    """Proportional allocation with a per-stratum floor, capped by stratum size.

    The floor keeps rare strata (e.g. europepmc/kept_miss, pre-2000) measurable; the
    proportional remainder keeps the sample representative of the corpus overall.
    """
    total = sum(sizes.values())
    if total == 0:
        return {}
    alloc = {s: min(floor, size) for s, size in sizes.items()}
    order = sorted(sizes, key=lambda k: -sizes[k])
    # Rounding leaves a remainder after one proportional pass, so keep passing until the
    # target is met or every stratum is exhausted.
    while (remaining := n - sum(alloc.values())) > 0:
        headroom = {s: sizes[s] - alloc[s] for s in sizes}
        spare = sum(headroom.values())
        if spare == 0:
            break
        for s in order:
            if remaining <= 0 or headroom[s] == 0:
                continue
            take = min(headroom[s], remaining, max(1, round(remaining * headroom[s] / spare)))
            alloc[s] += take
            remaining -= take
    return alloc


def stratified(conn: sqlite3.Connection, n: int, seed: int, floor: int = 20) -> list[Paper]:
    """Pick ~n kept works across strata, deterministic for a given (n, seed).

    Returned in shuffled order, not by work_id: work_ids sort by source and then by
    OpenAlex id (roughly by age), so any prefix of a sorted sample would be a biased
    subsample. Shuffled, `head -k` of the file is itself a valid stratified sample.
    """
    groups = _ids_by_stratum(conn)
    alloc = allocate({s: len(ids) for s, ids in groups.items()}, n, floor)
    rng = random.Random(seed)
    chosen: dict[str, str] = {}
    for s in sorted(groups):
        for work_id in rng.sample(groups[s], alloc.get(s, 0)):
            chosen[work_id] = s
    papers = sorted(_fetch(conn, chosen), key=lambda p: p.work_id)
    rng.shuffle(papers)
    return papers


def _fetch(conn: sqlite3.Connection, chosen: dict[str, str]) -> Iterator[Paper]:
    ids = list(chosen)
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT {ROW_COLUMNS} FROM works WHERE work_id IN ({placeholders})", batch
        )
        for work_id, title, year, doi, pmcid, via, status, study_type in rows:
            yield Paper(
                work_id=work_id,
                title=title,
                year=year,
                doi=doi,
                pmcid=pmcid,
                discovered_via=via,
                status=status,
                study_type=study_type,
                stratum=chosen[work_id],
            )


def write_jsonl(papers: Iterable[Paper], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for p in papers:
            f.write(json.dumps(p.__dict__) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[Paper]:
    with path.open() as f:
        return [Paper(**json.loads(line)) for line in f if line.strip()]
