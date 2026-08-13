"""The full relevance + label run over every kept work.

Resumable by construction: work is a kept work whose `relevance` IS NULL, so a restart
(crash, rate limit, reboot) re-judges nothing already stored. Verdicts land in the DB
batch by batch rather than at the end, so a run killed at any moment keeps its progress.

Order is a seeded shuffle of the pending ids, not work_id order: work_ids cluster by
source and roughly by age, so a partial run in id order would give biased per-source
statistics. Shuffled, whatever fraction is done is a fair sample of the whole.

Quota: `claude -p` spends the operator's subscription, and the loop that owns this run
also needs it. Before every round the runner reads ~/.claude/statusline-latest.json and
sleeps until reset if either window is above QUOTA_CEILING. A file older than STALE_S
carries no signal (usually nothing is updating it) and is treated as go, not stop.

That gate cannot see everything — a monthly spend cap makes every call fail instantly
and appears nowhere in rate_limits — so failure itself is the backstop: FAILURE_BRAKE
batches failing in a row stops the run. Without it a dead account burns the whole
pending list in futile spawns (338k works in 74 minutes, observed 2026-08-13).
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .classify import ClassifyError, classify_batch
from .models import Paper, Verdict
from .sample import ROW_COLUMNS, stratum_of

STATUSLINE = Path(os.environ.get("SANA_STATUSLINE", Path.home() / ".claude/statusline-latest.json"))
QUOTA_CEILING = 85.0
STALE_S = 1800.0
SLEEP_CHUNK_S = 300.0
FAILURE_BRAKE = 5

Log = Callable[[str], None]

# study_type from the publisher is better evidence than an abstract read; keep it and say
# so in label_source, which exists to record exactly that distinction.
UPDATE = """
UPDATE works SET
  relevance = ?,
  domain = ?,
  label_confidence = ?,
  study_type = COALESCE(study_type, ?),
  label_source = CASE WHEN study_type IS NULL THEN ? ELSE 'publisher' END
WHERE work_id = ?
"""

_PENDING = "FROM works WHERE status LIKE 'kept%' AND relevance IS NULL"


def quota_wait_s(now: float, path: Path = STATUSLINE) -> float:
    """Seconds to wait before spending more quota; 0 means go."""
    try:
        age = now - path.stat().st_mtime
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return 0.0
    if age > STALE_S:
        return 0.0
    windows = data.get("rate_limits")
    if not isinstance(windows, dict):
        return 0.0
    waits = [
        max(0.0, float(w.get("resets_at", 0.0)) - now) + 60.0
        for w in windows.values()
        if isinstance(w, dict) and float(w.get("used_percentage", 0.0)) > QUOTA_CEILING
    ]
    return max(waits, default=0.0)


def wait_for_quota(log: Log, sleep: Callable[[float], None] = time.sleep) -> float:
    """Block until both quota windows are under the ceiling. Returns seconds slept."""
    slept = 0.0
    while (wait := quota_wait_s(time.time())) > 0:
        chunk = min(wait, SLEEP_CHUNK_S)
        log(f"  quota above {QUOTA_CEILING:.0f}%; sleeping {chunk:.0f}s ({wait:.0f}s to reset)")
        sleep(chunk)
        slept += chunk
    return slept


def pending_ids(
    conn: sqlite3.Connection,
    seed: int = 7,
    *,
    via: Sequence[str] | None = None,
    status: str | None = None,
) -> list[str]:
    """Pending ids, shuffled. `via`/`status` restrict a run to one discovery stratum."""
    where, params = _PENDING, []
    if via:
        where += f" AND discovered_via IN ({','.join('?' * len(via))})"
        params += list(via)
    if status:
        where += " AND status = ?"
        params.append(status)
    rows = conn.execute(f"SELECT work_id {where} ORDER BY work_id", params).fetchall()
    ids = [str(r[0]) for r in rows]
    random.Random(seed).shuffle(ids)
    return ids


def load_papers(conn: sqlite3.Connection, ids: Sequence[str]) -> list[Paper]:
    """Papers for one batch, abstract attached. Rows with no abstract judge title-only."""
    cols = ", ".join(f"works.{c}" for c in ROW_COLUMNS.split(", "))
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT {cols}, abstracts.abstract FROM works"
        f" LEFT JOIN abstracts ON abstracts.work_id = works.work_id"
        f" WHERE works.work_id IN ({placeholders})",
        list(ids),
    ).fetchall()
    by_id = {
        work_id: Paper(
            work_id=work_id,
            title=title,
            year=year,
            doi=doi,
            pmcid=pmcid,
            discovered_via=via,
            status=status,
            study_type=study_type,
            stratum=stratum_of(via, status, year),
            abstract=abstract,
        )
        for work_id, title, year, doi, pmcid, via, status, study_type, abstract in rows
    }
    return [by_id[i] for i in ids if i in by_id]


def store_verdicts(conn: sqlite3.Connection, verdicts: Sequence[Verdict], model: str) -> int:
    if not verdicts:
        return 0
    source = f"claude-{model}"
    conn.executemany(
        UPDATE,
        [(v.relevance, v.domain, v.confidence, v.study_type, source, v.work_id) for v in verdicts],
    )
    conn.commit()
    return len(verdicts)


def _judge(papers: Sequence[Paper], model: str) -> tuple[list[Verdict], str]:
    """One batch, one retry. Returns (verdicts, last error); empty leaves the rows NULL."""
    error = ""
    for _ in range(2):
        try:
            return classify_batch(papers, model), ""
        except ClassifyError as e:
            error = str(e)
    return [], error


def run(
    conn: sqlite3.Connection,
    log: Log,
    *,
    model: str = "sonnet",
    workers: int = 4,
    batch_size: int = 60,
    limit: int | None = None,
    seed: int = 7,
    brake: int = FAILURE_BRAKE,
    via: Sequence[str] | None = None,
    status: str | None = None,
) -> tuple[int, int]:
    """Judge pending works until none are left (or `limit` are done). Returns (done, failed)."""
    ids = pending_ids(conn, seed, via=via, status=status)[:limit]
    batches = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]
    log(f"judging {len(ids)} pending works in {len(batches)} batches, {workers} workers, {model}")
    done = failed = streak = 0
    last_error = ""
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(batches), workers):
            wait_for_quota(log)
            group = [load_papers(conn, b) for b in batches[start : start + workers]]
            results = list(pool.map(lambda b: _judge(b, model), group))
            for batch, (verdicts, error) in zip(group, results, strict=True):
                if not verdicts:
                    failed += len(batch)
                    streak += 1
                    last_error = error
                    continue
                streak = 0
                done += store_verdicts(conn, verdicts, model)
            rate = done / max(1e-9, time.time() - started) * 60
            log(f"  {done}/{len(ids)} judged, {failed} failed, {rate:.0f} papers/min")
            if streak >= brake:
                log(f"brake: {streak} batches failed in a row, stopping. last error: {last_error}")
                break
    log(f"run done: {done} judged, {failed} left for the next run")
    return done, failed
