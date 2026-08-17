"""Findings extraction: the citable unit the Sana client consumes.

The client's bundle contract asks for findings, not passages — a plain-sentence claim,
mandatory caveats, and an anchor into the chunk that supports it. One `claude -p` call
reads a whole work (all its chunks, in order) and returns its own findings; chunks stay
the retrieval layer and the anchor target, they are just not what gets cited.

Anchors are computed here, never trusted from the model: the model names a chunk and
copies a quote, and `locate` finds that quote's char span in the stored chunk text
(exact match first, then a whitespace/punctuation-folded match, because models normalize
curly quotes and line breaks). A quote that cannot be located is a dropped finding, not
a stored anchor that points at nothing.

finding_id is a content hash of (work_id, normalized claim) — deliberately NOT derived
from chunk_id, which is `work_id#idx` and would change under any re-chunking, while the
client persists finding_id in conversations forever. Re-extraction of the same claim
therefore lands on the same row (INSERT OR REPLACE refreshes a moved anchor).

Validation is the extractor's own, applied per finding: empty or evasive caveats fail
(the contract's one hard rule), as does an unlocatable quote, a chunk that belongs to
another work, or a claim outside the length band. Failures drop that finding and are
counted; only a whole-call parse failure leaves the work pending for the next run.

Resumable like the other runners: work = a judged>=7 kept_text work with chunks and
`works.extracted_at` NULL, and extracted_at is stamped even when a work yields nothing.
Quota gating is shared with judging (judge.wait_for_quota) — same subscription.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from .judge import Log, wait_for_quota

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
TIMEOUT_S = 1800
BATCH_SIZE = 1
MIN_FINDINGS = 3
MAX_FINDINGS = 8
CLAIM_CHARS = (20, 300)
QUOTE_CHARS = (30, 1500)
CAVEATS_MIN_CHARS = 10
# Chunks are fed whole and in order; past this a work is truncated rather than dropped.
# p99 of the >=7 pool is ~12.5k words, so this affects a handful of outliers.
MAX_WORK_WORDS = 14000
FAILURE_BRAKE = 5

# Caveats that say nothing. The contract makes caveats mandatory precisely because the
# answer-time model must carry them, so "none" is a failed extraction, not a value.
_NULL_CAVEATS = re.compile(
    r"^(none|n/?a|nil|unknown|unclear|not\s+(stated|reported|specified|applicable|given)"
    r"|no\s+(caveats?|limitations?)\b.*)$",
    re.IGNORECASE,
)

# Model output normalizes typography and line breaks; folding both sides lets an
# otherwise-faithful quote still resolve to a real span.
_FOLD_CHARS = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    " ": " ",
    "­": "-",
    "ﬁ": "fi",
    "ﬂ": "fl",
}

INSTRUCTIONS = f"""You extract citable findings from research papers for Sana, a \
wellness companion: a chat app where ordinary people ask about their mental health, \
sleep, stress, pain and everyday lifestyle health. Each finding becomes a card the \
companion may cite in an answer, so it must stand on its own.

Each paper is given as its full text in numbered chunks, each tagged \
[chunk_id | section]. From each paper extract {MIN_FINDINGS}-{MAX_FINDINGS} findings \
that the paper itself reports.

Every finding has exactly these fields:
- claim: ONE plain sentence, readable by a non-expert, stating what this paper found.
  Give the direction and, where the text states it, the size ("cut insomnia severity by
  about 5 points on the ISI"). No citation markers, no jargon a lay reader would not
  know, no hedging preamble. Say only what your quoted span shows: keep the hedges the
  span keeps, and do not state a number that sits somewhere else in the paper.
- caveats: the population, duration and design limits a careful reader must attach to
  the claim, in 1-2 clauses ("40 adults with chronic insomnia, 8 weeks, no active
  control"). NEVER empty and never "none" — if you cannot state a caveat, drop the
  finding instead.
- chunk_id: the chunk whose text supports the claim, copied from its tag.
- quote: 1-3 consecutive sentences copied EXACTLY, character for character, from that
  one chunk. No paraphrase, no ellipses, no text joined across chunks. The quote must
  carry the claim by itself — the same direction and every number the claim states
  (scale denominators such as "out of 5" aside). A reader who sees only this quote must
  be able to check the claim. If no single span carries the whole claim, narrow the
  claim until one does.

Prefer results and conclusions over introduction and methods. Skip anything the paper
attributes to other studies — findings are this paper's own. If a paper reports fewer
than {MIN_FINDINGS} findings of its own, return only the ones it has.

Reply with ONLY a JSON array, one object per paper, in the given order:
[{{"paper": 1, "findings": [{{"claim": "...", "caveats": "...", \
"chunk_id": "W123#4", "quote": "..."}}]}}]
No prose, no markdown fences."""


class ExtractError(Exception):
    """Runner failure or malformed reply; the caller leaves those works pending."""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section: str
    text: str
    n_words: int


@dataclass(frozen=True)
class Work:
    """A work as the extractor sees it: metadata plus its chunks in order."""

    work_id: str
    title: str
    year: int | None
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True)
class Finding:
    finding_id: str
    work_id: str
    claim: str
    caveats: str
    anchor_chunk_id: str
    char_start: int
    char_end: int
    quote: str


@dataclass
class Usage:
    """What one `claude -p` call cost. Subscription auth, so cost_usd is a yardstick."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


@dataclass
class WorkResult:
    """One work's extraction outcome, whether or not it is stored."""

    work_id: str
    findings: list[Finding] = field(default_factory=list)
    drops: list[dict[str, str]] = field(default_factory=list)
    truncated: bool = False
    prompt_words: int = 0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def fold(text: str) -> tuple[str, list[int]]:
    """Typography- and whitespace-normalized text plus each kept char's original index."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = False
    for i, raw in enumerate(text):
        ch = _FOLD_CHARS.get(raw, raw)
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        for c in ch:  # ligature expansions map back to the one source char
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def locate(text: str, quote: str) -> tuple[int, int] | None:
    """Char span of `quote` in `text`, or None. Exact match first, folded match second."""
    exact = text.find(quote)
    if exact >= 0:
        return exact, exact + len(quote)
    folded_text, offsets = fold(text)
    folded_quote, _ = fold(quote.strip())
    if not folded_quote:
        return None
    at = folded_text.find(folded_quote)
    if at < 0:
        return None
    return offsets[at], offsets[at + len(folded_quote) - 1] + 1


def finding_id(work_id: str, claim: str) -> str:
    """Stable id: content hash of the work and its claim. Never involves chunk_id."""
    norm = " ".join(claim.split()).casefold()
    digest = hashlib.sha256(f"{work_id}\n{norm}".encode()).hexdigest()
    return f"f_{digest[:16]}"


def _work_block(i: int, work: Work) -> tuple[str, bool, int]:
    """One paper's prompt block, whether it was truncated, and its word count."""
    year = work.year if work.year is not None else "unknown"
    lines = [f"Paper {i + 1}\nTitle: {work.title}\nYear: {year}"]
    words = 0
    truncated = False
    for c in work.chunks:
        if words + c.n_words > MAX_WORK_WORDS and words:
            truncated = True
            break
        words += c.n_words
        lines.append(f"[{c.chunk_id} | {c.section}]\n{c.text}")
    return "\n\n".join(lines), truncated, words


def build_prompt(works: Sequence[Work]) -> tuple[str, list[tuple[bool, int]]]:
    blocks, meta = [], []
    for i, w in enumerate(works):
        block, truncated, words = _work_block(i, w)
        blocks.append(block)
        meta.append((truncated, words))
    body = "\n\n---\n\n".join(blocks)
    noun = "paper" if len(works) == 1 else "papers"
    return f"{INSTRUCTIONS}\n\nExtract findings from these {len(works)} {noun}:\n\n{body}", meta


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def validate(work: Work, raw: Sequence[Any]) -> tuple[list[Finding], list[dict[str, str]]]:
    """Turn one work's raw findings into stored rows; every drop is counted with a reason."""
    by_chunk = {c.chunk_id: c for c in work.chunks}
    kept: list[Finding] = []
    drops: list[dict[str, str]] = []
    seen: set[str] = set()

    def drop(reason: str, item: Any) -> None:
        drops.append({"reason": reason, "claim": str(item.get("claim", ""))[:120]})

    for item in raw:
        if not isinstance(item, dict):
            drops.append({"reason": "not_an_object", "claim": str(item)[:120]})
            continue
        if len(kept) >= MAX_FINDINGS:
            drop("over_limit", item)
            continue
        claim = str(item.get("claim", "")).strip()
        caveats = str(item.get("caveats", "")).strip()
        chunk_id = str(item.get("chunk_id", "")).strip()
        quote = str(item.get("quote", "")).strip()
        if not CLAIM_CHARS[0] <= len(claim) <= CLAIM_CHARS[1]:
            drop("claim_length", item)
            continue
        if len(caveats) < CAVEATS_MIN_CHARS or _NULL_CAVEATS.match(caveats):
            drop("empty_caveats", item)
            continue
        chunk = by_chunk.get(chunk_id)
        if chunk is None:
            drop("unknown_chunk", item)
            continue
        if not QUOTE_CHARS[0] <= len(quote) <= QUOTE_CHARS[1]:
            drop("quote_length", item)
            continue
        span = locate(chunk.text, quote)
        if span is None:
            drop("quote_not_found", item)
            continue
        fid = finding_id(work.work_id, claim)
        if fid in seen:
            drop("duplicate_claim", item)
            continue
        seen.add(fid)
        start, end = span
        kept.append(
            Finding(fid, work.work_id, claim, caveats, chunk_id, start, end, chunk.text[start:end])
        )
    return kept, drops


def parse_reply(
    reply: str, works: Sequence[Work]
) -> list[tuple[list[Finding], list[dict[str, str]]]]:
    try:
        parsed: Any = json.loads(_strip_fences(reply))
    except json.JSONDecodeError as e:
        raise ExtractError(f"reply is not JSON: {reply[:200]!r}") from e
    if not isinstance(parsed, list) or len(parsed) != len(works):
        got = len(parsed) if isinstance(parsed, list) else type(parsed).__name__
        raise ExtractError(f"expected {len(works)} paper objects, got {got}")
    out = []
    for i, (entry, work) in enumerate(zip(parsed, works, strict=True)):
        if not isinstance(entry, dict) or int(entry.get("paper", -1)) != i + 1:
            raise ExtractError(f"entry {i + 1} is not paper {i + 1}: {str(entry)[:120]!r}")
        raw = entry.get("findings")
        if not isinstance(raw, list):
            raise ExtractError(f"paper {i + 1} has no findings array")
        out.append(validate(work, raw))
    return out


def run_claude(prompt: str, model: str) -> tuple[str, Usage]:
    """One `claude -p` call. Returns the reply text and what the call cost."""
    cmd = [CLAUDE_BIN, "-p", prompt, "--model", model, "--output-format", "json"]
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ExtractError(f"claude -p failed to run: {e}") from e
    seconds = time.monotonic() - started
    if proc.returncode != 0:
        # Subscription and spend-limit refusals come out on stdout, not stderr, so an
        # stderr-only message reads as empty and hides why a whole run stopped.
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise ExtractError(f"claude -p exited {proc.returncode}: {detail[:200]}")
    try:
        wrapper = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ExtractError(f"claude output is not JSON: {proc.stdout[:200]!r}") from e
    if wrapper.get("is_error") or not isinstance(wrapper.get("result"), str):
        raise ExtractError(f"claude reported an error (subtype={wrapper.get('subtype')})")
    usage = wrapper.get("usage") or {}
    return str(wrapper["result"]), Usage(
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cache_read_tokens=int(usage.get("cache_read_input_tokens", 0)),
        cost_usd=float(wrapper.get("total_cost_usd", 0.0)),
        seconds=seconds,
    )


Runner = Callable[[str, str], tuple[str, Usage]]

PENDING = """
SELECT w.work_id FROM works w
WHERE w.relevance >= ? AND w.status = 'kept_text' AND w.extracted_at IS NULL
  AND EXISTS (SELECT 1 FROM chunks c WHERE c.work_id = w.work_id)
ORDER BY w.work_id
"""


def pending_ids(
    conn: sqlite3.Connection,
    *,
    min_relevance: int = 7,
    seed: int = 7,
    only: Sequence[str] | None = None,
) -> list[str]:
    """Unextracted ids, shuffled — a partial run stays a fair sample of the pool."""
    rows = conn.execute(PENDING, (min_relevance,)).fetchall()
    wanted = set(only) if only is not None else None
    ids = [str(r[0]) for r in rows if wanted is None or r[0] in wanted]
    random.Random(seed).shuffle(ids)
    return ids


def load_works(conn: sqlite3.Connection, ids: Sequence[str]) -> list[Work]:
    """Works with their chunks in order, in the given id order."""
    placeholders = ",".join("?" * len(ids))
    meta = {
        str(r[0]): (str(r[1]), r[2])
        for r in conn.execute(
            f"SELECT work_id, title, year FROM works WHERE work_id IN ({placeholders})", list(ids)
        )
    }
    chunks: dict[str, list[Chunk]] = {}
    for work_id, chunk_id, section, text, n_words in conn.execute(
        f"SELECT work_id, chunk_id, section, text, n_words FROM chunks"
        f" WHERE work_id IN ({placeholders}) ORDER BY work_id, idx",
        list(ids),
    ):
        chunks.setdefault(str(work_id), []).append(
            Chunk(str(chunk_id), str(section), str(text), int(n_words))
        )
    out = []
    for work_id in ids:
        if work_id in meta and chunks.get(work_id):
            title, year = meta[work_id]
            out.append(Work(work_id, title, year, tuple(chunks[work_id])))
    return out


def store(conn: sqlite3.Connection, work_ids: Sequence[str], findings: Iterable[Finding]) -> int:
    """Write findings and stamp the works — including works that produced nothing."""
    rows = [
        (
            f.finding_id,
            f.work_id,
            f.claim,
            f.caveats,
            f.anchor_chunk_id,
            f.char_start,
            f.char_end,
            f.quote,
        )
        for f in findings
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO findings (finding_id, work_id, claim, caveats,"
        " anchor_chunk_id, char_start, char_end, quote, extracted_at)"
        " VALUES (?,?,?,?,?,?,?,?, datetime('now'))",
        rows,
    )
    conn.executemany(
        "UPDATE works SET extracted_at = ? WHERE work_id = ?", [(_now(), w) for w in work_ids]
    )
    conn.commit()
    return len(rows)


def extract_batch(
    works: Sequence[Work], model: str, runner: Runner = run_claude
) -> tuple[list[WorkResult], Usage]:
    """One call over one batch of works. Raises ExtractError; the batch stays pending."""
    prompt, meta = build_prompt(works)
    reply, usage = runner(prompt, model)
    results = []
    for work, (kept, drops), (truncated, words) in zip(
        works, parse_reply(reply, works), meta, strict=True
    ):
        results.append(WorkResult(work.work_id, kept, drops, truncated, words))
    return results, usage


def _attempt(
    works: Sequence[Work], model: str, runner: Runner
) -> tuple[list[WorkResult], Usage, str]:
    """One batch, one retry. On failure returns no results and the last error."""
    error = ""
    total = Usage()
    for _ in range(2):
        try:
            results, usage = extract_batch(works, model, runner)
            _add(total, usage)
            return results, total, ""
        except ExtractError as e:
            error = str(e)
    return [], total, error


def _add(total: Usage, one: Usage) -> None:
    total.input_tokens += one.input_tokens
    total.output_tokens += one.output_tokens
    total.cache_read_tokens += one.cache_read_tokens
    total.cost_usd += one.cost_usd
    total.seconds += one.seconds


def run(
    conn: sqlite3.Connection,
    log: Log,
    *,
    model: str = "sonnet",
    workers: int = 2,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
    min_relevance: int = 7,
    seed: int = 7,
    only: Sequence[str] | None = None,
    dry_run: bool = False,
    report: Callable[[WorkResult, Usage], None] | None = None,
    runner: Runner = run_claude,
    brake: int = FAILURE_BRAKE,
) -> tuple[int, int, Usage]:
    """Extract until the pool is empty (or `limit` works are done).

    Returns (works done, works failed, total usage). `dry_run` runs the calls and the
    validation but writes nothing, which is how a calibration slice is measured.
    """
    ids = pending_ids(conn, min_relevance=min_relevance, seed=seed, only=only)[:limit]
    batches = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]
    log(f"extracting {len(ids)} works in {len(batches)} batches of {batch_size}, {model}")
    done = failed = streak = n_findings = 0
    total = Usage()
    last_error = ""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(batches), workers):
            wait_for_quota(log)
            group = [load_works(conn, b) for b in batches[start : start + workers]]
            outcomes = list(pool.map(lambda w: _attempt(w, model, runner), group))
            for works, (results, usage, error) in zip(group, outcomes, strict=True):
                _add(total, usage)
                if not results:
                    failed += len(works)
                    streak += 1
                    last_error = error
                    continue
                streak = 0
                done += len(results)
                n_findings += sum(len(r.findings) for r in results)
                if report is not None:
                    for r in results:
                        report(r, usage)
                if not dry_run:
                    store(
                        conn, [r.work_id for r in results], [f for r in results for f in r.findings]
                    )
            per_work = n_findings / max(1, done)
            log(
                f"  {done}/{len(ids)} works, {n_findings} findings ({per_work:.1f}/work),"
                f" {failed} failed, {total.output_tokens} out-tokens"
            )
            if streak >= brake:
                log(f"brake: {streak} batches failed in a row. last error: {last_error}")
                break
    log(f"run done: {done} works extracted, {n_findings} findings, {failed} left pending")
    return done, failed, total


def result_json(r: WorkResult, usage: Usage) -> str:
    """One report line: the findings, the drops, and what the call cost."""
    return json.dumps(
        {
            "work_id": r.work_id,
            "prompt_words": r.prompt_words,
            "truncated": r.truncated,
            "findings": [asdict(f) for f in r.findings],
            "drops": r.drops,
            "usage": asdict(usage),
        }
    )
