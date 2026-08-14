"""Section-aware chunking of cleaned texts into the rows the embedder will read.

Chunks never span a section boundary, so every chunk keeps a single honest
`section` label; within a section, paragraphs are packed greedily up to the target
and a paragraph too long to fit alone is split on sentence boundaries. No overlap:
paragraph boundaries are already the natural seam, and the golden-query eval is what
should decide whether overlap earns its storage.

Sizes are stated in tokens but measured in words through TOKENS_PER_WORD. The target
is 256, not the mission's ~350: measured on the golden queries, 256 retrieves better
(P@10 0.555 vs 0.460, P@3 0.700 vs 0.483) and 350 overran MiniLM's 512-token limit on
3.5% of chunks against 0.2% here. TOKENS_PER_WORD stays 1.35 even though the real
ratio through MiniLM's tokenizer is 1.474, because 1.35 is what produced the packing
that was evaluated — at this target it lands a 260-token median, p90 343.

Metadata is inherited by join, not by copy: relevance/domain/study_type/year live on
works and would go stale if duplicated here. The runner is resumable per work
(works.chunked_at, set even when a text yields nothing) and re-chunking a work
replaces its rows.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .clean import Block, clean, is_prose

TEXTS_DIR = Path(os.environ.get("SANA_TEXTS_DIR", "/sana-data/corpus/texts"))

TOKENS_PER_WORD = 1.35  # packing constant, not the real ratio (1.474) — see module docstring
TARGET_TOKENS = 256
MAX_TOKENS = 336  # leaves headroom under a 512-token encoder
MIN_TOKENS = 34  # below this a chunk is a caption or a stray line

TARGET_WORDS = int(TARGET_TOKENS / TOKENS_PER_WORD)
MAX_WORDS = int(MAX_TOKENS / TOKENS_PER_WORD)
MIN_WORDS = int(MIN_TOKENS / TOKENS_PER_WORD)


@dataclass(frozen=True)
class Budget:
    """Word budgets for one token target."""

    target_words: int
    max_words: int
    min_words: int


def budget(target_tokens: int = TARGET_TOKENS) -> Budget:
    """Budgets for a chunk-size experiment; the cap keeps TARGET_TOKENS' headroom ratio."""
    max_tokens = round(target_tokens * MAX_TOKENS / TARGET_TOKENS)
    return Budget(
        target_words=int(target_tokens / TOKENS_PER_WORD),
        max_words=int(max_tokens / TOKENS_PER_WORD),
        min_words=MIN_WORDS,
    )


COMMIT_WORKS = 200
POOL_CHUNKSIZE = 16

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[\"(\[]?[A-Z0-9])")

Log = Callable[[str], None]


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit of one paper."""

    work_id: str
    idx: int
    section: str
    heading: str | None
    text: str
    n_words: int


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text) if s.strip()]


def _split_long(text: str, max_words: int) -> Iterator[str]:
    """A paragraph over the cap, cut at sentence boundaries (hard cut if a single
    sentence is still too long)."""
    buf: list[str] = []
    size = 0
    for sentence in _sentences(text):
        words = sentence.split()
        if size and size + len(words) > max_words:
            yield " ".join(buf)
            buf, size = [], 0
        if len(words) > max_words:
            for start in range(0, len(words), max_words):
                yield " ".join(words[start : start + max_words])
            continue
        buf.append(sentence)
        size += len(words)
    if buf:
        yield " ".join(buf)


def chunk_blocks(
    work_id: str,
    blocks: Sequence[Block],
    *,
    target_words: int = TARGET_WORDS,
    max_words: int = MAX_WORDS,
    min_words: int = MIN_WORDS,
) -> list[Chunk]:
    """Pack cleaned blocks into chunks, never crossing a section boundary."""
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    section: str | None = None
    heading: str | None = None

    def flush() -> None:
        nonlocal buf, size
        text = " ".join(buf)
        # The prose test runs again here because clean() exempts short blocks from it,
        # and a run of short non-prose lines (a contributor list, an answer scale) packs
        # into a chunk long enough to judge.
        if buf and size >= min_words and section is not None and is_prose(text):
            chunks.append(Chunk(work_id, len(chunks), section, heading, text, size))
        buf, size = [], 0

    for block in blocks:
        if block.section != section:
            flush()
            section, heading = block.section, block.heading
        pieces = (
            list(_split_long(block.text, max_words))
            if len(block.text.split()) > max_words
            else [block.text]
        )
        for piece in pieces:
            words = len(piece.split())
            if size and size + words > max_words:
                flush()
                heading = block.heading
            buf.append(piece)
            size += words
            if size >= target_words:
                flush()
                heading = block.heading
    flush()
    return chunks


def chunk_text(work_id: str, raw: str, size: Budget | None = None) -> list[Chunk]:
    b = size or budget()
    blocks, _ = clean(raw)
    return chunk_blocks(
        work_id,
        blocks,
        target_words=b.target_words,
        max_words=b.max_words,
        min_words=b.min_words,
    )


def _chunk_file(job: tuple[str, str, Budget]) -> tuple[str, list[Chunk]]:
    work_id, text_path, size = job
    path = TEXTS_DIR / Path(text_path).name
    try:
        raw = path.read_text(errors="replace")
    except OSError:
        return work_id, []
    return work_id, chunk_text(work_id, raw, size)


PENDING = """
SELECT work_id, text_path FROM works
 WHERE text_path IS NOT NULL AND chunked_at IS NULL AND relevance >= ?
 ORDER BY relevance DESC
"""

INSERT = (
    "INSERT OR REPLACE INTO chunks"
    " (chunk_id, work_id, idx, section, heading, text, n_words) VALUES (?,?,?,?,?,?,?)"
)


def pending(
    conn: sqlite3.Connection, min_relevance: int, limit: int | None
) -> list[tuple[str, str]]:
    rows = conn.execute(PENDING, (min_relevance,)).fetchall()
    return [(str(r[0]), str(r[1])) for r in rows[:limit]]


def store(conn: sqlite3.Connection, results: Sequence[tuple[str, list[Chunk]]]) -> int:
    """Write each work's chunks and mark it done, replacing any earlier rows."""
    if not results:
        return 0
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    work_ids = [(work_id,) for work_id, _ in results]
    conn.executemany("DELETE FROM chunks WHERE work_id = ?", work_ids)
    conn.executemany(
        INSERT,
        [
            (f"{c.work_id}#{c.idx}", c.work_id, c.idx, c.section, c.heading, c.text, c.n_words)
            for _, chunks in results
            for c in chunks
        ],
    )
    conn.executemany(
        "UPDATE works SET chunked_at = ? WHERE work_id = ?", [(stamp, w) for (w,) in work_ids]
    )
    conn.commit()
    return sum(len(chunks) for _, chunks in results)


def run(
    conn: sqlite3.Connection,
    log: Log,
    *,
    min_relevance: int,
    workers: int = 8,
    limit: int | None = None,
    target_tokens: int = TARGET_TOKENS,
) -> tuple[int, int]:
    """Clean + chunk every pending work at or above min_relevance. Returns (works, chunks)."""
    todo = pending(conn, min_relevance, limit)
    size = budget(target_tokens)
    log(
        f"chunking {len(todo)} works (relevance >= {min_relevance}) on {workers} workers, "
        f"target {target_tokens} tokens ({size.target_words} words, cap {size.max_words})"
    )
    done = written = 0
    batch: list[tuple[str, list[Chunk]]] = []
    jobs = [(work_id, text_path, size) for work_id, text_path in todo]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_chunk_file, jobs, chunksize=POOL_CHUNKSIZE):
            batch.append(result)
            done += 1
            if len(batch) >= COMMIT_WORKS:
                written += store(conn, batch)
                batch = []
                log(f"  {done}/{len(todo)} works, {written} chunks")
    written += store(conn, batch)
    log(f"done: {done} works -> {written} chunks")
    return done, written
