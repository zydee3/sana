from __future__ import annotations

import sqlite3

from corpus import chunk, db
from corpus.clean import Block


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE works (work_id TEXT PRIMARY KEY, title TEXT, text_path TEXT);")
    db.migrate(conn)
    return conn


def _para(words: int, word: str = "sleep") -> str:
    # Function words matter: the cleaner's prose test also guards the packed chunk.
    filler = (word, "of", "the", "patients")
    return " ".join(filler[i % len(filler)] for i in range(words))


def test_chunks_never_cross_a_section() -> None:
    blocks = [
        Block("methods", "Methods", _para(40)),
        Block("results", "Results", _para(40)),
    ]
    chunks = chunk.chunk_blocks("W1", blocks, target_words=200, max_words=260, min_words=25)
    assert [c.section for c in chunks] == ["methods", "results"]
    assert [c.idx for c in chunks] == [0, 1]


def test_paragraphs_pack_up_to_the_target() -> None:
    blocks = [Block("results", "Results", _para(60)) for _ in range(6)]
    chunks = chunk.chunk_blocks("W1", blocks, target_words=200, max_words=260, min_words=25)
    assert [c.n_words for c in chunks] == [240, 120]


def test_long_paragraph_splits_on_sentences() -> None:
    para = " ".join(f"{_para(50)} number {i}." for i in range(8))
    chunks = chunk.chunk_blocks(
        "W1",
        [Block("discussion", "Discussion", para)],
        target_words=200,
        max_words=210,
        min_words=25,
    )
    assert len(chunks) > 1
    assert all(c.n_words <= 210 for c in chunks)
    # every sentence survives the split
    assert sum(c.text.count("number") for c in chunks) == 8


def test_short_tail_is_dropped() -> None:
    chunks = chunk.chunk_blocks(
        "W1",
        [Block("results", "Results", _para(10))],
        target_words=200,
        max_words=260,
        min_words=25,
    )
    assert chunks == []


def test_run_is_resumable_and_replaces_rows() -> None:
    conn = _conn()
    conn.execute("INSERT INTO works (work_id, title, text_path) VALUES ('W1','t','/x/a.txt')")
    conn.execute("UPDATE works SET relevance = 7 WHERE work_id = 'W1'")
    conn.commit()
    assert chunk.pending(conn, 5, None) == [("W1", "/x/a.txt")]

    # store() stands in for the pool: it is what marks the work done.
    made = [chunk.Chunk("W1", 0, "results", "Results", _para(100), 100)]
    assert chunk.store(conn, [("W1", made)]) == 1
    assert chunk.pending(conn, 5, None) == []
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1

    chunk.store(conn, [("W1", made + [chunk.Chunk("W1", 1, "results", None, _para(80), 80)])])
    assert conn.execute("SELECT count(*) FROM chunks WHERE work_id='W1'").fetchone()[0] == 2


def test_empty_text_still_marks_the_work_done() -> None:
    conn = _conn()
    conn.execute("INSERT INTO works (work_id, title, text_path) VALUES ('W2','t','/x/b.txt')")
    conn.execute("UPDATE works SET relevance = 9 WHERE work_id = 'W2'")
    conn.commit()
    chunk.store(conn, [("W2", [])])
    assert chunk.pending(conn, 5, None) == []
    assert conn.execute("SELECT chunked_at FROM works WHERE work_id='W2'").fetchone()[0]


def test_below_threshold_works_are_not_pending() -> None:
    conn = _conn()
    conn.execute("INSERT INTO works (work_id, title, text_path) VALUES ('W3','t','/x/c.txt')")
    conn.execute("UPDATE works SET relevance = 4 WHERE work_id = 'W3'")
    conn.commit()
    assert chunk.pending(conn, 5, None) == []


def test_budget_scales_the_cap_with_the_target() -> None:
    default, smaller = chunk.budget(), chunk.budget(128)
    assert (default.target_words, default.max_words) == (chunk.TARGET_WORDS, chunk.MAX_WORDS)
    assert smaller.target_words < default.target_words
    # The cap keeps the same headroom ratio, so a smaller target cannot clip the encoder.
    ratio = default.max_words / default.target_words
    assert abs(smaller.max_words / smaller.target_words - ratio) < 0.02


def test_a_smaller_target_shrinks_the_chunks() -> None:
    raw = "\n\n".join(f"{_para(60)} number {i}." for i in range(20))
    big = chunk.chunk_text("W1", raw)
    small = chunk.chunk_text("W1", raw, chunk.budget(128))
    assert max(c.n_words for c in small) <= chunk.budget(128).max_words
    assert max(c.n_words for c in small) < max(c.n_words for c in big)
    assert sum(c.n_words for c in small) == sum(c.n_words for c in big)


def test_packed_run_of_non_prose_lines_is_dropped() -> None:
    # Each line is short enough that clean() exempts it, but together they pack into a
    # chunk that is plainly not prose.
    lines = [Block("methods", None, "Writing - review & editing: Park JE, Baek CH.")] * 8
    assert chunk.chunk_blocks("w", lines) == []
