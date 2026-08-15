from __future__ import annotations

import sqlite3

from corpus import lexical


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, work_id TEXT, idx INTEGER, text TEXT);"
    )
    rows = [
        ("W1#0", "W1", 0, "stimulus control and sleep restriction shorten sleep onset latency"),
        ("W2#0", "W2", 0, "graded activity for chronic low back pain when imaging is normal"),
        ("W3#0", "W3", 0, "evening screen light suppresses melatonin and delays sleep"),
    ]
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?)", rows)
    conn.commit()
    return conn


def test_to_match_strips_punctuation_and_short_words() -> None:
    assert lexical.to_match("i can't fall asleep, i just lie there") == (
        '"can" OR "fall" OR "asleep" OR "just" OR "lie" OR "there"'
    )
    assert lexical.to_match("?? i am") == ""


def test_build_is_a_noop_when_already_current() -> None:
    conn = _db()
    logged: list[str] = []
    assert lexical.build(conn, logged.append) == 3
    assert lexical.build(conn, logged.append) == 3
    assert lexical.indexed(conn) == 3
    assert "already covers" in logged[-1]


def test_search_ranks_by_bm25_and_returns_chunk_ids() -> None:
    conn = _db()
    lexical.build(conn, lambda _m: None)
    # Two chunks mention sleep; the one that mentions it twice ranks first.
    assert lexical.search(conn, "why can't i sleep at night", 2)[0] == "W1#0"
    assert lexical.search(conn, "?!", 5) == []


def test_build_reindexes_chunks_added_after_the_last_build() -> None:
    conn = _db()
    lexical.build(conn, lambda _m: None)
    conn.execute(
        "INSERT INTO chunks VALUES (?,?,?,?)",
        ("W4#0", "W4", 0, "mindfulness based stress reduction lowers perceived stress"),
    )
    conn.commit()
    # count(*) on the external-content table reads `chunks`, so staleness is only
    # visible in the shadow docsize table.
    assert conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 4
    assert lexical.indexed(conn) == 3
    assert lexical.build(conn, lambda _m: None) == 4
    assert lexical.search(conn, "mindfulness stress reduction", 5) == ["W4#0"]


def test_rebuild_repopulates_after_the_chunk_set_changes() -> None:
    conn = _db()
    lexical.build(conn, lambda _m: None)
    conn.execute("DELETE FROM chunks WHERE chunk_id = 'W3#0'")
    conn.commit()
    assert lexical.build(conn, lambda _m: None, rebuild=True) == 2
    assert lexical.search(conn, "melatonin screen light", 5) == []


def test_rrf_prefers_what_both_arms_rank_and_keeps_arm_only_hits() -> None:
    dense = ["a", "b", "c"]
    bm25 = ["c", "d", "a"]
    assert lexical.rrf([dense, bm25], 4) == ["a", "c", "b", "d"]
