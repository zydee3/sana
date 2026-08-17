from __future__ import annotations

import sqlite3

import pytest

from corpus import db, quality


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE works (work_id TEXT PRIMARY KEY, title TEXT, status TEXT,"
        " evidence_grade INTEGER, relevance INTEGER);"
    )
    db.migrate(conn)
    return conn


def _store(conn: sqlite3.Connection, work_id: str, **cols: object) -> None:
    keys = ", ".join(cols)
    conn.execute(
        f"INSERT INTO works (work_id, {keys}) VALUES (?{',?' * len(cols)})",
        (work_id, *cols.values()),
    )
    conn.commit()


def _score(relevance: int | None, gate: float | None, grade: int | None) -> float:
    composed = quality.compose(relevance, gate, grade)
    assert composed is not None
    return composed[0]


def test_relevance_wins_over_gate_and_normalizes_by_ten() -> None:
    assert quality.compose(8, 0.99, None) == (0.8, quality.SONNET)


def test_gate_is_discounted_by_measured_precision() -> None:
    composed = quality.compose(None, 1.0, None)
    assert composed is not None
    assert composed[1] == quality.GATE
    assert composed[0] == pytest.approx(quality.GATE_PRECISION)


def test_no_signal_has_no_quality() -> None:
    assert quality.compose(None, None, 2) is None


def test_grade_only_discounts_and_unknown_reads_as_grade_one() -> None:
    top = _score(10, None, 1)
    assert top == 1.0
    assert _score(10, None, None) == top
    ladder = [_score(10, None, g) for g in (1, 2, 3, 4, 5)]
    assert ladder == sorted(ladder, reverse=True)
    assert ladder[-1] == pytest.approx(1.0 - 4 * quality.GRADE_PENALTY)


def test_stays_inside_zero_to_one() -> None:
    assert _score(0, None, 5) == 0.0
    assert _score(10, None, 1) <= 1.0


def test_sql_recompute_matches_the_python_spec() -> None:
    conn = _conn()
    rows = [
        ("W1", 8, None, None),
        ("W2", 7, None, 5),
        ("W3", None, 0.9, 2),
        ("W4", None, 0.4, None),
        ("W5", 10, 0.1, 1),
        ("W6", 0, None, 5),
    ]
    for work_id, relevance, gate, grade in rows:
        _store(conn, work_id, relevance=relevance, gate_p5=gate, evidence_grade=grade)
    _store(conn, "W7", relevance=None, gate_p5=None, evidence_grade=3)

    assert quality.recompute(conn) == len(rows)
    read = conn.execute("SELECT work_id, quality, quality_source FROM works")
    stored = {w: (q, s) for w, q, s in read}
    for work_id, relevance, gate, grade in rows:
        assert stored[work_id] == quality.compose(relevance, gate, grade), work_id
    assert stored["W7"] == (None, None)


def test_recompute_is_idempotent_and_follows_a_changed_signal() -> None:
    conn = _conn()
    _store(conn, "W1", relevance=7, gate_p5=None, evidence_grade=None)
    quality.recompute(conn)
    before = conn.execute("SELECT quality FROM works").fetchone()[0]
    quality.recompute(conn)
    assert conn.execute("SELECT quality FROM works").fetchone()[0] == before

    conn.execute("UPDATE works SET relevance = 9")
    quality.recompute(conn)
    assert conn.execute("SELECT quality FROM works").fetchone()[0] == 0.9
