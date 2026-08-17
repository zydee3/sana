from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from corpus import db, judge
from corpus.classify import ClassifyError
from corpus.models import Paper, Verdict


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE works (work_id TEXT PRIMARY KEY, title TEXT, year INTEGER, doi TEXT,"
        " pmcid TEXT, discovered_via TEXT, status TEXT, study_type TEXT, text_path TEXT);"
    )
    db.migrate(conn)
    return conn


def _work(conn: sqlite3.Connection, work_id: str, **kw: object) -> None:
    row = {
        "title": f"title {work_id}",
        "year": 2021,
        "doi": None,
        "pmcid": None,
        "discovered_via": "europepmc",
        "status": "kept_text",
        "study_type": None,
        "text_path": None,
        **kw,
    }
    conn.execute(
        "INSERT INTO works (work_id, title, year, doi, pmcid, discovered_via, status,"
        " study_type, text_path) VALUES (?,?,?,?,?,?,?,?,?)",
        (work_id, *row.values()),
    )
    conn.commit()


def _statusline(
    path: Path, five: float, seven: float, resets_at: float, age_s: float = 0.0
) -> Path:
    path.write_text(
        json.dumps(
            {
                "rate_limits": {
                    "five_hour": {"used_percentage": five, "resets_at": resets_at},
                    "seven_day": {"used_percentage": seven, "resets_at": resets_at},
                }
            }
        )
    )
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def test_quota_wait_is_zero_below_the_ceiling(tmp_path: Path) -> None:
    now = time.time()
    path = _statusline(tmp_path / "s.json", 35, 38, now + 3600)
    assert judge.quota_wait_s(now, path) == 0.0


def test_quota_wait_covers_the_later_reset(tmp_path: Path) -> None:
    now = time.time()
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "rate_limits": {
                    "five_hour": {"used_percentage": 92, "resets_at": now + 100},
                    "seven_day": {"used_percentage": 99, "resets_at": now + 900},
                }
            }
        )
    )
    assert judge.quota_wait_s(now, path) == 960.0


def test_stale_statusline_does_not_block(tmp_path: Path) -> None:
    now = time.time()
    path = _statusline(tmp_path / "s.json", 99, 99, now + 3600, age_s=judge.STALE_S + 60)
    assert judge.quota_wait_s(now, path) == 0.0


def test_missing_statusline_does_not_block(tmp_path: Path) -> None:
    assert judge.quota_wait_s(time.time(), tmp_path / "absent.json") == 0.0


def test_quota_refusal_matches_the_cli_spend_limit_text() -> None:
    refusal = "claude -p exited 1: You've hit your monthly spend limit · raise it at claude.ai"
    assert judge.is_quota_refusal(refusal)
    assert not judge.is_quota_refusal("claude -p exited 1: reply is not JSON")


def test_next_reset_rolls_a_stale_window_forward(tmp_path: Path) -> None:
    now = time.time()
    ahead = _statusline(tmp_path / "a.json", 99, 20, now + 600)
    assert judge.next_reset_s(now, ahead) == pytest.approx(660.0)
    # resets_at from two windows ago: the current window ends 1h from now, not in the past.
    behind = _statusline(tmp_path / "b.json", 99, 20, now - 2 * judge.WINDOW_S + 3600)
    assert judge.next_reset_s(now, behind) == pytest.approx(3660.0)


def test_next_reset_without_a_statusline_is_a_short_retry(tmp_path: Path) -> None:
    assert judge.next_reset_s(time.time(), tmp_path / "absent.json") == judge.UNKNOWN_RESET_S


def test_pending_skips_judged_and_non_kept_rows() -> None:
    conn = _conn()
    _work(conn, "W1")
    _work(conn, "W2", status="rejected")
    _work(conn, "W3")
    conn.execute("UPDATE works SET relevance = 5 WHERE work_id = 'W3'")
    conn.commit()
    assert judge.pending_ids(conn) == ["W1"]


def test_pending_can_be_restricted_to_one_stratum() -> None:
    conn = _conn()
    _work(conn, "W1", discovered_via="openalex")
    _work(conn, "W2", discovered_via="citation")
    _work(conn, "W3", discovered_via="europepmc")
    _work(conn, "W4", discovered_via="openalex", status="kept_miss")
    assert sorted(judge.pending_ids(conn, via=["openalex", "citation"], status="kept_text")) == [
        "W1",
        "W2",
    ]


def test_pending_can_be_restricted_to_a_drawn_sample() -> None:
    conn = _conn()
    for i in range(4):
        _work(conn, f"W{i}")
    conn.execute("UPDATE works SET relevance = 5 WHERE work_id = 'W1'")
    conn.commit()
    # W1 is in the sample but already judged, W9 is not a row at all: neither comes back.
    assert sorted(judge.pending_ids(conn, only=["W0", "W1", "W9"])) == ["W0"]


def test_store_keeps_a_publisher_study_type_and_says_so() -> None:
    conn = _conn()
    _work(conn, "W1", study_type="rct")
    _work(conn, "W2")
    judge.store_verdicts(
        conn,
        [Verdict("W1", 8, "sleep", "cohort", 0.7), Verdict("W2", 3, "pain", "opinion", 0.4)],
        "sonnet",
    )
    rows = dict(
        (r[0], r[1:])
        for r in conn.execute(
            "SELECT work_id, relevance, domain, study_type, label_source, label_confidence"
            " FROM works ORDER BY work_id"
        )
    )
    assert rows["W1"] == (8, "sleep", "rct", "publisher", 0.7)
    assert rows["W2"] == (3, "pain", "opinion", "claude-sonnet", 0.4)
    assert judge.pending_ids(conn) == []


def test_load_papers_preserves_batch_order_and_attaches_abstracts() -> None:
    conn = _conn()
    _work(conn, "W1")
    _work(conn, "W2")
    conn.execute(
        "INSERT INTO abstracts (work_id, abstract, source, fetched_at)"
        " VALUES ('W2','abs','text','t')"
    )
    conn.commit()
    papers = judge.load_papers(conn, ["W2", "W1"])
    assert [p.work_id for p in papers] == ["W2", "W1"]
    assert papers[0].abstract == "abs"
    assert papers[1].abstract is None


def test_a_batch_that_fails_twice_is_left_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def boom(papers: list[Paper], model: str) -> list[Verdict]:
        calls.append(model)
        raise ClassifyError("nope")

    monkeypatch.setattr(judge, "classify_batch", boom)
    paper = Paper("W1", "t", 2020, None, None, "europepmc", "kept_text", None, "s")
    assert judge._judge([paper], "sonnet") == ([], "nope")
    assert calls == ["sonnet", "sonnet"]


def test_the_brake_stops_a_run_whose_batches_all_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn()
    for i in range(60):
        _work(conn, f"W{i}")
    calls: list[int] = []

    def boom(papers: list[Paper], model: str) -> list[Verdict]:
        calls.append(len(papers))
        raise ClassifyError("You've hit your monthly spend limit")

    monkeypatch.setattr(judge, "classify_batch", boom)
    monkeypatch.setattr(judge, "wait_for_quota", lambda *a: 0.0)
    done, failed = judge.run(conn, lambda _: None, workers=2, batch_size=1, brake=3)
    assert (done, failed) == (0, 4)  # the round of 2 that trips the brake still finishes
    assert len(calls) == 8  # 4 batches x one retry each, not all 60
    assert len(judge.pending_ids(conn)) == 60  # a failed run consumes nothing


def test_a_success_resets_the_failure_streak(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn()
    for i in range(6):
        _work(conn, f"W{i}")
    seen: list[str] = []

    def flaky(papers: list[Paper], model: str) -> list[Verdict]:
        seen.append(papers[0].work_id)
        if len(seen) % 3 == 0:
            return [Verdict(p.work_id, 5, "sleep", "other", 0.5) for p in papers]
        raise ClassifyError("transient")

    monkeypatch.setattr(judge, "classify_batch", flaky)
    monkeypatch.setattr(judge, "wait_for_quota", lambda *a: 0.0)
    done, failed = judge.run(conn, lambda _: None, workers=1, batch_size=1, brake=2)
    assert done > 0 and done + failed == 6  # never tripped: every other batch succeeds
