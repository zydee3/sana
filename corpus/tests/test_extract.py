from __future__ import annotations

import json
import sqlite3
import subprocess
from typing import Any

import pytest

from corpus import db, extract
from corpus.extract import Chunk, ExtractError, Usage, Work

TEXT = (
    "Participants slept 42 minutes longer after eight weeks of therapy. "
    "The effect held at follow-up.\nA second paragraph mentions nothing useful."
)


def _work(work_id: str = "W1", n_chunks: int = 1) -> Work:
    chunks = tuple(
        Chunk(f"{work_id}#{i}", "results", TEXT, len(TEXT.split())) for i in range(n_chunks)
    )
    return Work(work_id, "A sleep trial", 2021, chunks)


def _raw(**over: Any) -> dict[str, Any]:
    item = {
        "claim": "Eight weeks of therapy added about 42 minutes of sleep a night.",
        "caveats": "40 adults with chronic insomnia, 8 weeks, no active control.",
        "chunk_id": "W1#0",
        "quote": "Participants slept 42 minutes longer after eight weeks of therapy.",
    }
    item.update(over)
    return item


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE works (work_id TEXT PRIMARY KEY, title TEXT, year INTEGER, status TEXT,"
        " relevance INTEGER);"
    )
    db.migrate(conn)
    return conn


def _store_work(conn: sqlite3.Connection, work: Work, *, relevance: int = 8) -> None:
    conn.execute(
        "INSERT INTO works (work_id, title, year, status, relevance) VALUES (?,?,?,?,?)",
        (work.work_id, work.title, work.year, "kept_text", relevance),
    )
    for i, c in enumerate(work.chunks):
        conn.execute(
            "INSERT INTO chunks (chunk_id, work_id, idx, section, heading, text, n_words)"
            " VALUES (?,?,?,?,?,?,?)",
            (c.chunk_id, work.work_id, i, c.section, None, c.text, c.n_words),
        )
    conn.commit()


def test_locate_finds_an_exact_quote() -> None:
    span = extract.locate(TEXT, "The effect held at follow-up.")
    assert span is not None and TEXT[span[0] : span[1]] == "The effect held at follow-up."


def test_locate_survives_typography_and_line_breaks() -> None:
    text = "the authors’ view\nwas that sleep improved"
    span = extract.locate(text, "the authors' view was that sleep improved")
    assert span == (0, len(text))


def test_locate_rejects_a_paraphrase() -> None:
    assert extract.locate(TEXT, "Participants slept much longer after therapy.") is None


def test_finding_id_is_stable_and_independent_of_the_chunk() -> None:
    a = extract.validate(_work(), [_raw()])[0][0]
    b = extract.validate(_work(n_chunks=2), [_raw(chunk_id="W1#1")])[0][0]
    assert a.finding_id == b.finding_id == extract.finding_id("W1", a.claim)
    assert a.anchor_chunk_id != b.anchor_chunk_id


def test_finding_id_differs_per_work() -> None:
    assert extract.finding_id("W1", "same claim") != extract.finding_id("W2", "same claim")


@pytest.mark.parametrize("caveats", ["", "none", "N/A", "Not reported", "  none  ", "no caveats"])
def test_empty_or_evasive_caveats_fail_validation(caveats: str) -> None:
    kept, drops = extract.validate(_work(), [_raw(caveats=caveats)])
    assert kept == [] and drops[0]["reason"] == "empty_caveats"


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"claim": "too short"}, "claim_length"),
        ({"chunk_id": "W9#0"}, "unknown_chunk"),
        ({"quote": "short"}, "quote_length"),
        (
            {"quote": "Participants slept much longer, the authors wrote in the paper."},
            "quote_not_found",
        ),
    ],
)
def test_validation_drops_with_a_reason(over: dict[str, Any], reason: str) -> None:
    kept, drops = extract.validate(_work(), [_raw(**over)])
    assert kept == [] and drops[0]["reason"] == reason


def test_validation_caps_findings_and_drops_duplicate_claims() -> None:
    kept, drops = extract.validate(_work(), [_raw(), _raw()])
    assert len(kept) == 1 and drops[0]["reason"] == "duplicate_claim"
    many = [_raw(claim=f"Therapy added about {i} minutes of sleep each night.") for i in range(10)]
    kept, drops = extract.validate(_work(), many)
    assert len(kept) == extract.MAX_FINDINGS
    assert [d["reason"] for d in drops] == ["over_limit"] * 2


def test_anchor_quote_is_the_stored_text_not_the_model_text() -> None:
    work = Work("W1", "t", 2020, (Chunk("W1#0", "results", "the authors’ view held", 4),))
    kept, _ = extract.validate(
        work,
        [
            _raw(
                chunk_id="W1#0",
                quote="the authors' view held and then some padding to clear the minimum",
            )
        ],
    )
    assert kept == []  # a quote longer than the chunk cannot be located
    kept, _ = extract.validate(
        Work("W1", "t", 2020, (Chunk("W1#0", "results", TEXT.replace("42", "forty-two"), 20),)),
        [_raw(quote="Participants slept forty-two minutes longer after eight weeks of therapy.")],
    )
    assert (
        kept[0].quote == "Participants slept forty-two minutes longer after eight weeks of therapy."
    )


def test_prompt_carries_chunk_tags_and_truncates_long_works(monkeypatch: Any) -> None:
    monkeypatch.setattr(extract, "MAX_WORK_WORDS", 15)
    work = _work(n_chunks=3)
    prompt, meta = extract.build_prompt([work])
    assert "[W1#0 | results]" in prompt and "[W1#2 | results]" not in prompt
    assert meta[0][0] is True and meta[0][1] <= 15 + work.chunks[0].n_words


def test_parse_reply_rejects_a_misaligned_or_malformed_reply() -> None:
    works = [_work("W1"), _work("W2")]
    with pytest.raises(ExtractError):
        extract.parse_reply(json.dumps([{"paper": 1, "findings": []}]), works)
    with pytest.raises(ExtractError):
        extract.parse_reply(
            json.dumps([{"paper": 2, "findings": []}, {"paper": 1, "findings": []}]), works
        )
    with pytest.raises(ExtractError):
        extract.parse_reply("not json", works[:1])
    with pytest.raises(ExtractError):
        extract.parse_reply(json.dumps([{"paper": 1}]), works[:1])


def test_parse_reply_accepts_fenced_json() -> None:
    reply = "```json\n" + json.dumps([{"paper": 1, "findings": [_raw()]}]) + "\n```"
    ((kept, drops),) = extract.parse_reply(reply, [_work()])
    assert len(kept) == 1 and not drops


def test_pending_skips_extracted_chunkless_and_low_relevance_works() -> None:
    conn = _conn()
    _store_work(conn, _work("W1"))
    _store_work(conn, _work("W2"), relevance=6)
    _store_work(conn, _work("W3"))
    conn.execute("UPDATE works SET extracted_at = '2026-01-01' WHERE work_id = 'W3'")
    conn.execute(
        "INSERT INTO works (work_id, title, year, status, relevance) VALUES ('W4','t',2020,"
        " 'kept_text', 9)"  # judged high but never chunked
    )
    conn.commit()
    assert extract.pending_ids(conn) == ["W1"]


def test_store_stamps_works_that_produced_nothing() -> None:
    conn = _conn()
    _store_work(conn, _work("W1"))
    extract.store(conn, ["W1"], [])
    assert extract.pending_ids(conn) == []
    assert conn.execute("SELECT count(*) FROM findings").fetchone()[0] == 0


def test_run_stores_findings_and_is_a_no_op_on_restart() -> None:
    conn = _conn()
    _store_work(conn, _work("W1"))
    calls: list[str] = []

    def runner(prompt: str, model: str) -> tuple[str, Usage]:
        calls.append(prompt)
        return json.dumps([{"paper": 1, "findings": [_raw()]}]), Usage(1000, 50, 0, 0.01, 1.0)

    done, failed, usage = extract.run(conn, lambda _m: None, runner=runner)
    assert (done, failed) == (1, 0) and usage.output_tokens == 50
    row = conn.execute("SELECT finding_id, char_start, char_end, quote FROM findings").fetchone()
    assert row[0].startswith("f_") and row[3] == TEXT[row[1] : row[2]]
    done, failed, _ = extract.run(conn, lambda _m: None, runner=runner)
    assert (done, failed) == (0, 0) and len(calls) == 1


def test_run_dry_run_writes_nothing_and_reports() -> None:
    conn = _conn()
    _store_work(conn, _work("W1"))
    seen: list[extract.WorkResult] = []

    def runner(prompt: str, model: str) -> tuple[str, Usage]:
        return json.dumps([{"paper": 1, "findings": [_raw()]}]), Usage(1000, 50, 0, 0.01, 1.0)

    extract.run(
        conn, lambda _m: None, runner=runner, dry_run=True, report=lambda r, u: seen.append(r)
    )
    assert conn.execute("SELECT count(*) FROM findings").fetchone()[0] == 0
    assert extract.pending_ids(conn) == ["W1"] and len(seen[0].findings) == 1


def test_run_leaves_a_failed_batch_pending_and_brakes() -> None:
    conn = _conn()
    for i in range(6):
        _store_work(conn, _work(f"W{i}"))

    def runner(prompt: str, model: str) -> tuple[str, Usage]:
        raise ExtractError("boom")

    done, failed, _ = extract.run(conn, lambda _m: None, runner=runner, workers=1, brake=2)
    assert done == 0 and failed == 2  # braked after two failed batches, rest untouched
    assert len(extract.pending_ids(conn)) == 6


def test_run_sleeps_off_a_quota_refusal_instead_of_braking() -> None:
    """A refused window is a wait: the same works are retried after the reset, not lost."""
    conn = _conn()
    _store_work(conn, _work("W1"))
    slept: list[float] = []
    calls: list[int] = []

    def runner(prompt: str, model: str) -> tuple[str, Usage]:
        calls.append(1)
        if len(calls) <= 2:  # _attempt retries once, so a refused batch is two calls
            raise ExtractError("claude -p exited 1: You've hit your monthly spend limit")
        return json.dumps([{"paper": 1, "findings": [_raw()]}]), Usage(1000, 50, 0, 0.01, 1.0)

    done, failed, _ = extract.run(
        conn, lambda _m: None, runner=runner, workers=1, brake=1, sleep=slept.append
    )
    assert (done, failed) == (1, 0) and len(slept) == 1
    assert extract.pending_ids(conn) == []


def test_run_stops_spending_a_window_at_the_budget() -> None:
    conn = _conn()
    for i in range(3):
        _store_work(conn, _work(f"W{i}"))
    slept: list[float] = []

    def runner(prompt: str, model: str) -> tuple[str, Usage]:
        return json.dumps([{"paper": 1, "findings": [_raw()]}]), Usage(1000, 50, 0, 0.01, 1.0)

    done, _, _ = extract.run(
        conn, lambda _m: None, runner=runner, workers=1, per_window=1, sleep=slept.append
    )
    assert done == 3 and len(slept) == 2  # one sleep between each work, none before the first


def test_nonzero_exit_reports_stdout_when_stderr_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI prints spend-limit refusals on stdout; the error must carry them."""

    class Proc:
        returncode = 1
        stdout = "You've hit your monthly spend limit"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Proc())
    with pytest.raises(ExtractError, match="monthly spend limit"):
        extract.run_claude("prompt", "sonnet")
