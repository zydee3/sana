from __future__ import annotations

import sqlite3
from typing import Any

from corpus import db, retraction
from corpus.models import Paper

NOW = "2026-08-18T05:00:00Z"


def _paper(work_id: str, *, doi: str | None = "10.0/a", pmcid: str | None = "PMC1") -> Paper:
    return Paper(
        work_id=work_id,
        title=f"title {work_id}",
        year=2020,
        doi=doi,
        pmcid=pmcid,
        discovered_via="",
        status="kept_text",
        study_type=None,
        stratum="",
    )


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE works (work_id TEXT PRIMARY KEY, title TEXT, year INTEGER, doi TEXT,"
        " pmcid TEXT, status TEXT, relevance INTEGER);"
    )
    db.migrate(conn)
    return conn


def _work(conn: sqlite3.Connection, work_id: str, **cols: Any) -> None:
    fields = {
        "title": f"title {work_id}",
        "year": 2020,
        "doi": f"10.0/{work_id}",
        "pmcid": f"PMC{work_id}",
        "status": "kept_text",
        "relevance": 8,
        **cols,
    }
    keys = ", ".join(fields)
    conn.execute(
        f"INSERT INTO works (work_id, {keys}) VALUES (?{',?' * len(fields)})",
        (work_id, *fields.values()),
    )
    conn.commit()


def _oa(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"results": results}


def _epmc(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"resultList": {"result": results}}


def test_openalex_reports_only_the_retracted_and_matches_doi_case_insensitively() -> None:
    papers = [_paper("W1", doi="10.0/AbC"), _paper("W2", doi="10.0/d")]
    got = retraction.fetch_openalex(
        papers,
        lambda _u: _oa(
            [
                {"doi": "https://doi.org/10.0/abc", "is_retracted": True},
                {"doi": "10.0/d", "is_retracted": False},
            ]
        ),
    )
    assert got == {"W1"}


def test_openalex_batch_skipped_when_no_paper_has_a_doi() -> None:
    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return _oa([])

    assert retraction.fetch_openalex([_paper("W1", doi=None)], fetch) == set()
    assert calls == []


def test_api_key_is_sent_when_set(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "k123")
    assert "api_key=k123" in retraction._oa_url(["10.0/a"])
    monkeypatch.delenv("OPENALEX_API_KEY")
    assert "api_key" not in retraction._oa_url(["10.0/a"])


def test_epmc_is_asked_even_when_openalex_says_clean() -> None:
    """The sources lag each other, so this is an OR over both, not a fallback."""
    papers = [_paper("W1", doi="10.0/a", pmcid="PMC1")]

    def fetch(url: str) -> dict[str, Any]:
        if "openalex" in url:
            return _oa([{"doi": "10.0/a", "is_retracted": False}])
        return _epmc([{"pmcid": "PMC1", "pubTypeList": {"pubType": ["Retracted Publication"]}}])

    assert retraction.check(papers, fetch) == (set(), {"W1"})


def test_epmc_ignores_other_pub_types() -> None:
    def fetch(_url: str) -> dict[str, Any]:
        return _epmc([{"pmcid": "PMC1", "pubTypeList": {"pubType": ["Review", "Comment"]}}])

    assert retraction.check([_paper("W1", doi=None)], fetch) == (set(), set())


def test_run_flips_status_stamps_the_rest_and_is_a_no_op_on_rerun() -> None:
    conn = _conn()
    _work(conn, "W1")
    _work(conn, "W2")
    _work(conn, "W3", relevance=3)  # below the pool threshold
    _work(conn, "W4", status="kept_miss")

    def fetch(url: str) -> dict[str, Any]:
        if "openalex" in url:
            return _oa([{"doi": "10.0/W1", "is_retracted": True}])
        return _epmc([])

    stats = retraction.run(conn, lambda _m: None, NOW, fetch=fetch)
    assert (stats["checked"], stats["retracted"], stats["openalex"]) == (2, 1, 1)
    assert conn.execute(
        "SELECT status, retraction_checked_at FROM works WHERE work_id='W1'"
    ).fetchone() == ("retracted", NOW)
    assert conn.execute(
        "SELECT status, retraction_checked_at FROM works WHERE work_id='W2'"
    ).fetchone() == ("kept_text", NOW)
    unchecked = "SELECT retraction_checked_at FROM works WHERE work_id='W3'"
    assert conn.execute(unchecked).fetchone() == (None,)
    assert retraction.run(conn, lambda _m: None, NOW, fetch=fetch)["checked"] == 0


def test_a_failed_slice_is_left_for_the_next_run() -> None:
    conn = _conn()
    _work(conn, "W1")

    def boom(_url: str) -> dict[str, Any]:
        raise TimeoutError("openalex down")

    stats = retraction.run(conn, lambda _m: None, NOW, fetch=boom)
    assert (stats["checked"], stats["deferred"]) == (0, 1)
    assert conn.execute("SELECT status, retraction_checked_at FROM works").fetchone() == (
        "kept_text",
        None,
    )


def test_a_stale_stamp_is_re_checked_and_a_fresh_one_is_not() -> None:
    conn = _conn()
    _work(conn, "W1", retraction_checked_at="2026-08-10T00:00:00Z")
    _work(conn, "W2", retraction_checked_at="2026-08-17T23:00:00Z")
    cutoff = "2026-08-11T00:00:00Z"

    def fetch(url: str) -> dict[str, Any]:
        if "openalex" in url:
            return _oa([{"doi": "10.0/W1", "is_retracted": True}])
        return _epmc([])

    stats = retraction.run(conn, lambda _m: None, NOW, fetch=fetch, stale_before=cutoff)
    assert (stats["checked"], stats["retracted"]) == (1, 1)
    assert conn.execute(
        "SELECT status, retraction_checked_at FROM works WHERE work_id='W1'"
    ).fetchone() == ("retracted", NOW)
    assert conn.execute(
        "SELECT retraction_checked_at FROM works WHERE work_id='W2'"
    ).fetchone() == ("2026-08-17T23:00:00Z",)


def test_a_retracted_work_is_never_re_opened_by_a_stale_sweep() -> None:
    conn = _conn()
    _work(conn, "W1", status="retracted", retraction_checked_at="2026-08-10T00:00:00Z")

    def boom(_url: str) -> dict[str, Any]:
        raise AssertionError("should not have been fetched")

    stats = retraction.run(conn, lambda _m: None, NOW, fetch=boom, stale_before=NOW)
    assert stats["checked"] == 0


def test_a_full_sweep_re_checks_every_stamped_pool_work() -> None:
    conn = _conn()
    _work(conn, "W1", retraction_checked_at="2026-08-17T23:00:00Z")
    _work(conn, "W2")  # never checked
    assert [p.work_id for p in retraction.pending(conn, stale_before=NOW)] == ["W1", "W2"]
    assert [p.work_id for p in retraction.pending(conn)] == ["W2"]
