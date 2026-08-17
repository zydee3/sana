from __future__ import annotations

import sqlite3
from typing import Any

from corpus import db, venue
from corpus.models import Paper


def _paper(work_id: str, *, doi: str | None = "10.0/a", pmcid: str | None = "PMC1") -> Paper:
    return Paper(
        work_id=work_id,
        title=f"title {work_id}",
        year=2020,
        doi=doi,
        pmcid=pmcid,
        discovered_via="openalex",
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


def _hit(doi: str, name: str) -> dict[str, Any]:
    return _oa([{"doi": doi, "primary_location": {"source": {"display_name": name}}}])


def test_openalex_maps_by_doi_case_insensitively() -> None:
    papers = [_paper("W1", doi="10.0/AbC")]
    got = venue.fetch_openalex(
        papers,
        lambda _u: _hit("https://doi.org/10.0/abc", "BMC Medicine"),
    )
    assert got == {"W1": "BMC Medicine"}


def test_openalex_skips_records_with_no_source_name() -> None:
    got = venue.fetch_openalex(
        [_paper("W1", doi="10.0/a")],
        lambda _u: _oa([{"doi": "10.0/a", "primary_location": {"source": {}}}]),
    )
    assert got == {}


def test_openalex_batch_skipped_when_no_paper_has_a_doi() -> None:
    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return _oa([])

    assert venue.fetch_openalex([_paper("W1", doi=None)], fetch) == {}
    assert calls == []


def test_api_key_is_sent_when_set(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "k123")
    assert "api_key=k123" in venue._oa_url(["10.0/a"])
    monkeypatch.delenv("OPENALEX_API_KEY")
    assert "api_key" not in venue._oa_url(["10.0/a"])


def test_resolve_falls_back_to_epmc_then_missing() -> None:
    papers = [
        _paper("W1", doi="10.0/a"),
        _paper("W2", doi=None, pmcid="PMC2"),
        _paper("W3", doi="10.0/c"),
    ]

    def fetch(url: str) -> dict[str, Any]:
        if "openalex" in url:
            return _hit("10.0/a", "Sleep")
        return {
            "resultList": {
                "result": [{"pmcid": "PMC2", "journalInfo": {"journal": {"title": "BMC medicine"}}}]
            }
        }

    assert venue.resolve(papers, fetch) == [
        ("W1", "Sleep", "openalex"),
        ("W2", "BMC medicine", "epmc"),
        ("W3", None, "missing"),
    ]


def test_run_is_a_no_op_on_rerun() -> None:
    conn = _conn()
    _work(conn, "W1")
    _work(conn, "W2", relevance=3)  # below the pool threshold
    _work(conn, "W3", status="kept_miss")

    def fetch(url: str) -> dict[str, Any]:
        if "openalex" in url:
            return _hit("10.0/W1", "Pain")
        return {"resultList": {"result": []}}

    attempted, resolved = venue.run(conn, lambda _m: None, fetch=fetch)
    assert (attempted, resolved) == (1, 1)
    assert conn.execute("SELECT venue, venue_source FROM works WHERE work_id='W1'").fetchone() == (
        "Pain",
        "openalex",
    )
    assert venue.run(conn, lambda _m: None, fetch=fetch) == (0, 0)


def test_missing_rows_are_closed_out_not_retried() -> None:
    conn = _conn()
    _work(conn, "W1")
    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return {"results": [], "resultList": {"result": []}}

    assert venue.run(conn, lambda _m: None, fetch=fetch) == (1, 0)
    assert conn.execute("SELECT venue, venue_source FROM works").fetchone() == (None, "missing")
    calls.clear()
    assert venue.run(conn, lambda _m: None, fetch=fetch) == (0, 0)
    assert calls == []


def test_a_failed_batch_is_left_for_the_next_run() -> None:
    conn = _conn()
    _work(conn, "W1")

    def boom(_url: str) -> dict[str, Any]:
        raise TimeoutError("epmc down")

    assert venue.run(conn, lambda _m: None, fetch=boom) == (1, 0)
    assert conn.execute("SELECT venue_source FROM works").fetchone() == (None,)
