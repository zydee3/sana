from __future__ import annotations

import json
import sqlite3
from typing import Any

from corpus import authors, bundle, db
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
        " pmcid TEXT, status TEXT, relevance INTEGER, authors TEXT);"
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
        "authors": None,
        **cols,
    }
    keys = ", ".join(fields)
    conn.execute(
        f"INSERT INTO works (work_id, {keys}) VALUES (?{',?' * len(fields)})",
        (work_id, *fields.values()),
    )
    conn.commit()


def _oa(doi: str, names: list[str]) -> dict[str, Any]:
    return {
        "results": [{"doi": doi, "authorships": [{"author": {"display_name": n}} for n in names]}]
    }


def test_openalex_maps_by_doi_case_insensitively() -> None:
    got = authors.fetch_openalex(
        [_paper("W1", doi="10.0/AbC")],
        lambda _u: _oa("https://doi.org/10.0/abc", ["Evelyn J. Bromet", "Irving Hwang"]),
    )
    assert got == {"W1": ["Evelyn J. Bromet", "Irving Hwang"]}


def test_openalex_skips_records_with_no_named_author() -> None:
    got = authors.fetch_openalex(
        [_paper("W1", doi="10.0/a")],
        lambda _u: {"results": [{"doi": "10.0/a", "authorships": [{"author": {}}]}]},
    )
    assert got == {}


def test_openalex_batch_skipped_when_no_paper_has_a_doi() -> None:
    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return {"results": []}

    assert authors.fetch_openalex([_paper("W1", doi=None)], fetch) == {}
    assert calls == []


def test_resolve_falls_back_to_epmc_then_missing() -> None:
    papers = [
        _paper("W1", doi="10.0/a"),
        _paper("W2", doi=None, pmcid="PMC2"),
        _paper("W3", doi="10.0/c"),
    ]

    def fetch(url: str) -> dict[str, Any]:
        if "openalex" in url:
            return _oa("10.0/a", ["Daisy R. Singla"])
        return {
            "resultList": {
                "result": [
                    {
                        "pmcid": "PMC2",
                        "authorList": {"author": [{"fullName": "Murray JK"}]},
                    }
                ]
            }
        }

    assert authors.resolve(papers, fetch) == [
        ("W1", ["Daisy R. Singla"], "openalex"),
        ("W2", ["Murray JK"], "epmc"),
        ("W3", None, "missing"),
    ]


def test_epmc_falls_back_to_the_comma_joined_author_string() -> None:
    from corpus import epmc

    got = epmc.fetch_authors(
        [_paper("W1", doi=None, pmcid="PMC1")],
        lambda _u: {
            "resultList": {"result": [{"pmcid": "PMC1", "authorString": "Murray JK, Knudson S."}]}
        },
    )
    assert got == {"W1": ["Murray JK", "Knudson S"]}


def test_run_writes_a_json_array_the_bundle_passes_through() -> None:
    conn = _conn()
    _work(conn, "W1")

    def fetch(url: str) -> dict[str, Any]:
        if "openalex" in url:
            return _oa("10.0/W1", ["Evelyn J. Bromet", "Irving Hwang"])
        return {"resultList": {"result": []}}

    assert authors.run(conn, lambda _m: None, fetch=fetch) == (1, 1)
    stored, source = conn.execute("SELECT authors, authors_source FROM works").fetchone()
    assert json.loads(stored) == ["Evelyn J. Bromet", "Irving Hwang"]
    assert source == "openalex"
    assert bundle.authors_array(stored) == ["Evelyn J. Bromet", "Irving Hwang"]


def test_works_the_crawler_already_named_are_not_touched() -> None:
    conn = _conn()
    _work(conn, "W1", authors="Murray JK, Knudson S.")
    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return {"results": [], "resultList": {"result": []}}

    assert authors.run(conn, lambda _m: None, fetch=fetch) == (0, 0)
    assert calls == []
    assert conn.execute("SELECT authors FROM works").fetchone() == ("Murray JK, Knudson S.",)


def test_run_is_a_no_op_on_rerun_and_skips_works_outside_the_pool() -> None:
    conn = _conn()
    _work(conn, "W1")
    _work(conn, "W2", relevance=3)  # below the pool threshold
    _work(conn, "W3", status="kept_miss")

    def fetch(url: str) -> dict[str, Any]:
        if "openalex" in url:
            return _oa("10.0/W1", ["Patrick W. Corrigan"])
        return {"resultList": {"result": []}}

    assert authors.run(conn, lambda _m: None, fetch=fetch) == (1, 1)
    assert authors.run(conn, lambda _m: None, fetch=fetch) == (0, 0)


def test_missing_rows_are_closed_out_not_retried() -> None:
    conn = _conn()
    _work(conn, "W1")
    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return {"results": [], "resultList": {"result": []}}

    assert authors.run(conn, lambda _m: None, fetch=fetch) == (1, 0)
    assert conn.execute("SELECT authors, authors_source FROM works").fetchone() == (None, "missing")
    calls.clear()
    assert authors.run(conn, lambda _m: None, fetch=fetch) == (0, 0)
    assert calls == []


def test_a_failed_batch_is_left_for_the_next_run() -> None:
    conn = _conn()
    _work(conn, "W1")

    def boom(_url: str) -> dict[str, Any]:
        raise TimeoutError("openalex down")

    assert authors.run(conn, lambda _m: None, fetch=boom) == (1, 0)
    assert conn.execute("SELECT authors_source FROM works").fetchone() == (None,)
