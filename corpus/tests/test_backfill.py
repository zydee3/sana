from __future__ import annotations

import sqlite3
from pathlib import Path

from corpus import backfill, db, epmc

MARK = "\x9f==============================\x9f"
BODY = "Background " + "the abstract text " * 20


def _conn(tmp_path: Path) -> sqlite3.Connection:
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


def test_head_abstract_takes_text_after_the_marker() -> None:
    raw = f"JOURNAL INFORMATION\n====\nPMCID: PMC1\n\nTitle\nAuthor A\n{MARK}\n{BODY}\n"
    got = backfill.head_abstract(raw)
    assert got is not None
    assert got.startswith("Background the abstract text")
    assert "PMC1" not in got and "Author A" not in got


def test_head_abstract_truncates_and_collapses_whitespace() -> None:
    raw = MARK + "\n" + ("word  \n" * 2000)
    got = backfill.head_abstract(raw)
    assert got is not None
    assert len(got) <= backfill.ABSTRACT_CHARS
    assert "  " not in got


def test_head_abstract_none_without_marker_or_when_too_short() -> None:
    assert backfill.head_abstract("no marker here, just body text " * 20) is None
    assert backfill.head_abstract(f"{MARK}\ntoo short") is None


def test_local_path_maps_container_path_to_texts_dir() -> None:
    assert backfill.local_path("/data/corpus/texts/W1.txt").name == "W1.txt"
    assert backfill.local_path("/data/corpus/texts/W1.txt").parent == backfill.TEXTS_DIR


def test_text_pass_is_resumable_and_rescues_missing_rows(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(backfill, "TEXTS_DIR", tmp_path)
    (tmp_path / "W1.txt").write_text(f"front matter\n{MARK}\n{BODY}")
    (tmp_path / "W2.txt").write_text("front matter with no marker at all")
    conn = _conn(tmp_path)
    _work(conn, "W1", text_path="/data/corpus/texts/W1.txt")
    _work(conn, "W2", text_path="/data/corpus/texts/W2.txt")
    # W2 was already tried against EPMC and missed; a local text may still rescue it.
    backfill.store(conn, [("W2", None, "missing")])

    scanned, found = backfill.text_pass(conn, lambda _: None)
    assert (scanned, found) == (2, 1)
    assert conn.execute("SELECT source FROM abstracts WHERE work_id='W1'").fetchone()[0] == "text"

    # Rerun: W1 is done, only the markerless W2 is rescanned, and it stays missing.
    scanned, found = backfill.text_pass(conn, lambda _: None)
    assert (scanned, found) == (1, 0)
    w2 = conn.execute("SELECT source FROM abstracts WHERE work_id='W2'").fetchone()
    assert w2[0] == "missing"


def test_pending_papers_skips_stored_and_optionally_requires_ids(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    _work(conn, "W1", pmcid="PMC1")
    _work(conn, "W2")
    _work(conn, "W3", doi="10.1/3")
    _work(conn, "W4", pmcid="PMC4", status="rejected")
    backfill.store(conn, [("W1", "stored", "epmc")])

    assert [p.work_id for p in backfill.pending_papers(conn, require_ids=True)] == ["W3"]
    assert [p.work_id for p in backfill.pending_papers(conn, require_ids=False)] == ["W2", "W3"]


def test_epmc_pass_records_misses_and_defers_failures(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    conn = _conn(tmp_path)
    _work(conn, "W1", pmcid="PMC1")
    _work(conn, "W2", pmcid="PMC2")
    monkeypatch.setattr(epmc, "fetch_batch", lambda batch: {"W1": "an abstract"})

    todo, found = backfill.epmc_pass(conn, lambda _: None, workers=1)
    assert (todo, found) == (2, 1)
    rows = dict(conn.execute("SELECT work_id, source FROM abstracts"))
    assert rows == {"W1": "epmc", "W2": "missing"}

    def boom(batch: object) -> dict[str, str]:
        raise OSError("network down")

    _work(conn, "W3", pmcid="PMC3")
    monkeypatch.setattr(epmc, "fetch_batch", boom)
    assert backfill.epmc_pass(conn, lambda _: None, workers=1) == (1, 0)
    assert backfill.pending_papers(conn, require_ids=True)[0].work_id == "W3"


def test_mark_missing_closes_out_unservable_works(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    _work(conn, "W1")
    assert backfill.mark_missing(conn, lambda _: None) == 1
    assert conn.execute("SELECT abstract, source FROM abstracts").fetchone() == (None, "missing")
    assert backfill.mark_missing(conn, lambda _: None) == 0
