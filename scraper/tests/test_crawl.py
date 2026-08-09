import sqlite3
from pathlib import Path

import pytest

from scraper import crawl, db, europepmc, pmc_oa
from scraper.models import Candidate


def _cand(work_id: str, **kw: object) -> Candidate:
    defaults: dict[str, object] = {"title": f"Paper {work_id}", "discovered_via": "openalex"}
    return Candidate(work_id=work_id, **{**defaults, **kw})  # type: ignore[arg-type]


@pytest.fixture()
def topic(tmp_path: Path) -> tuple[sqlite3.Connection, db.Topic]:
    conn = db.connect(tmp_path / "corpus.db")
    db.add_topic(conn, "sleep", "sleep")
    claimed = db.claim_next_topic(conn, recrawl_days=7)
    assert claimed is not None
    return conn, claimed


def test_parse_topics_reads_bullets_and_ignores_prose() -> None:
    text = (
        "# Corpus topics\n"
        "Notes about the format.\n"
        "- mental health treatment and access (T10272)\n"
        "- sleep quality\n"
        "  - resilience and mental health (T11761)\n"
        "-not a bullet\n"
    )
    assert crawl.parse_topics(text) == [
        ("mental health treatment and access", "T10272"),
        ("sleep quality", None),
        ("resilience and mental health", "T11761"),
    ]


def test_sync_topics_is_idempotent(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "corpus.db")
    text = "- sleep quality (T1)\n- anxiety\n"
    assert crawl.sync_topics(conn, text) == 2
    assert crawl.sync_topics(conn, text) == 2
    rows = conn.execute("SELECT name, openalex_id, added_by FROM topics ORDER BY id").fetchall()
    assert [(r["name"], r["openalex_id"], r["added_by"]) for r in rows] == [
        ("sleep quality", "T1", "config"),
        ("anxiety", None, "config"),
    ]


def test_dedupe_joins_on_any_identifier() -> None:
    a = _cand("W1", doi="10.1/x")
    b = _cand("doi:10.1/x", doi="10.1/x", discovered_via="europepmc")
    c = _cand("W2")
    assert crawl._dedupe([a, b, c]) == [a, c]


def test_study_type_takes_the_strongest_decisive_pub_type() -> None:
    rct = _cand("W1", pub_types=("Randomized Controlled Trial", "Journal Article"))
    review = _cand("W2", pub_types=("Journal Article", "Review", "Systematic Review"))
    unknown = _cand("W3", pub_types=("Journal Article", "research-article"))
    assert crawl._study_type(rct) == "rct"
    assert crawl._study_type(review) == "systematic_review"
    assert crawl._study_type(unknown) is None


def test_process_candidates_gates_fetches_and_records(
    topic: tuple[sqlite3.Connection, db.Topic],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn, t = topic
    monkeypatch.setattr(crawl, "REQUEST_DELAY_S", 0)
    monkeypatch.setattr(
        europepmc, "pmcids_for_dois", lambda dois, fetch=None: {"10.1/keep": "PMC1"}
    )
    monkeypatch.setattr(
        pmc_oa, "download_text", lambda pmcid: ("https://bucket/x.txt", "Full text.")
    )

    cands = [
        _cand("W1", is_retracted=True),
        _cand("W2", is_oa=False),
        _cand("W3", doi="10.1/keep", pub_types=("Systematic Review",)),
        _cand("W4"),  # no pmcid resolvable -> kept_miss
    ]
    fetched = crawl.process_candidates(conn, t, cands, tmp_path)

    assert fetched == 1
    rows = {r["work_id"]: r for r in conn.execute("SELECT * FROM works").fetchall()}
    assert rows["W1"]["status"] == "retracted"
    assert rows["W2"]["status"] == "rejected" and rows["W2"]["reject_reason"] == "not_open_access"
    assert rows["W3"]["status"] == "kept_text" and rows["W3"]["evidence_grade"] == 1
    assert rows["W3"]["pmcid"] == "PMC1" and rows["W3"]["text_source"] == "pmc_oa_txt"
    # kept without a decisive pub type: ungraded, not guessed
    assert rows["W4"]["status"] == "kept_miss"
    assert rows["W4"]["study_type"] is None and rows["W4"]["evidence_grade"] is None
    assert (tmp_path / "texts" / "W3.txt").read_text(encoding="utf-8") == "Full text."

    # a second pass sees everything and adds nothing
    assert crawl.process_candidates(conn, t, cands, tmp_path) == 0
    assert conn.execute("SELECT COUNT(*) FROM works").fetchone()[0] == 4


def test_drain_deferred_grades_from_rehydrated_metadata_and_fetches(
    topic: tuple[sqlite3.Connection, db.Topic],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn, t = topic
    monkeypatch.setattr(crawl, "REQUEST_DELAY_S", 0)
    db.record_work(conn, _cand("W1", doi="10.1/a"), t.id, status="candidate")
    db.record_work(conn, _cand("W2"), t.id, status="candidate")

    full = _cand("W1", doi="10.1/a", pmcid="PMC1", pub_types=("Meta-Analysis",))
    monkeypatch.setattr(europepmc, "records_for_dois", lambda dois, fetch=None: {"10.1/a": full})
    monkeypatch.setattr(pmc_oa, "download_text", lambda pmcid: ("https://bucket/x.txt", "Text."))

    joined: list[list[str]] = []

    def spy_join(dois: list[str], fetch: object = None) -> dict[str, str]:
        joined.append(list(dois))
        return {}

    monkeypatch.setattr(europepmc, "pmcids_for_dois", spy_join)
    assert crawl.drain_deferred(conn, tmp_path, limit=10) == 1

    # the rehydrated record carried the PMCID, so the join had nothing left to look up
    assert joined == []
    rows = {r["work_id"]: r for r in conn.execute("SELECT * FROM works").fetchall()}
    assert rows["W1"]["status"] == "kept_text" and rows["W1"]["evidence_grade"] == 1
    assert rows["W2"]["status"] == "kept_miss"  # no DOI to rehydrate, kept anyway
    assert db.deferred_candidates(conn, limit=10) == []


def test_drain_retires_candidates_recorded_under_another_id(
    topic: tuple[sqlite3.Connection, db.Topic],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn, t = topic
    monkeypatch.setattr(crawl, "REQUEST_DELAY_S", 0)
    monkeypatch.setattr(europepmc, "records_for_dois", lambda dois, fetch=None: {})
    db.record_work(conn, _cand("doi:10.1/a", doi="10.1/a"), t.id, status="candidate")
    db.record_work(conn, _cand("W1", doi="10.1/a"), t.id, status="kept_miss")

    crawl.drain_deferred(conn, tmp_path, limit=10)

    row = conn.execute("SELECT status, reject_reason FROM works WHERE work_id = 'doi:10.1/a'")
    assert tuple(row.fetchone()) == ("rejected", "duplicate")
    assert db.deferred_candidates(conn, limit=10) == []


def test_pass_summary_counts_every_status(
    topic: tuple[sqlite3.Connection, db.Topic],
    capsys: pytest.CaptureFixture[str],
) -> None:
    conn, t = topic
    db.record_work(conn, _cand("W1"), t.id, status="kept_miss")
    db.record_work(conn, _cand("W2"), t.id, status="candidate")
    db.record_work(conn, _cand("W3"), t.id, status="surprise")
    db.finish_topic(conn, t.id, "2026-01-01")

    crawl.log_pass_summary(conn)

    assert capsys.readouterr().out == (
        "pass: works kept_text=0 kept_miss=1 candidate=1 rejected=0 retracted=0 surprise=1;"
        " topics 1/1 crawled\n"
    )
