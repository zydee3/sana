import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scraper import db
from scraper.models import Candidate

T0 = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

CAND = Candidate(
    work_id="W123",
    title="A study",
    discovered_via="openalex",
    openalex_id="W123",
    doi="10.1/abc",
)


def _conn(tmp_path: Path) -> sqlite3.Connection:
    return db.connect(tmp_path / "corpus.db")


def test_add_topic_idempotent(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    db.add_topic(conn, "sleep", "sleep quality", "T1")
    db.add_topic(conn, "sleep", "different query")
    count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    assert count == 1


def test_claim_finish_and_recrawl(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    db.add_topic(conn, "sleep", "sleep quality", "T1")

    topic = db.claim_next_topic(conn, recrawl_days=7, now=T0)
    assert topic is not None and topic.name == "sleep" and topic.openalex_id == "T1"
    assert db.claim_next_topic(conn, recrawl_days=7, now=T0) is None  # active, not claimable

    db.finish_topic(conn, topic.id, "2026-08-01", now=T0)
    assert db.claim_next_topic(conn, recrawl_days=7, now=T0 + timedelta(days=3)) is None

    stale = db.claim_next_topic(conn, recrawl_days=7, now=T0 + timedelta(days=8))
    assert stale is not None and stale.watermark == "2026-08-01"


def test_recover_active_topics_requeues_orphans(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    db.add_topic(conn, "sleep", "sleep")
    claimed = db.claim_next_topic(conn, recrawl_days=7, now=T0)
    assert claimed is not None
    assert db.recover_active_topics(conn) == 1
    assert db.claim_next_topic(conn, recrawl_days=7, now=T0) is not None


def test_seen_matches_any_identifier(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    db.record_work(conn, CAND, topic_id=None, status="kept_miss")

    same_by_doi = Candidate(
        work_id="doi:10.1/abc", title="A study", discovered_via="europepmc", doi="10.1/abc"
    )
    other = Candidate(
        work_id="doi:10.9/zzz", title="Other", discovered_via="europepmc", doi="10.9/zzz"
    )
    assert db.seen(conn, CAND) is True
    assert db.seen(conn, same_by_doi) is True
    assert db.seen(conn, other) is False


def test_deferred_candidates_stay_unseen_until_judged(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    db.add_topic(conn, "sleep", "sleep")
    topic = db.claim_next_topic(conn, recrawl_days=7, now=T0)
    assert topic is not None
    db.record_work(conn, CAND, topic.id, status="candidate")

    assert db.seen(conn, CAND) is False  # a deferral is not a judgment
    rows = db.deferred_candidates(conn, limit=10)
    assert [r["work_id"] for r in rows] == ["W123"]

    db.record_work(conn, CAND, topic.id, status="kept_miss", study_type="rct", evidence_grade=2)
    row = conn.execute("SELECT status, evidence_grade FROM works").fetchone()
    assert (row["status"], row["evidence_grade"]) == ("kept_miss", 2)
    assert db.seen(conn, CAND) is True
    assert db.deferred_candidates(conn, limit=10) == []


def test_record_work_never_downgrades_a_judged_row(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    db.record_work(conn, CAND, None, status="kept_miss", study_type="rct", evidence_grade=2)
    db.set_fetched(conn, "W123", "texts/W123.txt", "pmc_oa_txt", now=T0)
    db.record_work(conn, CAND, None, status="rejected", reject_reason="triage")
    row = conn.execute("SELECT status, text_path FROM works").fetchone()
    assert (row["status"], row["text_path"]) == ("kept_text", "texts/W123.txt")


def test_topic_by_id(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    db.add_topic(conn, "sleep", "sleep quality", "T1")
    topic = db.topic_by_id(conn, 1)
    assert topic is not None and topic.name == "sleep" and topic.openalex_id == "T1"
    assert db.topic_by_id(conn, 99) is None


def test_fetch_and_expand_lifecycle(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    db.add_topic(conn, "sleep", "sleep")
    topic = db.claim_next_topic(conn, recrawl_days=7, now=T0)
    assert topic is not None
    db.record_work(conn, CAND, topic.id, status="kept_miss", study_type="rct", evidence_grade=2)

    assert db.unexpanded_kept(conn, topic.id, limit=5) == ["W123"]
    db.set_fetched(conn, "W123", "texts/W123.txt", "pmc_oa_txt", now=T0)
    row = conn.execute("SELECT status, text_path, evidence_grade FROM works").fetchone()
    assert (row["status"], row["text_path"], row["evidence_grade"]) == (
        "kept_text",
        "texts/W123.txt",
        2,
    )

    db.set_expanded(conn, "W123", now=T0)
    assert db.unexpanded_kept(conn, topic.id, limit=5) == []
