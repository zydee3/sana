from pathlib import Path

from scraper import db, report
from scraper.models import Candidate


def _cand(work_id: str) -> Candidate:
    return Candidate(work_id=work_id, title=f"Paper {work_id}", discovered_via="openalex")


def test_counts_reads_read_only(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "corpus.db")
    db.record_work(conn, _cand("W1"), topic_id=None, status="kept_miss")
    db.record_work(conn, _cand("W2"), topic_id=None, status="kept_miss")
    db.set_fetched(conn, "W2", "texts/W2.txt", "pmc_oa_txt")
    db.record_work(conn, _cand("W3"), topic_id=None, status="rejected")

    assert report._counts(tmp_path / "corpus.db") == (1, 3)


def test_post_count_formats_message(tmp_path: Path, monkeypatch: object) -> None:
    conn = db.connect(tmp_path / "corpus.db")
    db.record_work(conn, _cand("W1"), topic_id=None, status="kept_miss")
    db.set_fetched(conn, "W1", "texts/W1.txt", "pmc_oa_txt")

    sent: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(req: object, timeout: float = 0) -> FakeResponse:
        sent["url"] = req.full_url  # type: ignore[attr-defined]
        sent["auth"] = req.get_header("Authorization")  # type: ignore[attr-defined]
        sent["body"] = req.data  # type: ignore[attr-defined]
        return FakeResponse()

    import scraper.report as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)  # type: ignore[attr-defined]
    report.post_count(tmp_path / "corpus.db", "tok123", "999")

    assert sent["url"] == "https://discord.com/api/v10/channels/999/messages"
    assert sent["auth"] == "Bot tok123"
    assert b"scraped: 1 papers with text (1 tracked)" in sent["body"]  # type: ignore[operator]
