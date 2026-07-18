import json
from pathlib import Path

from scraper import corpus
from scraper.models import Paper

PAPER = Paper(
    pmcid="PMC7739073",
    title="A study of mindfulness",
    doi="10.1177/2164956120977827",
    authors="Smith J, Doe A.",
    year="2020",
    license="cc by-nc",
)


def test_save_and_dedupe(tmp_path: Path) -> None:
    assert corpus.has(tmp_path, PAPER.pmcid) is False

    pdf_path = corpus.save(tmp_path, PAPER, "https://example/PMC7739073.pdf", b"%PDF-fake")

    assert pdf_path.read_bytes() == b"%PDF-fake"
    assert corpus.has(tmp_path, PAPER.pmcid) is True

    meta = json.loads((tmp_path / "PMC7739073.json").read_text())
    assert meta["license"] == "cc by-nc"
    assert meta["pdf_url"] == "https://example/PMC7739073.pdf"
    assert meta["bytes"] == len(b"%PDF-fake")
