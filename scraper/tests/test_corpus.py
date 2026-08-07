from pathlib import Path

from scraper import corpus


def test_save_text_flattens_slashed_ids(tmp_path: Path) -> None:
    path = corpus.save_text(tmp_path, "doi:10.1037/a0018555", "Body text.")
    assert path == tmp_path / "texts" / "doi:10.1037_a0018555.txt"
    assert path.read_text(encoding="utf-8") == "Body text."


def test_text_path_is_stable(tmp_path: Path) -> None:
    assert corpus.text_path(tmp_path, "W123") == tmp_path / "texts" / "W123.txt"
