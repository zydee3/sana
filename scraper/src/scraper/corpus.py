"""The text store: one flat file per work under <corpus>/texts/.

Everything else about a work (metadata, provenance, status) lives in corpus.db;
these files hold only the article text the backend's retrieval side reads.
"""

from __future__ import annotations

from pathlib import Path


def _safe_name(work_id: str) -> str:
    # canonical ids can contain '/' (doi:10.x/y); keep filenames flat
    return work_id.replace("/", "_")


def text_path(corpus_dir: Path, work_id: str) -> Path:
    return corpus_dir / "texts" / f"{_safe_name(work_id)}.txt"


def save_text(corpus_dir: Path, work_id: str, text: str) -> Path:
    path = text_path(corpus_dir, work_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
