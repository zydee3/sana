"""The corpus store: plain-text articles plus a JSON metadata sidecar on disk.

One paper -> `<dir>/<pmcid>.txt` (the article's plain text) and `<dir>/<pmcid>.json`
(metadata). Presence of the `.txt` is the dedupe key. The store is deliberately dumb files
today; the backend's retrieval side owns chunking/embedding and can be pointed at whatever
this grows into.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import Paper


def _paths(corpus_dir: Path, pmcid: str) -> tuple[Path, Path]:
    return corpus_dir / f"{pmcid}.txt", corpus_dir / f"{pmcid}.json"


def has(corpus_dir: Path, pmcid: str) -> bool:
    txt, _ = _paths(corpus_dir, pmcid)
    return txt.exists()


def save(corpus_dir: Path, paper: Paper, source_url: str, text: str) -> Path:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    txt_path, meta_path = _paths(corpus_dir, paper.pmcid)
    txt_path.write_text(text, encoding="utf-8")
    meta = {**asdict(paper), "source_url": source_url, "chars": len(text)}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return txt_path
