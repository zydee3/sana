"""The corpus store: for now, PDFs plus a JSON metadata sidecar on disk.

One paper -> `<dir>/<pmcid>.pdf` and `<dir>/<pmcid>.json`. Presence of the PDF is the
dedupe key. The store is deliberately dumb files today; the backend's retrieval side owns
chunking/embedding and can be pointed at whatever this grows into.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import Paper


def _paths(corpus_dir: Path, pmcid: str) -> tuple[Path, Path]:
    return corpus_dir / f"{pmcid}.pdf", corpus_dir / f"{pmcid}.json"


def has(corpus_dir: Path, pmcid: str) -> bool:
    pdf, _ = _paths(corpus_dir, pmcid)
    return pdf.exists()


def save(corpus_dir: Path, paper: Paper, pdf_url: str, pdf_bytes: bytes) -> Path:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    pdf_path, meta_path = _paths(corpus_dir, paper.pmcid)
    pdf_path.write_bytes(pdf_bytes)
    meta = {**asdict(paper), "pdf_url": pdf_url, "bytes": len(pdf_bytes)}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return pdf_path
