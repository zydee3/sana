"""Local CPU embeddings for the chunk table — ONNX Runtime, no API calls.

Mission 4's constraint is that nothing here may talk to a model provider, so the
candidates are small sentence encoders exported to ONNX and run on the box's cores.
Each spec pins the exact file that gets downloaded (fp32 or int8) and the pooling the
model was trained with — CLS for the BGE family, mean for MiniLM; using the wrong one
silently costs recall rather than failing, so it lives in the spec, not in the caller.

Throughput comes from processes, not from one wide session: these models are small
enough that a 40-thread GEMM spends most of its time in synchronisation. Batches are
also length-sorted before padding, which is free and removes most of the wasted
compute on a corpus whose chunks vary 120-340 words.

Vectors are stored as one float32 .npy per model plus a parallel chunk_id list, which
is what the index benchmark (sqlite-vec vs FAISS) reads next.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from tokenizers import Tokenizer

from .http import get_bytes

MODELS_DIR = Path(os.environ.get("SANA_MODELS_DIR", "/sana-data/models"))
VECTORS_DIR = Path(os.environ.get("SANA_VECTORS_DIR", "/sana-data/corpus/vectors"))

MAX_LENGTH = 512
BATCH_SIZE = 32

Log = Callable[[str], None]
Floats = npt.NDArray[np.float32]


@dataclass(frozen=True)
class ModelSpec:
    """One embedding candidate: where its files live and how to pool its output."""

    name: str
    repo: str
    onnx_path: str
    pooling: str  # "cls" | "mean"
    dim: int
    query_prefix: str = ""
    doc_prefix: str = ""
    tokenizer_repo: str | None = None  # when the ONNX mirror lacks the original vocab


MODELS: dict[str, ModelSpec] = {
    "bge-small": ModelSpec(
        name="bge-small",
        repo="BAAI/bge-small-en-v1.5",
        onnx_path="onnx/model.onnx",
        pooling="cls",
        dim=384,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "bge-small-int8": ModelSpec(
        name="bge-small-int8",
        repo="Xenova/bge-small-en-v1.5",
        onnx_path="onnx/model_quantized.onnx",
        pooling="cls",
        dim=384,
        query_prefix="Represent this sentence for searching relevant passages: ",
        tokenizer_repo="BAAI/bge-small-en-v1.5",
    ),
    "minilm": ModelSpec(
        name="minilm",
        repo="sentence-transformers/all-MiniLM-L6-v2",
        onnx_path="onnx/model.onnx",
        pooling="mean",
        dim=384,
    ),
    "minilm-int8": ModelSpec(
        name="minilm-int8",
        repo="sentence-transformers/all-MiniLM-L6-v2",
        onnx_path="onnx/model_qint8_avx512_vnni.onnx",
        pooling="mean",
        dim=384,
    ),
    "nomic-int8": ModelSpec(
        name="nomic-int8",
        repo="nomic-ai/nomic-embed-text-v1.5",
        onnx_path="onnx/model_quantized.onnx",
        pooling="mean",
        dim=768,
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
    ),
}


def download(url: str, dest: Path, log: Log) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(get_bytes(url, timeout=600.0))
    tmp.rename(dest)


def ensure_model(spec: ModelSpec, log: Log = print) -> tuple[Path, Path]:
    """Fetch the ONNX graph + tokenizer into MODELS_DIR. Returns (onnx, tokenizer)."""
    root = MODELS_DIR / spec.name
    onnx = root / "model.onnx"
    tok = root / "tokenizer.json"
    download(f"https://huggingface.co/{spec.repo}/resolve/main/{spec.onnx_path}", onnx, log)
    tok_repo = spec.tokenizer_repo or spec.repo
    download(f"https://huggingface.co/{tok_repo}/resolve/main/tokenizer.json", tok, log)
    return onnx, tok


def pool(hidden: Floats, mask: npt.NDArray[np.int64], pooling: str) -> Floats:
    """Token states -> one L2-normalised vector per row, the way the model was trained."""
    if hidden.ndim == 2:  # the export already pools
        pooled = hidden
    elif pooling == "cls":
        pooled = hidden[:, 0]
    else:
        m = mask.astype(np.float32)[:, :, None]
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return np.asarray(pooled / np.clip(norms, 1e-12, None), dtype=np.float32)


class Embedder:
    """A loaded ONNX encoder. One per process — sessions are not cheap to build."""

    def __init__(self, spec: ModelSpec, threads: int = 1) -> None:
        self.spec = spec
        onnx, tok_path = ensure_model(spec, log=lambda _: None)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.tokenizer.enable_padding()

    def _forward(self, texts: Sequence[str]) -> Floats:
        encoded = self.tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feed: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.session.run(None, feed)[0]
        return pool(np.asarray(out, dtype=np.float32), mask, self.spec.pooling)

    def encode(
        self, texts: Sequence[str], *, batch_size: int = BATCH_SIZE, is_query: bool = False
    ) -> Floats:
        """Embed texts, returning L2-normalised rows in the input order."""
        if not texts:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        prefix = self.spec.query_prefix if is_query else self.spec.doc_prefix
        prepared = [prefix + t for t in texts]
        # Length-sorted batches: padding is per batch, so grouping similar lengths
        # removes most of the wasted compute. Order is restored before returning.
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]))
        out = np.zeros((len(prepared), self.spec.dim), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            out[idx] = self._forward([prepared[i] for i in idx])
        return out


_WORKER: Embedder | None = None


def _init_worker(spec: ModelSpec, threads: int) -> None:
    global _WORKER
    _WORKER = Embedder(spec, threads=threads)


def _encode_shard(texts: list[str]) -> Floats:
    assert _WORKER is not None
    return _WORKER.encode(texts)


def encode_parallel(
    spec: ModelSpec, texts: Sequence[str], *, workers: int, threads: int = 1
) -> Floats:
    """Embed texts across `workers` processes, each owning a `threads`-wide session."""
    if workers <= 1:
        return Embedder(spec, threads=threads).encode(texts)
    from concurrent.futures import ProcessPoolExecutor

    shard = max(1, len(texts) // (workers * 4))
    shards = [list(texts[i : i + shard]) for i in range(0, len(texts), shard)]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(spec, threads)
    ) as pool:
        parts = list(pool.map(_encode_shard, shards))
    return np.asarray(np.vstack(parts), dtype=np.float32)


@dataclass(frozen=True)
class BenchResult:
    model: str
    workers: int
    threads: int
    n: int
    seconds: float

    @property
    def per_second(self) -> float:
        return self.n / self.seconds


def bench(spec: ModelSpec, texts: Sequence[str], *, workers: int, threads: int = 1) -> BenchResult:
    start = time.monotonic()
    vecs = encode_parallel(spec, texts, workers=workers, threads=threads)
    elapsed = time.monotonic() - start
    assert vecs.shape == (len(texts), spec.dim)
    return BenchResult(spec.name, workers, threads, len(texts), elapsed)


def token_stats(spec: ModelSpec, texts: Sequence[str]) -> dict[str, float]:
    """Real tokens per word for this vocab — the check chunk.TOKENS_PER_WORD needs."""
    tok = Tokenizer.from_file(str(ensure_model(spec, log=lambda _: None)[1]))
    # Some tokenizer.json files ship their own truncation (MiniLM's is 128), which would
    # report every long chunk as exactly that length and hide the clipping being measured.
    tok.no_truncation()
    counts = [len(e.ids) for e in tok.encode_batch(list(texts))]
    words = [max(1, len(t.split())) for t in texts]
    ratios = np.array(counts, dtype=np.float64) / np.array(words, dtype=np.float64)
    arr = np.array(counts, dtype=np.float64)
    return {
        "tokens_per_word_mean": float(ratios.mean()),
        "tokens_per_word_p90": float(np.percentile(ratios, 90)),
        "tokens_median": float(np.median(arr)),
        "tokens_p90": float(np.percentile(arr, 90)),
        "tokens_max": float(arr.max()),
        "over_512": float((arr > MAX_LENGTH).mean()),
    }


def load_chunks(conn: sqlite3.Connection, limit: int | None = None) -> list[tuple[str, str]]:
    """(chunk_id, text) for every stored chunk, in a stable order."""
    sql = "SELECT chunk_id, text FROM chunks ORDER BY work_id, idx"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [(str(r[0]), str(r[1])) for r in conn.execute(sql)]


def vectors_path(model: str) -> Path:
    return VECTORS_DIR / f"{model}.npy"


def ids_path(model: str) -> Path:
    return VECTORS_DIR / f"{model}.ids.json"


def embed_all(
    conn: sqlite3.Connection,
    spec: ModelSpec,
    log: Log,
    *,
    workers: int,
    threads: int = 1,
    limit: int | None = None,
) -> Path:
    """Embed every chunk and store vectors + ids for the index benchmark."""
    rows = load_chunks(conn, limit)
    log(f"embedding {len(rows)} chunks with {spec.name} ({workers}x{threads})")
    start = time.monotonic()
    vecs = encode_parallel(spec, [t for _, t in rows], workers=workers, threads=threads)
    elapsed = time.monotonic() - start
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path(spec.name), vecs)
    ids_path(spec.name).write_text(json.dumps([c for c, _ in rows]))
    mb = vecs.nbytes / 1e6
    log(f"done: {len(rows)} chunks in {elapsed:.1f}s ({len(rows) / elapsed:.0f}/s), {mb:.1f}MB")
    return vectors_path(spec.name)


def load_vectors(model: str) -> tuple[Floats, list[str]]:
    vecs = np.load(vectors_path(model))
    ids = json.loads(ids_path(model).read_text())
    return np.asarray(vecs, dtype=np.float32), [str(i) for i in ids]


def search(
    query_vec: Floats, vecs: Floats, ids: Sequence[str], k: int = 10
) -> list[tuple[str, float]]:
    """Exact cosine top-k (vectors are normalised, so a dot product is the score)."""
    scores = vecs @ query_vec
    top = np.argsort(-scores)[:k]
    return [(ids[int(i)], float(scores[int(i)])) for i in top]
