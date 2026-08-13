"""Vector index benchmark — sqlite-vec vs FAISS on the stored chunk vectors.

Mission 4 asks which store the retrieval path should use, decided on recall@10,
query latency and index size rather than on which library fits the stack nicest.
The candidates are the exact baseline (numpy scan, what `corpus retrieve` does
today), sqlite-vec (brute force inside the SQLite file the rest of the pipeline
already lives in), and FAISS flat/HNSW/IVF.

Two things make the measurement honest:

* Latency is measured single-threaded (`OMP_NUM_THREADS=1`, faiss threads 1).
  A serving path answers one query at a time and multi-threading a brute-force
  scan only hides the asymptotics behind 40 cores.
* The corpus is chunked for 1,790 works so far; the full corpus is ~244k works,
  so the real 39,693-vector set says nothing about the shape at 5M. `synthetic`
  tiles the real vectors with gaussian jitter to reach a target N — recall on
  that set is a structural measurement (near-duplicate neighbourhoods), not a
  quality claim, and it is reported as such.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import numpy.typing as npt
import sqlite_vec

Floats = npt.NDArray[np.float32]
Log = Callable[[str], None]

INDEX_DIR = Path(os.environ.get("SANA_INDEX_DIR", "/sana-data/corpus/indexes"))

# HNSW/IVF knobs. M and efConstruction are the standard quality/size tradeoff;
# efSearch and nprobe are the per-query dials the bench sweeps.
BUILD_THREADS = int(os.environ.get("SANA_INDEX_BUILD_THREADS", os.cpu_count() or 1))

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = (32, 64, 128)
IVF_NPROBE = (8, 16, 32)


@dataclass
class Built:
    """A built index: how to query it, what it cost to build, what it occupies."""

    backend: str
    params: str
    build_seconds: float
    size_bytes: int
    search: Callable[[Floats, int], list[int]]


def synthetic(vecs: Floats, n: int, *, seed: int = 7, sigma: float = 0.05) -> Floats:
    """Tile `vecs` up to n rows with jitter, renormalised — a scale-shape probe."""
    if n <= len(vecs):
        return np.asarray(vecs[:n], dtype=np.float32)
    rng = np.random.default_rng(seed)
    reps = -(-n // len(vecs))
    out = np.tile(vecs, (reps, 1))[:n].astype(np.float32)
    out[len(vecs) :] += rng.normal(0.0, sigma, size=out[len(vecs) :].shape).astype(np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.asarray(out / np.clip(norms, 1e-12, None), dtype=np.float32)


def _npy_bytes(vecs: Floats) -> int:
    return int(vecs.nbytes)


def build_exact(vecs: Floats, _dir: Path) -> Built:
    """Numpy full scan — the ground truth every other backend is scored against."""
    start = time.monotonic()

    def search(q: Floats, k: int) -> list[int]:
        scores = vecs @ q
        top = np.argpartition(-scores, k)[:k]
        return [int(i) for i in top[np.argsort(-scores[top])]]

    return Built("exact", "numpy", time.monotonic() - start, _npy_bytes(vecs), search)


def build_sqlite_vec(vecs: Floats, out_dir: Path) -> Built:
    """vec0 virtual table in its own SQLite file (brute force, k-NN by MATCH)."""
    path = out_dir / "sqlite-vec.db"
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    start = time.monotonic()
    conn.execute(
        f"CREATE VIRTUAL TABLE v USING vec0(id INTEGER PRIMARY KEY, e float[{vecs.shape[1]}])"
    )
    for chunk_start in range(0, len(vecs), 10_000):
        block = vecs[chunk_start : chunk_start + 10_000]
        conn.executemany(
            "INSERT INTO v(id, e) VALUES (?, ?)",
            [(chunk_start + i, row.tobytes()) for i, row in enumerate(block)],
        )
        conn.commit()
    build = time.monotonic() - start

    def search(q: Floats, k: int) -> list[int]:
        rows = conn.execute(
            "SELECT id FROM v WHERE e MATCH ? AND k = ?", (q.astype(np.float32).tobytes(), k)
        ).fetchall()
        return [int(r[0]) for r in rows]

    return Built("sqlite-vec", "vec0 brute force", build, path.stat().st_size, search)


def _faiss_size(index: faiss.Index, path: Path) -> int:
    faiss.write_index(index, str(path))
    return path.stat().st_size


def build_faiss_flat(vecs: Floats, out_dir: Path) -> Built:
    start = time.monotonic()
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    build = time.monotonic() - start
    size = _faiss_size(index, out_dir / "faiss-flat.index")

    def search(q: Floats, k: int) -> list[int]:
        _, ids = index.search(q.reshape(1, -1), k)
        return [int(i) for i in ids[0]]

    return Built("faiss-flat", "IndexFlatIP", build, size, search)


def _faiss_search(index: faiss.Index) -> Callable[[Floats, int], list[int]]:
    def search(q: Floats, k: int) -> list[int]:
        _, ids = index.search(q.reshape(1, -1), k)
        return [int(i) for i in ids[0]]

    return search


def build_faiss_hnsw(vecs: Floats, out_dir: Path) -> list[Built]:
    """One graph, one Built per efSearch — the dial is query-time, not build-time."""
    start = time.monotonic()
    index = faiss.IndexHNSWFlat(vecs.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(vecs)
    build = time.monotonic() - start
    size = _faiss_size(index, out_dir / "faiss-hnsw.index")

    def at(ef: int) -> Built:
        def search(q: Floats, k: int) -> list[int]:
            index.hnsw.efSearch = ef
            return _faiss_search(index)(q, k)

        params = f"M={HNSW_M},efC={HNSW_EF_CONSTRUCTION},efS={ef}"
        return Built("faiss-hnsw", params, build, size, search)

    return [at(ef) for ef in HNSW_EF_SEARCH]


def build_faiss_ivf(vecs: Floats, out_dir: Path) -> list[Built]:
    """One trained index, one Built per nprobe."""
    nlist = max(1, min(4096, int(4 * np.sqrt(len(vecs)))))
    start = time.monotonic()
    quantizer = faiss.IndexFlatIP(vecs.shape[1])
    index = faiss.IndexIVFFlat(quantizer, vecs.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    train_n = min(len(vecs), max(nlist * 40, 10_000))
    index.train(vecs[:train_n])
    index.add(vecs)
    build = time.monotonic() - start
    size = _faiss_size(index, out_dir / "faiss-ivf.index")

    def at(nprobe: int) -> Built:
        def search(q: Floats, k: int) -> list[int]:
            index.nprobe = nprobe
            return _faiss_search(index)(q, k)

        return Built("faiss-ivf", f"nlist={nlist},nprobe={nprobe}", build, size, search)

    return [at(n) for n in IVF_NPROBE]


BUILDERS: dict[str, Callable[[Floats, Path], list[Built]]] = {
    "exact": lambda v, d: [build_exact(v, d)],
    "sqlite-vec": lambda v, d: [build_sqlite_vec(v, d)],
    "faiss-flat": lambda v, d: [build_faiss_flat(v, d)],
    "faiss-hnsw": build_faiss_hnsw,
    "faiss-ivf": build_faiss_ivf,
}


def recall_at_k(got: Sequence[Sequence[int]], truth: Sequence[Sequence[int]]) -> float:
    """Mean per-query overlap with the exact top-k — the only quality number here."""
    if not truth:
        return 0.0
    hits = [len(set(g) & set(t)) / max(1, len(t)) for g, t in zip(got, truth, strict=True)]
    return float(np.mean(hits))


@dataclass(frozen=True)
class BenchRow:
    backend: str
    params: str
    n: int
    build_seconds: float
    size_mb: float
    p50_ms: float
    p95_ms: float
    recall10: float

    def line(self) -> str:
        return (
            f"{self.backend:<12} n={self.n:<8} {self.params:<28} "
            f"build {self.build_seconds:7.1f}s  size {self.size_mb:8.1f}MB  "
            f"p50 {self.p50_ms:8.2f}ms  p95 {self.p95_ms:8.2f}ms  recall@10 {self.recall10:.3f}"
        )


def _time_queries(
    built: Built, queries: Floats, k: int, reps: int
) -> tuple[list[list[int]], list[float]]:
    results: list[list[int]] = []
    latencies: list[float] = []
    for rep in range(reps):
        for q in queries:
            start = time.perf_counter()
            ids = built.search(q, k)
            latencies.append((time.perf_counter() - start) * 1000)
            if rep == 0:
                results.append(ids)
    return results, latencies


def bench(
    vecs: Floats,
    queries: Floats,
    log: Log,
    *,
    backends: Sequence[str],
    k: int = 10,
    reps: int = 5,
    out_dir: Path | None = None,
) -> list[BenchRow]:
    """Build each backend over `vecs` and measure it on `queries`."""
    out_dir = out_dir or INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(1)
    truth, _ = _time_queries(build_exact(vecs, out_dir), queries, k, 1)
    rows: list[BenchRow] = []
    for name in backends:
        # Building is an offline batch job and gets the box; serving answers one
        # query at a time, so latency is measured on one thread.
        faiss.omp_set_num_threads(BUILD_THREADS)
        variants = BUILDERS[name](vecs, out_dir)
        faiss.omp_set_num_threads(1)
        for built in variants:
            got, lat = _time_queries(built, queries, k, reps)
            lat.sort()
            row = BenchRow(
                backend=built.backend,
                params=built.params,
                n=len(vecs),
                build_seconds=built.build_seconds,
                size_mb=built.size_bytes / 1e6,
                p50_ms=lat[len(lat) // 2],
                p95_ms=lat[int(len(lat) * 0.95)],
                recall10=recall_at_k(got, truth),
            )
            rows.append(row)
            log(row.line())
    return rows


def write_results(rows: Sequence[BenchRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.__dict__ for r in rows], indent=1))
