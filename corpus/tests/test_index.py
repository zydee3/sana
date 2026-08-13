from __future__ import annotations

import numpy as np
import pytest

from corpus import index


def _vectors(n: int = 400, dim: int = 32, seed: int = 3) -> index.Floats:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return np.asarray(v / np.linalg.norm(v, axis=1, keepdims=True), dtype=np.float32)


def test_synthetic_keeps_the_real_prefix_and_normalises() -> None:
    real = _vectors(100)
    grown = index.synthetic(real, 250)
    assert grown.shape == (250, real.shape[1])
    assert np.allclose(grown[:100], real, atol=1e-6)
    assert np.allclose(np.linalg.norm(grown, axis=1), 1.0, atol=1e-5)


def test_synthetic_truncates_when_asked_for_fewer() -> None:
    real = _vectors(100)
    assert index.synthetic(real, 40).shape == (40, real.shape[1])


@pytest.mark.parametrize("backend", sorted(index.BUILDERS))
def test_every_backend_finds_a_stored_vector_itself(backend: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    vecs = _vectors()
    built = index.BUILDERS[backend](vecs, tmp_path)[0]
    for row in (0, 17, 399):
        assert built.search(vecs[row], 5)[0] == row


def test_recall_at_k_counts_overlap_per_query() -> None:
    truth = [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert index.recall_at_k(truth, truth) == 1.0
    assert index.recall_at_k([[1, 2, 9, 9], [5, 6, 7, 8]], truth) == 0.75
    assert index.recall_at_k([], []) == 0.0


def test_bench_reports_one_row_per_variant_and_exact_backends_hit_recall_1(tmp_path) -> None:  # type: ignore[no-untyped-def]
    vecs = _vectors()
    queries = _vectors(5, seed=11)
    rows = index.bench(
        vecs,
        queries,
        lambda _: None,
        backends=["exact", "sqlite-vec", "faiss-flat", "faiss-ivf"],
        k=10,
        reps=2,
        out_dir=tmp_path,
    )
    assert [r.backend for r in rows] == ["exact", "sqlite-vec", "faiss-flat"] + ["faiss-ivf"] * len(
        index.IVF_NPROBE
    )
    # The brute-force backends are the same computation as the ground truth.
    for row in rows[:3]:
        assert row.recall10 == 1.0
        assert row.n == len(vecs)
        assert row.p50_ms > 0
