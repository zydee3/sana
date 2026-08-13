"""Embedding plumbing: pooling, batch order, search, chunk loading.

Nothing here downloads a model — the ONNX session is the one part that cannot be
unit-tested offline, so `Embedder.encode` is exercised with a stubbed `_forward`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import numpy as np
import pytest

from corpus import db, embed


def test_pool_cls_takes_first_token_and_normalises() -> None:
    hidden = np.array([[[3.0, 4.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.ones((1, 2), dtype=np.int64)
    out = embed.pool(hidden, mask, "cls")
    assert out[0].tolist() == pytest.approx([0.6, 0.8])


def test_pool_mean_ignores_padding() -> None:
    hidden = np.array([[[1.0, 0.0], [3.0, 0.0], [99.0, 99.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)
    out = embed.pool(hidden, mask, "mean")
    assert out[0].tolist() == pytest.approx([1.0, 0.0])  # padding row excluded


def test_pool_passes_through_a_pre_pooled_export() -> None:
    hidden = np.array([[0.0, 5.0]], dtype=np.float32)
    out = embed.pool(hidden, np.ones((1, 1), dtype=np.int64), "mean")
    assert out[0].tolist() == pytest.approx([0.0, 1.0])


def _stub_embedder(spec: embed.ModelSpec) -> embed.Embedder:
    """An Embedder whose forward pass encodes the text's length, no session needed."""
    e = embed.Embedder.__new__(embed.Embedder)
    e.spec = spec

    def forward(texts: Sequence[str]) -> embed.Floats:
        return np.array([[float(len(t)), 0.0] for t in texts], dtype=np.float32)

    e._forward = forward  # type: ignore[method-assign]
    return e


SPEC = embed.ModelSpec(name="stub", repo="x", onnx_path="y", pooling="cls", dim=2)


def test_encode_restores_input_order_after_length_sorting() -> None:
    # Length-sorted batching is the one place a chunk could get another chunk's
    # vector; the stub returns the input length so a mismatch is visible.
    texts = ["c" * n for n in (300, 5, 120, 40, 900, 7)]
    out = _stub_embedder(SPEC).encode(texts, batch_size=2)
    assert [row[0] for row in out] == [300, 5, 120, 40, 900, 7]


def test_encode_applies_the_query_prefix_only_to_queries() -> None:
    spec = embed.ModelSpec(
        name="stub", repo="x", onnx_path="y", pooling="cls", dim=2, query_prefix="Q: "
    )
    e = _stub_embedder(spec)
    assert e.encode(["abc"], is_query=True)[0][0] == len("Q: abc")
    assert e.encode(["abc"])[0][0] == len("abc")


def test_encode_of_nothing_is_an_empty_matrix() -> None:
    assert _stub_embedder(SPEC).encode([]).shape == (0, 2)


def test_search_returns_top_k_by_cosine_descending() -> None:
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    hits = embed.search(np.array([1.0, 0.0], dtype=np.float32), vecs, ["a", "b", "c"], k=2)
    assert [h[0] for h in hits] == ["a", "c"]
    assert hits[0][1] == pytest.approx(1.0)


def _db_with_chunks() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE works (work_id TEXT PRIMARY KEY)")
    conn.executescript(db.CHUNKS_SCHEMA)
    rows = [("w2:1", "w2", 1, "methods", None, "second work", 10)]
    rows += [(f"w1:{i}", "w1", i, "results", None, f"chunk {i}", 10) for i in (1, 0)]
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", rows)
    return conn


def test_load_chunks_is_ordered_by_work_then_index() -> None:
    got = embed.load_chunks(_db_with_chunks())
    assert [c for c, _ in got] == ["w1:0", "w1:1", "w2:1"]


def test_load_chunks_limit_truncates() -> None:
    assert len(embed.load_chunks(_db_with_chunks(), limit=2)) == 2


def test_every_model_spec_names_itself_consistently() -> None:
    assert all(name == spec.name for name, spec in embed.MODELS.items())
