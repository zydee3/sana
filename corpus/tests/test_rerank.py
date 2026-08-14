from __future__ import annotations

import numpy as np

from corpus import rerank


def test_union_keeps_first_seen_order_without_duplicates() -> None:
    dense = ["a", "b", "c"]
    bm25 = ["c", "d", "a", "e"]
    assert rerank.union(dense, bm25) == ["a", "b", "c", "d", "e"]
    assert rerank.union([], []) == []


def test_top_k_orders_by_score_and_breaks_ties_by_candidate_order() -> None:
    ids = ["a", "b", "c", "d"]
    assert rerank.top_k(ids, [0.1, 9.0, -3.0, 9.0], 3) == ["b", "d", "a"]
    assert rerank.top_k(ids, [1.0, 1.0, 1.0, 1.0], 2) == ["a", "b"]
    assert rerank.top_k([], [], 5) == []


def test_build_feed_carries_type_ids_only_when_the_graph_declares_them() -> None:
    ids = np.array([[101, 5, 102]], dtype=np.int64)
    mask = np.ones_like(ids)
    types = np.array([[0, 0, 1]], dtype=np.int64)
    bert = rerank.build_feed(ids, mask, types, {"input_ids", "attention_mask", "token_type_ids"})
    assert np.array_equal(bert["token_type_ids"], types)  # never zeros: the pair split matters
    xlmr = rerank.build_feed(ids, mask, types, {"input_ids", "attention_mask"})
    assert "token_type_ids" not in xlmr
    assert set(xlmr) == {"input_ids", "attention_mask"}


def test_every_reranker_spec_pins_a_file_and_a_length() -> None:
    for name, spec in rerank.RERANKERS.items():
        assert spec.name == name
        assert spec.onnx_path.endswith(".onnx")
        assert spec.max_length > 0
