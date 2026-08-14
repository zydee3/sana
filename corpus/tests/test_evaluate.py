from __future__ import annotations

from pathlib import Path

import numpy as np

from corpus import evaluate


def test_load_judgments_keys_by_pair_and_keeps_retired_prefixes(tmp_path: Path) -> None:
    path = tmp_path / "j.tsv"
    path.write_text("query_idx\tchunk_id\trelevant\n1\tW1#0\t1\n1\t350:W1#0\t0\n2\tW9#3\t0\n")
    judgments = evaluate.load_judgments(path)
    assert judgments == {(1, "W1#0"): 1, (1, "350:W1#0"): 0, (2, "W9#3"): 0}


def test_score_counts_unjudged_as_not_relevant_and_reports_them() -> None:
    judgments = {(1, "a"): 1, (1, "b"): 0}
    metrics, unjudged = evaluate.score([["a", "b", "c"]], judgments, k=3)
    assert metrics["p_at_k"] == 1 / 3
    assert metrics["hit_at_3"] == 1
    assert unjudged == [(1, "c")]


def test_score_averages_over_queries_and_indexes_them_from_one() -> None:
    judgments = {(1, "a"): 1, (1, "b"): 1, (2, "c"): 1, (2, "d"): 0}
    metrics, unjudged = evaluate.score([["a", "b"], ["c", "d"]], judgments, k=2)
    assert metrics["p_at_k"] == 0.75
    assert metrics["queries"] == 2
    assert unjudged == []


def test_evaluate_scores_exact_search_against_the_fixture(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    vecs = rng.normal(size=(50, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"W{i}#0" for i in range(len(vecs))]
    # Each query is a stored vector, so the top hit is that row by construction.
    queries = np.asarray(vecs[[3, 11]], dtype=np.float32)
    judgments = {(1, "W3#0"): 1, (2, "W11#0"): 0}
    rows, unjudged = evaluate.evaluate(
        vecs, ids, queries, judgments, lambda _m: None, backends=["exact"], k=1, out_dir=tmp_path
    )
    assert [r.backend for r in rows] == ["exact"]
    assert rows[0].p_at_k == 0.5
    assert rows[0].hit_at_3 == 1
    assert rows[0].unjudged == 0
    assert unjudged == []
