from __future__ import annotations

import sqlite3

import numpy as np

from corpus import distill


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE works (work_id TEXT PRIMARY KEY, title TEXT, relevance INTEGER,
          domain TEXT, label_source TEXT, status TEXT, discovered_via TEXT);
        CREATE TABLE abstracts (work_id TEXT PRIMARY KEY, abstract TEXT);
        """
    )
    return conn


def test_load_judged_takes_claude_labels_with_a_usable_abstract() -> None:
    conn = _conn()
    rows = [
        ("W1", "Sleep", 7, "sleep", "claude-sonnet", "kept_text", "openalex", "a" * 80),
        ("W2", "Publisher", 6, "pain", "publisher", "kept_text", "openalex", "b" * 80),
        ("W3", "Short", 5, "stress", "claude-sonnet", "kept_miss", "citation", "tiny"),
        ("W4", "Unjudged", None, None, None, "kept_text", "europepmc", "c" * 80),
    ]
    for row in rows:
        conn.execute("INSERT INTO works VALUES (?,?,?,?,?,?,?)", row[:7])
        conn.execute("INSERT INTO abstracts VALUES (?,?)", (row[0], row[7]))
    got = distill.load_judged(conn)
    assert [w.work_id for w in got] == ["W1"]
    assert got[0].text.startswith("Sleep\n\n")
    assert (got[0].status, got[0].via) == ("kept_text", "openalex")


def test_stratified_split_is_deterministic_and_holds_class_proportions() -> None:
    labels = np.array([0] * 80 + [1] * 20, dtype=np.int64)
    train, test = distill.stratified_split(labels, seed=3, test_fraction=0.2)
    assert len(train) == 80 and len(test) == 20
    assert set(train.tolist()) & set(test.tolist()) == set()
    assert int(np.sum(labels[test] == 1)) == 4  # 20% of each class, not of the whole
    assert np.array_equal(test, distill.stratified_split(labels, seed=3, test_fraction=0.2)[1])


def test_roc_auc_is_half_for_a_constant_scorer_and_one_for_a_perfect_one() -> None:
    truth = np.array([0, 0, 1, 1], dtype=np.int64)
    assert distill.roc_auc(truth, np.ones(4, dtype=np.float32)) == 0.5
    assert distill.roc_auc(truth, np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)) == 1.0
    assert distill.roc_auc(truth, np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)) == 0.0


def test_per_class_f1_counts_each_class_against_the_rest() -> None:
    truth = np.array([0, 0, 1, 1], dtype=np.int64)
    pred = np.array([0, 1, 1, 1], dtype=np.int64)
    got = distill.per_class_f1(truth, pred, ["no", "yes"])
    assert got["yes"] == {"precision": 2 / 3, "recall": 1.0, "f1": 0.8, "support": 2.0}
    assert got["no"]["recall"] == 0.5


def test_learning_curve_reports_one_row_per_size_capped_at_the_train_split() -> None:
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(0, 0.5, (60, 4)), rng.normal(2, 0.5, (60, 4))]).astype(np.float32)
    labels = np.array([0] * 60 + [1] * 60, dtype=np.int64)
    rows = distill.learning_curve(x, labels, [20, 96, 1000], repeats=2)
    assert [int(r["n_train"]) for r in rows] == [20, 96, 96]  # 96 = the whole train split
    assert all(0.0 <= r["accuracy_mean"] <= 1.0 for r in rows)
    assert rows[-1]["accuracy_spread"] == 0.0  # every repeat draws the full split


def test_fit_task_separates_two_clusters_and_reports_strata() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 0.1, (60, 4)), rng.normal(3, 0.1, (60, 4))]).astype(np.float32)
    labels = np.array([0] * 60 + [1] * 60, dtype=np.int64)
    strata = [("kept_text", "openalex")] * 60 + [("kept_miss", "citation")] * 60
    r = distill.fit_task(x, labels, ["no", "yes"], "toy", strata)
    assert r.accuracy == 1.0 and r.auc == 1.0
    assert r.n_train + r.n_test == 120
    assert r.majority_baseline == 0.5
    assert [row["threshold"] for row in r.thresholds] == list(distill.THRESHOLDS)
    assert set(r.by_stratum) == {"kept_text", "openalex", "kept_miss", "citation"}
    assert r.by_stratum["kept_text"]["n"] == 12  # 20% of the 60 rows in that stratum
