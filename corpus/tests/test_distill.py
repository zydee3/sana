from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence

import numpy as np

from corpus import distill
from corpus.embed import Floats


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE works (work_id TEXT PRIMARY KEY, title TEXT, relevance INTEGER,
          domain TEXT, label_source TEXT, status TEXT, discovered_via TEXT,
          gate_p5 REAL, gate_domain TEXT);
        CREATE TABLE abstracts (work_id TEXT PRIMARY KEY, abstract TEXT);
        """
    )
    return conn


def _gate_rows(conn: sqlite3.Connection) -> None:
    """One judged, one gated, one short-abstract, three pending."""
    rows = [
        ("W1", "Judged", 7, "sleep", "claude-sonnet", "kept_text", "openalex", "a" * 80),
        ("W2", "Gated", None, None, None, "kept_text", "europepmc", "b" * 80),
        ("W3", "Short", None, None, None, "kept_text", "europepmc", "tiny"),
        ("W4", "Rejected", None, None, None, "rejected", "europepmc", "d" * 80),
        ("W5", "Pending", None, None, None, "kept_text", "europepmc", "e" * 80),
        ("W6", "Pending", None, None, None, "kept_miss", "citation", "f" * 80),
        ("W7", "Pending", None, None, None, "kept_text", "openalex", "g" * 80),
    ]
    for row in rows:
        conn.execute(
            "INSERT INTO works (work_id, title, relevance, domain, label_source,"
            " status, discovered_via) VALUES (?,?,?,?,?,?,?)",
            row[:7],
        )
        conn.execute("INSERT INTO abstracts VALUES (?,?)", (row[0], row[7]))
    conn.execute("UPDATE works SET gate_p5 = 0.4, gate_domain = 'sleep' WHERE work_id = 'W2'")


def test_load_judged_takes_claude_labels_with_a_usable_abstract() -> None:
    conn = _conn()
    rows = [
        ("W1", "Sleep", 7, "sleep", "claude-sonnet", "kept_text", "openalex", "a" * 80),
        ("W2", "Publisher", 6, "pain", "publisher", "kept_text", "openalex", "b" * 80),
        ("W3", "Short", 5, "stress", "claude-sonnet", "kept_miss", "citation", "tiny"),
        ("W4", "Unjudged", None, None, None, "kept_text", "europepmc", "c" * 80),
    ]
    for row in rows:
        conn.execute(
            "INSERT INTO works (work_id, title, relevance, domain, label_source,"
            " status, discovered_via) VALUES (?,?,?,?,?,?,?)",
            row[:7],
        )
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


def test_pending_skips_judged_gated_rejected_and_short_rows() -> None:
    conn = _conn()
    _gate_rows(conn)
    got = distill.pending(conn, 10)
    assert [work_id for work_id, _ in got] == ["W5", "W6", "W7"]
    assert got[0][1].startswith("Pending\n\n")
    assert len(distill.pending(conn, 2)) == 2  # the slab bounds the read


def _toy_heads() -> distill.Heads:
    """Two separated clusters: the far one is relevance>=5 and domain 'pain'."""
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 0.05, (40, 2)), rng.normal(3, 0.05, (40, 2))]).astype(np.float32)
    works = [
        distill.JudgedWork(
            f"J{i}", "t", 2 if i < 40 else 8, "sleep" if i < 40 else "pain", "s", "v"
        )
        for i in range(80)
    ]
    return distill.fit_heads(works, x)


def _cluster_encoder(far: str) -> Callable[[Sequence[str]], Floats]:
    """Puts texts whose abstract is `far` in the high cluster, everything else in the low."""

    def encode(texts: Sequence[str]) -> Floats:
        rows = [[3.0, 3.0] if far in t else [0.0, 0.0] for t in texts]
        return np.array(rows, dtype=np.float32)

    return encode


def test_apply_heads_writes_a_probability_and_domain_then_is_a_no_op() -> None:
    conn = _conn()
    _gate_rows(conn)
    encode = _cluster_encoder("g" * 80)  # only W7
    assert distill.apply_heads(conn, _toy_heads(), encode, lambda _: None) == 3
    rows = dict(conn.execute("SELECT work_id, gate_p5 FROM works WHERE gate_p5 IS NOT NULL"))
    assert set(rows) == {"W2", "W5", "W6", "W7"}
    assert rows["W7"] > 0.9 and rows["W5"] < 0.1
    assert rows["W2"] == 0.4  # a work already carrying a gate score is never rescored
    domains = dict(conn.execute("SELECT work_id, gate_domain FROM works WHERE gate_p5 > 0"))
    assert (domains["W5"], domains["W7"]) == ("sleep", "pain")
    assert distill.apply_heads(conn, _toy_heads(), encode, lambda _: None) == 0


def test_apply_heads_stops_at_the_limit() -> None:
    conn = _conn()
    _gate_rows(conn)
    encode = _cluster_encoder("nothing matches")
    assert distill.apply_heads(conn, _toy_heads(), encode, lambda _: None, slab=2, limit=1) == 1
    assert conn.execute("SELECT count(*) FROM works WHERE gate_p5 IS NOT NULL").fetchone()[0] == 2
