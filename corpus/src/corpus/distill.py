"""Can a local classifier reproduce sonnet's judgments? — the option-(b) pilot.

Judging the whole corpus with `claude -p` costs weeks of the operator's subscription
(~1,000 papers per 7-day-window point). Option (b) is to judge a stratified sample and
distil the rest: embed title+abstract with the same local encoder retrieval already
uses, fit a cheap head on the sonnet labels, and apply it to the unjudged remainder.
This module measures whether that works — held-out agreement with sonnet, per stratum,
plus the precision/recall a keep-threshold would actually run at.

Nothing here calls a model provider, and nothing here writes labels to the DB: it is a
measurement that answers "is (b) viable", not the production classifier.

The head is multinomial logistic regression on L2-normalised embeddings. It is the
right first probe rather than a compromise — with a frozen encoder the features are
fixed, so anything more expressive is measuring the same 384 dimensions with more
parameters, and a linear number tells you whether the *encoder* carries the signal.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .embed import VECTORS_DIR, Floats, ModelSpec, encode_parallel

Log = Callable[[str], None]
Ints = npt.NDArray[np.int64]

TEST_FRACTION = 0.2
SEED = 0
# Sonnet saw the abstract truncated to classify.ABSTRACT_CHARS; the encoder truncates at
# 512 tokens anyway, so the cap here only keeps tokenisation off pathological rows.
ABSTRACT_CHARS = 4000
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


@dataclass(frozen=True)
class JudgedWork:
    """One sonnet-labelled work, with the strata a per-cell breakdown needs."""

    work_id: str
    text: str
    relevance: int
    domain: str
    status: str
    via: str


def load_judged(conn: sqlite3.Connection, limit: int | None = None) -> list[JudgedWork]:
    """Works carrying a claude relevance score and an abstract, in a stable order."""
    sql = """
        SELECT w.work_id, w.title, a.abstract, w.relevance, w.domain, w.status, w.discovered_via
        FROM works w JOIN abstracts a USING (work_id)
        WHERE w.relevance IS NOT NULL AND w.label_source LIKE 'claude-%'
          AND a.abstract IS NOT NULL AND LENGTH(a.abstract) > 50
        ORDER BY w.work_id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [
        JudgedWork(
            work_id=str(r[0]),
            text=f"{r[1]}\n\n{str(r[2])[:ABSTRACT_CHARS]}",
            relevance=int(r[3]),
            domain=str(r[4] or "other-medical"),
            status=str(r[5]),
            via=str(r[6]),
        )
        for r in conn.execute(sql)
    ]


def _cache_paths(model: str) -> tuple[Path, Path]:
    return VECTORS_DIR / f"distill-{model}.npy", VECTORS_DIR / f"distill-{model}.ids.json"


def embed_works(
    spec: ModelSpec, works: Sequence[JudgedWork], log: Log, *, workers: int, threads: int = 1
) -> Floats:
    """Embed title+abstract, reusing the cache when it covers exactly these works."""
    vec_path, ids_path = _cache_paths(spec.name)
    ids = [w.work_id for w in works]
    if vec_path.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == ids:
        log(f"reusing cached embeddings {vec_path}")
        return np.asarray(np.load(vec_path), dtype=np.float32)
    log(f"embedding {len(works)} abstracts with {spec.name} ({workers}x{threads})")
    vecs = encode_parallel(spec, [w.text for w in works], workers=workers, threads=threads)
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vecs)
    ids_path.write_text(json.dumps(ids))
    return vecs


def stratified_split(
    labels: Ints, *, seed: int = SEED, test_fraction: float = TEST_FRACTION
) -> tuple[Ints, Ints]:
    """Train/test row indices holding each class's proportion, deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for value in np.unique(labels):
        rows = np.flatnonzero(labels == value)
        rng.shuffle(rows)
        cut = int(round(len(rows) * test_fraction))
        test.extend(rows[:cut].tolist())
        train.extend(rows[cut:].tolist())
    return np.array(sorted(train), dtype=np.int64), np.array(sorted(test), dtype=np.int64)


def per_class_f1(truth: Ints, pred: Ints, classes: Sequence[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(classes):
        tp = float(np.sum((pred == i) & (truth == i)))
        fp = float(np.sum((pred == i) & (truth != i)))
        fn = float(np.sum((pred != i) & (truth == i)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[name] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return out


def roc_auc(truth: Ints, score: Floats) -> float:
    """Rank-based AUC (ties averaged); 0.5 when one class is absent."""
    pos, neg = float(np.sum(truth == 1)), float(np.sum(truth == 0))
    if not pos or not neg:
        return 0.5
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # Average ranks within tied score groups so a constant scorer reads 0.5, not 1.0.
    sorted_scores = score[order]
    start = 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return float((ranks[truth == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


@dataclass
class TaskResult:
    """Held-out agreement for one distillation target."""

    task: str
    classes: list[str]
    n_train: int
    n_test: int
    accuracy: float
    majority_baseline: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    auc: float | None = None
    thresholds: list[dict[str, float]] = field(default_factory=list)
    by_stratum: dict[str, dict[str, float]] = field(default_factory=dict)


def _threshold_table(truth: Ints, prob: Floats) -> list[dict[str, float]]:
    """What a keep-threshold would actually do: how much is kept, and how clean it is."""
    rows = []
    for t in THRESHOLDS:
        keep = prob >= t
        kept = float(keep.sum())
        tp = float(np.sum(keep & (truth == 1)))
        rows.append(
            {
                "threshold": t,
                "kept_fraction": kept / len(truth),
                "precision": tp / kept if kept else 0.0,
                "recall": tp / float(np.sum(truth == 1)),
            }
        )
    return rows


def fit_task(
    x: Floats,
    labels: Ints,
    classes: Sequence[str],
    task: str,
    strata: Sequence[tuple[str, str]],
    *,
    seed: int = SEED,
) -> TaskResult:
    """Fit logistic regression on a stratified split and score the held-out rows."""
    from sklearn.linear_model import LogisticRegression

    train, test = stratified_split(labels, seed=seed)
    model = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    model.fit(x[train], labels[train])
    prob = np.asarray(model.predict_proba(x[test]), dtype=np.float32)
    pred = np.asarray(model.classes_[prob.argmax(axis=1)], dtype=np.int64)
    truth = labels[test]
    counts = np.bincount(labels[train], minlength=len(classes))

    per_class = per_class_f1(truth, pred, classes)
    result = TaskResult(
        task=task,
        classes=list(classes),
        n_train=len(train),
        n_test=len(test),
        accuracy=float(np.mean(pred == truth)),
        majority_baseline=float(np.mean(truth == int(counts.argmax()))),
        macro_f1=float(np.mean([c["f1"] for c in per_class.values()])),
        per_class=per_class,
    )
    if len(classes) == 2:
        positive = prob[:, list(model.classes_).index(1)]
        result.auc = roc_auc(truth, positive)
        result.thresholds = _threshold_table(truth, positive)
    for key in sorted({s for pair in (strata[i] for i in test) for s in pair}):
        rows = np.array([key in strata[i] for i in test])
        result.by_stratum[key] = {
            "n": float(rows.sum()),
            "accuracy": float(np.mean(pred[rows] == truth[rows])) if rows.any() else 0.0,
        }
    return result


def pilot(
    works: Sequence[JudgedWork], x: Floats, log: Log, *, seed: int = SEED
) -> list[TaskResult]:
    """The three targets a keep/route decision needs: relevance >=5, >=7, and domain."""
    strata = [(w.status, w.via) for w in works]
    domains = sorted({w.domain for w in works})
    results = []
    for cut in (5, 7):
        labels = np.array([int(w.relevance >= cut) for w in works], dtype=np.int64)
        classes = [f"<{cut}", f">={cut}"]
        results.append(fit_task(x, labels, classes, f"relevance>={cut}", strata, seed=seed))
        auc = results[-1].auc or 0.0
        log(f"relevance>={cut}: acc {results[-1].accuracy:.3f} auc {auc:.3f}")
    labels = np.array([domains.index(w.domain) for w in works], dtype=np.int64)
    results.append(fit_task(x, labels, domains, "domain", strata, seed=seed))
    log(f"domain: acc {results[-1].accuracy:.3f} macro-f1 {results[-1].macro_f1:.3f}")
    return results
