"""Processing CLI. Each subcommand is one pipeline step; long runs are meant to be
nohup'd with their stdout as the logfile.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import (
    backfill,
    chunk,
    compare,
    db,
    distill,
    embed,
    epmc,
    evaluate,
    index,
    judge,
    lexical,
    rerank,
    sample,
)
from .classify import BATCH_SIZE, ClassifyError, classify_batch
from .models import Paper, Verdict


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    print(f"{_now()} {msg}", flush=True)


def cmd_sample(args: argparse.Namespace) -> int:
    conn = db.connect(args.db, read_only=True)
    papers = sample.stratified(conn, args.n, args.seed, args.floor)
    written = sample.write_jsonl(papers, args.out)
    strata = Counter(p.stratum for p in papers)
    _log(f"sampled {written} works into {args.out} (seed={args.seed})")
    for s, c in sorted(strata.items()):
        _log(f"  {s}: {c}")
    return 0


def cmd_abstracts(args: argparse.Namespace) -> int:
    papers = sample.read_jsonl(args.sample)
    conn = db.connect(args.db)
    have = {r[0] for r in conn.execute("SELECT work_id FROM abstracts WHERE abstract IS NOT NULL")}
    todo = [p for p in papers if p.work_id not in have]
    _log(f"{len(papers)} sampled, {len(papers) - len(todo)} already stored, fetching {len(todo)}")
    found = 0
    for start in range(0, len(todo), epmc.BATCH):
        batch = todo[start : start + epmc.BATCH]
        got = epmc.fetch_abstracts(batch)
        backfill.store(
            conn,
            [
                (p.work_id, got.get(p.work_id), "epmc" if p.work_id in got else "missing")
                for p in batch
            ],
        )
        found += len(got)
        _log(f"  {start + len(batch)}/{len(todo)} fetched, {found} with abstracts")
    _log(f"done: {found}/{len(todo)} abstracts stored")
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    if args.phase in ("text", "all"):
        backfill.text_pass(conn, _log, args.limit)
    if args.phase in ("epmc", "all"):
        backfill.epmc_pass(conn, _log, args.workers, args.limit)
    if args.phase == "all":
        backfill.mark_missing(conn, _log)
    rows = conn.execute("SELECT source, count(*) FROM abstracts GROUP BY 1").fetchall()
    _log(f"abstracts by source: {dict(rows)}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    done, failed = judge.run(
        conn,
        _log,
        model=args.model,
        workers=args.workers,
        batch_size=args.batch,
        limit=args.limit,
        via=args.via,
        status=args.status,
        only=args.ids.read_text().split() if args.ids else None,
    )
    dist = dict(
        conn.execute(
            "SELECT relevance, count(*) FROM works WHERE relevance IS NOT NULL GROUP BY 1"
        ).fetchall()
    )
    _log(f"corpus relevance histogram: {dist}")
    return 1 if failed and not done else 0


def cmd_chunk(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    chunk.run(
        conn,
        _log,
        min_relevance=args.min_relevance,
        workers=args.workers,
        limit=args.limit,
        target_tokens=args.target_tokens,
    )
    sections = dict(
        conn.execute("SELECT section, count(*) FROM chunks GROUP BY 1 ORDER BY 2 DESC").fetchall()
    )
    total, mean, low, high = conn.execute(
        "SELECT count(*), avg(n_words), min(n_words), max(n_words) FROM chunks"
    ).fetchone()
    if total:
        _log(f"corpus chunks: {total} total, mean {mean:.0f} words, range {low}-{high}")
        _log(f"  by section: {sections}")
    return 0


def cmd_embed_bench(args: argparse.Namespace) -> int:
    conn = db.connect(args.db, read_only=True)
    texts = [t for _, t in embed.load_chunks(conn, args.n)]
    _log(f"benchmarking {len(texts)} chunks on {len(args.models)} models")
    for name in args.models:
        spec = embed.MODELS[name]
        embed.ensure_model(spec, _log)
        stats = embed.token_stats(spec, texts)
        _log(f"{name} tokens: {json.dumps({k: round(v, 3) for k, v in stats.items()})}")
        for workers in args.workers:
            r = embed.bench(spec, texts, workers=workers, threads=args.threads)
            rate = f"{r.per_second:.0f} chunks/s ({r.seconds:.1f}s)"
            _log(f"  {name} {workers}x{args.threads}: {rate}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    conn = db.connect(args.db, read_only=True)
    embed.embed_all(
        conn,
        embed.MODELS[args.model],
        _log,
        workers=args.workers,
        threads=args.threads,
        limit=args.limit,
    )
    return 0


def _read_queries(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [q.strip() for q in lines if q.strip() and not q.startswith("#")]


def cmd_retrieve(args: argparse.Namespace) -> int:
    conn = db.connect(args.db, read_only=True)
    spec = embed.MODELS[args.model]
    vecs, ids = embed.load_vectors(args.model)
    queries = [args.query] if args.query else _read_queries(args.queries)
    embedder = embed.Embedder(spec, threads=args.threads)
    latencies: list[float] = []
    results = []
    for q in queries:
        start = time.monotonic()
        qv = embedder.encode([q], is_query=True)[0]
        hits = embed.search(qv, vecs, ids, args.k)
        latencies.append((time.monotonic() - start) * 1000)
        rows = []
        for chunk_id, score in hits:
            r = conn.execute(
                "SELECT c.work_id, c.section, c.text, w.title, w.relevance, w.domain, w.year "
                "FROM chunks c JOIN works w USING (work_id) WHERE c.chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "score": round(score, 4),
                    "work_id": r[0],
                    "section": r[1],
                    "title": r[3],
                    "relevance": r[4],
                    "domain": r[5],
                    "year": r[6],
                    "text": r[2] if args.full else r[2][:400],
                }
            )
        results.append({"query": q, "hits": rows})
        _log(f"{q[:60]!r}: top score {hits[0][1]:.3f}" if hits else f"{q!r}: no hits")
    lat = sorted(latencies)
    _log(f"query p50 {lat[len(lat) // 2]:.1f}ms max {lat[-1]:.1f}ms over {len(ids)} vectors")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        _log(f"wrote {args.out}")
    return 0


def _chunk_texts(conn: sqlite3.Connection, chunk_ids: Sequence[str]) -> list[str]:
    """Chunk text for each id, in the given order."""
    holes = ",".join("?" * len(chunk_ids))
    rows = dict(
        conn.execute(f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({holes})", chunk_ids)
    )
    return [str(rows[cid]) for cid in chunk_ids]


def cmd_eval(args: argparse.Namespace) -> int:
    conn = db.connect(args.db, read_only=True)
    spec = embed.MODELS[args.model]
    vecs, ids = embed.load_vectors(args.model)
    queries = _read_queries(args.queries)[: args.queries_limit]
    # One query per encode call, like the serving path: dynamic int8 quantization takes
    # its activation scales over the whole input tensor, so batching queries together
    # perturbs each one (~0.99 cosine, ~11% of the top-10 churns) and the eval would
    # score vectors no request will ever produce.
    embedder = embed.Embedder(spec, threads=args.threads)
    qv = np.vstack([embedder.encode([q], is_query=True) for q in queries])
    judgments = evaluate.load_judgments(args.judgments)
    _log(f"{len(vecs)} vectors ({spec.name}), {len(queries)} queries, {len(judgments)} judgments")
    rows: list[evaluate.EvalRow] = []
    unjudged: dict[tuple[int, str], None] = {}
    if "dense" in args.rankers:
        dense_rows, missing = evaluate.evaluate(
            vecs, ids, qv, judgments, _log, backends=args.backends, k=args.k
        )
        rows.extend(dense_rows)
        unjudged.update(dict.fromkeys(missing))
    encoder: rerank.CrossEncoder | None = None

    def ranked(name: str, i: int, q: str, encoder: rerank.CrossEncoder | None) -> list[str]:
        """One non-dense arm's ranking for one query."""
        if name == "bm25":
            return lexical.search(conn, q, args.depth)[: args.k]
        dense = [cid for cid, _ in embed.search(qv[i], vecs, ids, args.depth)]
        if name == "hybrid":
            return lexical.rrf([dense, lexical.search(conn, q, args.depth)], args.k)
        if name == "rerank":
            cands = dense
        else:
            cands = rerank.union(dense, lexical.search(conn, q, args.depth))
        assert encoder is not None
        return rerank.top_k(cands, encoder.score(q, _chunk_texts(conn, cands)), args.k)

    for name in [r for r in args.rankers if r != "dense"]:
        if name.startswith("rerank") and encoder is None:
            encoder = rerank.CrossEncoder(rerank.RERANKERS[args.reranker], threads=args.threads)
        start = time.monotonic()
        hits = [ranked(name, i, q, encoder) for i, q in enumerate(queries)]
        per_query = (time.monotonic() - start) / max(1, len(queries))
        params = (
            f"{args.reranker}, depth {args.depth}, {per_query * 1000:.0f}ms/q"
            if name.startswith("rerank")
            else f"fts5 porter, depth {args.depth}"
        )
        row, missing = evaluate.score_hits(
            hits,
            judgments,
            backend=name,
            params=params,
            n=len(vecs),
            k=args.k,
        )
        rows.append(row)
        unjudged.update(dict.fromkeys(missing))
        _log(row.line())
    if args.out:
        args.out.write_text(json.dumps([r.__dict__ for r in rows], indent=1))
        _log(f"wrote {args.out}")
    if args.dump_unjudged:
        pairs = []
        for query_idx, chunk_id in unjudged:
            r = conn.execute(
                "SELECT c.text, c.section, w.title FROM chunks c JOIN works w USING (work_id) "
                "WHERE c.chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            pairs.append(
                {
                    "query_idx": query_idx,
                    "query": queries[query_idx - 1],
                    "chunk_id": chunk_id,
                    "section": r[1] if r else None,
                    "title": r[2] if r else None,
                    "text": r[0] if r else None,
                }
            )
        args.dump_unjudged.write_text("\n".join(json.dumps(p) for p in pairs))
        _log(f"wrote {len(pairs)} unjudged pairs to {args.dump_unjudged}")
    return 0


def cmd_lexical(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    n = lexical.build(conn, _log, rebuild=args.rebuild)
    _log(f"chunks_fts covers {n} chunks")
    return 0


def cmd_index_bench(args: argparse.Namespace) -> int:
    spec = embed.MODELS[args.model]
    vecs, _ids = embed.load_vectors(args.model)
    queries = _read_queries(args.queries)
    qv = embed.Embedder(spec, threads=1).encode(queries, is_query=True)
    _log(f"{len(vecs)} vectors ({spec.name}, dim {vecs.shape[1]}), {len(queries)} golden queries")
    rows: list[index.BenchRow] = []
    for n in args.scales:
        at_scale = index.synthetic(vecs, n) if n != len(vecs) else vecs
        kind = "real" if n <= len(vecs) else "synthetic (jittered copies)"
        _log(f"--- n={len(at_scale)} {kind} ---")
        rows.extend(
            index.bench(at_scale, qv, _log, backends=args.backends, k=args.k, reps=args.reps)
        )
    if args.out:
        index.write_results(rows, args.out)
        _log(f"wrote {args.out}")
    return 0


def _with_abstracts(conn: sqlite3.Connection, papers: list[Paper], *, require: bool) -> list[Paper]:
    ids = [p.work_id for p in papers]
    text: dict[str, str] = {}
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT work_id, abstract FROM abstracts WHERE work_id IN ({placeholders})", batch
        )
        text.update({r[0]: r[1] for r in rows if r[1]})
    return [
        Paper(**{**p.__dict__, "abstract": text.get(p.work_id)})
        for p in papers
        if not require or p.work_id in text
    ]


def cmd_classify(args: argparse.Namespace) -> int:
    papers = sample.read_jsonl(args.sample)
    if args.ids:
        wanted = {line.strip() for line in args.ids.read_text().splitlines() if line.strip()}
        papers = [p for p in papers if p.work_id in wanted]
    conn = db.connect(args.db, read_only=True)
    papers = _with_abstracts(conn, papers, require=True)[: args.limit]
    _log(f"classifying {len(papers)} papers with model={args.model} batch={args.batch}")
    verdicts: list[Verdict] = []
    failed = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for start in range(0, len(papers), args.batch):
            batch = papers[start : start + args.batch]
            try:
                got = classify_batch(batch, args.model)
            except ClassifyError as e:
                _log(f"  batch at {start} failed ({e}); retrying once")
                try:
                    got = classify_batch(batch, args.model)
                except ClassifyError as e2:
                    failed += len(batch)
                    _log(f"  batch at {start} failed again ({e2}); skipped")
                    continue
            for v in got:
                f.write(json.dumps(v.__dict__) + "\n")
            f.flush()
            verdicts.extend(got)
            kept = sum(v.relevance >= compare.KEEP_THRESHOLD for v in got)
            _log(f"  {start + len(batch)}/{len(papers)} judged, {kept}/{len(got)} >= 7 this batch")
    dist = Counter(v.relevance for v in verdicts)
    _log(f"done: {len(verdicts)} verdicts, {failed} skipped -> {args.out}")
    _log(f"  relevance histogram: {dict(sorted(dist.items()))}")
    _log(f"  domains: {dict(Counter(v.domain for v in verdicts).most_common())}")
    return 1 if failed and not verdicts else 0


def _read_verdicts(path: Path) -> list[Verdict]:
    with path.open() as f:
        return [Verdict(**json.loads(line)) for line in f if line.strip()]


def cmd_distill(args: argparse.Namespace) -> int:
    conn = db.connect(args.db, read_only=True)
    spec = embed.MODELS[args.model]
    works = distill.load_judged(conn, args.limit)
    _log(f"{len(works)} judged works with abstracts")
    x = distill.embed_works(spec, works, _log, workers=args.workers, threads=args.threads)
    results = distill.pilot(works, x, _log, seed=args.seed)
    for r in results:
        _log(
            f"{r.task}: n_train={r.n_train} n_test={r.n_test} acc={r.accuracy:.3f} "
            f"(majority {r.majority_baseline:.3f}) macro-f1={r.macro_f1:.3f}"
        )
        for name, m in r.per_class.items():
            _log(
                f"  {name}: p={m['precision']:.3f} r={m['recall']:.3f} "
                f"f1={m['f1']:.3f} n={int(m['support'])}"
            )
        for row in r.thresholds:
            _log(
                f"  t={row['threshold']:.1f}: kept={row['kept_fraction']:.3f} "
                f"precision={row['precision']:.3f} recall={row['recall']:.3f}"
            )
        for name, m in r.by_stratum.items():
            _log(f"  [{name}] n={int(m['n'])} acc={m['accuracy']:.3f}")
    curve = []
    if args.curve:
        labels = np.array([int(w.relevance >= 5) for w in works], dtype=np.int64)
        curve = distill.learning_curve(x, labels, args.curve, seed=args.seed)
        _log("learning curve, relevance>=5 (same held-out split):")
        for row in curve:
            _log(
                f"  n_train={int(row['n_train'])}: acc={row['accuracy_mean']:.3f} "
                f"(spread {row['accuracy_spread']:.3f}) auc={row['auc_mean']:.3f}"
            )
    if args.out:
        payload = {"tasks": [r.__dict__ for r in results], "learning_curve": curve}
        args.out.write_text(json.dumps(payload, indent=1))
        _log(f"wrote {args.out}")
    return 0


GATE_TOTALS = """
SELECT count(*), avg(gate_p5),
       sum(gate_p5 >= 0.3), sum(gate_p5 >= 0.5), sum(gate_p5 >= 0.7)
FROM works WHERE gate_p5 IS NOT NULL
"""

GATE_BY = """
SELECT {col}, count(*), avg(gate_p5), avg(gate_p5 >= 0.3)
FROM works WHERE gate_p5 IS NOT NULL GROUP BY 1 ORDER BY 2 DESC
"""


def cmd_gate(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    spec = embed.MODELS[args.model]
    judged = distill.load_judged(conn)
    _log(f"fitting the gate on {len(judged)} judged works ({spec.name})")
    x = distill.embed_works(spec, judged, _log, workers=args.workers, threads=args.threads)
    heads = distill.fit_heads(judged, x, seed=args.seed)

    def encode(texts: Sequence[str]) -> embed.Floats:
        return embed.encode_parallel(spec, texts, workers=args.workers, threads=args.threads)

    start = time.monotonic()
    n = distill.apply_heads(conn, heads, encode, _log, slab=args.slab, limit=args.limit)
    elapsed = time.monotonic() - start
    _log(f"scored {n} works in {elapsed:.0f}s ({n / elapsed:.0f}/s)" if n else "nothing pending")

    total, mean, t3, t5, t7 = conn.execute(GATE_TOTALS).fetchone()
    if not total:
        return 0
    _log(f"gated corpus: {total} works, mean p5 {mean:.3f}")
    for t, kept in ((0.3, t3), (0.5, t5), (0.7, t7)):
        _log(f"  p5>={t}: {kept} works ({kept / total:.1%})")
    for col in ("discovered_via", "status", "gate_domain"):
        for name, n_rows, avg_p5, keep in conn.execute(GATE_BY.format(col=col)):
            _log(f"  [{col}={name}] n={n_rows} mean p5 {avg_p5:.3f} p5>=0.3 {keep:.1%}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a, b = _read_verdicts(args.a), _read_verdicts(args.b)
    ag = compare.agreement(a, b, args.threshold)
    _log(f"{args.a.name} vs {args.b.name}: n={ag.n}")
    _log(f"  relevance mean|diff|={ag.mean_abs_diff:.2f} within1={ag.within_1:.1%}")
    _log(f"  keep@{args.threshold} agree={ag.keep_agree:.1%}")
    _log(f"  domain agree={ag.domain_agree:.1%} study_type agree={ag.study_type_agree:.1%}")
    if args.disagreements:
        ids = compare.disagreements(a, b)
        args.disagreements.write_text("\n".join(ids) + "\n")
        _log(f"  {len(ids)} disagreements -> {args.disagreements}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="corpus", description="corpus processing steps")
    p.add_argument("--db", type=Path, default=db.DEFAULT_DB)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="stratified sample of kept works -> jsonl")
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--floor", type=int, default=20, help="minimum rows per stratum")
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_sample)

    s = sub.add_parser("abstracts", help="rehydrate abstracts from Europe PMC into corpus.db")
    s.add_argument("--sample", type=Path, required=True)
    s.set_defaults(func=cmd_abstracts)

    s = sub.add_parser("backfill", help="abstracts for every kept work (local text, then EPMC)")
    s.add_argument("--phase", choices=("text", "epmc", "all"), default="all")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int, default=None, help="cap works per phase (for smoke runs)")
    s.set_defaults(func=cmd_backfill)

    s = sub.add_parser("judge", help="full relevance + label run into corpus.db (resumable)")
    s.add_argument("--model", default="sonnet")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--batch", type=int, default=BATCH_SIZE)
    s.add_argument("--limit", type=int, default=None, help="cap works this run")
    s.add_argument("--via", nargs="+", default=None, help="only these discovery sources")
    s.add_argument("--status", default=None, help="only this kept status")
    s.add_argument("--ids", type=Path, default=None, help="only these work ids (one per line)")
    s.set_defaults(func=cmd_judge)

    s = sub.add_parser("chunk", help="clean + section-aware chunk stored texts (resumable)")
    s.add_argument("--min-relevance", type=int, default=5, help="only works scored at least this")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--limit", type=int, default=None, help="cap works this run")
    s.add_argument(
        "--target-tokens",
        type=int,
        default=chunk.TARGET_TOKENS,
        help="chunk size target; use a scratch --db when trying a different one",
    )
    s.set_defaults(func=cmd_chunk)

    s = sub.add_parser("embed-bench", help="throughput + tokenization stats per candidate model")
    s.add_argument("--models", nargs="+", default=list(embed.MODELS), choices=list(embed.MODELS))
    s.add_argument("--n", type=int, default=2000, help="chunks to embed per configuration")
    s.add_argument("--workers", nargs="+", type=int, default=[1, 8, 20, 40])
    s.add_argument("--threads", type=int, default=1, help="ONNX intra-op threads per worker")
    s.set_defaults(func=cmd_embed_bench)

    s = sub.add_parser("embed", help="embed every chunk into a vectors file")
    s.add_argument("--model", default="bge-small", choices=list(embed.MODELS))
    s.add_argument("--workers", type=int, default=20)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_embed)

    s = sub.add_parser("retrieve", help="exact top-k search over a vectors file")
    s.add_argument("--model", default="bge-small", choices=list(embed.MODELS))
    s.add_argument("--query", default=None)
    s.add_argument("--queries", type=Path, default=None, help="one query per line")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--full", action="store_true", help="dump whole chunk text, not a preview")
    s.add_argument("--out", type=Path, default=None)
    s.set_defaults(func=cmd_retrieve)

    s = sub.add_parser("eval", help="P@k on the golden set, per index backend")
    s.add_argument("--model", default="minilm-int8", choices=list(embed.MODELS))
    s.add_argument("--queries", type=Path, required=True, help="one query per line")
    s.add_argument("--judgments", type=Path, required=True, help="query_idx/chunk_id/relevant tsv")
    s.add_argument(
        "--queries-limit",
        type=int,
        default=None,
        help="score only the first N queries (negative controls sit at the tail)",
    )
    s.add_argument("--backends", nargs="+", default=["exact"], choices=list(index.BUILDERS))
    s.add_argument(
        "--rankers",
        nargs="+",
        default=["dense"],
        choices=["dense", "bm25", "hybrid", "rerank", "rerank-union"],
        help="dense runs --backends; bm25/hybrid/rerank-union need `corpus lexical`",
    )
    s.add_argument(
        "--reranker",
        default="ms-marco-minilm",
        choices=list(rerank.RERANKERS),
        help="cross encoder for the rerank arms",
    )
    s.add_argument("--depth", type=int, default=50, help="candidates per arm before fuse/rerank")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--out", type=Path, default=None)
    s.add_argument(
        "--dump-unjudged", type=Path, default=None, help="jsonl of pairs needing a verdict"
    )
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("lexical", help="build the FTS5 index over chunk text")
    s.add_argument("--rebuild", action="store_true", help="drop and repopulate")
    s.set_defaults(func=cmd_lexical)

    s = sub.add_parser("index-bench", help="sqlite-vec vs FAISS on the stored vectors")
    s.add_argument("--model", default="minilm-int8", choices=list(embed.MODELS))
    s.add_argument("--queries", type=Path, required=True, help="one query per line")
    s.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[39693],
        help="vector counts to test; above the stored count the set is synthetic",
    )
    s.add_argument(
        "--backends", nargs="+", default=list(index.BUILDERS), choices=list(index.BUILDERS)
    )
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--reps", type=int, default=5, help="timing repetitions per query")
    s.add_argument("--out", type=Path, default=None)
    s.set_defaults(func=cmd_index_bench)

    s = sub.add_parser("classify", help="score relevance + labels with claude -p")
    s.add_argument("--sample", type=Path, required=True)
    s.add_argument("--model", default="sonnet")
    s.add_argument("--batch", type=int, default=BATCH_SIZE)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--ids", type=Path, default=None, help="file of work_ids to restrict to")
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_classify)

    s = sub.add_parser("distill", help="held-out agreement of a local classifier vs sonnet")
    s.add_argument("--model", default="minilm-int8", choices=list(embed.MODELS))
    s.add_argument("--workers", type=int, default=20)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--seed", type=int, default=distill.SEED)
    s.add_argument("--curve", type=int, nargs="*", help="training sizes for a learning curve")
    s.add_argument("--out", type=Path, default=None)
    s.set_defaults(func=cmd_distill)

    s = sub.add_parser("gate", help="score unjudged kept works with the distilled head (resumable)")
    s.add_argument("--model", default="minilm-int8", choices=list(embed.MODELS))
    s.add_argument("--workers", type=int, default=20)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--slab", type=int, default=distill.SLAB, help="works per write+commit")
    s.add_argument("--limit", type=int, default=None, help="cap works this run")
    s.add_argument("--seed", type=int, default=distill.SEED)
    s.set_defaults(func=cmd_gate)

    s = sub.add_parser("compare", help="agreement between two classify runs")
    s.add_argument("--a", type=Path, required=True)
    s.add_argument("--b", type=Path, required=True)
    s.add_argument("--threshold", type=int, default=compare.KEEP_THRESHOLD)
    s.add_argument("--disagreements", type=Path, default=None)
    s.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
