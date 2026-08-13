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
from datetime import UTC, datetime
from pathlib import Path

from . import backfill, chunk, compare, db, embed, epmc, index, judge, sample
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
    s.set_defaults(func=cmd_judge)

    s = sub.add_parser("chunk", help="clean + section-aware chunk stored texts (resumable)")
    s.add_argument("--min-relevance", type=int, default=5, help="only works scored at least this")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--limit", type=int, default=None, help="cap works this run")
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
