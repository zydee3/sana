"""CLI for the corpus crawler.

python -m scraper                     # the service mode: sync $SANA_TOPICS, poll forever
python -m scraper add-topic "sleep and adolescents" --openalex-topic T10272
python -m scraper run --once          # drain the queue, then exit

Topics arrive either via `add-topic` or the SANA_TOPICS env var (bullet lines,
`- <topic name> (Txxxx)` with an optional OpenAlex topic id), synced into the queue
at startup — in k3s that env comes from the sana-topics ConfigMap.

Ingest needs no model: papers are kept as discovered and graded from publisher
metadata. Config (env): SANA_CORPUS (default ./corpus), SANA_TOPICS,
OPENALEX_API_KEY, POLL_SECONDS, RECRAWL_DAYS, and — for the hourly
scraped-count message — DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID (unset = off).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import crawl, db, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scraper", description="Build Sana's research corpus.")
    parser.add_argument(
        "--corpus-dir",
        default=os.environ.get("SANA_CORPUS", "corpus"),
        help="corpus root (default ./corpus or $SANA_CORPUS); holds corpus.db + texts/",
    )
    sub = parser.add_subparsers(dest="command")

    add = sub.add_parser("add-topic", help="enqueue a topic to crawl")
    add.add_argument("name", help='topic label, e.g. "sleep and adolescents"')
    add.add_argument("--query", help="search expression (default: the name)")
    add.add_argument("--openalex-topic", help="OpenAlex topic id (e.g. T10272) for cheap discovery")

    run = sub.add_parser("run", help="crawl queued topics (the default command)")
    run.add_argument("--once", action="store_true", help="drain the queue then exit")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:] or ["run"])
    corpus_dir = Path(args.corpus_dir)
    conn = db.connect(corpus_dir / "corpus.db")

    if args.command == "add-topic":
        db.add_topic(conn, args.name, args.query or args.name, args.openalex_topic)
        print(f"queued: {args.name}")
        return 0

    recovered = db.recover_active_topics(conn)
    if recovered:
        print(f"recovered {recovered} topics claimed by a previous worker", flush=True)
    crawl.sync_topics(conn, os.environ.get("SANA_TOPICS", ""))
    recrawl_days = int(os.environ.get("RECRAWL_DAYS", "7"))
    if getattr(args, "once", False):
        while crawl.run_once(conn, corpus_dir, recrawl_days):
            pass
        return 0
    token = os.environ.get("DISCORD_BOT_TOKEN")
    channel_id = os.environ.get("DISCORD_CHANNEL_ID")
    if token and channel_id:
        report.start_reporter(corpus_dir / "corpus.db", token, channel_id)
        print("discord reporter started", flush=True)
    poll_seconds = int(os.environ.get("POLL_SECONDS", "300"))
    crawl.run_loop(conn, corpus_dir, poll_seconds, recrawl_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
