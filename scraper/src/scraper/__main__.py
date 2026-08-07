"""CLI for the corpus crawler.

python -m scraper add-topic "sleep and adolescents" --openalex-topic T10272
python -m scraper run --once          # drain the queue, then exit
python -m scraper run                 # keep polling (the k3s service mode)

Triage runs through Claude Code headless (`claude -p`), so it needs the claude CLI
on PATH with logged-in credentials; without it papers are kept with
metadata-derived grades. Config (env): SANA_CORPUS (default ./corpus), CLAUDE_BIN,
OPENALEX_API_KEY, POLL_SECONDS, RECRAWL_DAYS.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import crawl, db, triage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scraper", description="Build Sana's research corpus.")
    parser.add_argument(
        "--corpus-dir",
        default=os.environ.get("SANA_CORPUS", "corpus"),
        help="corpus root (default ./corpus or $SANA_CORPUS); holds corpus.db + texts/",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add-topic", help="enqueue a topic to crawl")
    add.add_argument("name", help='topic label, e.g. "sleep and adolescents"')
    add.add_argument("--query", help="search expression (default: the name)")
    add.add_argument("--openalex-topic", help="OpenAlex topic id (e.g. T10272) for cheap discovery")

    run = sub.add_parser("run", help="crawl queued topics")
    run.add_argument("--once", action="store_true", help="drain the queue then exit")

    args = parser.parse_args(argv)
    corpus_dir = Path(args.corpus_dir)
    conn = db.connect(corpus_dir / "corpus.db")

    if args.command == "add-topic":
        db.add_topic(conn, args.name, args.query or args.name, args.openalex_topic)
        print(f"queued: {args.name}")
        return 0

    use_triage = triage.available()
    if not use_triage:
        print("warning: claude CLI not found; keeping papers with metadata-derived grades")
    recrawl_days = int(os.environ.get("RECRAWL_DAYS", "7"))
    if args.once:
        while crawl.run_once(conn, corpus_dir, use_triage, recrawl_days):
            pass
        return 0
    poll_seconds = int(os.environ.get("POLL_SECONDS", "300"))
    crawl.run_loop(conn, corpus_dir, use_triage, poll_seconds, recrawl_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
