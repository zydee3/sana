"""CLI: pull open-access research PDFs into the corpus.

python -m scraper "mindfulness anxiety"        # pull one paper
python -m scraper "sleep quality" -n 3         # pull up to three
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import corpus, europepmc, pmc_oa


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scraper",
        description="Pull open-access research PDFs into Sana's corpus.",
    )
    parser.add_argument("query", help='search terms, e.g. "mindfulness anxiety"')
    parser.add_argument("-n", "--limit", type=int, default=1, help="max papers to pull (default 1)")
    parser.add_argument(
        "--corpus-dir",
        default=os.environ.get("SANA_CORPUS", "corpus"),
        help="output dir (default ./corpus or $SANA_CORPUS)",
    )
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    papers = europepmc.search(args.query, args.limit)
    if not papers:
        print(f"no open-access papers found for: {args.query}", file=sys.stderr)
        return 1

    for p in papers:
        if corpus.has(corpus_dir, p.pmcid):
            print(f"skip  {p.pmcid}  (already in corpus)")
            continue
        try:
            url, data = pmc_oa.download_pdf(p.pmcid)
        except LookupError as e:
            print(f"miss  {p.pmcid}  ({e})")
            continue
        path = corpus.save(corpus_dir, p, url, data)
        print(f"saved {p.pmcid}  {len(data) // 1024} KB  -> {path}")
        print(f"      {p.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
