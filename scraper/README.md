# scraper

Python ingester that builds Sana's research corpus: it collects open-access scientific papers
(wellness / health) and lands them in a store the backend retrieves from, so the AI's replies
are grounded in published research.

The seam with the rest of Sana is **data, not code**: the scraper writes the corpus, the backend
reads it. The corpus is public data — it sits outside the per-user encryption boundary.

## What's built (walking skeleton)

Pull one open-access paper's PDF by search query:

```bash
make setup                           # uv sync (creates the venv, installs dev tools)
make run QUERY="mindfulness anxiety"  # -> ./corpus/PMC*.pdf + PMC*.json
make run QUERY="sleep quality" ARGS="-n 3"
make check                           # ruff + mypy + pytest (the verification gate)
```

Each paper lands as `corpus/<pmcid>.pdf` plus a `corpus/<pmcid>.json` sidecar (pmcid, title,
doi, authors, year, **license**, source URL). Re-running skips papers already in the corpus.

### How it fetches

1. **Discovery** — Europe PMC search API, filtered to `IN_EPMC:Y AND OPEN_ACCESS:Y`, gives the
   PMCID and metadata.
2. **Fetch** — the PMC open-access dataset on AWS (`pmc-oa-opendata` S3 bucket), keyed per paper
   at `PMC<id>.<version>/`, serves the PDF over plain HTTPS.

This path avoids Europe PMC's `?pdf=render` endpoint (returning 404s as of 2026-07-18) and NCBI
FTP (blocked in many environments). No API key required.

Runtime dependencies: **none** — the standard library (`urllib`, `json`, `xml`) covers HTTP and
parsing. Dev tools (ruff, mypy, pytest) are managed by `uv`.

## Not built yet

- **Scraping vs. crawling**: this pulls papers you name via a query. Continuous / automated
  crawling (seed → expand by citation or topic, incrementally, on a schedule) is the next step —
  the effective crawl algorithm is still to be designed.
- Corpus store beyond flat PDFs (a queryable index / schema), and its own container.
- Chunking / embedding is owned by the backend's retrieval side, not here.
- Only papers whose PDF exists in the PMC OA bucket are retrieved; full-text XML fallback and
  other sources (OpenAlex, Semantic Scholar) are not wired in.
