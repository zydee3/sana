# scraper

Python crawler that builds Sana's research corpus: it discovers open-access scientific papers
(mental health / wellness, broadly), judges what is worth keeping, and lands graded plain-text
articles in a store the backend retrieves from, so the AI's replies are grounded in published
research. Design: `../docs/crawler.md` (what and why), `../docs/crawler-impl.md` (how).

The seam with the rest of Sana is **data, not code**: the scraper writes the corpus, the backend
reads it. The corpus is public data — it sits outside the per-user encryption boundary.

## What's built (crawler v1)

A topic-queue worker. Anyone enqueues a topic; the worker drains the queue, one pipeline pass
per topic: discover → gate → triage → fetch → store → citation expansion.

```bash
make setup                                    # uv sync (creates the venv, installs dev tools)
make run ARGS='add-topic "sleep and adolescents" --openalex-topic T10272'
make run ARGS='run --once'                    # drain the queue, then exit
make run ARGS='run'                           # keep polling (the service mode)
make check                                    # ruff + mypy + pytest (the verification gate)
```

Config (env): `SANA_CORPUS` (corpus root, default `./corpus`), `CLAUDE_BIN` (path to the
claude CLI, default `claude` on PATH), `OPENALEX_API_KEY` (raises the free OpenAlex budget
10x), `POLL_SECONDS`, `RECRAWL_DAYS`, `DISCORD_BOT_TOKEN` + `DISCORD_CHANNEL_ID` (hourly
scraped-count message; unset = off).

### The pipeline

1. **Discover** — OpenAlex works filtered by the topic's OpenAlex topic id (1 credit/page;
   the API is metered), plus Europe PMC search with `FIRST_IDATE` windows as the incremental
   mechanism. Watermark per topic; re-crawls only see new work.
2. **Gate** — retracted papers are excluded; non-open-access rejected. Every candidate becomes
   a `works` row (kept, rejected, or missed) so re-crawls never re-judge old papers.
3. **Triage** — Claude judges relevance + study type from title/abstract, batched through
   **Claude Code headless** (`claude -p`, the same runtime as the backend), so it uses the
   operator's existing Claude credentials — no separate metered API key. Study type maps to
   evidence grade 1 (meta-analysis) .. 5 (case report/opinion), a retrieval weight, never a
   drop reason. Without the claude CLI, or on a failed call, papers keep metadata-derived
   grades / defer to the next pass.
4. **Fetch** — PMC OA bucket plain text first (needs a PMCID, joined from the DOI via Europe
   PMC — OpenAlex no longer carries PMCIDs), then Europe PMC `fullTextXML`; neither → the work
   is kept as metadata-only (`kept_miss`) and retried on later passes.
5. **Store** — `corpus/texts/<work_id>.txt` + a row in `corpus/corpus.db` (SQLite, WAL):
   identity across sources, provenance, grade, misses, watermarks.
6. **Expand** — depth-1 citation walk (citers + references via OpenAlex) from kept works,
   bounded per pass, through the same gate.

Python dependencies: **none** — the standard library (`urllib`, `sqlite3`, `subprocess`,
`json`, `xml`) covers HTTP, storage, and parsing. Triage shells out to the `claude` CLI
(optional at runtime; the pipeline degrades to metadata grades without it). Dev tools
(ruff, mypy, pytest) are managed by `uv`.

## Deployment (k3s)

`make deploy` builds the image (python + claude CLI), imports it into k3s, renders
`topics.md` into the `sana-topics` ConfigMap, applies `deploy.yaml`, and restarts. The
container runs `python -m scraper` with no args: sync `$SANA_TOPICS` → drain the queue →
poll forever. Data lands on the hostPath `/sana-data` (symlink to the data disk); Claude
credentials are a live hostPath mount of `~/.claude/.credentials.json` (read-only, no
subPath — a copied secret goes stale when tokens rotate). Edit `topics.md`, `make deploy`
again to ship new topics.

## Not built yet

- Topic taxonomy beyond the seed list in `topics.md`.
- Backend enqueue path (inserting into the shared SQLite `topics` table).
- Re-triage of deferred candidates and retry of metadata-only misses.
- Text normalization beyond the sources' own extraction; a queryable index beyond SQLite.
- Chunking / embedding is owned by the backend's retrieval side, not here.
