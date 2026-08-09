# Crawler: Implementation Spec

Companion to `crawler.md`. That doc says what and why; this doc says how: components,
data model, APIs, and the run loop. Code under `scraper/` is the source of truth once built.

## Shape of the system

One long-running worker (k3s Deployment, 1 replica) draining a **topic queue**:

1. Pop the next pending topic from the `topics` table.
2. Run the pipeline for that topic (discover → gate → fetch → store → expand).
3. Mark the topic done with a watermark; pop the next.
4. Queue empty: sleep `POLL_SECONDS`, re-check. Topics past `RECRAWL_DAYS` re-enter as pending.

Anyone can enqueue a topic by inserting a row — the human taxonomy pass, or the backend when
a chat touches a topic the corpus doesn't cover. The insert is the whole integration contract;
the crawler does not know or care who asked.

## Storage

One hostPath volume (`/sana-data/corpus`), two things on it:

- `texts/<work_id>.txt` — full text, write-once flat files.
- `corpus.db` — SQLite (WAL mode), source of truth for everything else: metadata, seen-set,
  misses, rejections, watermarks. The JSON sidecars are retired; the DB replaces them.

Backend retrieval later mounts the same volume read-only: query `corpus.db` to choose works,
read `texts/` for content.

### Tables

```sql
topics(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,            -- human label, e.g. "sleep and adolescents"
  query TEXT NOT NULL,           -- the search expression sent to sources
  openalex_id TEXT,              -- OpenAlex topic id (e.g. T10272); enables 10x-cheaper discovery
  status TEXT NOT NULL,          -- pending | active | done
  added_by TEXT NOT NULL,        -- taxonomy | backend | manual
  last_crawled_at TEXT,          -- ISO timestamp; drives re-crawl
  watermark TEXT,                -- newest date seen per source; next pass starts here
  openalex_cursor TEXT,          -- resume point of an unfinished sweep; NULL = start at the head
  epmc_cursor TEXT               -- same, for the Europe PMC cursorMark
)

works(
  work_id TEXT PRIMARY KEY,      -- canonical id: OpenAlex W-id, else doi:<doi>, else pmcid:<id>
  openalex_id TEXT, doi TEXT, pmcid TEXT,
  title TEXT NOT NULL,
  year INTEGER, authors TEXT, license TEXT,
  topic_id INTEGER REFERENCES topics(id),   -- what pulled it in
  discovered_via TEXT NOT NULL,  -- openalex | europepmc | citation
  status TEXT NOT NULL,          -- candidate | rejected | kept_miss | kept_text | retracted
  reject_reason TEXT,            -- filter name or 'triage'; NULL unless rejected
  study_type TEXT,               -- model-inferred
  evidence_grade INTEGER,        -- 1..5, NULL until triaged
  triage_confidence REAL,        -- model self-rating 0..1
  text_path TEXT,                -- NULL unless status = kept_text
  text_source TEXT,              -- pmc_oa_txt | oa_url | NULL
  fetched_at TEXT,
  expanded_at TEXT               -- NULL = citation walk hasn't visited this work yet
)
```

The in-memory seen-set from `crawler.md` is `SELECT work_id FROM works` loaded at startup;
the table is its durable form. Rejections stay as rows so a re-crawl never re-pays triage
for a paper already judged.

## Pipeline stages (per topic)

### 1. Discover

Two sources behind one interface (`discover(topic) -> iterator[Candidate]`). Facts below
verified against the live APIs (2026-08-07).

- **OpenAlex** (primary, for topics with an `openalex_id`):
  `GET api.openalex.org/works?filter=primary_topic.id:<Tid>,is_oa:true,from_publication_date:<watermark>`
  with `select=` field trimming, `sort=publication_date:desc`, cursor paging (`cursor=*`,
  per-page up to 200). Topic-id filtering costs 1 credit/page; `search=` and `*.search`
  filters cost 10 credits/page — prefer topic ids. The `/topics` entity (~4,500 curated
  topics, searchable) is where taxonomy rows get their `openalex_id`.
  Take: id, DOI, title, year, authors, `is_retracted`, `open_access`, abstract
  (inverted index — reconstruct by sorting positions). **`ids.pmcid` is no longer
  populated** — PMCIDs come from the Europe PMC join below.
  Publication dates include future-dated in-press records (seen: 2030); clamp or tolerate.
- **Europe PMC** (biomedical depth, and all query-only topics): the existing
  `europepmc.search()` plus a `FIRST_IDATE:[<watermark> TO *]` clause — index-date
  windows are the free incremental mechanism. `cursorMark=*` paging, pageSize up to 1000
  (verified). `resultType=core` also returns `pubTypeList` (labels like "Systematic
  Review") — a free mechanical evidence-grade signal — plus abstract and license.
- **Join**: OpenAlex candidates carry DOIs; batch them into one EPMC query
  (`DOI:"..." OR DOI:"..."`, ~20 per call, free) to obtain PMCIDs for fetch. Papers
  published in the last weeks usually have no PMCID yet (PMC deposition lags months) —
  they become `kept_miss` and resolve on a later pass.

Every candidate is normalized to the canonical `work_id` (OpenAlex → DOI → PMCID, first
available) before the seen-check. Already in `works`: skip, regardless of which source or
name it arrived under.

### 2. Gate — mechanical

Cheap checks first, in order; first failure writes a `rejected` row with `reject_reason`:

1. Retracted (`is_retracted` from OpenAlex) → status `retracted`, never fetched.
2. Not open-access / no usable text route.
3. Outside the topic's recency window, if the topic sets one.

### 3. Gate — model triage

Claude (Sonnet) judges title + abstract, batched N candidates per call, via **Claude Code
headless** (`claude -p --output-format json`) — the same runtime the backend uses, so triage
authenticates with the operator's existing Claude credentials (subscription or key) instead
of a separate metered API key. The prompt demands a bare JSON array; per paper:
`relevant: bool`, `study_type`, `confidence: 0..1`. A malformed reply defers the batch.
Study type maps to grade:

| Grade | Study types |
|---|---|
| 1 | meta-analysis, systematic review |
| 2 | randomized controlled trial |
| 3 | cohort, case-control |
| 4 | cross-sectional, observational |
| 5 | case report, opinion, qualitative |

Not relevant → `rejected` (reason `triage`). Relevant → keep with grade. The grade is a
weight for retrieval, never a drop. The exact mapping and the keep-threshold on
`confidence` are tunable — start permissive, tighten against real triage output
(open question in `crawler.md`).

### 4. Fetch

In order; first success wins:

1. PMC OA bucket plain text — existing `pmc_oa.download_text()` (needs a PMCID).
   Verified fresh: papers indexed weeks ago already have `.txt` in the bucket.
2. Europe PMC `GET /rest/<pmcid>/fullTextXML` — JATS XML, strip to text (verified live).
3. Neither → status `kept_miss`: full metadata row, no text, fetchable later.

Publisher `oa_url` fetching is dropped for v1: publishers bot-block plain HTTP clients
(verified 403), and real extraction would need a dependency. `kept_miss` covers the gap.

### 5. Store

Write `texts/<work_id>.txt`, then the `works` row (`kept_text`, `text_path`,
`text_source`, `fetched_at`) — file first, row second, so a crash never yields a row
pointing at a missing file. Normalization beyond the source's own text extraction is
deferred (open question in `crawler.md`).

### 6. Expand

After a topic's search results are processed: for each `kept_text` work with
`expanded_at IS NULL`, pull `referenced_works` and citers
(`filter=cites:<id>`) from OpenAlex, run them through the same gate as candidates
(`discovered_via = citation`), stamp `expanded_at`. Bounds: depth 1 per pass and
`MAX_EXPAND_PER_TOPIC`; hitting a cap logs a coverage line and sets the work aside for
the next pass — caps are recorded, never silent.

## Deployment

- k3s Deployment, 1 replica, image built from `scraper/` (own Dockerfile, per monorepo rule).
- Volume: hostPath `/sana-data/corpus` mounted read-write.
- Image: python3.12-slim + Node 22 + the claude CLI; UID-1000 user. Credentials are a
  **live hostPath mount** of `~/.claude/.credentials.json` (read-only, no subPath) — never
  a copied secret: the CLI rotates OAuth tokens and snapshots start returning 401.
- Topics: `scraper/topics.md` (one bullet per topic, optional `(Txxxx)` OpenAlex id) is
  rendered into the `sana-topics` ConfigMap by `make deploy` and read from `$SANA_TOPICS`
  at startup (idempotent sync into the queue). Topics claimed by a killed worker are
  re-queued on startup.
- `strategy: Recreate` — corpus.db has exactly one writer.
- Config (env): `SANA_CORPUS`, `SANA_TOPICS`, `CLAUDE_BIN`, `OPENALEX_API_KEY`,
  `POLL_SECONDS`, `RECRAWL_DAYS`, `DISCORD_BOT_TOKEN`/`DISCORD_CHANNEL_ID`.
- Observability: hourly Discord message (count of works with text / tracked) posted by a
  daemon thread; best-effort, never blocks the crawl. Token is copied at deploy time from
  the discord-bot namespace secret; unset disables the reporter.
- `make deploy` (scraper/ or root): build image → import into k3s containerd → render
  ConfigMap → apply → restart. A page cap bounds the work per pass, not the sweep: the
  stopping cursor of each source is stored on the topic and the pass resumes from it, so
  a topic mid-sweep is re-claimable immediately and skips the re-crawl interval. The
  watermark is held until a sweep finishes, so a window is only ever recorded as covered
  once it truly is. A 4xx on a resumed pass drops the cursors and restarts from the head.
- Logs to stdout; `kubectl logs` is the interface. Metrics endpoint deferred until the
  backend sets the pattern (`backend.md`).

The CLI drives the same code path as the service — no drift between manual and service
runs: `python -m scraper add-topic "<name>" [--query Q] [--openalex-topic Tid]` enqueues;
`python -m scraper run --once` drains the queue and exits; `run` polls forever.

## Rate limits and retries

- **OpenAlex is metered** (verified 2026-08-07): anonymous = $0.10/day (1,000 credits),
  free API key = $1/day. Measured costs: entity lookup (DOI/W-id) = 0 credits and
  unlimited; filtered list = 1 credit/page; any text search = 10 credits/page. Budget
  headers (`x-ratelimit-*`) arrive on every response — log remaining credits each pass,
  and stop discovery for the day when they run out (recorded, per the no-silent-caps rule).
- Europe PMC and the PMC OA bucket: free, no rate-limit headers; keep a fixed
  inter-request delay as courtesy. No parallelism in v1.
- HTTP: retry 429/5xx with exponential backoff, small cap; then record the failure and
  move on. A source being down fails the topic pass, not the process.
- Triage: batch `claude -p` calls; on failure leave candidates untriaged (`candidate`
  status) for the next pass rather than guessing. Spend lands on the operator's Claude
  plan (subscription usage windows apply), not a per-token API bill.

## Flagged uncertainties

Resolved by live verification (2026-08-07):

- `from_updated_date` is Premium-only, confirmed ("Plan upgrade required"). Incremental
  crawl uses EPMC `FIRST_IDATE` windows instead; OpenAlex falls back to
  `from_publication_date` watermarks.
- Abstract inverted-index reconstruction verified trivial; some works have no abstract —
  triage on title alone, prompt must say so.
- Publisher `oa_url` fetching confirmed bot-blocked (403) — dropped for v1, see Fetch.

Still open; each has a fallback:

1. **Two writers, one SQLite file**: crawler writes everything; backend inserts topics.
   WAL supports multi-process on a local filesystem (hostPath qualifies), but this is
   untested here — verify under k3s before relying on it. Fallback: backend enqueues via
   a tiny HTTP endpoint on the crawler instead of touching the DB.
2. **Retraction recall**: `is_retracted` covers discovery-time screening only. Already-
   stored works retracted later are caught on re-crawl passes at the earliest — no
   real-time signal. Acceptable for v1; noted so it isn't mistaken for full coverage.
3. **OpenAlex key mechanics**: how the free API key is passed (header vs `api_key`
   param) is undocumented in what was checked; verify when a key exists. Anonymous tier
   works today and the code must run keyless.

## Not in this spec

Topic taxonomy contents (human pass, `crawler.md`) · chunking/embedding (backend) ·
text-cleaning depth · queryable index beyond SQLite · distilled summary layer ·
how the backend detects uncovered chat topics (backend-side; its output is just a
`topics` insert).
