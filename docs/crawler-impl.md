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
  status TEXT NOT NULL,          -- pending | active | done
  added_by TEXT NOT NULL,        -- taxonomy | backend | manual
  last_crawled_at TEXT,          -- ISO timestamp; drives re-crawl
  watermark TEXT                 -- newest publication date seen; next pass starts here
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

Two sources behind one interface (`discover(topic) -> iterator[Candidate]`):

- **OpenAlex** (primary): `GET api.openalex.org/works?search=<query>` with
  `filter=is_oa:true,from_publication_date:<watermark>`, cursor paging (`cursor=*`),
  `mailto=` param for the polite pool. Take: id, DOI, PMCID (`ids` object), title, year,
  authors, `is_retracted`, `best_oa_location` URL, abstract.
- **Europe PMC** (biomedical depth): the existing `europepmc.search()`, unchanged, mapped
  into the same Candidate shape.

Every candidate is normalized to the canonical `work_id` (OpenAlex → DOI → PMCID, first
available) before the seen-check. Already in `works`: skip, regardless of which source or
name it arrived under.

### 2. Gate — mechanical

Cheap checks first, in order; first failure writes a `rejected` row with `reject_reason`:

1. Retracted (`is_retracted` from OpenAlex) → status `retracted`, never fetched.
2. Not open-access / no usable text route.
3. Outside the topic's recency window, if the topic sets one.

### 3. Gate — model triage

Claude (Sonnet) judges title + abstract, batched N candidates per call. Output per paper,
enforced via tool-use schema: `relevant: bool`, `study_type`, `confidence: 0..1`.
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
2. `best_oa_location` URL from OpenAlex — fetch and extract text.
3. Neither → status `kept_miss`: full metadata row, no text, fetchable later.

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
- Secret: Anthropic API key (the existing `claude-credentials` k3s secret or a sibling).
- Config (env): `SANA_CORPUS`, `ANTHROPIC_API_KEY`, `OPENALEX_MAILTO`, `POLL_SECONDS`,
  `RECRAWL_DAYS`.
- Logs to stdout; `kubectl logs` is the interface. Metrics endpoint deferred until the
  backend sets the pattern (`backend.md`).

The one-shot CLI (`python -m scraper "<query>"`) stays: it enqueues a topic and runs the
loop once — same code path, no drift between manual and service runs.

## Rate limits and retries

- OpenAlex polite pool: ~10 req/s, 100k/day. Stay far under with a fixed inter-request
  delay; no parallelism in v1.
- HTTP: retry 429/5xx with exponential backoff, small cap; then record the failure and
  move on. A source being down fails the topic pass, not the process.
- Anthropic: batch triage calls; on failure leave candidates untriaged (`candidate`
  status) for the next pass rather than guessing.

## Flagged uncertainties

Verify these during build; each has a fallback:

1. **OpenAlex incremental filter**: `from_updated_date` may be a Premium-only filter.
   Fallback (assumed in this spec): watermark on `from_publication_date` — misses
   later-edited records, acceptable.
2. **Abstract format**: OpenAlex returns abstracts as an inverted index that must be
   reconstructed; some works have none. Triage on title alone → lower confidence, and the
   prompt must say the abstract is missing.
3. **Two writers, one SQLite file**: crawler writes everything; backend inserts topics.
   WAL supports multi-process on a local filesystem (hostPath qualifies), but this is
   untested here — verify under k3s before relying on it. Fallback: backend enqueues via
   a tiny HTTP endpoint on the crawler instead of touching the DB.
4. **Retraction recall**: `is_retracted` covers discovery-time screening only. Already-
   stored works retracted later are caught on re-crawl passes at the earliest — no
   real-time signal. Acceptable for v1; noted so it isn't mistaken for full coverage.
5. **OA-URL text extraction** (fetch fallback 2): arbitrary publisher HTML/PDF → text is
   the messiest part of the pipeline and may need a dependency (e.g. trafilatura),
   breaking the scraper's stdlib-only rule. Decide when reached; `kept_miss` is the
   escape hatch until then.

## Not in this spec

Topic taxonomy contents (human pass, `crawler.md`) · chunking/embedding (backend) ·
text-cleaning depth · queryable index beyond SQLite · distilled summary layer ·
how the backend detects uncovered chat topics (backend-side; its output is just a
`topics` insert).
