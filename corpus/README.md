# sana-corpus

Processing side of the research corpus: reads `corpus.db` + `texts/` written by
`scraper/`, and turns them into retrieval-ready data — relevance scores, study-type
labels, cleaned section-aware chunks, and a CPU-only vector index.

The scraper stays stdlib-only because it runs in the ingest path; this project may take
dependencies (embedding runtime, vector index).

## Data it reads and writes

- `$SANA_CORPUS_DB` (default `/sana-data/corpus/corpus.db`) — `works` rows plus the
  columns this project adds (`relevance`, `domain`, `label_source`, `label_confidence`)
  and the `abstracts` table. Migrations apply on connect.
- `/sana-data/corpus/texts/*.txt` — article text, input to cleaning and chunking.

## CLI

    make run ARGS='sample --n 1000 --seed 7 --out /sana-data/corpus/calibration/sample.jsonl'
    make run ARGS='abstracts --sample <sample.jsonl>'        # rehydrate abstracts from Europe PMC
    make run ARGS='classify --sample <sample.jsonl> --model haiku --limit 200 --out <run.jsonl>'
    make run ARGS='compare --a <run-a.jsonl> --b <run-b.jsonl> --disagreements <out.jsonl>'

Model calls go through Claude Code headless (`claude -p`) on the operator's
subscription — never a metered API key. Embeddings are local models only.
