# scraper

Python ingester that builds Sana's research corpus: it collects scientific papers (wellness /
health) and lands them in a store the backend retrieves from, so the AI's replies are grounded
in published research.

Not built yet. What is decided:

- Python, self-contained in this directory (own build system; eventually its own container).
- The seam with the rest of Sana is **data, not code**: the scraper writes the corpus, the
  backend reads it. The corpus schema is the contract.
- The corpus is public data — it sits outside the per-user encryption boundary.

Open: paper sources (PubMed / Europe PMC / Semantic Scholar expose real APIs — likely API
ingestion rather than HTML scraping, also cleaner licensing) · corpus store and schema ·
chunking/embedding strategy (owned by the backend's retrieval side).
