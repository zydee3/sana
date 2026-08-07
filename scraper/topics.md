# Corpus topics

One topic per bullet line. An optional trailing `(Txxxx)` is the topic's OpenAlex id
(find them at https://api.openalex.org/topics?search=...) — with it, discovery uses the
cheap topic filter; without it, only Europe PMC keyword search runs. Everything that is
not a bullet line is ignored, so notes like this are fine.

`make deploy` renders this file into the sana-topics ConfigMap; the crawler syncs it
into its queue at startup. Removing a line here does not remove already-crawled topics.

- mental health treatment and access (T10272)
- resilience and mental health (T11761)
- mindfulness and compassion interventions (T10708)
