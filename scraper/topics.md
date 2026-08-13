# Corpus topics

One topic per bullet line. An optional trailing `(Txxxx)` is the topic's OpenAlex id
(find them at https://api.openalex.org/topics?search=...) — with it, discovery uses the
cheap topic filter; without it, only Europe PMC keyword search runs. Everything that is
not a bullet line is ignored, so notes like this are fine.

Everything after `::` is that topic's Europe PMC query. Write it field-scoped: a bare
name searches every field including reference lists, which is how the first sweep of
"mental health treatment and access" pulled 124k works at 30% relevance — a paper that
merely cites mental-health work matches. TITLE_ABS restricts to what the paper is about
(measured: 30% -> 76% relevant, and the pool it has to walk drops from 492k to 38k).
Changing a query here restarts that topic's sweep from the head of the new result set.

`make deploy` renders this file into the sana-topics ConfigMap; the crawler syncs it
into its queue at startup. Removing a line here does not remove already-crawled topics.

- mental health treatment and access (T10272) :: TITLE_ABS:"mental health" AND (TITLE_ABS:"treatment" OR TITLE_ABS:"psychotherapy" OR TITLE_ABS:"intervention")
- resilience and mental health (T11761) :: TITLE_ABS:"resilience" AND TITLE_ABS:"mental health"
- mindfulness and compassion interventions (T10708)
