"""The crawl worker: drain the topic queue, one pipeline pass per topic.

Per topic: discover (OpenAlex topic filter + Europe PMC query) → mechanical gate
(retraction, open access) → PMCID join → fetch text (PMC bucket, then fullTextXML)
→ store → bounded citation expansion. Every candidate becomes a works row — kept,
rejected, or missed — so re-crawls never re-judge old papers.

Fetch-first: nothing here calls a model. Topic-scoped discovery is the relevance
filter (it lets some off-topic papers through, accepted), and the study type comes
from publisher metadata where that is decisive. Ingest therefore keeps running
through model quota limits and credential outages, which is what actually stalled
it. Works left ungraded are the work list for out-of-band enrichment (triage.py).
"""

from __future__ import annotations

import re
import sqlite3
import time
import urllib.error
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from . import corpus, db, europepmc, openalex, pmc_oa
from .models import GRADE_BY_STUDY_TYPE, Candidate

JOIN_BATCH = 20
EXPAND_WORKS_PER_PASS = 10
REFS_PER_WORK = 10
REQUEST_DELAY_S = 0.2
# backlog works pushed through the fetch path per idle poll; bounded by fetch time
# (~1s each), not model spend
DRAIN_PER_PASS = 500

# stable leading order for the pass summary; unknown statuses are appended, never dropped
SUMMARY_STATUSES = ("kept_text", "kept_miss", "candidate", "rejected", "retracted")

# Publisher-declared pub types that decide a study type on their own. Anything else
# leaves the study type unknown rather than guessing "other" — a null grade is the
# enrichment work list, a wrong grade is silent.
STUDY_TYPE_BY_PUB_TYPE = {
    "meta-analysis": "meta_analysis",
    "network meta-analysis": "meta_analysis",
    "systematic review": "systematic_review",
    "systematic-review": "systematic_review",
    "randomized controlled trial": "rct",
    "randomized-controlled-trial": "rct",
    "observational study": "observational",
    "case reports": "case_report",
    "editorial": "opinion",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


# one topic per bullet line; optional trailing (Txxxx) is its OpenAlex topic id
_TOPIC_LINE = re.compile(r"^-\s+(.+?)(?:\s+\((T\d+)\))?\s*$")


def parse_topics(text: str) -> list[tuple[str, str | None, str | None]]:
    """Bullet lines -> (name, openalex_id|None, epmc query|None); prose ignored.

    Everything after a `::` is the Europe PMC query for that topic, which lets a
    line name a topic in prose while searching in field-scoped syntax. Without it
    the name is the query, which searches every field including reference lists.
    """
    out = []
    for line in text.splitlines():
        head, sep, query = line.strip().partition("::")
        m = _TOPIC_LINE.match(head.strip())
        if m:
            out.append((m.group(1), m.group(2), query.strip() if sep else None))
    return out


def sync_topics(conn: sqlite3.Connection, text: str) -> int:
    """Enqueue every configured topic, and re-point ones whose query changed."""
    topics = parse_topics(text)
    for name, openalex_id, query in topics:
        db.add_topic(conn, name, query or name, openalex_id, added_by="config")
        if db.set_topic_query(conn, name, query or name):
            _log(f"config: topic {name} re-pointed at a new query, sweep restarted")
    if topics:
        _log(f"config: {len(topics)} topics synced")
    return len(topics)


def _dedupe(cands: list[Candidate]) -> list[Candidate]:
    """In-run dedupe across sources: any shared identifier means the same paper."""
    out: list[Candidate] = []
    ids: set[str] = set()
    for c in cands:
        keys = [k for k in (c.work_id, c.doi, c.pmcid) if k]
        if any(k in ids for k in keys):
            continue
        ids.update(keys)
        out.append(c)
    return out


def discover(topic: db.Topic) -> tuple[list[Candidate], tuple[str | None, str | None]]:
    """This pass's candidates, plus each source's resume cursor (None = source exhausted).

    Both sources resume from the topic's stored cursor, so a page cap bounds the work
    per pass instead of the sweep: successive passes walk deeper into the same result
    set rather than re-reading its head.
    """
    cands: list[Candidate] = []
    oa_cursor: str | None = None
    if topic.openalex_id:
        oa, oa_cursor = openalex.works_by_topic(
            topic.openalex_id, topic.watermark, cursor=topic.openalex_cursor
        )
        cands.extend(oa)
        if oa_cursor:
            _log(f"coverage: openalex page cap hit for topic {topic.name}")
    ep, ep_cursor = europepmc.search_window(topic.query, topic.watermark, cursor=topic.epmc_cursor)
    cands.extend(ep)
    if ep_cursor:
        _log(f"coverage: europepmc page cap hit for topic {topic.name}")
    return _dedupe(cands), (oa_cursor, ep_cursor)


def _study_type(c: Candidate) -> str | None:
    """Study type from publisher-declared pub types, or None when they don't decide it.

    Strongest match wins: Europe PMC tags a systematic review as both "Review" and
    "Systematic Review", and a paper can be both an RCT and a meta-analysis source.
    """
    graded = [
        st for pt in c.pub_types if (st := STUDY_TYPE_BY_PUB_TYPE.get(pt.lower())) is not None
    ]
    return min(graded, key=lambda st: GRADE_BY_STUDY_TYPE[st]) if graded else None


def _join_pmcids(cands: list[Candidate]) -> list[Candidate]:
    """Fill missing PMCIDs from DOIs via Europe PMC (OpenAlex no longer carries them)."""
    missing = [c.doi for c in cands if not c.pmcid and c.doi]
    found: dict[str, str] = {}
    for start in range(0, len(missing), JOIN_BATCH):
        found.update(europepmc.pmcids_for_dois(missing[start : start + JOIN_BATCH]))
        time.sleep(REQUEST_DELAY_S)
    return [
        replace(c, pmcid=found[c.doi.lower()])
        if not c.pmcid and c.doi and c.doi.lower() in found
        else c
        for c in cands
    ]


def _fetch_text(c: Candidate) -> tuple[str, str] | None:
    """(text_source, text) via PMC bucket then fullTextXML, or None -> kept_miss."""
    if not c.pmcid:
        return None
    try:
        _, text = pmc_oa.download_text(c.pmcid)
        return ("pmc_oa_txt", text)
    except (LookupError, OSError):
        pass
    fallback = europepmc.full_text(c.pmcid)
    return ("epmc_fulltext_xml", fallback) if fallback else None


def process_candidates(
    conn: sqlite3.Connection,
    topic: db.Topic,
    cands: list[Candidate],
    corpus_dir: Path,
) -> int:
    """Gate → join → fetch → store. Returns how many works got text."""
    fresh = [c for c in cands if not db.seen(conn, c)]
    survivors: list[Candidate] = []
    for c in fresh:
        if c.is_retracted:
            db.record_work(conn, c, topic.id, status="retracted")
        elif not c.is_oa:
            db.record_work(conn, c, topic.id, status="rejected", reject_reason="not_open_access")
        else:
            survivors.append(c)
    kept = _join_pmcids(survivors)
    fetched = 0
    for c in kept:
        study_type = _study_type(c)
        db.record_work(
            conn,
            c,
            topic.id,
            status="kept_miss",
            study_type=study_type,
            evidence_grade=GRADE_BY_STUDY_TYPE[study_type] if study_type else None,
        )
        result = _fetch_text(c)
        if result:
            source, text = result
            path = corpus.save_text(corpus_dir, c.work_id, text)
            db.set_fetched(conn, c.work_id, str(path), source)
            fetched += 1
        time.sleep(REQUEST_DELAY_S)
    _log(
        f"topic {topic.name}: {len(cands)} discovered, {len(fresh)} new, "
        f"{len(kept)} kept, {fetched} with text"
    )
    return fetched


def _candidate_from_row(row: sqlite3.Row) -> Candidate:
    """Rebuild a Candidate from its works row; it already passed the mechanical gate."""
    return Candidate(
        work_id=row["work_id"],
        title=row["title"],
        discovered_via=row["discovered_via"],
        openalex_id=row["openalex_id"],
        doi=row["doi"],
        pmcid=row["pmcid"],
        year=row["year"],
        authors=row["authors"],
        license=row["license"],
    )


def _rehydrate(cands: list[Candidate]) -> list[Candidate]:
    """Works rows keep no pub types, so re-fetch them — they carry the study type.

    The same record supplies a PMCID when the row lacks one, which spares the join
    a second lookup over the same DOIs.
    """
    dois = [c.doi for c in cands if c.doi]
    found: dict[str, Candidate] = {}
    for start in range(0, len(dois), JOIN_BATCH):
        found.update(europepmc.records_for_dois(dois[start : start + JOIN_BATCH]))
        time.sleep(REQUEST_DELAY_S)
    out = []
    for c in cands:
        r = found.get(c.doi.lower()) if c.doi else None
        out.append(replace(c, pub_types=r.pub_types, pmcid=c.pmcid or r.pmcid) if r else c)
    return out


def drain_deferred(conn: sqlite3.Connection, corpus_dir: Path, limit: int = DRAIN_PER_PASS) -> int:
    """Push works a past triage outage left as 'candidate' through the fetch path."""
    rows = db.deferred_candidates(conn, limit)
    fetched = 0
    for topic_id in dict.fromkeys(int(r["topic_id"]) for r in rows):
        topic = db.topic_by_id(conn, topic_id)
        if topic is None:
            continue
        cands = [_candidate_from_row(r) for r in rows if r["topic_id"] == topic_id]
        # recorded since deferral under another identifier: retire so it stops taking a slot
        for c in cands:
            if db.seen(conn, c):
                db.record_work(conn, c, topic_id, status="rejected", reject_reason="duplicate")
        _log(f"drain: {len(cands)} deferred candidates for topic {topic.name}")
        fetched += process_candidates(conn, topic, _rehydrate(cands), corpus_dir)
    return fetched


def _rejected_request(e: OSError) -> bool:
    """4xx other than rate limiting — the request was bad, retrying it verbatim won't help."""
    return isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429


def expand(conn: sqlite3.Connection, topic: db.Topic, corpus_dir: Path) -> None:
    """Depth-1 citation walk from this topic's kept works, through the same gate."""
    ids = db.unexpanded_kept(conn, topic.id, EXPAND_WORKS_PER_PASS)
    for openalex_id in ids:
        cands = openalex.citers(openalex_id)
        refs = openalex.referenced_ids(openalex_id)
        if len(refs) > REFS_PER_WORK:
            _log(f"coverage: {openalex_id} has {len(refs)} refs; walking first {REFS_PER_WORK}")
        for ref in refs[:REFS_PER_WORK]:
            work = openalex.work_by_id(ref)
            if work:
                cands.append(work)
            time.sleep(REQUEST_DELAY_S)
        db.set_expanded(conn, openalex_id)
        process_candidates(conn, topic, _dedupe(cands), corpus_dir)


def run_once(
    conn: sqlite3.Connection,
    corpus_dir: Path,
    recrawl_days: int,
) -> bool:
    """Claim and crawl one topic. Returns False when there was nothing to do or the
    pass failed — either way the caller should back off before trying again."""
    topic = db.claim_next_topic(conn, recrawl_days)
    if topic is None:
        return False
    pass_date = datetime.now(UTC).date().isoformat()
    try:
        cands, cursors = discover(topic)
    except OSError as e:
        if _rejected_request(e):
            # a source refusing the request itself is where a cursor it no longer
            # accepts lands; keeping that cursor would stall the sweep forever
            db.set_sweep_cursors(conn, topic.id, None, None)
            _log(f"topic {topic.name}: sweep cursor rejected; restarting from the head")
        return _fail(conn, topic, e)
    try:
        process_candidates(conn, topic, cands, corpus_dir)
    except OSError as e:
        # the pages walked here are not stored yet: banking their cursor would move
        # the sweep past works nothing recorded, so leave it and re-walk the range
        return _fail(conn, topic, e)
    # this range is stored, so paging progress can be banked: expansion failing after
    # it must not cost the sweep the pages it already walked
    db.set_sweep_cursors(conn, topic.id, *cursors)
    try:
        expand(conn, topic, corpus_dir)
    except OSError as e:
        return _fail(conn, topic, e)
    if any(cursors):
        # a capped sweep is incomplete: hold the watermark so the window is not
        # recorded as covered, and let the stored cursors carry the next pass on
        _log(f"topic {topic.name}: sweep capped; resumes from stored cursor")
        db.finish_topic(conn, topic.id, None)
    else:
        # watermark = pass start date; overlap next window absorbed by dedupe
        db.finish_topic(conn, topic.id, pass_date)
    return True


def _fail(conn: sqlite3.Connection, topic: db.Topic, e: OSError) -> bool:
    """A source being down fails the topic pass, not the process.

    The old watermark stands, so the missed window is retried. An unfinished sweep is
    claimable at once, so report failure and let the caller sleep — otherwise a down
    (or 429ing) source gets hammered.
    """
    _log(f"topic {topic.name} failed ({e}); watermark unchanged")
    db.finish_topic(conn, topic.id, None)
    return False


def log_pass_summary(conn: sqlite3.Connection) -> None:
    """One grep-able line of corpus state per pass — the cheap way to read a trend."""
    counts = db.status_counts(conn)
    extra = sorted(set(counts) - set(SUMMARY_STATUSES))
    body = " ".join(f"{s}={counts.get(s, 0)}" for s in (*SUMMARY_STATUSES, *extra))
    done, total = db.topic_progress(conn)
    _log(f"pass: works {body}; topics {done}/{total} crawled")


def run_loop(
    conn: sqlite3.Connection,
    corpus_dir: Path,
    poll_seconds: int,
    recrawl_days: int,
) -> None:
    _log("crawler: draining topic queue")
    while True:
        idle = not run_once(conn, corpus_dir, recrawl_days)
        if idle:
            # idle time is when deferred works get worked off, not the re-crawl interval
            try:
                drain_deferred(conn, corpus_dir)
            except OSError as e:
                _log(f"drain failed ({e})")
        log_pass_summary(conn)
        if idle:
            time.sleep(poll_seconds)
