"""The crawl worker: drain the topic queue, one pipeline pass per topic.

Per topic: discover (OpenAlex topic filter + Europe PMC query) → mechanical gate
(retraction, open access) → model triage → PMCID join → fetch text (PMC bucket,
then fullTextXML) → store → bounded citation expansion. Every candidate becomes a
works row — kept, rejected, or missed — so re-crawls never re-judge old papers.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from . import corpus, db, europepmc, openalex, pmc_oa
from .models import Candidate
from .triage import GRADE_BY_STUDY_TYPE, TriageError, Verdict, triage

JOIN_BATCH = 20
EXPAND_WORKS_PER_PASS = 10
REFS_PER_WORK = 10
REQUEST_DELAY_S = 0.2

# Publisher-declared pub types that map cleanly onto a study type (used when no
# model triage is available; the model's judgment supersedes these).
STUDY_TYPE_BY_PUB_TYPE = {
    "meta-analysis": "meta_analysis",
    "systematic review": "systematic_review",
    "systematic-review": "systematic_review",
    "randomized controlled trial": "rct",
    "randomized-controlled-trial": "rct",
    "case reports": "case_report",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


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


def discover(topic: db.Topic) -> list[Candidate]:
    cands: list[Candidate] = []
    if topic.openalex_id:
        oa, truncated = openalex.works_by_topic(topic.openalex_id, topic.watermark)
        cands.extend(oa)
        if truncated:
            _log(f"coverage: openalex page cap hit for topic {topic.name}; rest next pass")
    ep, truncated = europepmc.search_window(topic.query, topic.watermark)
    cands.extend(ep)
    if truncated:
        _log(f"coverage: europepmc page cap hit for topic {topic.name}; rest next pass")
    return _dedupe(cands)


def _metadata_verdict(c: Candidate) -> Verdict:
    """No-triage fallback: keep, grading from publisher-declared type when clear."""
    for pt in c.pub_types:
        study_type = STUDY_TYPE_BY_PUB_TYPE.get(pt.lower())
        if study_type:
            return Verdict(relevant=True, study_type=study_type, confidence=0.0)
    return Verdict(relevant=True, study_type="other", confidence=0.0)


def _judge(
    conn: sqlite3.Connection, topic_id: int, cands: list[Candidate], use_triage: bool
) -> list[tuple[Candidate, Verdict]]:
    """Model triage via claude -p; on failure leave candidates for the next pass."""
    if not cands:
        return []
    if not use_triage:
        return [(c, _metadata_verdict(c)) for c in cands]
    try:
        verdicts = triage(cands)
    except TriageError as e:
        _log(f"triage failed ({e}); {len(cands)} candidates deferred to next pass")
        for c in cands:
            db.record_work(conn, c, topic_id, status="candidate")
        return []
    kept: list[tuple[Candidate, Verdict]] = []
    for c, v in zip(cands, verdicts, strict=True):
        if v.relevant:
            kept.append((c, v))
        else:
            db.record_work(conn, c, topic_id, status="rejected", reject_reason="triage")
    return kept


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
    use_triage: bool,
) -> int:
    """Gate → triage → join → fetch → store. Returns how many works got text."""
    fresh = [c for c in cands if not db.seen(conn, c)]
    survivors: list[Candidate] = []
    for c in fresh:
        if c.is_retracted:
            db.record_work(conn, c, topic.id, status="retracted")
        elif not c.is_oa:
            db.record_work(conn, c, topic.id, status="rejected", reject_reason="not_open_access")
        else:
            survivors.append(c)
    kept = _join_pmcids_paired(_judge(conn, topic.id, survivors, use_triage))
    fetched = 0
    for c, v in kept:
        db.record_work(
            conn,
            c,
            topic.id,
            status="kept_miss",
            study_type=v.study_type,
            evidence_grade=GRADE_BY_STUDY_TYPE.get(v.study_type),
            triage_confidence=v.confidence or None,
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


def _join_pmcids_paired(
    kept: list[tuple[Candidate, Verdict]],
) -> list[tuple[Candidate, Verdict]]:
    joined = _join_pmcids([c for c, _ in kept])
    return list(zip(joined, (v for _, v in kept), strict=True))


def expand(conn: sqlite3.Connection, topic: db.Topic, corpus_dir: Path, use_triage: bool) -> None:
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
        process_candidates(conn, topic, _dedupe(cands), corpus_dir, use_triage)


def run_once(
    conn: sqlite3.Connection,
    corpus_dir: Path,
    use_triage: bool,
    recrawl_days: int,
) -> bool:
    """Claim and crawl one topic. Returns False when the queue is empty."""
    topic = db.claim_next_topic(conn, recrawl_days)
    if topic is None:
        return False
    pass_date = datetime.now(UTC).date().isoformat()
    try:
        cands = discover(topic)
        process_candidates(conn, topic, cands, corpus_dir, use_triage)
        expand(conn, topic, corpus_dir, use_triage)
        # watermark = pass start date; overlap on the next window is absorbed by dedupe
        db.finish_topic(conn, topic.id, pass_date)
    except OSError as e:
        # a source being down fails the topic pass, not the process; the old
        # watermark stands so the missed window is retried at the next re-crawl
        _log(f"topic {topic.name} failed ({e}); watermark unchanged")
        db.finish_topic(conn, topic.id, None)
    return True


def run_loop(
    conn: sqlite3.Connection,
    corpus_dir: Path,
    use_triage: bool,
    poll_seconds: int,
    recrawl_days: int,
) -> None:
    _log("crawler: draining topic queue")
    while True:
        if not run_once(conn, corpus_dir, use_triage, recrawl_days):
            time.sleep(poll_seconds)
