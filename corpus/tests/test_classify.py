from __future__ import annotations

import json

import pytest

from corpus import compare, epmc
from corpus.classify import ClassifyError, build_prompt, classify_batch, parse_verdicts
from corpus.models import Paper, Verdict


def paper(i: int, abstract: str | None = "an abstract") -> Paper:
    return Paper(
        work_id=f"W{i}",
        title=f"title {i}",
        year=2021,
        doi=f"10.1/{i}",
        pmcid=f"PMC{i}",
        discovered_via="europepmc",
        status="kept_text",
        study_type=None,
        stratum="europepmc/kept_text/2020s",
        abstract=abstract,
    )


def verdict_json(n: int, relevance: int = 8) -> str:
    return json.dumps(
        [
            {
                "paper": i + 1,
                "relevance": relevance,
                "domain": "sleep",
                "study_type": "rct",
                "confidence": 0.9,
            }
            for i in range(n)
        ]
    )


def test_prompt_includes_every_paper_and_truncates() -> None:
    papers = [paper(1), paper(2, "x" * 5000)]
    prompt = build_prompt(papers)
    assert "Paper 1" in prompt and "Paper 2" in prompt
    assert "x" * 1500 in prompt and "x" * 1501 not in prompt


def test_missing_abstract_is_marked_not_dropped() -> None:
    assert "(no abstract available)" in build_prompt([paper(1, None)])


def test_parse_accepts_fenced_json() -> None:
    reply = f"```json\n{verdict_json(2)}\n```"
    assert len(parse_verdicts(reply, [paper(1), paper(2)])) == 2


def test_parse_rejects_wrong_count_and_misalignment() -> None:
    with pytest.raises(ClassifyError):
        parse_verdicts(verdict_json(1), [paper(1), paper(2)])
    shifted = json.dumps(
        [{"paper": 2, "relevance": 5, "domain": "sleep", "study_type": "rct", "confidence": 0.5}]
    )
    with pytest.raises(ClassifyError):
        parse_verdicts(shifted, [paper(1)])


def test_parse_rejects_out_of_vocabulary_labels() -> None:
    bad_domain = json.dumps(
        [
            {
                "paper": 1,
                "relevance": 5,
                "domain": "nutrition",
                "study_type": "rct",
                "confidence": 0.5,
            }
        ]
    )
    with pytest.raises(ClassifyError):
        parse_verdicts(bad_domain, [paper(1)])
    bad_range = json.dumps(
        [{"paper": 1, "relevance": 11, "domain": "sleep", "study_type": "rct", "confidence": 0.5}]
    )
    with pytest.raises(ClassifyError):
        parse_verdicts(bad_range, [paper(1)])


def test_classify_batch_keeps_work_ids() -> None:
    papers = [paper(1), paper(2)]
    verdicts = classify_batch(papers, "haiku", run=lambda prompt, model: verdict_json(2))
    assert [v.work_id for v in verdicts] == ["W1", "W2"]


def test_agreement_and_disagreements() -> None:
    a = [Verdict("W1", 8, "sleep", "rct", 0.9), Verdict("W2", 3, "pain", "cohort", 0.5)]
    b = [Verdict("W1", 7, "sleep", "rct", 0.8), Verdict("W2", 8, "sleep", "cohort", 0.6)]
    ag = compare.agreement(a, b)
    assert ag.n == 2
    assert ag.mean_abs_diff == pytest.approx(3.0)
    assert ag.within_1 == pytest.approx(0.5)
    assert ag.keep_agree == pytest.approx(0.5)
    assert ag.study_type_agree == pytest.approx(1.0)
    assert compare.disagreements(a, b) == ["W2"]


def test_epmc_batch_maps_by_pmcid_and_doi() -> None:
    papers = [paper(1), Paper(**{**paper(2).__dict__, "pmcid": None})]
    response = {
        "resultList": {
            "result": [
                {"pmcid": "PMC1", "doi": "10.1/1", "abstractText": "<h4>Background</h4>one two"},
                {"pmcid": None, "doi": "10.1/2", "abstractText": "two"},
                {"pmcid": "PMC9", "doi": "10.1/9", "abstractText": "unrelated"},
            ]
        }
    }
    got = epmc.fetch_batch(papers, fetch=lambda url: response)
    assert got == {"W1": "Background: one two", "W2": "two"}


def test_epmc_skips_papers_without_ids() -> None:
    no_ids = Paper(**{**paper(3).__dict__, "pmcid": None, "doi": None})
    assert epmc.fetch_batch([no_ids], fetch=lambda url: {}) == {}
