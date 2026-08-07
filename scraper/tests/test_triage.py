import json

import pytest

from scraper import triage
from scraper.models import Candidate

CANDS = [
    Candidate(work_id="W1", title="Paper one", discovered_via="openalex", abstract="Abs one"),
    Candidate(
        work_id="W2",
        title="Paper two",
        discovered_via="europepmc",
        pub_types=("Systematic Review",),
    ),
]

TWO_VERDICTS = json.dumps(
    [
        {"relevant": True, "study_type": "meta_analysis", "confidence": 0.9},
        {"relevant": False, "study_type": "other", "confidence": 0.6},
    ]
)


def test_triage_batch_parses_verdicts_in_order() -> None:
    prompts: list[str] = []

    def run(prompt: str) -> str:
        prompts.append(prompt)
        return TWO_VERDICTS

    verdicts = triage.triage_batch(CANDS, run=run)
    assert [v.relevant for v in verdicts] == [True, False]
    assert verdicts[0].evidence_grade == 1
    assert "Paper one" in prompts[0] and "Systematic Review" in prompts[0]
    assert "ONLY a JSON array" in prompts[0]


def test_triage_batch_strips_markdown_fences() -> None:
    fenced = f"```json\n{TWO_VERDICTS}\n```"
    verdicts = triage.triage_batch(CANDS, run=lambda p: fenced)
    assert len(verdicts) == 2


def test_triage_batch_rejects_wrong_count() -> None:
    one = json.dumps([{"relevant": True, "study_type": "other", "confidence": 0.5}])
    with pytest.raises(triage.TriageError, match="expected 2"):
        triage.triage_batch(CANDS, run=lambda p: one)


def test_triage_batch_rejects_unknown_study_type() -> None:
    bad = json.dumps([{"relevant": True, "study_type": "vibes", "confidence": 0.5}] * 2)
    with pytest.raises(triage.TriageError, match="unknown study_type"):
        triage.triage_batch(CANDS, run=lambda p: bad)


def test_triage_batch_rejects_prose_reply() -> None:
    with pytest.raises(triage.TriageError, match="not a JSON array"):
        triage.triage_batch(CANDS, run=lambda p: "Sure! Here are my verdicts...")


def test_extract_result_unwraps_claude_output() -> None:
    ok = json.dumps({"is_error": False, "subtype": "success", "result": "[1]"})
    assert triage._extract_result(ok) == "[1]"

    err = json.dumps({"is_error": True, "subtype": "error_during_execution", "result": ""})
    with pytest.raises(triage.TriageError, match="claude reported an error"):
        triage._extract_result(err)

    with pytest.raises(triage.TriageError, match="not JSON"):
        triage._extract_result("claude: command crashed")


def test_triage_splits_batches() -> None:
    calls: list[int] = []

    def run(prompt: str) -> str:
        n = prompt.count("Title:")
        calls.append(n)
        return json.dumps([{"relevant": True, "study_type": "other", "confidence": 0.5}] * n)

    many = [Candidate(work_id=f"W{i}", title=f"P{i}", discovered_via="openalex") for i in range(11)]
    verdicts = triage.triage(many, run=run)
    assert len(verdicts) == 11
    assert calls == [8, 3]
