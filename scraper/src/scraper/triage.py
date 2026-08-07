"""Model triage: relevance + study type from title/abstract, batched per call.

Judgment runs through Claude Code headless (`claude -p`) — the same runtime the
backend uses — so it authenticates with the operator's existing Claude credentials
(subscription or key) instead of requiring a separate metered API key. The prompt
demands a bare JSON array; anything else is a TriageError and the caller leaves
the candidates untriaged for the next pass.

Study type maps to the evidence grade (1 strongest .. 5 weakest) that retrieval
weights — the grade is never a drop reason.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .models import Candidate

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
MODEL = "sonnet"
BATCH_SIZE = 8
ABSTRACT_CHARS = 1500
TIMEOUT_S = 300

GRADE_BY_STUDY_TYPE = {
    "meta_analysis": 1,
    "systematic_review": 1,
    "rct": 2,
    "cohort": 3,
    "case_control": 3,
    "cross_sectional": 4,
    "observational": 4,
    "case_report": 5,
    "opinion": 5,
    "qualitative": 5,
    "other": 5,
}

INSTRUCTIONS = (
    "You triage research papers for the corpus behind a mental-health and wellness "
    "companion. Relevant topics are broad: psychology, mental health, wellness, sleep, "
    "social determinants of health, and developmental/life-circumstance factors that "
    "shape wellbeing. Judge each paper from its title and abstract only. A paper is "
    "relevant if its findings could ground advice or understanding about a person's "
    "mental health or wellbeing. Classify the study type from what the text states; "
    "use 'other' when unclear. Confidence is your certainty in both judgments, 0 to 1.\n\n"
    "Reply with ONLY a JSON array, one object per paper in the given order, shaped "
    '[{"relevant": true, "study_type": "rct", "confidence": 0.9}, ...]. study_type must '
    f"be one of: {', '.join(GRADE_BY_STUDY_TYPE)}. No prose, no markdown fences."
)

Run = Callable[[str], str]


@dataclass(frozen=True)
class Verdict:
    relevant: bool
    study_type: str
    confidence: float

    @property
    def evidence_grade(self) -> int:
        return GRADE_BY_STUDY_TYPE[self.study_type]


class TriageError(Exception):
    """Runner failure or malformed verdicts; the caller leaves candidates untriaged."""


def available() -> bool:
    return shutil.which(CLAUDE_BIN) is not None


def _extract_result(stdout: str) -> str:
    """Unwrap `claude -p --output-format json`: {"is_error": ..., "result": "..."}."""
    try:
        wrapper = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise TriageError(f"claude output is not JSON: {stdout[:200]!r}") from e
    if wrapper.get("is_error") or not isinstance(wrapper.get("result"), str):
        raise TriageError(f"claude reported an error (subtype={wrapper.get('subtype')})")
    return str(wrapper["result"])


def _run_claude(prompt: str) -> str:
    cmd = [CLAUDE_BIN, "-p", prompt, "--model", MODEL, "--output-format", "json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise TriageError(f"claude -p failed to run: {e}") from e
    if proc.returncode != 0:
        raise TriageError(f"claude -p exited {proc.returncode}: {proc.stderr.strip()[:200]}")
    return _extract_result(proc.stdout)


def _paper_block(i: int, c: Candidate) -> str:
    abstract = (c.abstract or "").strip()[:ABSTRACT_CHARS] or "(no abstract available)"
    hint = f"\nPublisher-declared types: {', '.join(c.pub_types)}" if c.pub_types else ""
    return f"Paper {i + 1}\nTitle: {c.title}\nAbstract: {abstract}{hint}"


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse_verdicts(reply: str, expected: int) -> list[Verdict]:
    try:
        raw: Any = json.loads(_strip_fences(reply))
    except json.JSONDecodeError as e:
        raise TriageError(f"reply is not a JSON array: {reply[:200]!r}") from e
    if not isinstance(raw, list) or len(raw) != expected:
        got = len(raw) if isinstance(raw, list) else type(raw).__name__
        raise TriageError(f"expected {expected} verdicts, got {got}")
    verdicts = []
    for v in raw:
        study_type = str(v["study_type"])
        if study_type not in GRADE_BY_STUDY_TYPE:
            raise TriageError(f"unknown study_type {study_type!r}")
        verdicts.append(Verdict(bool(v["relevant"]), study_type, float(v["confidence"])))
    return verdicts


def triage_batch(candidates: Sequence[Candidate], run: Run = _run_claude) -> list[Verdict]:
    """One `claude -p` call judging up to BATCH_SIZE candidates."""
    prompt = (
        INSTRUCTIONS
        + "\n\nTriage these papers:\n\n"
        + "\n\n".join(_paper_block(i, c) for i, c in enumerate(candidates))
    )
    return _parse_verdicts(run(prompt), len(candidates))


def triage(candidates: Sequence[Candidate], run: Run = _run_claude) -> list[Verdict]:
    """Judge all candidates in batches; verdicts align with the input order."""
    out: list[Verdict] = []
    for start in range(0, len(candidates), BATCH_SIZE):
        out.extend(triage_batch(candidates[start : start + BATCH_SIZE], run))
    return out
