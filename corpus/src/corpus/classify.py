"""Relevance + label judging through Claude Code headless (`claude -p`).

One call judges a whole batch: the ~36k-token harness overhead per `claude -p` process
dominates the abstracts themselves, so batches are large (BATCH_SIZE) rather than the
8 the crawler's triage used. Output is a bare JSON array carrying the paper index, so a
model that drops or reorders an entry is caught instead of silently misaligning scores.

Never a metered API key: `claude -p` authenticates with the operator's subscription.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from .models import DOMAINS, STUDY_TYPES, Paper, Verdict

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
BATCH_SIZE = 60
ABSTRACT_CHARS = 1500
TIMEOUT_S = 900

INSTRUCTIONS = f"""You score research papers for the retrieval corpus behind Sana, a \
wellness companion: a chat app where ordinary people ask about their mental health, \
sleep, stress, pain, and everyday lifestyle health. Judge each paper from its title and \
abstract only.

For each paper give:
- relevance: 0-10, how useful this paper is for grounding an answer to such a user.
  10 = directly answers a question a user would plausibly ask (e.g. an RCT of an
  insomnia treatment, a meta-analysis of exercise for depression).
  7-9 = solidly on-topic, would support or qualify an answer.
  4-6 = related background: mechanism, epidemiology, or a special population a user
  probably is not asking about.
  1-3 = human health but not this app's scope (e.g. oncology surgery technique).
  0 = not about human health at all, or unusable (retracted notice, erratum, protocol
  with no findings, non-English body).
  Score the paper's usefulness, not its quality — a weak study on-topic still scores high.
- domain: one of {", ".join(DOMAINS)}. Use off-topic only with relevance <= 2.
- study_type: one of {", ".join(STUDY_TYPES)}. Use other when the abstract does not say.
- confidence: 0.0-1.0, your certainty in relevance and study_type together.

Reply with ONLY a JSON array, one object per paper, in the given order:
[{{"paper": 1, "relevance": 8, "domain": "sleep", "study_type": "rct", "confidence": 0.9}}]
No prose, no markdown fences."""

Run = Callable[[str, str], str]


class ClassifyError(Exception):
    """Runner failure or malformed verdicts; the caller leaves the batch unjudged."""


def _extract_result(stdout: str) -> str:
    """Unwrap `claude -p --output-format json`: {"is_error": ..., "result": "..."}."""
    try:
        wrapper = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ClassifyError(f"claude output is not JSON: {stdout[:200]!r}") from e
    if wrapper.get("is_error") or not isinstance(wrapper.get("result"), str):
        raise ClassifyError(f"claude reported an error (subtype={wrapper.get('subtype')})")
    return str(wrapper["result"])


def run_claude(prompt: str, model: str) -> str:
    cmd = [CLAUDE_BIN, "-p", prompt, "--model", model, "--output-format", "json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ClassifyError(f"claude -p failed to run: {e}") from e
    if proc.returncode != 0:
        raise ClassifyError(f"claude -p exited {proc.returncode}: {proc.stderr.strip()[:200]}")
    return _extract_result(proc.stdout)


def _paper_block(i: int, p: Paper) -> str:
    abstract = (p.abstract or "").strip()[:ABSTRACT_CHARS] or "(no abstract available)"
    year = p.year if p.year is not None else "unknown"
    return f"Paper {i + 1}\nYear: {year}\nTitle: {p.title}\nAbstract: {abstract}"


def build_prompt(papers: Sequence[Paper]) -> str:
    blocks = "\n\n".join(_paper_block(i, p) for i, p in enumerate(papers))
    return f"{INSTRUCTIONS}\n\nScore these {len(papers)} papers:\n\n{blocks}"


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def parse_verdicts(reply: str, papers: Sequence[Paper]) -> list[Verdict]:
    try:
        raw: Any = json.loads(_strip_fences(reply))
    except json.JSONDecodeError as e:
        raise ClassifyError(f"reply is not a JSON array: {reply[:200]!r}") from e
    if not isinstance(raw, list) or len(raw) != len(papers):
        got = len(raw) if isinstance(raw, list) else type(raw).__name__
        raise ClassifyError(f"expected {len(papers)} verdicts, got {got}")
    out: list[Verdict] = []
    for i, (v, paper) in enumerate(zip(raw, papers, strict=True)):
        try:
            if int(v["paper"]) != i + 1:
                raise ClassifyError(f"verdict {i + 1} claims to be paper {v['paper']}")
            relevance, domain = int(v["relevance"]), str(v["domain"])
            study_type, confidence = str(v["study_type"]), float(v["confidence"])
        except (KeyError, TypeError, ValueError) as e:
            raise ClassifyError(f"verdict {i + 1} malformed: {v!r}") from e
        if not 0 <= relevance <= 10:
            raise ClassifyError(f"relevance out of range: {relevance}")
        if domain not in DOMAINS:
            raise ClassifyError(f"unknown domain {domain!r}")
        if study_type not in STUDY_TYPES:
            raise ClassifyError(f"unknown study_type {study_type!r}")
        out.append(Verdict(paper.work_id, relevance, domain, study_type, confidence))
    return out


def classify_batch(papers: Sequence[Paper], model: str, run: Run = run_claude) -> list[Verdict]:
    """One `claude -p` call scoring up to BATCH_SIZE papers."""
    return parse_verdicts(run(build_prompt(papers), model), papers)
