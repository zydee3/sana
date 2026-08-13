"""Deterministic cleaning of a stored full text into labelled prose blocks.

The PMC-to-text converter writes a fixed shape: a front-matter header (journal +
article metadata), a `\x9f=…=\x9f` rule, then the article as blank-line-separated
blocks where a heading is a short standalone line. All 200 sampled pilot texts carry
the rule, so the rule is the front-matter boundary; a file without one is kept whole.

Three kinds of noise are removed, in descending order of volume: administrative
sections (references, acknowledgements, funding, ethics, …) identified by their
heading, tab-delimited table rows, and stray numbered reference lines in files whose
reference list has no heading. Median byte reduction on the pilot set is 39%.

What survives those structural rules is text that reads as a paragraph but is not
prose — a quoted PubMed search strategy, a questionnaire answer scale, a space-aligned
table — and it was taking golden-query top-10 slots. A block is kept only if it looks
like English prose (function-word share); a LaTeX equation blob is cut out in place,
since the prose around it is real.

Sections are canonicalised to abstract/introduction/methods/results/discussion/
conclusion so a chunk can be filtered or weighted by where in the paper it came from.
A non-canonical heading inherits its parent section ("Survey Data" under "Methods"),
which is what drops the unclassified share from 66% to 2%.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

MARK = re.compile("\x9f=+\x9f")
_WS = re.compile(r"\s+")
_PARA_SPLIT = re.compile(r"\n\s*\n")

# A heading is a short standalone line: no internal newline, no tabs, no closing
# punctuation. Long enough limits that numbered subsection titles still qualify.
HEADING_CHARS = 90
HEADING_WORDS = 10
_NUM_PREFIX = re.compile(r"^\s*(\d+(\.\d+)*\.?|[IVXLC]+\.|[A-Z]\.)\s+")

# Abstract detection: the region after the rule is the abstract until a section tag
# repeats (structured abstracts reuse Methods/Results as body headings do) or the
# region outgrows any real abstract.
ABSTRACT_CHARS = 2_500

# Administrative sections. Prefix match — the tail varies too much to enumerate
# ("Ethics approval and consent to participate", "Funding for the study").
_BOILER_PREFIX = re.compile(
    r"^(references?|bibliography|literature cited"
    r"|acknowledge?ments?|funding|financial support|grant support"
    r"|conflicts? of interest|competing interests?|declarations?|disclosures?"
    r"|data availability|availability of data|code availability"
    r"|authors?.? (contributions?|information|notes?)|contributor information"
    r"|ethics?|ethical|consent|informed consent|human and animal rights"
    r"|orcid|publisher.?s note|provenance and peer review|copyright|permissions"
    r"|footnotes?|abbreviations|trial registration|clinical trial number"
    r"|registration|peer review)\b",
    re.I,
)
# Matched whole, because a prefix rule would also eat "Supplementary analyses".
_BOILER_EXACT = re.compile(
    r"^(supplement(ary|al)( material| information| data| files?)?|supporting information"
    r"|online only material|figures?|tables?|appendix|glossary)$",
    re.I,
)

# PLOS and F1000 append the whole peer-review correspondence after the article: a
# revision DOI, then decision letters, referee reports and author responses, each
# carrying its own copyright block. It is strictly post-article, so the first marker
# truncates the file. This was 2% of pilot works and every residual metadata leak.
_APPENDIX = re.compile(
    r"^(\*{3,}$|DOI: 10\.\S+\.p?r\d+"  # PLOS/F1000 .rNNN, Cambridge .prN
    r"|decision letter|author response|acceptance letter|referee report"
    r"|reviewer response|peer review history|reviewer #\s*\d|submission version)",
    re.I,
)

_CANON_PATTERNS = (
    ("abstract", r"abstract|summary"),
    ("introduction", r"introduction|background|objectives?|aims?|purpose|rationale"),
    (
        "methods",
        r"(materials? and )?methods?|methodology|study design|design|participants"
        r"|subjects|measures|instruments|procedures?|statistical analys[ei]s"
        r"|data (collection|analysis)|sample",
    ),
    ("results", r"results?|findings|outcomes"),
    ("discussion", r"discussion|interpretation|limitations?|strengths?"),
    ("conclusion", r"conclusions?|implications?|recommendations?"),
)
CANONICAL = tuple(tag for tag, _ in _CANON_PATTERNS)
_CANON = tuple((tag, re.compile(rf"^({pat})$", re.I)) for tag, pat in _CANON_PATTERNS)

# Reference-list fallback for files with no References heading: a numbered line long
# enough to be a citation and carrying a year.
_REF_NUM = re.compile(r"^\s*\d{1,3}[.)]?\s+")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
REF_LINE_CHARS = 60
REF_BLOCK_SHARE = 0.6
TABLE_TABS = 2

# Non-prose that survives the structural rules and still reaches the embedder, in the
# order it was found ranking in golden-query top-10s: a quoted PubMed search strategy,
# a bare questionnaire answer scale, and table rows whose file used spaces, not tabs.
# The LaTeX blob is different — it sits inside real prose, so it is cut out, not
# grounds for dropping the block.
_LATEX_BLOB = re.compile(r"\\documentclass.*?\\end\{document\}", re.S)
_FIELD_TAG = re.compile(r"\[\s*(title/abstract|mesh terms?|tiab|all fields)\s*\]", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
QUERY_TAGS = 2  # per sentence; one is a passing mention, a run of them is the strategy
_CHECKBOX = re.compile(r"[(\[]\s*[)\]]")
CHECKBOXES = 4

# English function words: prose runs ~30% of them, a table row or a scale near zero.
# Short blocks are exempt — the share is noise below a paragraph's worth of words.
STOPWORDS = frozenset(
    "the of and in to a was were is are for with that this we our by on as at from be"
    " been between than not it its these those their has have had".split()
)
MIN_STOPWORD_SHARE = 0.08
PROSE_MIN_WORDS = 25


@dataclass(frozen=True)
class Block:
    """One paragraph of article prose, with where it sits in the paper."""

    section: str
    heading: str | None
    text: str


def strip_numbering(heading: str) -> str:
    return _NUM_PREFIX.sub("", heading.strip()).rstrip(":").strip()


def is_subsection(heading: str) -> bool:
    """True for a numbered heading two levels deep or more ("2.1.3. Outcomes").

    Such a heading must not retag its section: "Outcomes" nested under "2. Materials and
    Methods" names a methods subsection, and taking it for a results heading dragged the
    rest of the paper's methods along with it.
    """
    m = re.match(r"^\s*(\d+(?:\.\d+)+)\.?\s+\S", heading)
    return bool(m and m.group(1).count(".") >= 1)


def canonical_section(heading: str) -> str | None:
    """The canonical section a heading names, or None if it is a subsection title."""
    plain = strip_numbering(heading)
    for tag, pattern in _CANON:
        if pattern.match(plain):
            return tag
    return None


def is_heading(block: str) -> bool:
    text = block.strip()
    if not text or "\n" in text or "\t" in block:
        return False
    return (
        len(text) <= HEADING_CHARS
        and len(text.split()) <= HEADING_WORDS
        and not text.endswith((".", ";", ","))
    )


def is_boilerplate(heading: str) -> bool:
    plain = strip_numbering(heading)
    return bool(_BOILER_PREFIX.match(plain) or _BOILER_EXACT.match(plain))


def _is_reference_line(line: str) -> bool:
    return len(line) > REF_LINE_CHARS and bool(_REF_NUM.match(line)) and bool(_YEAR.search(line))


def _is_reference_block(lines: list[str]) -> bool:
    hits = sum(_is_reference_line(line) for line in lines)
    return hits >= max(1, len(lines) * REF_BLOCK_SHARE)


def strip_latex(text: str) -> str:
    """Cut the converter's LaTeX preamble+equation blob out of the prose around it."""
    return _WS.sub(" ", _LATEX_BLOB.sub(" ", text)).strip()


def strip_search_strategy(text: str) -> str:
    """Drop the sentences that are a quoted PubMed query, keep the prose around them.

    Systematic reviews print their search strategy inside Methods, so the block is
    usually half prose; only the sentences carrying a run of field tags go.
    """
    if len(_FIELD_TAG.findall(text)) < QUERY_TAGS:
        return text
    kept = [s for s in _SENTENCE.split(text) if len(_FIELD_TAG.findall(s)) < QUERY_TAGS]
    return " ".join(kept).strip()


def stopword_share(text: str) -> float:
    words = [w.strip(".,;:()[]\"'“”").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0
    return sum(w in STOPWORDS for w in words) / len(words)


def is_prose(text: str) -> bool:
    """False for answer scales and for table rows that reached here without tabs."""
    if len(_CHECKBOX.findall(text)) >= CHECKBOXES:
        return False
    if len(text.split()) < PROSE_MIN_WORDS:
        return True
    return stopword_share(text) >= MIN_STOPWORD_SHARE


def article_body(raw: str) -> str:
    """Everything after the converter's front-matter rule (whole file if absent)."""
    mark = MARK.search(raw)
    return raw[mark.end() :] if mark else raw


def clean(raw: str) -> tuple[list[Block], Counter[str]]:
    """Front matter, administrative sections, tables and reference lists removed."""
    blocks: list[Block] = []
    dropped: Counter[str] = Counter()
    heading: str | None = None
    # Body section is tracked apart from the abstract region so an "Abstract" heading
    # never becomes the inherited parent of body prose.
    section = "other"
    seen: set[str] = set()
    abstract_chars = 0
    in_abstract = True
    skipping = False

    for raw_block in _PARA_SPLIT.split(article_body(raw)):
        if not raw_block.strip():
            continue
        if is_heading(raw_block):
            title = raw_block.strip()
            if _APPENDIX.match(strip_numbering(title)):
                dropped["appendix_truncated"] += 1
                break
            if is_boilerplate(title):
                skipping = True
                dropped["boilerplate_section"] += 1
                continue
            skipping = False
            if is_subsection(title):
                heading = title
                continue
            tag = canonical_section(title)
            if in_abstract and (tag in seen or abstract_chars > ABSTRACT_CHARS):
                in_abstract = False
            if tag:
                seen.add(tag)
            heading = title
            if tag and tag != "abstract":
                section = tag
            continue
        if skipping:
            dropped["boilerplate_block"] += 1
            continue
        lines = [line for line in raw_block.splitlines() if line.strip()]
        if _is_reference_block(lines):
            dropped["reference_block"] += 1
            continue
        prose = [line for line in lines if line.count("\t") < TABLE_TABS]
        dropped["table_line"] += len(lines) - len(prose)
        if not prose:
            dropped["table_block"] += 1
            continue
        text = strip_search_strategy(strip_latex(_WS.sub(" ", " ".join(prose)).strip()))
        if not text:
            dropped["search_strategy"] += 1
            continue
        if not is_prose(text):
            dropped["non_prose"] += 1
            continue
        blocks.append(Block("abstract" if in_abstract else section, heading, text))
        if in_abstract:
            abstract_chars += len(text)
            if abstract_chars > ABSTRACT_CHARS:
                in_abstract = False
    return blocks, dropped
