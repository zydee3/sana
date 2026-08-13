from __future__ import annotations

from corpus.clean import Block, canonical_section, clean, is_boilerplate, is_heading

RULE = "\x9f==============================\x9f"


def _doc(*blocks: str) -> str:
    front = "JOURNAL INFORMATION\n==============================\nISSN: 1234-5678\n"
    return f"\n{front}\n{RULE}\n" + "\n\n".join(blocks) + "\n"


def _texts(blocks: list[Block]) -> list[str]:
    return [b.text for b in blocks]


def test_front_matter_is_dropped() -> None:
    blocks, _ = clean(_doc("Abstract", "We studied " + "sleep " * 30))
    assert not any("ISSN" in t for t in _texts(blocks))


def test_no_rule_keeps_whole_file() -> None:
    blocks, _ = clean("Introduction\n\n" + "Nothing was delimited. " * 20)
    assert blocks and blocks[0].text.startswith("Nothing was delimited")


def test_structured_abstract_then_body() -> None:
    blocks, _ = clean(
        _doc(
            "Background",
            "Abstract background sentence. " * 5,
            "Methods",
            "Abstract methods sentence. " * 5,
            "INTRODUCTION",
            "Body introduction sentence. " * 20,
            "METHODS",
            "Body methods sentence. " * 20,
        )
    )
    sections = [b.section for b in blocks]
    # The repeat of a section tag ends the abstract region.
    assert sections == ["abstract", "abstract", "introduction", "methods"]


def test_subsection_inherits_parent_section() -> None:
    # An unstructured abstract only ends on size: nothing else distinguishes it from a
    # first body section, so the body paragraph here is longer than ABSTRACT_CHARS.
    blocks, _ = clean(
        _doc(
            "Abstract",
            "An abstract. " * 5,
            "Methods",
            "Body methods sentence. " * 200,
            "Survey Data",
            "How the survey ran. " * 20,
        )
    )
    assert blocks[0].section == "abstract"
    assert blocks[-1].section == "methods"
    assert blocks[-1].heading == "Survey Data"


def test_numbered_subsection_does_not_retag_its_parent() -> None:
    blocks, _ = clean(
        _doc(
            "Abstract",
            "An abstract. " * 5,
            "2. Materials and Methods",
            "Body methods sentence. " * 200,
            "2.1.3. Outcomes",
            "What we measured. " * 20,
            "2.2. Search Strategy",
            "How we searched. " * 20,
        )
    )
    assert [b.section for b in blocks[-2:]] == ["methods", "methods"]
    assert blocks[-1].heading == "2.2. Search Strategy"


def test_boilerplate_sections_and_their_bodies_go() -> None:
    blocks, dropped = clean(
        _doc(
            "Conclusion",
            "We conclude something useful. " * 20,
            "Ethics approval and consent to participate",
            "Approved by the committee. " * 10,
            "References",
            "1. Smith J. A study of things. Journal 2019 12 34 45\n"
            "2. Jones K. Another study. Journal 2020 1 2 3",
        )
    )
    assert _texts(blocks) == [("We conclude something useful. " * 20).strip()]
    assert dropped["boilerplate_section"] == 2
    assert dropped["boilerplate_block"] == 2


def test_headingless_reference_list_is_dropped() -> None:
    tail = (
        "31 Zion A. S. De Meersman R. A home-based resistance training program for elderly"
        " patients Clinical Autonomic Research 2003 13 4 286 292\n"
        "32 Tangtrongkanti T. Effects of a knowledge enhancement program on fitness"
        " Journal of Physical Education 2020 23 1 51 61"
    )
    blocks, dropped = clean(_doc("Conclusion", "Real prose here. " * 20, tail))
    assert dropped["reference_block"] == 1
    assert len(blocks) == 1


def test_table_rows_are_dropped_but_captions_survive() -> None:
    table = "Variable\tAge\tGender\tTotal\n" + "\n".join(f"row{i}\t1\t2\t3" for i in range(4))
    blocks, dropped = clean(_doc("Results", "Findings prose. " * 20, table))
    assert dropped["table_block"] == 1
    assert dropped["table_line"] == 5
    assert len(blocks) == 1


def test_long_abstract_region_is_capped() -> None:
    # Worst case: no heading ever repeats, so only the char cap ends the region — and an
    # "Abstract" heading must not become the inherited parent of what follows.
    blocks, _ = clean(_doc("Abstract", "word " * 3000, "Some Subsection", "more prose " * 30))
    assert blocks[0].section == "abstract"
    assert blocks[1].section == "other"


def test_peer_review_appendix_truncates_the_article() -> None:
    blocks, dropped = clean(
        _doc(
            "Conclusion",
            "The article's real conclusion. " * 20,
            "DOI: 10.5256/f1000research.139716.r161398",
            "Reviewer response for version 1",
            "Copyright: 2023 Referee. " * 20,
            "Discussion",
            "Text that only exists inside the referee report. " * 20,
        )
    )
    assert len(blocks) == 1
    assert dropped["appendix_truncated"] == 1


def test_cambridge_review_doi_truncates() -> None:
    blocks, _ = clean(
        _doc(
            "Conclusion", "Real conclusion prose. " * 20, "DOI: 10.1017/gmh.2024.117.pr3", "x " * 40
        )
    )
    assert len(blocks) == 1


def test_plos_star_separator_truncates() -> None:
    blocks, _ = clean(
        _doc("Results", "Real findings. " * 20, "**********", "Reviewer #1: Yes, in every respect.")
    )
    assert _texts(blocks) == [("Real findings. " * 20).strip()]


def test_heading_and_canonical_recognition() -> None:
    assert is_heading("3.5. Ethical Considerations")
    assert not is_heading("We measured sleep quality with the PSQI.")
    assert not is_heading("Variable\tAge\tTotal")
    assert canonical_section("4.1 Materials and Methods") == "methods"
    assert canonical_section("Statistical Analyses") == "methods"
    assert canonical_section("Survey Data") is None
    assert is_boilerplate("Funding for the study")
    assert is_boilerplate("References and Recommended Reading")
    assert not is_boilerplate("Supplementary analyses")
