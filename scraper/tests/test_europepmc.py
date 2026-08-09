from typing import Any

from scraper import europepmc

RECORD = {
    "pmcid": "PMC9975722",
    "doi": "10.3389/fpsyg.2023.994205",
    "title": "Mindfulness and anxiety: a meta-analysis. ",
    "authorString": "Xu F, Zhu W.",
    "pubYear": "2023",
    "license": "cc by",
    "isOpenAccess": "Y",
    "abstractText": "Background: ...",
    "pubTypeList": {"pubType": ["Systematic Review", "Journal Article"]},
}

JATS = b"""<?xml version="1.0" encoding="UTF-8"?>
<article><front><article-meta><title-group>
<article-title>A title</article-title></title-group></article-meta></front>
<body><sec><title>Intro</title><p>First <italic>finding</italic>.</p></sec></body></article>"""


def test_from_epmc_maps_core_record() -> None:
    c = europepmc.from_epmc(RECORD)
    assert c.work_id == "doi:10.3389/fpsyg.2023.994205"
    assert c.pmcid == "PMC9975722" and c.doi == "10.3389/fpsyg.2023.994205"
    assert c.year == 2023 and c.is_oa is True
    assert c.pub_types == ("Systematic Review", "Journal Article")
    assert c.title == "Mindfulness and anxiety: a meta-analysis."


def test_search_window_adds_idate_clause_and_pages() -> None:
    urls: list[str] = []

    def fetch(url: str) -> Any:
        urls.append(url)
        if len(urls) == 1:
            return {"resultList": {"result": [RECORD]}, "nextCursorMark": "c2"}
        return {"resultList": {"result": []}, "nextCursorMark": "c2"}

    cands, next_cursor = europepmc.search_window("mindfulness", "2026-07-01", fetch=fetch)
    assert len(cands) == 1 and next_cursor is None
    assert "FIRST_IDATE%3A%5B2026-07-01+TO+%2A%5D" in urls[0]
    assert "cursorMark=c2" in urls[1]


def test_search_window_returns_the_cap_cursor_and_resumes_from_it() -> None:
    urls: list[str] = []

    def fetch(url: str) -> Any:
        urls.append(url)
        return {"resultList": {"result": [RECORD]}, "nextCursorMark": f"c{len(urls) + 1}"}

    _, next_cursor = europepmc.search_window("mindfulness", None, fetch=fetch, max_pages=2)
    assert next_cursor == "c3"
    assert "cursorMark=%2A" in urls[0]

    urls.clear()
    europepmc.search_window("mindfulness", None, fetch=fetch, max_pages=1, cursor="c3")
    assert "cursorMark=c3" in urls[0]


def test_pmcids_for_dois_batches_and_lowercases() -> None:
    def fetch(url: str) -> Any:
        assert "DOI%3A%2210.1%2Fabc%22+OR+DOI%3A%2210.2%2Fdef%22" in url
        hits = [{"doi": "10.1/ABC", "pmcid": "PMC1"}, {"doi": "10.2/def"}]
        return {"resultList": {"result": hits}}

    assert europepmc.pmcids_for_dois(["10.1/abc", "10.2/def"], fetch=fetch) == {"10.1/abc": "PMC1"}
    assert europepmc.pmcids_for_dois([], fetch=fetch) == {}


def test_full_text_strips_jats_body() -> None:
    assert europepmc.full_text("PMC1", fetch_bytes=lambda url: JATS) == "Intro First finding ."


def test_full_text_none_on_missing_or_bad_xml() -> None:
    def raise_404(url: str) -> bytes:
        raise OSError("404")

    assert europepmc.full_text("PMC1", fetch_bytes=raise_404) is None
    assert europepmc.full_text("PMC1", fetch_bytes=lambda url: b"not xml") is None
