import email.message
import urllib.error
from typing import Any

import pytest

from scraper import openalex

INVERTED = {"anxiety": [1], "Mindfulness": [0], "helps": [2]}
WORK: dict[str, Any] = {
    "id": "https://openalex.org/W2109631588",
    "doi": "https://doi.org/10.1037/a0018555",
    "display_name": "The effect of mindfulness-based therapy ",
    "publication_year": 2010,
    "is_retracted": False,
    "open_access": {"is_oa": True},
    "abstract_inverted_index": INVERTED,
}


def test_rebuild_abstract_orders_by_position() -> None:
    assert openalex.rebuild_abstract(INVERTED) == "Mindfulness anxiety helps"
    assert openalex.rebuild_abstract(None) is None


def test_from_openalex_normalizes_ids() -> None:
    c = openalex.from_openalex(WORK)
    assert c.work_id == "W2109631588"
    assert c.openalex_id == "W2109631588"
    assert c.doi == "10.1037/a0018555"
    assert c.title == "The effect of mindfulness-based therapy"
    assert c.year == 2010 and c.is_oa is True and c.is_retracted is False
    assert c.abstract == "Mindfulness anxiety helps"


def test_works_by_topic_pages_until_cursor_ends() -> None:
    urls: list[str] = []

    def fetch(url: str) -> Any:
        urls.append(url)
        if len(urls) == 1:
            return {"results": [WORK], "meta": {"next_cursor": "abc"}}
        page2 = [{**WORK, "id": "https://openalex.org/W2"}]
        return {"results": page2, "meta": {"next_cursor": None}}

    cands, next_cursor = openalex.works_by_topic("T10272", "2026-07-01", fetch=fetch)
    assert [c.work_id for c in cands] == ["W2109631588", "W2"]
    assert next_cursor is None
    assert "primary_topic.id%3AT10272" in urls[0]
    assert "from_publication_date%3A2026-07-01" in urls[0]
    assert "cursor=abc" in urls[1]


def test_works_by_topic_returns_the_cap_cursor_and_resumes_from_it() -> None:
    urls: list[str] = []

    def fetch(url: str) -> Any:
        urls.append(url)
        return {"results": [WORK], "meta": {"next_cursor": f"page{len(urls) + 1}"}}

    _, next_cursor = openalex.works_by_topic("T10272", None, fetch=fetch, max_pages=2)
    assert next_cursor == "page3"
    assert "cursor=%2A" in urls[0]  # a fresh sweep starts at "*"

    urls.clear()
    openalex.works_by_topic("T10272", None, fetch=fetch, max_pages=1, cursor="page3")
    assert "cursor=page3" in urls[0]


def test_referenced_ids_shortens_urls() -> None:
    def fetch(url: str) -> Any:
        return {"referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"]}

    assert openalex.referenced_ids("W9", fetch=fetch) == ["W1", "W2"]


def test_work_by_id_treats_a_missing_work_as_absent_not_broken() -> None:
    def gone(url: str) -> Any:
        raise urllib.error.HTTPError(url, 404, "Not Found", email.message.Message(), None)

    def refused(url: str) -> Any:
        raise urllib.error.HTTPError(url, 403, "Forbidden", email.message.Message(), None)

    assert openalex.work_by_id("W1", fetch=gone) is None
    with pytest.raises(urllib.error.HTTPError):
        openalex.work_by_id("W1", fetch=refused)
