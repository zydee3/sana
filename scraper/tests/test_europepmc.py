from scraper.europepmc import _parse_search

SAMPLE = {
    "resultList": {
        "result": [
            {
                "pmcid": "PMC7739073",
                "title": "  A study of mindfulness  ",
                "doi": "10.1177/2164956120977827",
                "authorString": "Smith J, Doe A.",
                "firstPublicationDate": "2020-12-13",
                "license": "cc by-nc",
            },
            # No pmcid -> not retrievable from the PMC bucket, so it is dropped.
            {"title": "Abstract-only record", "id": "12345"},
        ]
    }
}


def test_parse_keeps_only_records_with_pmcid() -> None:
    papers = _parse_search(SAMPLE)
    assert len(papers) == 1
    p = papers[0]
    assert p.pmcid == "PMC7739073"
    assert p.title == "A study of mindfulness"  # stripped
    assert p.year == "2020"  # derived from firstPublicationDate
    assert p.license == "cc by-nc"


def test_parse_empty() -> None:
    assert _parse_search({}) == []
