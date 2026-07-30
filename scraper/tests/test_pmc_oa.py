from scraper.pmc_oa import _pick_key

LISTING = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Contents><Key>PMC7739073.1/PMC7739073.1.json</Key></Contents>
  <Contents><Key>PMC7739073.1/PMC7739073.1.pdf</Key></Contents>
  <Contents><Key>PMC7739073.1/PMC7739073.1.txt</Key></Contents>
  <Contents><Key>PMC7739073.1/PMC7739073.1.xml</Key></Contents>
</ListBucketResult>"""

NO_TXT = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Contents><Key>PMC13120320.1/PMC13120320.1.xml</Key></Contents>
</ListBucketResult>"""


def test_pick_text_key() -> None:
    assert _pick_key(LISTING, ".txt") == "PMC7739073.1/PMC7739073.1.txt"


def test_pick_text_key_absent() -> None:
    assert _pick_key(NO_TXT, ".txt") is None
