from scraper.pmc_oa import _pick_pdf_key

LISTING = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Contents><Key>PMC7739073.1/PMC7739073.1.json</Key></Contents>
  <Contents><Key>PMC7739073.1/PMC7739073.1.pdf</Key></Contents>
  <Contents><Key>PMC7739073.1/PMC7739073.1.xml</Key></Contents>
</ListBucketResult>"""

NO_PDF = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Contents><Key>PMC13120320.1/PMC13120320.1.xml</Key></Contents>
</ListBucketResult>"""


def test_pick_pdf_key() -> None:
    assert _pick_pdf_key(LISTING) == "PMC7739073.1/PMC7739073.1.pdf"


def test_pick_pdf_key_absent() -> None:
    assert _pick_pdf_key(NO_PDF) is None
