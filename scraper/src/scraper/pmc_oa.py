"""Fetch: download a paper's plain text from the PMC open-access dataset on AWS.

The `pmc-oa-opendata` S3 bucket is keyed per paper at `PMC<id>.<version>/`, holding a
ready-made plain-text `.txt` alongside the PDF/XML/JSON. We list the paper's prefix to
discover the versioned `.txt` key, then download it — no PDF or XML parsing needed. This
avoids Europe PMC's `?pdf=render` endpoint (currently returning 404s) and NCBI FTP
(blocked in many environments).
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET

from .http import get_bytes

BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"


def _pick_key(listing_xml: bytes, suffix: str) -> str | None:
    """Return the first object key ending in `suffix` in an S3 ListBucketResult, or None."""
    root = ET.fromstring(listing_xml)
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == "Key" and el.text and el.text.endswith(suffix):
            return el.text
    return None


def text_key(pmcid: str) -> str | None:
    params = urllib.parse.urlencode({"list-type": "2", "prefix": f"{pmcid}."})
    return _pick_key(get_bytes(f"{BUCKET}/?{params}"), ".txt")


def download_text(pmcid: str) -> tuple[str, str]:
    """Return (source_url, text). Raises LookupError if no plain text is in the bucket."""
    key = text_key(pmcid)
    if key is None:
        raise LookupError(f"no open-access plain text in PMC bucket for {pmcid}")
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    return url, get_bytes(url).decode("utf-8", "replace")
