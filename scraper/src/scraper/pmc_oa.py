"""Fetch: download a paper's PDF from the PMC open-access dataset on AWS.

The `pmc-oa-opendata` S3 bucket is keyed per paper at `PMC<id>.<version>/`, holding the
PDF, XML, and JSON. We list the paper's prefix to discover the versioned PDF key, then
download it. This avoids Europe PMC's `?pdf=render` endpoint (currently returning 404s) and
NCBI FTP (blocked in many environments).
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET

from .http import get_bytes

BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"


def _pick_pdf_key(listing_xml: bytes) -> str | None:
    """Return the first `.pdf` object key in an S3 ListBucketResult, or None."""
    root = ET.fromstring(listing_xml)
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == "Key" and el.text and el.text.endswith(".pdf"):
            return el.text
    return None


def pdf_key(pmcid: str) -> str | None:
    params = urllib.parse.urlencode({"list-type": "2", "prefix": f"{pmcid}."})
    return _pick_pdf_key(get_bytes(f"{BUCKET}/?{params}"))


def download_pdf(pmcid: str) -> tuple[str, bytes]:
    """Return (source_url, pdf_bytes). Raises LookupError if no PDF is in the bucket."""
    key = pdf_key(pmcid)
    if key is None:
        raise LookupError(f"no open-access PDF in PMC bucket for {pmcid}")
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    return url, get_bytes(url)
