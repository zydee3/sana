"""Tiny HTTP GET helper over the standard library, same contract as scraper/http.py.

Duplicated rather than imported: the two projects ship as separate wheels/images.
429 is not retried (credits exhausted); transient 5xx and connection resets get one
more try with linear backoff.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

USER_AGENT = "sana-corpus/0.1"
RETRY_STATUS = frozenset({500, 502, 503, 504})
ATTEMPTS = 3
BACKOFF_S = 2.0


def _fetch(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted https hosts)
        return bytes(resp.read())


def get_bytes(url: str, timeout: float = 30.0) -> bytes:
    for attempt in range(1, ATTEMPTS):
        try:
            return _fetch(url, timeout)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS:
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(BACKOFF_S * attempt)
    return _fetch(url, timeout)


def get_json(url: str, timeout: float = 30.0) -> Any:
    return json.loads(get_bytes(url, timeout))
