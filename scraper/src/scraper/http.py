"""Tiny HTTP GET helpers over the standard library (no third-party deps)."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

USER_AGENT = "sana-scraper/0.1"


def get_bytes(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted https hosts)
        return bytes(resp.read())


def get_json(url: str, timeout: float = 30.0) -> Any:
    return json.loads(get_bytes(url, timeout))
