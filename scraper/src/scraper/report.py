"""Hourly Discord status line: how many targets the crawler has scraped.

A daemon thread posts one short message per interval to a Discord channel via the
bot API. Reporting is best-effort observability — failures are logged and never
touch the crawl. Enabled only when DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID are set.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import urllib.request
from pathlib import Path

from .http import USER_AGENT

API_URL = "https://discord.com/api/v10/channels/{channel_id}/messages"


def _counts(db_path: Path) -> tuple[int, int]:
    """(works with text, works total) from a read-only connection."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        with_text = conn.execute(
            "SELECT COUNT(*) FROM works WHERE status = 'kept_text'"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM works").fetchone()[0]
        return int(with_text), int(total)
    finally:
        conn.close()


def post_count(db_path: Path, token: str, channel_id: str, timeout: float = 30.0) -> None:
    with_text, total = _counts(db_path)
    body = json.dumps({"content": f"scraped: {with_text} papers with text ({total} tracked)"})
    req = urllib.request.Request(
        API_URL.format(channel_id=channel_id),
        data=body.encode(),
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout):  # noqa: S310 (fixed https host)
        pass


def start_reporter(
    db_path: Path, token: str, channel_id: str, interval_s: int = 3600
) -> threading.Thread:
    """Post the count every interval on a daemon thread; errors are logged, not raised."""

    def loop() -> None:
        ticker = threading.Event()
        while True:
            try:
                post_count(db_path, token, channel_id)
            except (OSError, sqlite3.Error) as e:
                print(f"discord report failed: {e}", flush=True)
            ticker.wait(interval_s)

    thread = threading.Thread(target=loop, name="discord-reporter", daemon=True)
    thread.start()
    return thread
