"""Publishes the client bundle: the manifest plus the works/findings JSONL the Mac app pulls.

Wire format and rules come from the client's bundle contract; two of them shape
everything here:

- Only works that are both citable and retrievable ship: status kept_text, a composed
  quality, and at least one finding. kept_miss works have no text and therefore no
  anchor, and retracted works never appear as live rows.
- A row the client already has can only be withdrawn by a tombstone, because bundles
  apply by primary key: dropping a row from the payload leaves it in the client's DB
  forever. So every bundle carries the full tombstone set — every id in the `shipped`
  ledger that is no longer shippable — rather than only the ids that changed since some
  base version. Full bundles stay self-sufficient, and a client any number of versions
  behind converges from the latest one alone.
- IDs are stable from the first publish. finding_id is already a content hash of
  (work_id, claim); work_id is the crawler's OpenAlex id. Nothing here derives an id
  from row order, chunking or bundle version.

The bundle is content-addressed: payload filenames carry a digest of the uncompressed
rows, so republishing unchanged data rewrites nothing and the client's applied
bundle_id stays valid.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zstandard

from . import embed

# 2: evidence_grade is derived for every labeled work, so the quality scalar dropped for
# 65% of shipped works (mean 0.744 -> 0.664). The formula is unchanged, but tau_q has to
# be re-swept, which is what the version exists to signal.
SCHEMA_VERSION = 2
ZSTD_LEVEL = 19

# The server profile's encoder. The client ignores it today (it embeds claim text with
# its own model) but the contract wants the descriptor present from day one, so it can
# detect a mismatch if it ever adopts MiniLM for parity.
ENCODER_MODEL = "minilm-int8"

DEFAULT_OUT = Path("/sana-data/corpus/bundles")
MANIFEST_NAME = "latest.json"

# kept_text + a quality + at least one finding. The three conditions are the whole
# inclusion rule; everything else in the corpus is out of the client profile.
WORKS_SQL = """
SELECT w.work_id, w.doi, w.title, w.venue, w.year, w.authors, w.quality, w.quality_source,
       w.study_type, w.evidence_grade
FROM works w
WHERE w.status = 'kept_text' AND w.quality IS NOT NULL
  AND EXISTS (SELECT 1 FROM findings f WHERE f.work_id = w.work_id)
ORDER BY w.work_id
"""

# section comes from the anchor's chunk rather than the finding, so a re-chunk that
# moves a claim into a differently-labelled section cannot leave a stale label behind.
FINDINGS_SQL = """
SELECT f.finding_id, f.work_id, f.claim, f.caveats, f.anchor_chunk_id, c.section,
       f.char_start, f.char_end, f.quote
FROM findings f
JOIN chunks c ON c.chunk_id = f.anchor_chunk_id
JOIN works w ON w.work_id = f.work_id
WHERE w.status = 'kept_text' AND w.quality IS NOT NULL
ORDER BY f.finding_id
"""

SHIPPED_SQL = "SELECT row_id FROM shipped WHERE kind = ?"
RECORD_SQL = "INSERT OR IGNORE INTO shipped (kind, row_id, first_shipped) VALUES (?, ?, ?)"


class BundleError(RuntimeError):
    """A bundle that would violate the contract; publishing stops rather than shipping."""


@dataclass(frozen=True)
class Published:
    manifest: dict[str, Any]
    changed: bool  # False when the payload digest matched what is already published


def authors_array(raw: str | None) -> list[str]:
    """Canonical wire format is a JSON array of strings.

    The crawler stored the publisher's own string ("Murray JK, Knudson S."), which is
    comma-joined; a few sources use semicolons. Names keep their trailing initials-dot.
    """
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    if text.startswith("["):
        parsed = json.loads(text)
        return [str(a).strip() for a in parsed if str(a).strip()]
    sep = ";" if ";" in text else ","
    return [part.strip() for part in text.split(sep) if part.strip()]


def work_row(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    work_id, doi, title, venue, year, authors, quality, quality_source, study_type, grade = row
    return {
        "work_id": work_id,
        "doi": doi,
        "title": title,
        # Rehydrated by venue.py, not stored by the crawler. Still null for the works
        # neither OpenAlex nor EPMC could name; the key is always present.
        "venue": venue,
        "year": year,
        "authors": authors_array(authors),
        "quality": quality,
        "quality_source": quality_source,
        "study_type": study_type,
        "evidence_grade": grade,
    }


def finding_row(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    fid, work_id, claim, caveats, chunk_id, section, start, end, quote = row
    return {
        "finding_id": fid,
        "work_id": work_id,
        "claim": claim,
        "caveats": caveats,
        # No `page`: the texts are plain-text extractions with no page boundaries. The
        # contract's required minimum is section + span + quote.
        "anchor": {
            "chunk_id": chunk_id,
            "section": section,
            "char_start": start,
            "char_end": end,
            "quote": quote,
        },
    }


def tombstone_rows(conn: sqlite3.Connection, kind: str, live: set[str]) -> list[dict[str, Any]]:
    """`{id: ..., deleted: true}` for every shipped row of `kind` that is no longer live."""
    key = f"{kind}_id"
    gone = sorted({str(r[0]) for r in conn.execute(SHIPPED_SQL, (kind,))} - live)
    return [{key: row_id, "deleted": True} for row_id in gone]


def record_shipped(conn: sqlite3.Connection, kind: str, live: set[str], now: str) -> int:
    """Add live ids to the ledger. Tombstoned ids stay: they must ship in every bundle."""
    before = conn.execute("SELECT count(*) FROM shipped WHERE kind = ?", (kind,)).fetchone()[0]
    conn.executemany(RECORD_SQL, [(kind, row_id, now) for row_id in sorted(live)])
    conn.commit()
    return int(
        conn.execute("SELECT count(*) FROM shipped WHERE kind = ?", (kind,)).fetchone()[0] - before
    )


def split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(live, tombstones) — the two row shapes that share a payload file."""
    return [r for r in rows if not r.get("deleted")], [r for r in rows if r.get("deleted")]


def validate(works: list[dict[str, Any]], findings: list[dict[str, Any]]) -> None:
    """Contract invariants, checked before anything is written."""
    ids = {w["work_id"] for w in works}
    if len(ids) != len(works):
        raise BundleError("duplicate work_id in bundle")
    for w in works:
        q = w["quality"]
        if not isinstance(q, float) or not 0.0 <= q <= 1.0:
            raise BundleError(f"{w['work_id']}: quality {q!r} outside [0,1]")
        if not w["quality_source"]:
            raise BundleError(f"{w['work_id']}: quality without a source")
    seen: set[str] = set()
    for f in findings:
        if f["finding_id"] in seen:
            raise BundleError(f"duplicate finding_id {f['finding_id']}")
        seen.add(f["finding_id"])
        if f["work_id"] not in ids:
            raise BundleError(f"{f['finding_id']}: anchors to unshipped work {f['work_id']}")
        if not f["claim"].strip() or not f["caveats"].strip():
            raise BundleError(f"{f['finding_id']}: empty claim or caveats")
        if not f["anchor"]["quote"].strip():
            raise BundleError(f"{f['finding_id']}: empty anchor quote")
    orphans = ids - {f["work_id"] for f in findings}
    if orphans:
        raise BundleError(f"{len(orphans)} works ship with no findings (e.g. {sorted(orphans)[0]})")


def validate_tombstones(live: list[dict[str, Any]], dead: list[dict[str, Any]], key: str) -> None:
    """A tombstone contradicting a live row would make the client's apply order decide."""
    live_ids = {r[key] for r in live}
    dead_ids = {r[key] for r in dead}
    both = live_ids & dead_ids
    if both:
        raise BundleError(f"{len(both)} rows are both live and tombstoned (e.g. {sorted(both)[0]})")
    if len(dead_ids) != len(dead):
        raise BundleError("duplicate tombstone")


def jsonl(rows: list[dict[str, Any]]) -> bytes:
    lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows]
    return ("\n".join(lines) + "\n").encode() if lines else b""


def digest(works: bytes, findings: bytes) -> str:
    """Digest of the payloads, length-prefixed so the pair cannot be re-split ambiguously."""
    h = hashlib.sha256()
    for payload in (works, findings):
        h.update(str(len(payload)).encode())
        h.update(b"\n")
        h.update(payload)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def encoder_descriptor(models_dir: Path | None = None) -> dict[str, Any]:
    spec = embed.MODELS[ENCODER_MODEL]
    root = models_dir if models_dir is not None else embed.MODELS_DIR
    onnx = root / spec.name / "model.onnx"
    if not onnx.exists():
        raise BundleError(f"encoder {spec.name} not present at {onnx}; cannot describe it")
    return {"name": spec.name, "dim": spec.dim, "sha256": sha256_file(onnx)}


@dataclass(frozen=True)
class Payloads:
    works: bytes
    findings: bytes
    counts: dict[str, int]
    tombstones: dict[str, int]
    live_work_ids: set[str]
    live_finding_ids: set[str]


def build(conn: sqlite3.Connection) -> Payloads:
    """Serialize the shippable population plus the tombstones for everything withdrawn."""
    works = [work_row(r) for r in conn.execute(WORKS_SQL)]
    findings = [finding_row(r) for r in conn.execute(FINDINGS_SQL)]
    validate(works, findings)
    work_ids = {w["work_id"] for w in works}
    finding_ids = {f["finding_id"] for f in findings}
    dead_works = tombstone_rows(conn, "work", work_ids)
    dead_findings = tombstone_rows(conn, "finding", finding_ids)
    validate_tombstones(works, dead_works, "work_id")
    validate_tombstones(findings, dead_findings, "finding_id")
    return Payloads(
        works=jsonl(works + dead_works),
        findings=jsonl(findings + dead_findings),
        counts={"works": len(works), "findings": len(findings)},
        tombstones={"works": len(dead_works), "findings": len(dead_findings)},
        live_work_ids=work_ids,
        live_finding_ids=finding_ids,
    )


def publish(
    conn: sqlite3.Connection,
    out_dir: Path,
    now: str,
    *,
    models_dir: Path | None = None,
) -> Published:
    """Write the bundle into out_dir. A no-op when the payload digest is already published.

    `now` is minute-precision UTC ("2026-08-17T22:40Z") and only names the bundle; the
    identity that matters is the digest, which is a pure function of the rows.

    Writes: publishing records what it shipped in the `shipped` ledger, which is how the
    next bundle knows what it owes a tombstone.
    """
    payloads = build(conn)
    works, findings = payloads.works, payloads.findings
    if not payloads.counts["works"]:
        raise BundleError("nothing to publish: no work has both a quality and a finding")
    content = digest(works, findings)
    files = {
        "works": f"works-{content[:12]}.jsonl.zst",
        "findings": f"findings-{content[:12]}.jsonl.zst",
    }

    manifest_path = out_dir / MANIFEST_NAME
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text())
        if current.get("files") == files and current.get("schema_version") == SCHEMA_VERSION:
            return Published(current, changed=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": f"b_{now}_{content[:6]}",
        # Full bundle, always: it carries every live row and every tombstone, so there is
        # no chain to walk. `replaces` stays null until deltas are worth their complexity.
        "replaces": None,
        "created_at": now,
        "encoder": encoder_descriptor(models_dir),
        "counts": payloads.counts,
        # Rows in the payloads beyond `counts` — withdrawn ids the client marks deleted.
        "tombstones": payloads.tombstones,
        "files": files,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    for key, payload in (("works", works), ("findings", findings)):
        path = out_dir / files[key]
        if not path.exists():  # content-addressed: same name means same bytes
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(compressor.compress(payload))
            tmp.rename(path)
    tmp_manifest = manifest_path.with_suffix(".part")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp_manifest.rename(manifest_path)
    # After the files exist: a crash between the two leaves the ledger short, and the next
    # run re-derives the same payloads and records them. Claiming first could not be undone.
    record_shipped(conn, "work", payloads.live_work_ids, now)
    record_shipped(conn, "finding", payloads.live_finding_ids, now)
    return Published(manifest, changed=True)


def read_published(out_dir: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """The client's side of the wire: manifest plus the decompressed payloads it names."""
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text())
    decompressor = zstandard.ZstdDecompressor()
    payloads = {
        key: decompressor.decompress((out_dir / name).read_bytes())
        for key, name in manifest["files"].items()
    }
    return manifest, payloads


def verify(out_dir: Path) -> dict[str, Any]:
    """Re-read what was published, as a client would, and check it against its manifest."""
    manifest, payloads = read_published(out_dir)
    content = digest(payloads["works"], payloads["findings"])
    if manifest["files"]["works"] != f"works-{content[:12]}.jsonl.zst":
        raise BundleError("published payload does not match its content-addressed name")
    rows = {k: [json.loads(line) for line in v.splitlines()] for k, v in payloads.items()}
    works, dead_works = split(rows["works"])
    findings, dead_findings = split(rows["findings"])
    validate(works, findings)
    validate_tombstones(works, dead_works, "work_id")
    validate_tombstones(findings, dead_findings, "finding_id")
    if len(works) != manifest["counts"]["works"] or len(findings) != manifest["counts"]["findings"]:
        raise BundleError("manifest counts disagree with the payloads")
    declared = manifest.get("tombstones") or {"works": 0, "findings": 0}
    if len(dead_works) != declared["works"] or len(dead_findings) != declared["findings"]:
        raise BundleError("manifest tombstone counts disagree with the payloads")
    compressed = sum((out_dir / n).stat().st_size for n in manifest["files"].values())
    return {
        "bundle_id": manifest["bundle_id"],
        "works": len(works),
        "findings": len(findings),
        "tombstones": {"works": len(dead_works), "findings": len(dead_findings)},
        "bytes_raw": len(payloads["works"]) + len(payloads["findings"]),
        "bytes_compressed": compressed,
    }


# The published quote must still be the bytes the chunk holds at the anchor's span: it is
# what the client renders as evidence under a claim, and a re-clean or re-chunk moves text
# under a stored span silently.
ANCHOR_SQL = """
SELECT f.finding_id, substr(c.text, f.char_start + 1, f.char_end - f.char_start)
FROM findings f JOIN chunks c ON c.chunk_id = f.anchor_chunk_id
WHERE f.finding_id IN ({placeholders})
"""

STATUS_SQL = "SELECT work_id, status FROM works WHERE work_id IN ({placeholders})"

SLICE = 500


def _by_id(conn: sqlite3.Connection, sql: str, ids: list[str]) -> dict[str, Any]:
    """Look up ids in slices — a bundle carries more of them than SQLite takes per query."""
    out: dict[str, Any] = {}
    for start in range(0, len(ids), SLICE):
        chunk = ids[start : start + SLICE]
        query = sql.format(placeholders=",".join("?" * len(chunk)))
        out.update({str(row[0]): row[1] for row in conn.execute(query, chunk)})
    return out


def audit(conn: sqlite3.Connection, out_dir: Path) -> dict[str, Any]:
    """Read the published bundle as a client would and check it against corpus.db.

    verify() proves a bundle agrees with its own manifest; a tampered or truncated
    payload is all it can catch. This proves the bundle still says what the database
    says — that no shipped row drifted from its source, no retracted work is live, no
    anchor quote has moved, and every id the ledger claims to have shipped is either
    live or tombstoned. Drift against the database is reported separately from the
    problems, because it is not a fault: it just means a republish is owed.
    """
    manifest, payloads = read_published(out_dir)
    rows = {k: [json.loads(line) for line in v.splitlines()] for k, v in payloads.items()}
    works, dead_works = split(rows["works"])
    findings, dead_findings = split(rows["findings"])
    shipped_works = {w["work_id"]: w for w in works}
    shipped_findings = {f["finding_id"]: f for f in findings}

    problems: list[str] = []
    db_works = {str(r[0]): work_row(r) for r in conn.execute(WORKS_SQL)}
    db_findings = {str(r[0]): finding_row(r) for r in conn.execute(FINDINGS_SQL)}

    for shipped, current, label in (
        (shipped_works, db_works, "work"),
        (shipped_findings, db_findings, "finding"),
    ):
        for row_id, row in sorted(shipped.items()):
            live = current.get(row_id)
            if live is None:  # left the shippable population: drift, reported below
                continue
            fields = sorted(k for k in live if live[k] != row.get(k))
            if fields:
                problems.append(f"{label} {row_id}: {', '.join(fields)} differ from the db")

    statuses = _by_id(conn, STATUS_SQL, sorted(shipped_works))
    problems += [
        f"work {wid} is {statuses[wid]} and still ships as a live row"
        for wid in sorted(shipped_works)
        if statuses.get(wid) != "kept_text"
    ]

    spans = _by_id(conn, ANCHOR_SQL, sorted(shipped_findings))
    problems += [
        f"finding {fid}: anchor quote is not the text at its span"
        for fid, row in sorted(shipped_findings.items())
        if spans.get(fid) != row["anchor"]["quote"]
    ]

    ledger_size = ledger_uncovered = 0
    for kind, key, live_ids, dead in (
        ("work", "work_id", set(shipped_works), dead_works),
        ("finding", "finding_id", set(shipped_findings), dead_findings),
    ):
        ledger = {str(r[0]) for r in conn.execute(SHIPPED_SQL, (kind,))}
        uncovered = sorted(ledger - live_ids - {r[key] for r in dead})
        ledger_size += len(ledger)
        ledger_uncovered += len(uncovered)
        problems += [
            f"{kind} {row_id} shipped once but is neither live nor tombstoned"
            for row_id in uncovered
        ]

    return {
        "bundle_id": manifest["bundle_id"],
        "counts": {"works": len(works), "findings": len(findings)},
        "tombstones": {"works": len(dead_works), "findings": len(dead_findings)},
        "checked": {
            "works": len(shipped_works),
            "findings": len(shipped_findings),
            "ledger": ledger_size,
        },
        "drift": {
            "works_added": len(set(db_works) - set(shipped_works)),
            "works_removed": len(set(shipped_works) - set(db_works)),
            "findings_added": len(set(db_findings) - set(shipped_findings)),
            "findings_removed": len(set(shipped_findings) - set(db_findings)),
        },
        "uncovered": ledger_uncovered,
        "problems": problems,
    }
