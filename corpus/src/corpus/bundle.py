"""Publishes the client bundle: the manifest plus the works/findings JSONL the Mac app pulls.

Wire format and rules come from the client's bundle contract; two of them shape
everything here:

- Only works that are both citable and retrievable ship: status kept_text, a composed
  quality, and at least one finding. kept_miss works have no text and therefore no
  anchor, and retracted works never appear at all — tombstones for rows already shipped
  arrive with deltas, one release later.
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

SCHEMA_VERSION = 1
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
SELECT w.work_id, w.doi, w.title, w.year, w.authors, w.quality, w.quality_source,
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
    work_id, doi, title, year, authors, quality, quality_source, study_type, grade = row
    return {
        "work_id": work_id,
        "doi": doi,
        "title": title,
        # Not stored by the crawler; rehydrating it from OpenAlex/EPMC is the next
        # contract item. The key ships as null so its arrival is not a shape change.
        "venue": None,
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


def build(conn: sqlite3.Connection) -> tuple[bytes, bytes, int, int]:
    """Serialize the shippable population. Returns (works, findings, n_works, n_findings)."""
    works = [work_row(r) for r in conn.execute(WORKS_SQL)]
    findings = [finding_row(r) for r in conn.execute(FINDINGS_SQL)]
    validate(works, findings)
    return jsonl(works), jsonl(findings), len(works), len(findings)


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
    """
    works, findings, n_works, n_findings = build(conn)
    if not n_works:
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
        # Full bundle. Deltas (and the tombstones they carry) come one release later.
        "replaces": None,
        "created_at": now,
        "encoder": encoder_descriptor(models_dir),
        "counts": {"works": n_works, "findings": n_findings},
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
    return Published(manifest, changed=True)


def verify(out_dir: Path) -> dict[str, Any]:
    """Re-read what was published, as a client would, and check it against its manifest."""
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text())
    payloads: dict[str, bytes] = {}
    decompressor = zstandard.ZstdDecompressor()
    for key, name in manifest["files"].items():
        payloads[key] = decompressor.decompress((out_dir / name).read_bytes())
    content = digest(payloads["works"], payloads["findings"])
    if manifest["files"]["works"] != f"works-{content[:12]}.jsonl.zst":
        raise BundleError("published payload does not match its content-addressed name")
    works = [json.loads(line) for line in payloads["works"].splitlines()]
    findings = [json.loads(line) for line in payloads["findings"].splitlines()]
    validate(works, findings)
    if len(works) != manifest["counts"]["works"] or len(findings) != manifest["counts"]["findings"]:
        raise BundleError("manifest counts disagree with the payloads")
    compressed = sum((out_dir / n).stat().st_size for n in manifest["files"].values())
    return {
        "bundle_id": manifest["bundle_id"],
        "works": len(works),
        "findings": len(findings),
        "bytes_raw": len(payloads["works"]) + len(payloads["findings"]),
        "bytes_compressed": compressed,
    }
