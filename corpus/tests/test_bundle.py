from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from corpus import bundle, db

NOW = "2026-08-17T22:40Z"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE works (work_id TEXT PRIMARY KEY, doi TEXT, title TEXT, year INTEGER,"
        " authors TEXT, status TEXT, study_type TEXT, evidence_grade INTEGER,"
        " relevance INTEGER);"
    )
    db.migrate(conn)
    return conn


def _work(
    conn: sqlite3.Connection, work_id: str, *, status: str = "kept_text", **cols: object
) -> None:
    fields = {
        "doi": f"10.0/{work_id}",
        "title": f"title {work_id}",
        "year": 2020,
        "authors": "Murray JK, Knudson S.",
        "status": status,
        "study_type": "rct",
        "evidence_grade": 2,
        "quality": 0.8,
        "quality_source": "sonnet",
        **cols,
    }
    keys = ", ".join(fields)
    conn.execute(
        f"INSERT INTO works (work_id, {keys}) VALUES (?{',?' * len(fields)})",
        (work_id, *fields.values()),
    )
    conn.commit()


def _finding(
    conn: sqlite3.Connection, fid: str, work_id: str, *, caveats: str = "adults only"
) -> None:
    chunk_id = f"{work_id}#0"
    conn.execute(
        "INSERT OR IGNORE INTO chunks (chunk_id, work_id, idx, section, text, n_words)"
        " VALUES (?, ?, 0, 'results', 'body text here', 3)",
        (chunk_id, work_id),
    )
    conn.execute(
        "INSERT INTO findings (finding_id, work_id, claim, caveats, anchor_chunk_id,"
        " char_start, char_end, quote, extracted_at) VALUES (?, ?, ?, ?, ?, 0, 4, 'body', ?)",
        (fid, work_id, f"claim {fid}", caveats, chunk_id, NOW),
    )
    conn.commit()


def _models_dir(tmp_path: Path) -> Path:
    onnx = tmp_path / "models" / bundle.ENCODER_MODEL / "model.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"fake onnx graph")
    return tmp_path / "models"


def _populated(tmp_path: Path) -> sqlite3.Connection:
    conn = _conn()
    _work(conn, "W1")
    _finding(conn, "f_a", "W1")
    _finding(conn, "f_b", "W1")
    return conn


def test_authors_parse_into_an_array() -> None:
    assert bundle.authors_array("Murray JK, Knudson S.") == ["Murray JK", "Knudson S."]
    assert bundle.authors_array("Murray JK; Knudson S") == ["Murray JK", "Knudson S"]
    assert bundle.authors_array('["A B", "C D"]') == ["A B", "C D"]
    assert bundle.authors_array(None) == []
    assert bundle.authors_array("  ") == []


def test_only_citable_and_retrievable_works_ship() -> None:
    conn = _conn()
    _work(conn, "W1")
    _finding(conn, "f_a", "W1")
    _work(conn, "W2")  # no findings
    _work(conn, "W3", status="kept_miss")
    _finding(conn, "f_c", "W3")
    _work(conn, "W4", status="retracted")
    _finding(conn, "f_d", "W4")
    _work(conn, "W5", quality=None, quality_source=None)
    _finding(conn, "f_e", "W5")

    works_b, findings_b, n_works, n_findings = bundle.build(conn)
    assert (n_works, n_findings) == (1, 1)
    row = json.loads(works_b.splitlines()[0])
    assert row["work_id"] == "W1"
    assert row["authors"] == ["Murray JK", "Knudson S."]
    assert row["venue"] is None  # unset in the fixture; the key ships either way
    anchor = json.loads(findings_b.splitlines()[0])["anchor"]
    assert anchor == {
        "chunk_id": "W1#0",
        "section": "results",
        "char_start": 0,
        "char_end": 4,
        "quote": "body",
    }


def test_a_rehydrated_venue_ships_on_the_work_record() -> None:
    conn = _conn()
    _work(conn, "W1", venue="BMC Medicine", venue_source="openalex")
    _finding(conn, "f_a", "W1")
    works_b, _, _, _ = bundle.build(conn)
    assert json.loads(works_b.splitlines()[0])["venue"] == "BMC Medicine"


def test_empty_caveats_never_ship() -> None:
    conn = _conn()
    _work(conn, "W1")
    _finding(conn, "f_a", "W1", caveats="  ")
    with pytest.raises(bundle.BundleError, match="empty claim or caveats"):
        bundle.build(conn)


def test_publish_writes_a_manifest_a_client_can_verify(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    published = bundle.publish(conn, out, NOW, models_dir=_models_dir(tmp_path))
    assert published.changed
    m = published.manifest
    assert m["schema_version"] == bundle.SCHEMA_VERSION
    assert m["replaces"] is None
    assert m["bundle_id"].startswith(f"b_{NOW}_")
    assert m["counts"] == {"works": 1, "findings": 2}
    assert m["encoder"]["dim"] == 384
    assert json.loads((out / "latest.json").read_text()) == m
    stats = bundle.verify(out)
    assert stats["works"] == 1 and stats["findings"] == 2


def test_republishing_unchanged_data_is_a_no_op(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    models = _models_dir(tmp_path)
    first = bundle.publish(conn, out, NOW, models_dir=models)
    again = bundle.publish(conn, out, "2026-09-01T00:00Z", models_dir=models)
    assert not again.changed
    assert again.manifest["bundle_id"] == first.manifest["bundle_id"]


def test_new_data_produces_a_new_bundle_id_and_new_files(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    models = _models_dir(tmp_path)
    first = bundle.publish(conn, out, NOW, models_dir=models)
    _work(conn, "W2")
    _finding(conn, "f_c", "W2")
    second = bundle.publish(conn, out, "2026-09-01T00:00Z", models_dir=models)
    assert second.changed
    assert second.manifest["bundle_id"] != first.manifest["bundle_id"]
    assert second.manifest["files"] != first.manifest["files"]
    # The first bundle's payloads stay readable: names are content-addressed.
    assert (out / first.manifest["files"]["works"]).exists()
    assert bundle.verify(out)["works"] == 2


def test_ids_do_not_move_when_other_rows_arrive(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    before = [json.loads(line) for line in bundle.build(conn)[1].splitlines()]
    _work(conn, "W0")  # sorts ahead of W1
    _finding(conn, "f_z", "W0")
    after = {json.loads(line)["finding_id"] for line in bundle.build(conn)[1].splitlines()}
    assert {f["finding_id"] for f in before} <= after


def test_verify_rejects_a_tampered_payload(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    published = bundle.publish(conn, out, NOW, models_dir=_models_dir(tmp_path))
    import zstandard

    path = out / published.manifest["files"]["findings"]
    rows = zstandard.ZstdDecompressor().decompress(path.read_bytes()).splitlines()
    path.write_bytes(zstandard.ZstdCompressor().compress(rows[0] + b"\n"))
    with pytest.raises(bundle.BundleError):
        bundle.verify(out)


def test_missing_encoder_stops_the_publish(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    with pytest.raises(bundle.BundleError, match="not present"):
        bundle.publish(conn, tmp_path / "bundles", NOW, models_dir=tmp_path / "absent")


def test_publishing_nothing_is_an_error(tmp_path: Path) -> None:
    conn = _conn()
    with pytest.raises(bundle.BundleError, match="nothing to publish"):
        bundle.publish(conn, tmp_path / "bundles", NOW, models_dir=_models_dir(tmp_path))
