from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
import zstandard

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

    built = bundle.build(conn)
    assert built.counts == {"works": 1, "findings": 1}
    row = json.loads(built.works.splitlines()[0])
    assert row["work_id"] == "W1"
    assert row["authors"] == ["Murray JK", "Knudson S."]
    assert row["venue"] is None  # unset in the fixture; the key ships either way
    anchor = json.loads(built.findings.splitlines()[0])["anchor"]
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
    assert json.loads(bundle.build(conn).works.splitlines()[0])["venue"] == "BMC Medicine"


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
    before = [json.loads(line) for line in bundle.build(conn).findings.splitlines()]
    _work(conn, "W0")  # sorts ahead of W1
    _finding(conn, "f_z", "W0")
    after = {json.loads(line)["finding_id"] for line in bundle.build(conn).findings.splitlines()}
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


def _rows(out: Path, manifest: dict[str, object], key: str) -> list[dict[str, object]]:
    import zstandard

    files = manifest["files"]
    assert isinstance(files, dict)
    raw = zstandard.ZstdDecompressor().decompress((out / files[key]).read_bytes())
    return [json.loads(line) for line in raw.splitlines()]


def test_a_retracted_work_leaves_as_a_tombstone_with_its_findings(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    models = _models_dir(tmp_path)
    bundle.publish(conn, out, NOW, models_dir=models)
    _work(conn, "W2")
    _finding(conn, "f_c", "W2")
    bundle.publish(conn, out, "2026-09-01T00:00Z", models_dir=models)

    conn.execute("UPDATE works SET status='retracted' WHERE work_id='W1'")
    conn.commit()
    third = bundle.publish(conn, out, "2026-09-02T00:00Z", models_dir=models)
    assert third.manifest["counts"] == {"works": 1, "findings": 1}
    assert third.manifest["tombstones"] == {"works": 1, "findings": 2}
    works = _rows(out, third.manifest, "works")
    assert {"work_id": "W1", "deleted": True} in works
    assert [w["work_id"] for w in works if not w.get("deleted")] == ["W2"]
    dead = {f["finding_id"] for f in _rows(out, third.manifest, "findings") if f.get("deleted")}
    assert dead == {"f_a", "f_b"}
    assert bundle.verify(out)["tombstones"] == {"works": 1, "findings": 2}


def test_tombstones_repeat_in_every_later_bundle(tmp_path: Path) -> None:
    """A client several versions behind must converge from the latest bundle alone."""
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    models = _models_dir(tmp_path)
    bundle.publish(conn, out, NOW, models_dir=models)
    _work(conn, "W2")
    _finding(conn, "f_c", "W2")
    conn.execute("UPDATE works SET status='retracted' WHERE work_id='W1'")
    conn.commit()
    bundle.publish(conn, out, "2026-09-01T00:00Z", models_dir=models)
    _work(conn, "W3")
    _finding(conn, "f_d", "W3")
    later = bundle.publish(conn, out, "2026-09-02T00:00Z", models_dir=models)
    assert later.manifest["tombstones"] == {"works": 1, "findings": 2}


def test_a_work_that_never_shipped_leaves_no_tombstone(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    models = _models_dir(tmp_path)
    _work(conn, "W2", status="retracted")
    _finding(conn, "f_c", "W2")
    published = bundle.publish(conn, out, NOW, models_dir=models)
    assert published.manifest["tombstones"] == {"works": 0, "findings": 0}


def test_the_ledger_records_only_what_was_written(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    bundle.publish(conn, out, NOW, models_dir=_models_dir(tmp_path))
    assert conn.execute("SELECT count(*) FROM shipped WHERE kind='work'").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM shipped WHERE kind='finding'").fetchone()[0] == 2
    _work(conn, "W2")
    _finding(conn, "f_c", "W2")
    with pytest.raises(bundle.BundleError, match="cannot describe it"):
        bundle.publish(conn, out, NOW, models_dir=tmp_path / "absent")
    assert conn.execute("SELECT count(*) FROM shipped").fetchone()[0] == 3


def test_verify_rejects_a_manifest_that_undercounts_tombstones(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    models = _models_dir(tmp_path)
    bundle.publish(conn, out, NOW, models_dir=models)
    conn.execute("UPDATE works SET status='retracted' WHERE work_id='W1'")
    conn.commit()
    _work(conn, "W2")
    _finding(conn, "f_c", "W2")
    manifest = bundle.publish(conn, out, "2026-09-01T00:00Z", models_dir=models).manifest
    manifest["tombstones"] = {"works": 0, "findings": 0}
    (out / "latest.json").write_text(json.dumps(manifest))
    with pytest.raises(bundle.BundleError, match="tombstone counts"):
        bundle.verify(out)


def test_audit_passes_on_a_freshly_published_bundle(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    bundle.publish(conn, out, NOW, models_dir=_models_dir(tmp_path))
    report = bundle.audit(conn, out)
    assert report["problems"] == []
    assert report["checked"] == {"works": 1, "findings": 2, "ledger": 3}
    assert not any(report["drift"].values())


def test_audit_reports_db_drift_without_calling_it_a_problem(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    bundle.publish(conn, out, NOW, models_dir=_models_dir(tmp_path))
    _work(conn, "W2")
    _finding(conn, "f_c", "W2")
    report = bundle.audit(conn, out)
    assert report["problems"] == []
    assert report["drift"]["works_added"] == 1 and report["drift"]["findings_added"] == 1


def test_audit_catches_a_shipped_row_that_drifted_from_the_db(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    bundle.publish(conn, out, NOW, models_dir=_models_dir(tmp_path))
    conn.execute("UPDATE works SET quality = 0.1 WHERE work_id = 'W1'")
    conn.commit()
    problems = bundle.audit(conn, out)["problems"]
    assert problems == ["work W1: quality differ from the db"]


def test_audit_catches_a_retracted_work_still_live_in_the_published_bundle(tmp_path: Path) -> None:
    """The Gap 3 guarantee, checked against the bundle on disk rather than the publisher."""
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    bundle.publish(conn, out, NOW, models_dir=_models_dir(tmp_path))
    conn.execute("UPDATE works SET status = 'retracted' WHERE work_id = 'W1'")
    conn.commit()
    problems = bundle.audit(conn, out)["problems"]
    assert "work W1 is retracted and still ships as a live row" in problems


def test_audit_catches_an_anchor_quote_that_no_longer_matches_its_chunk(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    bundle.publish(conn, out, NOW, models_dir=_models_dir(tmp_path))
    # A re-clean shifts the chunk text under a stored span; the bundle keeps the old quote.
    conn.execute("UPDATE chunks SET text = 'xxxx body text here' WHERE chunk_id = 'W1#0'")
    conn.commit()
    problems = bundle.audit(conn, out)["problems"]
    assert sorted(problems) == [
        "finding f_a: anchor quote is not the text at its span",
        "finding f_b: anchor quote is not the text at its span",
    ]


def test_audit_catches_a_shipped_id_that_is_neither_live_nor_tombstoned(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    bundle.publish(conn, out, NOW, models_dir=_models_dir(tmp_path))
    # A ledger id with no live row and no tombstone is a row the client can never lose.
    conn.execute(
        "INSERT INTO shipped (kind, row_id, first_shipped) VALUES ('finding', 'f_gone', ?)", (NOW,)
    )
    conn.commit()
    problems = bundle.audit(conn, out)["problems"]
    assert problems == ["finding f_gone shipped once but is neither live nor tombstoned"]


def _content(manifest: dict[str, object]) -> str:
    files = manifest["files"]
    assert isinstance(files, dict)
    return str(files["works"])[len("works-") : -len(".jsonl.zst")]


def test_diff_reports_what_a_client_applies_between_two_bundles(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    models = _models_dir(tmp_path)
    first = bundle.publish(conn, out, NOW, models_dir=models)
    _work(conn, "W2")
    _finding(conn, "f_c", "W2")
    conn.execute("UPDATE works SET title='a corrected title' WHERE work_id='W1'")
    conn.commit()
    second = bundle.publish(conn, out, "2026-09-01T00:00Z", models_dir=models)

    report = bundle.diff(out, _content(first.manifest), _content(second.manifest))
    assert report["problems"] == []
    works = report["kinds"]["works"]
    assert works["added"] == 1 and works["carried"] == 1 and works["vanished"] == 0
    assert works["changed_fields"] == {"title": 1}
    findings = report["kinds"]["findings"]
    assert findings["added"] == 1 and findings["changed_fields"] == {}


def test_diff_counts_a_retraction_as_tombstoned_not_vanished(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    models = _models_dir(tmp_path)
    first = bundle.publish(conn, out, NOW, models_dir=models)
    _work(conn, "W2")
    _finding(conn, "f_c", "W2")
    conn.execute("UPDATE works SET status='retracted' WHERE work_id='W1'")
    conn.commit()
    second = bundle.publish(conn, out, "2026-09-01T00:00Z", models_dir=models)

    report = bundle.diff(out, _content(first.manifest), _content(second.manifest))
    assert report["problems"] == []
    assert report["kinds"]["works"]["tombstoned"] == 1
    assert report["kinds"]["works"]["vanished"] == 0
    assert report["kinds"]["findings"]["tombstoned"] == 2


def test_diff_flags_a_row_that_left_without_a_tombstone(tmp_path: Path) -> None:
    """The unrecoverable case: the client keeps a dropped row in its DB forever."""
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    models = _models_dir(tmp_path)
    first = bundle.publish(conn, out, NOW, models_dir=models)
    _work(conn, "W2")
    _finding(conn, "f_c", "W2")
    second = bundle.publish(conn, out, "2026-09-01T00:00Z", models_dir=models)
    # Rewrite the second payload as if W1 had simply been dropped from the works file.
    rows = [w for w in _rows(out, second.manifest, "works") if w["work_id"] != "W1"]
    name = str(second.manifest["files"]["works"])
    (out / name).write_bytes(zstandard.ZstdCompressor().compress(bundle.jsonl(rows)))

    report = bundle.diff(out, _content(first.manifest), _content(second.manifest))
    assert report["kinds"]["works"]["vanished"] == 1
    assert report["problems"] == ["work_id W1 left the bundle with no tombstone"]


def test_history_orders_published_pairs_oldest_first(tmp_path: Path) -> None:
    conn = _populated(tmp_path)
    out = tmp_path / "bundles"
    models = _models_dir(tmp_path)
    first = bundle.publish(conn, out, NOW, models_dir=models)
    _work(conn, "W2")
    _finding(conn, "f_c", "W2")
    second = bundle.publish(conn, out, "2026-09-01T00:00Z", models_dir=models)
    for name in ("works", "findings"):
        os.utime(out / str(first.manifest["files"][name]), (1000, 1000))
        os.utime(out / str(second.manifest["files"][name]), (2000, 2000))

    assert bundle.history(out) == [_content(first.manifest), _content(second.manifest)]
