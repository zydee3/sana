from __future__ import annotations

import sqlite3

from corpus import db, sample


def _corpus(n_per_group: int = 50) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE works (work_id TEXT PRIMARY KEY, title TEXT, year INTEGER, doi TEXT,"
        " pmcid TEXT, discovered_via TEXT, status TEXT, study_type TEXT);"
    )
    rows = []
    for via in ("openalex", "europepmc"):
        for status in ("kept_text", "kept_miss", "rejected"):
            for i in range(n_per_group):
                wid = f"{via}-{status}-{i}"
                rows.append((wid, f"title {wid}", 2021 if i % 2 else 1998, None, None, via, status))
    conn.executemany(
        "INSERT INTO works (work_id, title, year, doi, pmcid, discovered_via, status)"
        " VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return conn


def test_year_bucket_boundaries() -> None:
    assert sample.year_bucket(None) == "unknown"
    assert sample.year_bucket(1999) == "pre2000"
    assert sample.year_bucket(2000) == "2000s"
    assert sample.year_bucket(2019) == "2010s"
    assert sample.year_bucket(2026) == "2020s"


def test_allocate_respects_floor_and_size() -> None:
    alloc = sample.allocate({"big": 1000, "tiny": 3}, n=100, floor=20)
    assert alloc["tiny"] == 3  # capped by stratum size
    assert alloc["big"] >= 20
    assert sum(alloc.values()) <= 103


def test_stratified_is_deterministic_and_excludes_rejected() -> None:
    conn = _corpus()
    a = sample.stratified(conn, n=60, seed=7, floor=5)
    b = sample.stratified(conn, n=60, seed=7, floor=5)
    assert [p.work_id for p in a] == [p.work_id for p in b]
    assert all(p.status.startswith("kept") for p in a)
    assert len({p.stratum for p in a}) == 8  # 2 via x 2 kept status x 2 year buckets
    assert sample.stratified(conn, n=60, seed=8, floor=5) != a


def test_jsonl_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    papers = sample.stratified(_corpus(), n=20, seed=1, floor=2)
    path = tmp_path / "s.jsonl"
    assert sample.write_jsonl(papers, path) == len(papers)
    assert sample.read_jsonl(path) == papers


def test_migrate_is_idempotent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "c.db"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE works (work_id TEXT PRIMARY KEY, title TEXT);")
    assert "works.relevance" in db.migrate(conn)
    assert db.migrate(conn) == []
    cols = {r[1] for r in conn.execute("PRAGMA table_info(works)")}
    assert {"relevance", "domain", "label_source", "label_confidence"} <= cols
    assert conn.execute("SELECT count(*) FROM abstracts").fetchone()[0] == 0


def test_allocate_meets_target_when_headroom_allows() -> None:
    sizes = {"big": 5000, "mid": 300, "small": 40, "tiny": 3}
    assert sum(sample.allocate(sizes, n=1000, floor=20).values()) == 1000


def test_allocate_stops_at_available_rows() -> None:
    sizes = {"a": 10, "b": 5}
    assert sum(sample.allocate(sizes, n=1000, floor=20).values()) == 15


def test_prefix_of_sample_spans_strata() -> None:
    papers = sample.stratified(_corpus(200), n=200, seed=3, floor=10)
    head = papers[:40]
    assert len({p.stratum for p in head}) >= 6  # shuffled, so a prefix is not one stratum
