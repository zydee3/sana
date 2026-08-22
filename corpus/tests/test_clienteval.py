from __future__ import annotations

import sqlite3
from pathlib import Path

from corpus import bundle, clienteval, db, evaluate

NOW = "2026-08-22T16:40Z"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE works (work_id TEXT PRIMARY KEY, doi TEXT, title TEXT, year INTEGER,"
        " authors TEXT, status TEXT, study_type TEXT, evidence_grade INTEGER,"
        " relevance INTEGER);"
    )
    db.migrate(conn)
    conn.execute(
        "INSERT INTO works (work_id, doi, title, year, authors, status, study_type,"
        " evidence_grade, quality, quality_source)"
        " VALUES ('W1', '10.0/W1', 'a title', 2020, 'Murray JK', 'kept_text', 'rct', 2, 0.8,"
        " 'sonnet')"
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, work_id, idx, section, text, n_words)"
        " VALUES ('W1#0', 'W1', 0, 'results', 'body text here', 3)"
    )
    for fid in ("f_a", "f_b"):
        conn.execute(
            "INSERT INTO findings (finding_id, work_id, claim, caveats, anchor_chunk_id,"
            " char_start, char_end, quote, extracted_at) VALUES (?, 'W1', ?, 'adults only',"
            " 'W1#0', 0, 4, 'body', ?)",
            (fid, f"claim {fid}", NOW),
        )
    conn.commit()
    return conn


def _published(tmp_path: Path) -> Path:
    onnx = tmp_path / "models" / bundle.ENCODER_MODEL / "model.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"fake onnx graph")
    out = tmp_path / "bundles"
    bundle.publish(_conn(), out, NOW, models_dir=tmp_path / "models")
    return out


def test_cards_carry_the_work_fields_the_client_renders(tmp_path: Path) -> None:
    bundle_id, cards = clienteval.load_cards(_published(tmp_path))
    assert bundle_id.startswith("b_")
    assert [c.finding_id for c in cards] == ["f_a", "f_b"]
    assert cards[0].claim == "claim f_a" and cards[0].caveats == "adults only"
    assert cards[0].quality == 0.8 and cards[0].title == "a title"


def test_an_unjudged_row_is_not_scored_as_irrelevant(tmp_path: Path) -> None:
    path = tmp_path / "j.tsv"
    path.write_text("query_idx\tfinding_id\trelevant\n1\tf_a\t1\n1\tf_b\t?\n")
    judgments = clienteval.load_judgments(path)
    assert judgments == {(1, "f_a"): 1}
    row, missing = evaluate.score_hits(
        [["f_a", "f_b"]], judgments, backend="claims", params="m", n=2, k=2
    )
    assert row.p_at_k == 0.5 and missing == [(1, "f_b")]
    assert "P@2" in row.line()


def test_dump_carries_known_verdicts_and_marks_the_rest(tmp_path: Path) -> None:
    _, cards = clienteval.load_cards(_published(tmp_path))
    path = tmp_path / "hits.tsv"
    n = clienteval.dump([[("f_a", 0.9), ("f_b", 0.4)]], cards, path, judgments={(1, "f_a"): 1})
    rows = [line.split("\t") for line in path.read_text().splitlines()[1:]]
    assert n == 2
    assert [r[1] for r in rows] == ["f_a", "f_b"]
    assert [r[2] for r in rows] == ["1", "?"]
    assert clienteval.load_judgments(path) == {(1, "f_a"): 1}
