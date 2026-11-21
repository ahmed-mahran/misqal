import json
import sqlite3
from pathlib import Path

from misqal.harness import judge_bird as jb
from misqal.harness.bench import Task


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "s.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, c TEXT)")
        con.execute("CREATE TABLE l (o_id INTEGER, qty INTEGER)")
        con.executemany("INSERT INTO o VALUES (?,?)", [(1, "a"), (2, "b")])
        con.executemany("INSERT INTO l VALUES (?,?)", [(1, 1), (1, 2), (2, 3)])
    return path


def test_prompt_shows_both_golds_and_answer(tmp_path: Path) -> None:
    db = _db(tmp_path)
    t = Task("7", "bird", "s", db, "How many orders?", evidence="orders refers to o",
             gold_sql="SELECT COUNT(DISTINCT o.id) FROM o JOIN l ON l.o_id = o.id",
             original_gold_sql="SELECT COUNT(o.id) FROM o JOIN l ON l.o_id = o.id", category="inflated")
    text = jb.user_prompt(t, "lines repeat orders", "SELECT COUNT(*) FROM l")
    assert "Hint: orders refers to o" in text and "CREATE TABLE o" in text
    assert "Original (over-counting) gold SQL" in text and "Why the original over-counts: lines repeat" in text
    assert text.rstrip().endswith("3")


def test_select_by_outcome(tmp_path: Path) -> None:
    db = _db(tmp_path)
    t = Task("7", "bird", "s", db, "q", gold_sql="g", original_gold_sql="o", category="inflated")
    run = tmp_path / "run"
    run.mkdir()
    rows = [{"task_id": "7", "status": "completed", "pred_sql": "SELECT 1", "set_ex": False, "original_set_ex": False},
            {"task_id": "7", "status": "completed", "pred_sql": "select  1", "set_ex": False, "original_set_ex": False},
            {"task_id": "7", "status": "completed", "pred_sql": "SELECT 2", "set_ex": False, "original_set_ex": True},
            {"task_id": "7", "status": "completed", "pred_sql": "SELECT 3", "set_ex": True, "original_set_ex": False}]
    (run / "results.jsonl").write_text("\n".join(map(json.dumps, rows)))
    other = jb.select([run], {"7": t}, "other", 0, 0)
    assert list(other) == [("7", "select 1")]  # same SQL up to case and spacing: judged once
    cal = jb.select([run], {"7": t}, "calibration", 5, 0)
    assert sorted(v["status"] for v in cal.values()) == ["correct", "trapped"]


def test_verdict_parses() -> None:
    v = jb.Verdict.model_validate_json('{"fanout": true, "cause": "overcount", "reason": "x"}')
    assert v.fanout and v.cause == "overcount"
