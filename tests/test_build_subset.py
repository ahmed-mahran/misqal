from __future__ import annotations

import sqlite3
from pathlib import Path

from misqal.fanout.build_subset import build, judge_name
from misqal.harness.bench import Task


def test_judge_name_keeps_model_family_only() -> None:
    assert judge_name("mygw:team-claude-opus-5-5") == "claude-opus-5-5"
    assert judge_name("mygw:openai/gpt-5.5") == "gpt-5.5"
    assert judge_name("openai:gpt-4.1-mini") == "gpt-4.1-mini"


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "shop.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript("""
            CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer TEXT);
            CREATE TABLE lines (line_id INTEGER PRIMARY KEY, order_id INTEGER);
            INSERT INTO orders VALUES (1, 'ann'), (2, 'bob');
            INSERT INTO lines VALUES (1, 1), (2, 1), (3, 1), (4, 2);
        """)
    return path


GOLD = "SELECT COUNT(o.order_id) FROM orders o JOIN lines l ON o.order_id = l.order_id"
FIX = "SELECT COUNT(DISTINCT o.order_id) FROM orders o JOIN lines l ON o.order_id = l.order_id"
OTHER_FIX = "SELECT COUNT(*) FROM orders"


def _label(qid: str, model: str, verdict: str, sql: str | None = None) -> dict:
    return {"id": qid, "model": model, "verdict": verdict, "reason": "r", "corrected_sql": sql}


def test_categories_and_adjudication(tmp_path: Path) -> None:
    db = _db(tmp_path)
    tasks = {q: Task(q, "bird", "shop", db, "How many orders?", gold_sql=GOLD)
             for q in ("1", "2", "3", "4", "5")}
    labels = [
        _label("1", "gw:claude-x", "inflated", FIX), _label("1", "gw:gpt-x", "inflated", OTHER_FIX),
        _label("2", "gw:claude-x", "intended"), _label("2", "gw:gpt-x", "intended"),
        _label("3", "gw:claude-x", "inflated", FIX), _label("3", "gw:gpt-x", "unclear"),
        _label("4", "gw:claude-x", "inflated", GOLD), _label("4", "gw:gpt-x", "inflated", GOLD),
        _label("5", "gw:claude-x", "inflated", FIX), _label("5", "gw:gpt-x", "inflated", "SELECT 7"),
    ]
    rows = {r["id"]: r for r in build(tasks, labels, {"choose_correction": {"5": "gpt-x"},
                                                      "other_error": ["5"]})}
    # same result from both corrections (2 orders) -> inflated, 4 joined rows / 2 = 2.0
    assert rows["1"]["category"] == "inflated" and rows["1"]["gold_to_corrected"] == 2.0
    assert rows["2"]["category"] == "intended"
    assert rows["3"]["category"] == "contested"
    assert rows["4"]["category"] == "benign"  # corrections reproduce the gold answer
    assert rows["5"]["corrected_sql"] == "SELECT 7" and rows["5"]["other_error"]
    assert set(rows["1"]["verdicts"]) == {"claude-x", "gpt-x"}
