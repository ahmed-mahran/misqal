"""Spider 2.0-Lite local (SQLite) loader and result-table scorer.

Gold answers are CSV execution results, not gold SQL. Scoring ports the column-
vector match from the official `evaluation_suite/evaluate.py` (stdlib only; we
do not import anything under `data/`).
"""

from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpiderTask:
    id: str
    db_id: str
    question: str
    evidence: str = ""  # external knowledge document text, if any
    knowledge_file: str | None = None
    family: str = ""
    round: int = 0


def load_spider2_local(root: Path) -> list[SpiderTask]:
    """Load the 135 `local*` tasks from `spider2-lite.jsonl` under `root`."""
    docs = root / "resource" / "documents"
    tasks: list[SpiderTask] = []
    for line in (root / "spider2-lite.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        iid = r["instance_id"]
        if not str(iid).startswith("local"):
            continue
        ek = r.get("external_knowledge") or None
        evidence = ""
        if ek:
            path = docs / ek
            evidence = path.read_text() if path.exists() else ek
        tasks.append(SpiderTask(
            id=str(iid), db_id=r["db"], question=r["question"], evidence=evidence,
            knowledge_file=ek, family=str(iid),
        ))
    return tasks


def spider2_db_path(root: Path, db_id: str) -> Path:
    return root / "resource" / "databases" / "spider2-localdb" / f"{db_id}.sqlite"


def load_eval_standards(root: Path) -> dict[str, dict]:
    path = root / "evaluation_suite" / "gold" / "spider2lite_eval.jsonl"
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        out[item["instance_id"]] = item
    return out


def resolve_gold_paths(root: Path, instance_id: str) -> list[Path]:
    gold_dir = root / "evaluation_suite" / "gold" / "exec_result"
    single = gold_dir / f"{instance_id}.csv"
    if single.exists():
        return [single]
    pat = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    return sorted(p for p in gold_dir.iterdir() if pat.match(p.name))


def _coerce(cell: str) -> Any:
    if cell is None or cell == "":
        return None
    try:
        if re.fullmatch(r"-?\d+", cell):
            return int(cell)
        float(cell)  # accept scientific notation
        if any(c in cell for c in ".eE"):
            return float(cell)
    except ValueError:
        pass
    return cell


def load_csv_table(path: Path) -> list[list[Any]]:
    """Row-major table (header skipped)."""
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    return [[_coerce(c) for c in row] for row in rows[1:]]


def _normalize(value: Any) -> Any:
    return 0 if value is None else value


def _vectors_match(v1: list, v2: list, *, ignore_order: bool, tol: float = 1e-2) -> bool:
    a = [_normalize(x) for x in v1]
    b = [_normalize(x) for x in v2]
    if ignore_order:
        key = lambda x: (x is None, str(x), isinstance(x, (int, float)))
        a, b = sorted(a, key=key), sorted(b, key=key)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x is None and y is None:
            continue
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if not math.isclose(float(x), float(y), abs_tol=tol):
                return False
        elif x != y:
            return False
    return True


def compare_tables(
    pred: list[list[Any]],
    gold: list[list[Any]],
    condition_cols: list[int] | None = None,
    ignore_order: bool = False,
) -> bool:
    """True if every gold column (optionally a subset) matches some pred column."""
    if condition_cols:
        gold = [[row[i] for i in condition_cols] for row in gold]
    if not gold and not pred:
        return True
    if not gold or not pred:
        return False
    t_gold = list(map(list, zip(*gold)))
    t_pred = list(map(list, zip(*pred)))
    return all(
        any(_vectors_match(gv, pv, ignore_order=ignore_order) for pv in t_pred)
        for gv in t_gold
    )


def compare_to_any_gold(
    pred: list[list[Any]],
    gold_tables: list[list[list[Any]]],
    condition_cols: Any = None,
    ignore_order: bool = False,
) -> bool:
    if not gold_tables:
        return False
    # Official multi-gold: condition_cols may be a list-of-lists aligned with golds.
    if condition_cols in (None, [], [[]], [None]):
        per = [None] * len(gold_tables)
    elif len(gold_tables) > 1 and not all(isinstance(x, list) for x in condition_cols):
        per = [condition_cols] * len(gold_tables)
    elif len(gold_tables) > 1 and all(isinstance(x, list) for x in condition_cols):
        # [[1], [0], [0]] or [0, 1] — only treat as per-gold if elements are lists
        if condition_cols and isinstance(condition_cols[0], list):
            per = list(condition_cols) + [None] * max(0, len(gold_tables) - len(condition_cols))
        else:
            per = [condition_cols] * len(gold_tables)
    else:
        per = [condition_cols] * len(gold_tables)

    for i, gold in enumerate(gold_tables):
        cols = per[i]
        if cols is not None and not isinstance(cols, list):
            cols = [cols]
        if compare_tables(pred, gold, cols, ignore_order):
            return True
    return False


def run_sql_to_table(db: Path, sql: str) -> list[list[Any]] | None:
    if not sql or not sql.strip():
        return None
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            cur = con.execute(sql)
            return [list(row) for row in cur.fetchall()]
    except sqlite3.Error:
        return None


def spider2_match(
    root: Path,
    db: Path,
    instance_id: str,
    pred_sql: str | None,
    standards: dict[str, dict] | None = None,
) -> bool:
    """Execute `pred_sql` and score against any gold CSV for `instance_id`."""
    pred = run_sql_to_table(db, pred_sql or "")
    if pred is None:
        return False
    golds = [load_csv_table(p) for p in resolve_gold_paths(root, instance_id)]
    if not golds:
        return False
    std = (standards or load_eval_standards(root)).get(instance_id, {})
    return compare_to_any_gold(
        pred, golds, std.get("condition_cols"), bool(std.get("ignore_order", False)),
    )


def spider2_match_table(
    root: Path,
    instance_id: str,
    pred: list[list[Any]],
    standards: dict[str, dict] | None = None,
) -> bool:
    golds = [load_csv_table(p) for p in resolve_gold_paths(root, instance_id)]
    if not golds:
        return False
    std = (standards or load_eval_standards(root)).get(instance_id, {})
    return compare_to_any_gold(
        pred, golds, std.get("condition_cols"), bool(std.get("ignore_order", False)),
    )
