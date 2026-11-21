"""Lenient re-scoring of saved runs: the same answer up to formatting.

The strict scores (`set_ex`, `multiset_ex`, trap `outcome`) compare rows exactly, so an answer that
rounds a number differently or orders its columns differently counts as wrong. Here each row is
normalized before comparing: numbers rounded to 2 decimals (integers and floats compare equal) and
the values of a row sorted, so column order does not matter. Extra or missing columns still count as
wrong. Rows compare as a multiset on the trap suite (as its strict score) and as a set on BIRD (as
BIRD's own EX). Every distinct predicted query is run once; results go to a separate file and the saved runs
are not changed.

    uv run python -m misqal.harness.lenient --bench bird-fanout-train --runs runs/<train-run> ... \
        --out runs/lenient-train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from misqal.harness.bench import QueryTimeout, load_tasks, run_query


def _norm_value(v: Any) -> str:
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, (int, float)):
        return f"{round(float(v), 2):.2f}"
    return "" if v is None else str(v).strip()


def normalize(rows: list[tuple], as_set: bool = False) -> Counter | frozenset:
    normed = (tuple(sorted(_norm_value(v) for v in r)) for r in rows)
    return frozenset(normed) if as_set else Counter(normed)


def _result(db: Path, sql: str | None, timeout: float, as_set: bool = False) -> Counter | frozenset | None:
    if not sql:
        return None
    try:
        return normalize(run_query(db, sql, timeout)[1], as_set)
    except (sqlite3.Error, QueryTimeout):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    tasks = {t.id: t for t in load_tasks(args.bench)}
    as_set = args.bench != "traps"
    rows = [r for run in args.runs for r in map(json.loads, (run / "results.jsonl").read_text().splitlines())
            if r["status"] != "error" and r["task_id"] in tasks]
    refs: dict[str, dict[str, Any]] = {}

    def ref(tid: str) -> dict[str, Any]:
        t = tasks[tid]
        golds = [g for g in (t.gold_sql, *t.alt_gold_sqls) if g]
        return {"gold": [_result(t.db_path, g, args.timeout, as_set) for g in golds],
                "trap": _result(t.db_path, t.naive_sql or (t.original_gold_sql if t.original_gold_sql != t.gold_sql
                                                           else None), args.timeout, as_set)}

    with ThreadPoolExecutor(args.workers) as pool:
        ids = sorted({r["task_id"] for r in rows})
        refs = dict(zip(ids, pool.map(ref, ids)))
        keys = sorted({(r["task_id"], " ".join((r.get("pred_sql") or "").split())) for r in rows})
        preds = dict(zip(keys, pool.map(lambda k: _result(tasks[k[0]].db_path, k[1], args.timeout, as_set), keys)))

    with args.out.open("w") as out:
        for r in rows:
            pred = preds[(r["task_id"], " ".join((r.get("pred_sql") or "").split()))]
            ref_ = refs[r["task_id"]]
            ok = pred is not None and any(g is not None and pred == g for g in ref_["gold"])
            trap = pred is not None and ref_["trap"] is not None and pred == ref_["trap"]
            out.write(json.dumps({"model": r["model"], "arm": r["arm"], "task_id": r["task_id"],
                                  "rep": r.get("rep", 0), "lenient_ex": ok, "lenient_trap": trap}) + "\n")
    print(f"{len(rows)} runs, {len(keys)} distinct queries -> {args.out}")


if __name__ == "__main__":
    main()
