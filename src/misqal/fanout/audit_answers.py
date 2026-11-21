"""Run the fan-out audit over the SQL that models submitted, as an offline stand-in for a verifier.

For each distinct (question, submitted SQL) the audit says whether the query realizes fan-out
(definite), counts with COUNT(*) where only some joined tables repeat (ambiguous), or counts a
column whose values repeat (repeated values). Reported against the answer's outcome: how many
answers that reproduce the trapped gold the audit would flag (what a verifier could catch), and how
many correct answers it would flag (false alarms).

    uv run python -m misqal.fanout.audit_answers --bench bird-fanout-20251106 \
        --runs runs/<run> ... --out runs/audit-answers-fanout.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from misqal.fanout.audit import audit_query
from misqal.harness.bench import load_tasks


def _key(task_id: str, sql: str) -> tuple[str, str]:
    return task_id, " ".join(sql.lower().split())


def flag(audit: dict) -> str:
    if audit.get("parse_error"):
        return "unparsed"
    if audit["realized"]:
        return "definite"
    if audit.get("realized_ambiguous"):
        return "ambiguous"
    return "repeated_values" if audit.get("repeated_values") else "none"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    tasks = {t.id: t for t in load_tasks(args.bench, args.root)}
    rows = [r for run in args.runs for r in map(json.loads, (run / "results.jsonl").read_text().splitlines())
            if r["status"] != "error" and r.get("sql_source") == "submitted" and r.get("pred_sql")
            and r["task_id"] in tasks]
    cache: dict[tuple[str, str], dict] = {}
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            a = json.loads(line)
            cache[_key(a["id"], a["sql"])] = a
    todo = {_key(r["task_id"], r["pred_sql"]): r for r in rows} .keys() - cache.keys()
    sql_of = {_key(r["task_id"], r["pred_sql"]): r["pred_sql"] for r in rows}

    def run(k: tuple[str, str]) -> dict:
        t = tasks[k[0]]
        return audit_query(t.id, t.db_id, t.db_path, sql_of[k]).as_dict()

    with ThreadPoolExecutor(args.workers) as pool, args.out.open("a") as out:
        for k, a in zip(list(todo), pool.map(run, list(todo))):
            cache[k] = a
            out.write(json.dumps(a, default=str) + "\n")
    print(f"{len(rows)} submitted answers, {len(sql_of)} distinct, {len(todo)} audited now")

    groups: dict[str, list[dict]] = {}
    for r in rows:
        a = cache[_key(r["task_id"], r["pred_sql"])]
        cat = r.get("category") or "all"
        right = bool(r.get("set_ex"))
        groups.setdefault(f"{cat}: reproduces trapped gold", []).append(a) if r.get("original_set_ex") else None
        groups.setdefault(f"{cat}: {'correct' if right else 'wrong'}", []).append(a)
    for g in sorted(groups):
        c = Counter(flag(a) for a in groups[g])
        n = len(groups[g])
        print(f"{g:42s} n={n:5d}  " + "  ".join(
            f"{f} {c[f]} ({100 * c[f] / n:.0f}%)" for f in ("definite", "ambiguous", "repeated_values",
                                                             "none", "unparsed") if c[f]))


if __name__ == "__main__":
    main()
