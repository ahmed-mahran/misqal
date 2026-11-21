"""LLM-judged fan-out in wrong BIRD-FanOut answers that do not reproduce the distorted gold.

On BIRD-FanOut, a run is scored *trapped* only when its result equals BIRD's original (distorted)
gold. An answer can over-count in the same way and also differ for another reason (a filter, a
column); it then counts as "other". This judges each distinct (question, submitted SQL) of those
answers: does it count or add some entity more than once because rows repeat (fan-out), and what
is the main cause of the difference from the corrected gold.

The judge sees the question and its hint, the schema of the question's tables, the corrected gold
and BIRD's original gold with their results and the benchmark's note on why the original
over-counts, and the answer with its result. It does not see the static audit's flag, so the two
stay independent checks.

``--select other`` judges the "other" answers; ``--select calibration`` judges a seeded sample of
answers with known status (trapped answers: over-count; correct answers: not) to measure the
judge; ``--ids FILE`` restricts to listed keys (JSON lines with task_id and sql), for a second
judge on a sample.

    uv run python -m misqal.harness.judge_bird runs/<train-run> ... \
        --model <judge> --select other --out runs/judge-bird-train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from misqal.fanout.label import _preview
from misqal.harness.agent import known_tables, schema_ddl
from misqal.harness.bench import Task, load_tasks
from misqal.harness.models import DAILY_BUDGET, load_dotenv, make_model, over_budget
from misqal.paths import shipped

CAUSES = ("overcount", "undercount", "wrong_grain", "misread_question", "sql_mistake", "acceptable")

SYSTEM = """You review wrong answers of a text-to-SQL benchmark (SQLite). Every question in it is
one where joining or grouping tables repeats rows, so a careless query counts or adds the same
entity more than once (fan-out). The benchmark's original gold SQL made that mistake; the corrected
gold does not. The answer you judge matches neither result. Decide:

1. "fanout": true if the answer counts, adds, averages or ranks some entity more than once because
   rows repeat (through a join, or through repeated rows in one table) where the question means
   each entity once; false otherwise. Judge the answer's own SQL and result, not its similarity to
   the original gold: an answer can over-count through a different path, or can avoid the original
   over-count and be wrong for another reason.
2. "cause": the main reason the answer differs from the corrected gold:
   - "overcount": rows or amounts counted more than once (fan-out);
   - "undercount": rows or values treated as duplicates when they are distinct, so real rows or
     amounts are lost;
   - "wrong_grain": relationships handled right, but grouped or broken down at another grain;
   - "misread_question": a wrong filter, measure, ranking or set of columns, unrelated to how
     rows repeat;
   - "sql_mistake": the intent is right but the SQL does not express it (rounding, NULLs, ties,
     join type, ...);
   - "acceptable": a defensible answer to the question as written that the gold does not accept.
If the answer both over-counts and has another error, "fanout" is true and "cause" is the error
that matters more for the result."""

JSON_INSTRUCTION = (
    "\n\nReply with one JSON object and nothing else, with keys: \"fanout\" (true or false), "
    "\"cause\" (" + " | ".join(f'"{c}"' for c in CAUSES) + "), \"reason\" (string, two sentences "
    "at most).")


class Verdict(BaseModel):
    fanout: bool
    cause: Literal[CAUSES]  # type: ignore[valid-type]
    reason: str = Field(default="")


def _key(task_id: str, sql: str) -> tuple[str, str]:
    return task_id, " ".join(sql.lower().split())


def _notes(bench: str) -> dict[str, str]:
    """The benchmark's note on why each original gold over-counts (first judge's reason)."""
    out = {}
    for row in map(json.loads, shipped(f"benchmarks/{bench}/subset.jsonl").read_text().splitlines()):
        reasons = row.get("reasons") or {}
        if isinstance(reasons, str):
            reasons = json.loads(reasons)
        out[str(row["id"])] = next(iter(reasons.values()), "") if reasons else ""
    return out


@lru_cache(maxsize=None)
def _cached_preview(db: Path, sql: str) -> str:
    """Results of the reference queries, computed once per question."""
    return _preview(db, sql)[1]


def user_prompt(task: Task, note: str, sql: str) -> str:
    gold = _cached_preview(task.db_path, task.gold_sql)
    _, pred = _preview(task.db_path, sql)
    original = ""
    if task.original_gold_sql and task.original_gold_sql != task.gold_sql:
        original = (f"\nOriginal (over-counting) gold SQL:\n{task.original_gold_sql}\n\n"
                    f"Original gold result:\n{_cached_preview(task.db_path, task.original_gold_sql)}\n"
                    + (f"\nWhy the original over-counts: {note}\n" if note else ""))
    return (f"Question: {task.question}\n"
            + (f"Hint: {task.evidence}\n" if task.evidence else "")
            + f"\nSchema:\n{schema_ddl(task.db_path, known_tables(task))}\n"
            + f"\nCorrected gold SQL:\n{task.gold_sql}\n\nCorrected gold result:\n{gold}\n"
            + original
            + f"\nAnswer SQL:\n{sql}\n\nAnswer result:\n{pred}\n")


def judge(model_spec: str, task: Task, note: str, sql: str, budget: float,
          budget_start: Path | None) -> dict[str, Any]:
    row: dict[str, Any] = {"task_id": task.id, "sql": sql, "model": model_spec}
    if over_budget(model_spec, budget, budget_start):
        return {**row, "error": "daily budget reached"}
    model = make_model(model_spec, max_tokens=4096, temperature=None)
    messages = [("system", SYSTEM + JSON_INSTRUCTION), ("user", user_prompt(task, note, sql))]
    error = None
    for _ in range(2):  # one retry on malformed JSON
        try:
            reply = model.invoke(messages)
            text = reply.text if hasattr(reply, "text") else str(reply.content)
            start, end = text.find("{"), text.rfind("}")
            usage = getattr(reply, "usage_metadata", None) or {}
            return {**row, **Verdict.model_validate_json(text[start:end + 1]).model_dump(),
                    "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}
        except Exception as e:  # provider or parse errors are recorded, not fatal
            error = f"{type(e).__name__}: {e}"[:300]
    return {**row, "error": error}


def outcome(r: dict) -> str:
    if r.get("set_ex"):
        return "correct"
    return "trapped" if r.get("original_set_ex") else "other"


def select(runs: list[Path], tasks: dict[str, Task], how: str, n: int, seed: int
           ) -> dict[tuple[str, str], dict[str, str]]:
    """Distinct answers to judge, with their known status."""
    found: dict[str, dict[tuple[str, str], str]] = {"other": {}, "trapped": {}, "correct": {}}
    for run in runs:
        for line in (run / "results.jsonl").read_text().splitlines():
            r = json.loads(line)
            t = tasks.get(r["task_id"])
            if (r["status"] == "error" or not r.get("pred_sql") or t is None
                    or t.category != "inflated"):
                continue
            found[outcome(r)].setdefault(_key(r["task_id"], r["pred_sql"]), r["pred_sql"])
    if how == "other":
        return {k: {"sql": s, "status": "other"} for k, s in found["other"].items()}
    rng = random.Random(seed)
    out = {}
    for status in ("trapped", "correct"):
        keys = sorted(found[status])
        for k in rng.sample(keys, min(n, len(keys))):
            out[k] = {"sql": found[status][k], "status": status}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run folders with results.jsonl")
    p.add_argument("--bench", default="bird-fanout-train")
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--select", choices=["other", "calibration"], default="other")
    p.add_argument("--calibration-size", type=int, default=300, help="per known status")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ids", type=Path, help="JSON lines with task_id and sql: judge only these")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--daily-budget", type=float, default=DAILY_BUDGET)
    p.add_argument("--budget-start", type=Path, metavar="FILE")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    load_dotenv()
    tasks = {t.id: t for t in load_tasks(args.bench)}
    notes = _notes(args.bench)
    answers = select(args.runs, tasks, args.select, args.calibration_size, args.seed)
    if args.ids:
        keep = {_key(r["task_id"], r["sql"]) for r in map(json.loads, args.ids.read_text().splitlines())}
        answers = {k: v for k, v in answers.items() if k in keep}
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            if "fanout" in r:
                done.add((r["model"], *_key(r["task_id"], r["sql"])))
    jobs = [(m, k) for k in sorted(answers) for m in args.model if (m, *k) not in done]
    print(f"{len(answers)} distinct answers × {len(args.model)} judges: {len(jobs)} to judge")

    def run(job: tuple[str, tuple[str, str]]) -> dict[str, Any]:
        m, k = job
        row = judge(m, tasks[k[0]], notes.get(k[0], ""), answers[k]["sql"], args.daily_budget,
                    args.budget_start)
        return {**row, "status": answers[k]["status"]}

    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.out.open("a") as out:
        for i, row in enumerate(pool.map(run, jobs), 1):
            if "fanout" in row:
                out.write(json.dumps(row) + "\n")
                out.flush()
            else:
                errors += 1
            if i % 200 == 0:
                print(f"[{i}/{len(jobs)}] errors {errors}", flush=True)
    rows = [json.loads(line) for line in args.out.read_text().splitlines()]
    print(f"errors this run: {errors}")
    for m in args.model:
        mine = [r for r in rows if r["model"] == m]
        for status in sorted({r["status"] for r in mine}):
            rs = [r for r in mine if r["status"] == status]
            print(m, status, len(rs), "fan-out:", sum(r["fanout"] for r in rs),
                  dict(Counter(r["cause"] for r in rs).most_common()))


if __name__ == "__main__":
    main()
