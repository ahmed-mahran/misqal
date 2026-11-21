"""LLM-judged causes of wrong trap-suite answers that are not the trap ("other" outcomes).

Each distinct (question, submitted SQL) among the "other" rows of the given run folders is judged
once, against the question, its written trap explanation and solution, the schema, and both
results. The cause is one of:

- ``data_model_overcount``: assumed a relationship the data does not have, so rows are counted or
  added more than once (a fan-out the trap check did not match);
- ``data_model_undercount``: assumed rows or values are duplicates when the data says they are
  distinct (de-duplicating real rows, DISTINCT on a measure, joining on a non-key);
- ``wrong_grain``: right relationships, but the result is grouped or broken down at another grain
  than asked;
- ``misread_question``: wrong filter, measure, ranking or columns, unrelated to the data model;
- ``sql_mistake``: the intent is right but the SQL is wrong (rounding, NULL handling, ties, ...);
- ``acceptable``: a defensible answer to the question as written.

    uv run python -m misqal.harness.judge_other runs/<traps-run> ... \
        --model <judge> --out runs/judge-other.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from misqal.fanout.label import _preview
from misqal.harness.bench import BENCH_ROOTS, TRAP_QUESTIONS, Task, load_traps
from misqal.harness.models import DAILY_BUDGET, load_dotenv, make_model, over_budget

CAUSES = ("data_model_overcount", "data_model_undercount", "wrong_grain", "misread_question",
          "sql_mistake", "acceptable")

SYSTEM = """You review wrong answers of a text-to-SQL benchmark (SQLite) about counting rows the
right number of times. Each question comes with a written explanation of the trap it sets and how
it should be solved, the gold SQL, and an answer that neither matches the gold nor falls into the
trap as written. Decide the main cause of the difference:
- "data_model_overcount": the answer assumes a relationship the data does not have, so rows are
  counted or added more than once;
- "data_model_undercount": the answer treats rows or values as duplicates when the data says they
  are distinct (de-duplicating real rows, DISTINCT on a measure, joining on a column that is not a
  key), so rows or amounts are lost;
- "wrong_grain": relationships are handled right, but the result is grouped or broken down at
  another grain than the question asks;
- "misread_question": a wrong filter, measure, ranking, or set of columns, unrelated to how the
  tables relate;
- "sql_mistake": the intent is right but the SQL does not express it (rounding, NULL handling,
  ties, join type, ...);
- "acceptable": a defensible answer to the question as written, which the gold does not accept.
Base the decision on the SQL and the results, not on the answer's wording."""

JSON_INSTRUCTION = (
    "\n\nReply with one JSON object and nothing else, with keys: \"cause\" ("
    + " | ".join(f'"{c}"' for c in CAUSES) + "), \"reason\" (string, two sentences at most).")


class Cause(BaseModel):
    cause: Literal[CAUSES]  # type: ignore[valid-type]
    reason: str = Field(default="")


def _key(task_id: str, sql: str) -> tuple[str, str]:
    return task_id, " ".join(sql.lower().split())


def _schema(db: Path) -> str:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        return "\n".join(s + ";" for (s,) in con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"))


def user_prompt(task: Task, doc: dict[str, Any], sql: str) -> str:
    _, gold = _preview(task.db_path, task.gold_sql)
    _, pred = _preview(task.db_path, sql)
    return (f"Question: {task.question}\n\nTrap: {doc['trap_explanation']}\n\n"
            f"Solution: {doc['solution']}\n"
            + (f"\nVague wording: {doc['vague_reading']}\n" if doc.get("vague_reading") else "")
            + f"\nSchema:\n{_schema(task.db_path)}\n"
            + f"\nGold SQL:\n{task.gold_sql}\n\nGold result:\n{gold}\n"
            + f"\nAnswer SQL:\n{sql}\n\nAnswer result:\n{pred}\n")


def judge(model_spec: str, task: Task, doc: dict[str, Any], sql: str, budget: float,
          budget_start: Path | None) -> dict[str, Any]:
    row: dict[str, Any] = {"task_id": task.id, "sql": sql, "model": model_spec}
    if over_budget(model_spec, budget, budget_start):
        return {**row, "error": "daily budget reached"}
    model = make_model(model_spec, max_tokens=2048, temperature=None)
    messages = [("system", SYSTEM + JSON_INSTRUCTION), ("user", user_prompt(task, doc, sql))]
    error = None
    for _ in range(2):  # one retry on malformed JSON
        try:
            reply = model.invoke(messages)
            text = reply.text if hasattr(reply, "text") else str(reply.content)
            start, end = text.find("{"), text.rfind("}")
            return {**row, **Cause.model_validate_json(text[start:end + 1]).model_dump()}
        except Exception as e:  # provider or parse errors are recorded, not fatal
            error = f"{type(e).__name__}: {e}"[:300]
    return {**row, "error": error}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run folders with results.jsonl")
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--daily-budget", type=float, default=DAILY_BUDGET)
    p.add_argument("--budget-start", type=Path, metavar="FILE")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    load_dotenv()
    tasks = {t.id: t for t in load_traps(BENCH_ROOTS["traps"])}
    docs = {q["id"]: q for q in json.loads(TRAP_QUESTIONS.read_text())}
    answers: dict[tuple[str, str], str] = {}
    for run in args.runs:
        for line in (run / "results.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("outcome") == "other" and r.get("pred_sql"):
                answers.setdefault(_key(r["task_id"], r["pred_sql"]), r["pred_sql"])
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            if "cause" in r:
                done.add((r["model"], *_key(r["task_id"], r["sql"])))
    jobs = [(m, k) for k in answers for m in args.model if (m, *k) not in done]
    print(f"{len(answers)} distinct answers × {len(args.model)} judges: {len(jobs)} to judge")
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.out.open("a") as out:
        for row in pool.map(lambda j: judge(j[0], tasks[j[1][0]], docs[j[1][0]], answers[j[1]],
                                            args.daily_budget, args.budget_start), jobs):
            if "cause" in row:
                out.write(json.dumps(row) + "\n")
                out.flush()
    rows = [json.loads(line) for line in args.out.read_text().splitlines()]
    for m in args.model:
        print(m, dict(Counter(r["cause"] for r in rows if r["model"] == m).most_common()))


if __name__ == "__main__":
    main()
