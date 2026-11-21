"""Review BIRD-FanOut gold SQL for errors other than fan-out, and fix them.

BIRD-FanOut corrected only the fan-out of each gold query; other mistakes of BIRD's gold (a dropped
filter, a hint formula left out, a broken join) stayed. This judges every harness question's gold:

- "correct": the gold answers the question as written, following the hint;
- "error": the gold gives a wrong answer to a clear question; the judge writes fixed SQL, and for
  fan-out-corrected questions also the trap: the fixed SQL with the original's over-count put back,
  so a fixed question can still be scored as trapped;
- "ambiguous": the question supports several readings and the gold picks a defensible one (the
  judge writes SQL for the other reading);
- "flawed": no answer is right (the question or hint contradicts itself or the data, or the shown
  tables cannot answer it).

Each fix is checked mechanically: it runs, changes the gold's result, and (fan-out-corrected
questions) still counts each entity once while the trap repeats rows. ``--stage 1`` runs cheap
judges on every question; ``--stage 2`` runs strong judges on the questions any stage-1 judge did
not call correct (``--ids``). ``decide`` combines two judges. Where stage 2 leaves a question
disputed, ``--reconcile`` runs one more round in which each strong judge sees both reviews (without
the judges' names) and gives a final one.

    uv run python -m misqal.fanout.gold_review --bench bird-fanout-train \
        --model <judge-a> <judge-b> --out runs/gold-review-train-s1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from misqal.fanout.audit import audit_query
from misqal.harness.agent import known_tables, schema_ddl
from misqal.harness.bench import QueryTimeout, Task, load_tasks, run_query
from misqal.harness.models import DAILY_BUDGET, load_dotenv, make_model, over_budget

VERDICTS = ("correct", "error", "ambiguous", "flawed")

SYSTEM = """You review the gold SQL of a text-to-SQL benchmark (SQLite). The benchmark is about
fan-out: joins or grouping that repeat rows, so a careless query counts or adds the same entity more
than once. How each gold handles repeated rows has already been checked and is right; do not change
it. Your job is to find any other error in the gold: does it answer the question as written?

The hint is part of the question: it maps words to columns, values and formulas. Follow it, unless
it is plainly broken (a value that matches nothing because of a typo, a condition that can never
hold, a formula that contradicts the question's own words).

Verdict:
- "correct": the gold answers the question, following the hint. Presentation does not count:
  column order, aliases, rounding, a fraction for a percentage, an extra or missing helper column,
  ties broken differently.
- "error": the question (with its hint) is clear and the gold answers it wrongly: a missing, extra
  or wrong filter; a wrong column, table or join condition; the hint's formula left out or changed;
  wrong arithmetic (integer division, text compared as numbers); returning a count where the
  question asks for a list, or one arbitrary row where it asks for all or for the highest; and so
  on. Write "fixed_sql": the gold with the error fixed, changing nothing else and keeping its
  handling of repeated rows.
- "ambiguous": the question supports more than one reasonable reading and the gold picks one of
  them. Write "alt_sql" for the other main reading.
- "flawed": no answer can be called right: the question contradicts itself, the hint and the
  question contradict each other with no way to tell which is meant, or the tables shown cannot
  answer it. A broken hint that the question's own words repair (a typo, say) is an "error".

Use only the tables in the schema shown. Results are shown for the first rows only."""

TRAP_RULE = """

This question's gold was corrected for fan-out: the original gold, shown too, counts repeated rows.
If your verdict is "error", also write "trap_sql": your fixed SQL with the original gold's
over-count put back (the same joins and aggregates as the original), so that it differs from
"fixed_sql" only in how repeated rows are counted."""

JSON_INSTRUCTION = (
    "\n\nReply with one JSON object and nothing else, with keys: \"verdict\" ("
    + " | ".join(f'"{v}"' for v in VERDICTS) + "), \"problem\" (string: what is wrong or ambiguous, "
    "two sentences at most; empty if correct), \"fixed_sql\" (string or null), \"trap_sql\" (string "
    "or null), \"alt_sql\" (string or null).")


class Review(BaseModel):
    verdict: Literal[VERDICTS]  # type: ignore[valid-type]
    problem: str = Field(default="")
    fixed_sql: str | None = None
    trap_sql: str | None = None
    alt_sql: str | None = None


def fanout_corrected(task: Task) -> bool:
    return bool(task.original_gold_sql) and task.original_gold_sql != task.gold_sql


@lru_cache(maxsize=None)
def _rows(db: Path, sql: str) -> tuple[list[str], list[tuple]] | str:
    try:
        return run_query(db, sql, 60.0)
    except (sqlite3.Error, QueryTimeout) as e:
        return f"error: {e}"[:300]


def _show(res: tuple[list[str], list[tuple]] | str, limit: int = 10) -> str:
    if isinstance(res, str):
        return res
    cols, rows = res
    out = [" | ".join(cols)] + [" | ".join(map(str, r)) for r in rows[:limit]]
    if len(rows) > limit:
        out.append(f"... ({len(rows)} rows)")
    return "\n".join(out)


def _samples(db: Path, tables: tuple[str, ...], n: int = 3) -> str:
    out = []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        for t in tables:
            cur = con.execute(f'SELECT * FROM "{t}" LIMIT {n}')
            cols = [d[0] for d in cur.description]
            rows = [" | ".join(str(v)[:40] for v in r) for r in cur.fetchall()]
            out.append(f"{t}:\n" + "\n".join([" | ".join(cols), *rows]))
    return "\n\n".join(out)


def user_prompt(task: Task) -> str:
    tables = known_tables(task)
    text = (f"Question: {task.question}\n"
            + (f"Hint: {task.evidence}\n" if task.evidence else "")
            + f"\nSchema:\n{schema_ddl(task.db_path, tables)}\n"
            + f"\nSample rows:\n{_samples(task.db_path, tables)}\n"
            + f"\nGold SQL:\n{task.gold_sql}\n\nGold result:\n{_show(_rows(task.db_path, task.gold_sql))}\n")
    if fanout_corrected(task):
        text += (f"\nOriginal gold SQL (counts repeated rows):\n{task.original_gold_sql}\n\n"
                 f"Original gold result:\n{_show(_rows(task.db_path, task.original_gold_sql))}\n")
    return text


def _set(res: Any) -> frozenset | None:
    return None if isinstance(res, str) else frozenset(res[1])


def checks(task: Task, r: Review) -> dict[str, Any]:
    """Mechanical checks of a review's SQL."""
    out: dict[str, Any] = {}
    gold = _set(_rows(task.db_path, task.gold_sql))
    for name in ("fixed_sql", "trap_sql", "alt_sql"):
        sql = getattr(r, name)
        if not sql:
            continue
        res = _rows(task.db_path, sql)
        key = name.removesuffix("_sql")
        out[f"{key}_runs"] = not isinstance(res, str)
        out[f"{key}_preview"] = _show(res)
        if not isinstance(res, str):
            out[f"{key}_changes_gold"] = _set(res) != gold
    if r.fixed_sql and r.trap_sql and out.get("fixed_runs") and out.get("trap_runs"):
        out["trap_differs"] = _set(_rows(task.db_path, r.trap_sql)) != _set(_rows(task.db_path, r.fixed_sql))
    if fanout_corrected(task) and r.fixed_sql and out.get("fixed_runs"):
        out["fixed_repeats_rows"] = audit_query(task.id, task.db_id, task.db_path, r.fixed_sql).realized
    return out


RECONCILE = """

Two reviewers already reviewed this gold and did not agree (on the verdict, or on a fix that
returns the same result). Their reviews are shown with the results of their SQL. They may both be
wrong. Check them against the question, the hint and the data, then give your own final review."""


def prior_reviews(prior: list[dict[str, Any]]) -> str:
    """Earlier reviews of one question, without the judges' names."""
    out = []
    for name, r in zip("AB", prior):
        part = [f"\nReviewer {name}: {r['verdict']}" + (f". {r['problem']}" if r.get("problem") else "")]
        for key in ("fixed", "trap", "alt"):
            if r.get(f"{key}_sql"):
                part.append(f"{key}_sql:\n{r[f'{key}_sql']}\nResult:\n{r.get(f'{key}_preview', '')}")
                if key == "fixed" and r.get("fixed_repeats_rows"):
                    part.append("(This fix still counts repeated rows.)")
        out.append("\n".join(part))
    return "\n".join(out) + "\n"


def review(model_spec: str, task: Task, budget: float, budget_start: Path | None,
           prior: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"task_id": task.id, "model": model_spec, "category": task.category}
    if over_budget(model_spec, budget, budget_start):
        return {**row, "error": "daily budget reached"}
    model = make_model(model_spec, max_tokens=8192, temperature=None)
    system = (SYSTEM + (TRAP_RULE if fanout_corrected(task) else "") + (RECONCILE if prior else "")
              + JSON_INSTRUCTION)
    user = user_prompt(task) + (prior_reviews(prior) if prior else "")
    messages = [("system", system), ("user", user)]
    error = None
    for _ in range(2):  # one retry on malformed JSON
        try:
            reply = model.invoke(messages)
            text = reply.text if hasattr(reply, "text") else str(reply.content)
            start, end = text.find("{"), text.rfind("}")
            r = Review.model_validate_json(text[start:end + 1])
            break
        except Exception as e:  # provider or parse errors are recorded, not fatal
            error = f"{type(e).__name__}: {e}"[:300]
    else:
        return {**row, "error": error}
    usage = getattr(reply, "usage_metadata", None) or {}
    return {**row, **r.model_dump(), **checks(task, r),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}


def usable_fix(r: dict[str, Any]) -> bool:
    return (r.get("verdict") == "error" and bool(r.get("fixed_runs")) and bool(r.get("fixed_changes_gold"))
            and not r.get("fixed_repeats_rows"))


def degenerate(rows: Any) -> bool:
    """A result that says nothing: no rows, or only 0 and NULL."""
    return not rows or all(v in (0, None) for row in rows for v in row)


def decide(task: Task, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Combine two judges' reviews of one question.

    ok: both correct (an error whose fix returns the gold's result counts as correct). fixed: both say error and their fixes return the same result (set); check: that
    result is empty, 0 or NULL, so agreement proves little (left out until checked on the data). ambiguous
    / flawed: both say so. Anything else is disputed. A fixed fan-out-corrected question keeps a
    trap only when both traps run, differ from the fix, and agree."""
    def verdict(r: dict[str, Any]) -> str | None:
        # An error whose fix returns the gold's result does not change the answer on this data.
        if r.get("verdict") == "error" and r.get("fixed_runs") and r.get("fixed_changes_gold") is False:
            return "correct"
        return r.get("verdict")

    va, vb = verdict(a), verdict(b)
    out: dict[str, Any] = {"task_id": task.id, "verdicts": {a["model"]: va, b["model"]: vb},
                           "problems": {a["model"]: a.get("problem", ""), b["model"]: b.get("problem", "")}}
    if va == vb == "correct":
        return {**out, "status": "ok"}
    if va == vb == "error":
        if not (usable_fix(a) and usable_fix(b)):
            return {**out, "status": "disputed", "why": "a fix does not run, change the gold, or keep "
                    "the fan-out handling"}
        fa, fb = (_set(_rows(task.db_path, r["fixed_sql"])) for r in (a, b))
        if fa != fb:
            return {**out, "status": "disputed", "why": "the fixes return different results"}
        if degenerate(fa):
            # Two fixes can agree on nothing (e.g. the same mistyped literal): check on the data.
            return {**out, "status": "check", "fixed_sql": a["fixed_sql"],
                    "why": "both fixes return no answer (empty, 0 or NULL): check against the data"}
        out.update(status="fixed", fixed_sql=a["fixed_sql"], fixed_by=a["model"])
        if fanout_corrected(task):
            ta, tb = (_rows(task.db_path, r["trap_sql"]) if r.get("trap_sql") else "missing"
                      for r in (a, b))
            if (not isinstance(ta, str) and _set(ta) == _set(tb) and a.get("trap_differs")
                    and b.get("trap_differs")):
                out["trap_sql"] = a["trap_sql"]
        return out
    if va == vb and va in ("ambiguous", "flawed"):
        return {**out, "status": va, "alt_sql": [r["alt_sql"] for r in (a, b) if r.get("alt_sql")]}
    return {**out, "status": "disputed", "why": "the judges disagree"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bench", default="bird-fanout-train")
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--ids", type=Path, help="JSON list of task ids: review only these")
    p.add_argument("--reconcile", type=Path, metavar="REVIEWS",
                   help="earlier reviews (JSON lines): each judge sees both and gives a final review")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--daily-budget", type=float, default=DAILY_BUDGET)
    p.add_argument("--budget-start", type=Path, metavar="FILE")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    load_dotenv()
    tasks = {t.id: t for t in load_tasks(args.bench)}
    if args.ids:
        keep = set(json.loads(args.ids.read_text()))
        tasks = {k: t for k, t in tasks.items() if k in keep}
    prior: dict[str, list[dict[str, Any]]] = {}
    if args.reconcile:
        for r in map(json.loads, args.reconcile.read_text().splitlines()):
            if r["task_id"] in tasks:
                prior.setdefault(r["task_id"], []).append(r)
    done = set()
    if args.out.exists():
        done = {(r["model"], r["task_id"]) for r in map(json.loads, args.out.read_text().splitlines())
                if "verdict" in r}
    jobs = [(m, tid) for tid in sorted(tasks) for m in args.model if (m, tid) not in done]
    print(f"{len(tasks)} questions × {len(args.model)} judges: {len(jobs)} to review", flush=True)

    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.out.open("a") as out:
        for i, row in enumerate(pool.map(
                lambda j: review(j[0], tasks[j[1]], args.daily_budget, args.budget_start,
                                 sorted(prior.get(j[1], []), key=lambda r: r["model"]) or None),
                jobs), 1):
            if "verdict" in row:
                out.write(json.dumps(row) + "\n")
                out.flush()
            else:
                errors += 1
                print("error:", row["task_id"], row["model"], row.get("error"), flush=True)
            if i % 100 == 0:
                print(f"[{i}/{len(jobs)}] errors {errors}", flush=True)
    rows = [json.loads(line) for line in args.out.read_text().splitlines()]
    for m in args.model:
        mine = [r for r in rows if r["model"] == m]
        fixes = [r for r in mine if r["verdict"] == "error"]
        print(m, dict(Counter(r["verdict"] for r in mine)), "usable fixes:",
              sum(usable_fix(r) for r in fixes), "/", len(fixes))


if __name__ == "__main__":
    main()
