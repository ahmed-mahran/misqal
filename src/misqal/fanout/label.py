"""LLM-judged intent labels and corrected SQL for fan-out-exposed benchmark questions.

For each question whose gold SQL the audit flags (rows actually repeat on the data: realized join
fan-out, an ambiguous ``COUNT(*)``, or a counted column whose values repeat), a judge decides whether the question means distinct entities of the repeated table (the gold
answer is **inflated**) or deliberately counts the joined rows (**intended**), and writes corrected
SQL for inflated cases. Corrected SQL is then checked mechanically: it must run, the audit must no
longer flag it, and its result is compared with the gold result.

    uv run python -m misqal.fanout.label --audit runs/bird_audit.jsonl \
        --model <judge-a> <judge-b> --out runs/bird_labels.jsonl
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

from misqal.fanout.audit import audit_query
from misqal.harness.bench import BENCH_ROOTS, QueryTimeout, Task, load_bird, run_query
from misqal.harness.models import load_dotenv, make_model, over_budget

SYSTEM = """You review gold SQL answers of a text-to-SQL benchmark (SQLite) for fan-out.

Fan-out: when a table is joined to another table on a key that is not unique there, each of its
rows is repeated once per matching row. Duplicate-sensitive aggregates (COUNT, SUM, AVG without
DISTINCT) over a repeated table then count or add the same entity several times, and ORDER BY ...
LIMIT N can return the same entity in several of the N slots.

Repeated rows are not always a mistake. Counting rows through a parent key is often deliberate:
"how many votes does the post have" counts vote rows, even if written COUNT(posts.Id). Decide from
the question (and hint) what is being counted or aggregated:
- "inflated": the question means distinct entities of the repeated table (e.g. "how many
  patients"), or a per-entity value, so the repetition changes the intended answer;
- "intended": the question means the joined rows themselves (e.g. "how many lab tests",
  "number of votes"), so the repetition is the intended semantics;
- "unclear": the question genuinely supports both readings.

Also say whether the intended reading is resolvable from the data model: the thing the question
counts or aggregates has its own identity in the schema (a table or column where it is unique) and
the data shows which rows repeat it, so an analyst who checks keys and one-to-many relations would
pick the intended reading without asking the user. Not resolvable: both readings stay plausible
even with full knowledge of the keys (e.g. the hint prescribes a row-level formula).

If "inflated", write corrected SQLite SQL that answers the question as intended, changing only what
the fan-out requires (COUNT(DISTINCT ...), aggregating before the join, DISTINCT entities before
LIMIT, ...). Keep the same output columns and filters; do not fix unrelated issues."""


class Judgement(BaseModel):
    counted_entity: str = Field(description="What the question counts or aggregates over, in words.")
    verdict: Literal["inflated", "intended", "unclear"]
    reason: str = Field(description="One or two sentences.")
    resolvable: bool = Field(description="The intended reading follows from the data model.")
    resolvable_reason: str = Field(default="", description="One sentence.")
    corrected_sql: str | None = Field(
        default=None, description="Corrected SQLite query when the verdict is 'inflated'.")


def _schema_for(db: Path, tables: set[str]) -> str:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        rows = con.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
    return "\n".join(sql + ";" for name, sql in rows if name.lower() in tables and sql)


def _preview(db: Path, sql: str, limit: int = 10) -> tuple[list[tuple] | None, str]:
    try:
        cols, rows = run_query(db, sql, 60.0)
    except (sqlite3.Error, QueryTimeout) as e:
        return None, f"error: {e}"
    shown = [" | ".join(cols)] + [" | ".join(map(str, r)) for r in rows[:limit]]
    if len(rows) > limit:
        shown.append(f"... ({len(rows)} rows)")
    return rows, "\n".join(shown)


def facts(audit: dict[str, Any]) -> str:
    """The audit's findings, stated neutrally."""
    lines = []
    for b in audit["blocks"]:
        if not (b.get("realized_multiplied") and (any(g.get("realized") for g in b["aggs"])
                                                  or b.get("top_n_realized"))):
            continue
        t = b["tables"]
        for e in b["edges"]:
            for near, far, far_cols, far_unique in (
                    (e["left"], e["right"], e["right_cols"], e["right_unique"]),
                    (e["right"], e["left"], e["left_cols"], e["left_unique"])):
                if far_unique is False:
                    lines.append(f"- A row of `{t.get(near, near)}` ({near}) can match several rows "
                                 f"of `{t.get(far, far)}` ({far}) on {', '.join(far_cols)}.")
        rep = ", ".join(f"`{t.get(a, a)}` ({a})" for a in b["realized_multiplied"])
        lines.append(f"- With the query's joins and filters, rows of {rep} repeat on this data.")
        for g in b["aggs"]:
            if g.get("realized") and g.get("ambiguous"):
                lines.append(f"- `{g['sql']}` counts the joined rows: rows of the tables above "
                             "repeat in it, the other tables' rows do not.")
            elif g.get("realized"):
                lines.append(f"- `{g['sql']}` therefore sees repeated rows.")
        if b.get("top_n_realized"):
            lines.append("- ORDER BY ... LIMIT can return the same output row more than once.")
    for b in audit["blocks"]:
        for g in b["aggs"]:
            if g.get("repeated_values") and not g.get("realized"):
                lines.append(f"- In `{g['sql']}` the counted column takes the same value on several "
                             "of the rows counted, so it counts rows, not distinct values.")
    return "\n".join(lines)


def user_prompt(task: Task, audit: dict[str, Any]) -> str:
    tables = {v.lower() for b in audit["blocks"] for v in b["tables"].values()}
    _, gold_preview = _preview(task.db_path, task.gold_sql or "")
    return (f"Question: {task.question}\n"
            + (f"Hint: {task.evidence}\n" if task.evidence else "")
            + f"\nSchema:\n{_schema_for(task.db_path, tables)}\n"
            + f"\nGold SQL:\n{task.gold_sql}\n\nGold result:\n{gold_preview}\n"
            + f"\nFacts from the data:\n{facts(audit)}\n")


def check_correction(task: Task, sql: str | None) -> dict[str, Any]:
    if not sql:
        return {}
    rows, preview = _preview(task.db_path, sql)
    out: dict[str, Any] = {"corrected_runs": rows is not None, "corrected_preview": preview}
    if rows is None:
        return out
    gold, _ = _preview(task.db_path, task.gold_sql or "")
    qa = audit_query(task.id, task.db_id, task.db_path, sql)
    out.update(corrected_still_realized=qa.realized,
               changed_answer=gold is None or Counter(rows) != Counter(gold))
    return out


JSON_INSTRUCTION = (
    "\n\nReply with one JSON object and nothing else, with keys: \"counted_entity\" (string), "
    "\"verdict\" (\"inflated\" | \"intended\" | \"unclear\"), \"reason\" (string), "
    "\"resolvable\" (true | false), \"resolvable_reason\" (string), "
    "\"corrected_sql\" (string, or null unless the verdict is \"inflated\").")


def _parse(text: str) -> Judgement:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    return Judgement.model_validate_json(text[start:end + 1])


def judge(model_spec: str, task: Task, audit: dict[str, Any],
          temperature: float | None = None) -> dict[str, Any]:
    """JSON in the reply rather than forced tool calls: works the same on every provider."""
    row: dict[str, Any] = {"id": task.id, "db_id": task.db_id, "model": model_spec,
                           "temperature": temperature}
    if over_budget(model_spec):
        return {**row, "error": "daily budget reached"}
    model = make_model(model_spec, max_tokens=4096, temperature=temperature)
    messages = [("system", SYSTEM + JSON_INSTRUCTION), ("user", user_prompt(task, audit))]
    error = None
    for _ in range(2):  # one retry on malformed JSON
        try:
            reply = model.invoke(messages)
            j = _parse(reply.text if hasattr(reply, "text") else str(reply.content))
            break
        except Exception as e:  # provider or parse errors are recorded, not fatal
            error = f"{type(e).__name__}: {e}"[:300]
    else:
        return {**row, "error": error}
    row.update(j.model_dump())
    usage = getattr(reply, "usage_metadata", None) or {}
    row.update(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
    if j.verdict == "inflated":
        row.update(check_correction(task, j.corrected_sql))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--root", type=Path, default=BENCH_ROOTS["bird"])
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--temperature", type=float, help="default: provider default")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    load_dotenv()
    audits = {a["id"]: a for a in map(json.loads, args.audit.read_text().splitlines())
              if a["realized"] or a.get("realized_ambiguous") or a.get("repeated_values")}
    tasks = [t for t in load_bird(args.root) if t.id in audits]
    jobs = [(m, t) for t in tasks for m in args.model]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(
            lambda job: judge(job[0], job[1], audits[job[1].id], args.temperature), jobs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    for m in args.model:
        mine = [r for r in rows if r["model"] == m]
        verdicts = Counter(r.get("verdict", "error") for r in mine)
        fixed = [r for r in mine if r.get("verdict") == "inflated"]
        print(f"{m}: {dict(verdicts)}; corrections run {sum(bool(r.get('corrected_runs')) for r in fixed)}"
              f"/{len(fixed)}, clean {sum(r.get('corrected_still_realized') is False for r in fixed)}"
              f", changed answer {sum(bool(r.get('changed_answer')) for r in fixed)}")
    if len(args.model) == 2:
        by_id: dict[str, dict[str, str]] = {}
        for r in rows:
            by_id.setdefault(r["id"], {})[r["model"]] = r.get("verdict", "error")
        agree = sum(len(set(v.values())) == 1 for v in by_id.values())
        print(f"agreement: {agree}/{len(by_id)}")


if __name__ == "__main__":
    main()
