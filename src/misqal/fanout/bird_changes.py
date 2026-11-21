"""Compare two BIRD dev releases: which answer changes fix fan-out, judged against the audit.

Mechanical part: for every question, whether the gold answer changed between the releases and
what the audit flags in each release. Judged part (two LLM judges), for the changes that matter:

- **flagged and fixed**: the old gold was flagged by the audit, the new gold is not and gives a
  different answer. Did the release fix a fan-out error, and with the question's meaning kept?
- **missed**: the old gold was not flagged, but the new gold adds a DISTINCT aggregate and gives a
  different answer: candidate fan-out fixes the audit did not see (recall).

    uv run python -m misqal.fanout.bird_changes --old data/bird/dev --old-audit runs/bird_audit.jsonl \
        --new <dev_20251106 root> --new-audit <its audit> --model <judge-a> <judge-b> --out <file>
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from misqal.fanout.label import _preview, _schema_for, facts
from misqal.harness.bench import QueryTimeout, Task, _normalize, load_bird, run_query
from misqal.harness.models import load_dotenv, make_model, over_budget

SYSTEM = """You compare two releases of a text-to-SQL benchmark question (SQLite): the old question
with its gold SQL, and the new question with its gold SQL after a human review.

Fan-out: when a table is joined to another on a key that is not unique there, or when a column
repeats within its own table, a duplicate-sensitive aggregate (COUNT, SUM, AVG without DISTINCT)
can count or add the same entity several times.

Decide:
- "same_meaning": does the new question ask the same thing as the old one (wording or typo fixes,
  or making the intended meaning explicit), rather than a different question?
- "change": "fan_out_fix" if the old gold counted or aggregated repeated rows where the question
  means each entity once, and the new gold fixes that; "other_fix" if the new gold fixes some other
  mistake; "different_question" if the answer changed because the question changed.
- "resolvable": reading the OLD question, is the intended reading decidable from the data model?
  True when the thing it counts or aggregates has its own identity in the schema (a table or column
  where it is unique) and the data shows which rows repeat it, so an analyst who checks keys and
  one-to-many relations would choose the new gold's reading without asking the user."""

JSON_INSTRUCTION = (
    "\n\nReply with one JSON object and nothing else, with keys: \"same_meaning\" (true | false), "
    "\"change\" (\"fan_out_fix\" | \"other_fix\" | \"different_question\"), \"resolvable\" "
    "(true | false), \"reason\" (string, two sentences at most).")


class ChangeJudgement(BaseModel):
    same_meaning: bool
    change: Literal["fan_out_fix", "other_fix", "different_question"]
    resolvable: bool
    reason: str = Field(default="")


def _distinct_aggs(sql: str) -> int:
    return len(re.findall(r"(count|sum|avg)\s*\(\s*distinct", sql, re.I))


def _flag(audit: dict[str, Any]) -> str:
    if audit["realized"]:
        return "definite"
    if audit.get("realized_ambiguous"):
        return "ambiguous"
    return "repeated_values" if audit.get("repeated_values") else "none"


def _answer(task: Task, sql: str) -> Counter | None:
    try:
        return _normalize(run_query(task.db_path, sql, 60.0)[1])
    except (sqlite3.Error, QueryTimeout):
        return None


def compare(old: dict[str, Task], new: dict[str, Task], old_audit: dict[str, dict],
            new_audit: dict[str, dict]) -> list[dict[str, Any]]:
    rows = []
    for qid, o in old.items():
        n = new[qid]
        same_sql = " ".join(o.gold_sql.split()).lower() == " ".join(n.gold_sql.split()).lower()
        a, b = (None, None) if same_sql else (_answer(o, o.gold_sql), _answer(n, n.gold_sql))
        changed = not same_sql and a is not None and b is not None and a != b
        row = {"id": qid, "db_id": o.db_id, "old_flag": _flag(old_audit[qid]),
               "new_flag": _flag(new_audit[qid]), "answer_changed": changed,
               "question_changed": " ".join(o.question.split()) != " ".join(n.question.split())}
        if changed and row["old_flag"] != "none" and row["new_flag"] == "none":
            row["candidate"] = "flagged_fixed"
        elif (changed and row["old_flag"] == "none"
              and _distinct_aggs(n.gold_sql) > _distinct_aggs(o.gold_sql)):
            row["candidate"] = "missed"
        rows.append(row)
    return rows


def user_prompt(o: Task, n: Task, old_audit: dict[str, Any]) -> str:
    tables = {v.lower() for b in old_audit["blocks"] for v in b["tables"].values()}
    with sqlite3.connect(f"file:{o.db_path}?mode=ro", uri=True) as con:
        names = {r[0].lower() for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    tables |= {t for t in names if re.search(rf"\b{re.escape(t)}\b", n.gold_sql, re.I)}
    _, old_res = _preview(o.db_path, o.gold_sql)
    _, new_res = _preview(n.db_path, n.gold_sql)
    found = facts(old_audit)
    return (f"OLD question: {o.question}\n" + (f"OLD hint: {o.evidence}\n" if o.evidence else "")
            + f"\nNEW question: {n.question}\n" + (f"NEW hint: {n.evidence}\n" if n.evidence else "")
            + f"\nSchema:\n{_schema_for(o.db_path, tables)}\n"
            + f"\nOLD gold SQL:\n{o.gold_sql}\n\nOLD result:\n{old_res}\n"
            + f"\nNEW gold SQL:\n{n.gold_sql}\n\nNEW result:\n{new_res}\n"
            + (f"\nFacts from the data about the OLD gold:\n{found}\n" if found else ""))


def judge(model_spec: str, o: Task, n: Task, old_audit: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"id": o.id, "model": model_spec}
    if over_budget(model_spec):
        return {**row, "error": "daily budget reached"}
    model = make_model(model_spec, max_tokens=2048, temperature=None)
    messages = [("system", SYSTEM + JSON_INSTRUCTION), ("user", user_prompt(o, n, old_audit))]
    error = None
    for _ in range(2):  # one retry on malformed JSON
        try:
            reply = model.invoke(messages)
            text = reply.text if hasattr(reply, "text") else str(reply.content)
            start, end = text.find("{"), text.rfind("}")
            return {**row, **ChangeJudgement.model_validate_json(text[start:end + 1]).model_dump()}
        except Exception as e:  # provider or parse errors are recorded, not fatal
            error = f"{type(e).__name__}: {e}"[:300]
    return {**row, "error": error}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--old-audit", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--new-audit", type=Path, required=True)
    p.add_argument("--model", nargs="*", default=[])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    load_dotenv()
    read = lambda path: {a["id"]: a for a in map(json.loads, path.read_text().splitlines())}
    old_audit, new_audit = read(args.old_audit), read(args.new_audit)
    old = {t.id: t for t in load_bird(args.old)}
    new = {t.id: t for t in load_bird(args.new)}
    rows = compare(old, new, old_audit, new_audit)
    jobs = [(m, r["id"]) for r in rows if r.get("candidate") for m in args.model]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verdicts = list(pool.map(lambda j: judge(j[0], old[j[1]], new[j[1]], old_audit[j[1]]), jobs))
    by_id: dict[str, list[dict]] = {}
    for v in verdicts:
        by_id.setdefault(v["id"], []).append(v)
    for r in rows:
        if r["id"] in by_id:
            r["judges"] = by_id[r["id"]]
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    flags = Counter(r["old_flag"] for r in rows)
    changed = Counter(r["old_flag"] for r in rows if r["answer_changed"])
    for f in flags:
        print(f"old flag {f:16s} n={flags[f]:5d}  answer changed {changed[f]:4d} "
              f"({100 * changed[f] / flags[f]:.0f}%)")
    for cand in ("flagged_fixed", "missed"):
        mine = [r for r in rows if r.get("candidate") == cand]
        agree = Counter(tuple(sorted({(v.get("change"), v.get("same_meaning")) for v in r.get("judges", [])}))
                        for r in mine)
        print(f"{cand}: {len(mine)} questions; judge (change, same_meaning) sets: "
              + "; ".join(f"{k}: {v}" for k, v in agree.most_common(6)))


if __name__ == "__main__":
    main()
