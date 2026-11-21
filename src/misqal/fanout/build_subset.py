"""Assemble the fan-out-sensitive BIRD subset from the audit, two judges' labels and adjudication.

Categories:
- ``inflated``: both judges say the question means distinct entities, so the gold answer counts
  repeated rows; ``corrected_sql`` is the corrected gold (both judges' corrections return the same
  result, or the adjudication picks one). ``other_error`` marks corrections that also fix a
  non-fan-out mistake of the gold query.
- ``intended``: both judges say the joined rows are what the question counts.
- ``contested``: the judges disagree or find it unclear (typically: the question asks for distinct
  entities while the hint prescribes a row-level formula).
- ``benign``: inflated in form, but removing the repetition does not change the answer on this data.
- ``fixed_in_release`` (with ``--changes``): a later release of the benchmark fixed a fan-out error
  of the gold, and both judges agree the fix keeps the question's meaning. ``gold_sql`` is the
  release's fixed SQL and ``naive_sql`` the earlier gold, i.e. the trap its annotators fell into;
  ``old_question`` is set when the release also reworded the question.

Every row records ``resolvable``: both judges say the intended reading follows from the data model
(the counted thing has its own key, and the data shows which rows repeat it). With
``--escalated``, questions the first judges left unsure are labelled by stronger judges instead
(``escalated`` is true on those rows).

    uv run python -m misqal.fanout.build_subset --audit runs/bird_audit.jsonl \
        --labels runs/bird_labels.jsonl --adjudication benchmarks/bird-fanout/adjudication.json \
        --out benchmarks/bird-fanout/subset.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from misqal.harness.bench import BENCH_ROOTS, QueryTimeout, Task, load_bird, run_query


def judge_name(model_spec: str) -> str:
    """Model family only: ``<gateway>:<prefix>-claude-opus-5-5`` -> ``claude-opus-5-5``."""
    name = model_spec.split(":", 1)[-1]
    m = re.search(r"(claude|gpt|gemini|llama|qwen|deepseek|kimi|glm|mistral)[\w.\-]*", name)
    return m.group(0) if m else name


def _result(task: Task, sql: str | None) -> Counter | None:
    if not sql:
        return None
    try:
        _, rows = run_query(task.db_path, sql, 60.0)
    except (sqlite3.Error, QueryTimeout):
        return None
    return Counter(tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows)


def _scalar(c: Counter | None) -> float | None:
    if c is None or sum(c.values()) != 1:
        return None
    (row,) = c
    return float(row[0]) if len(row) == 1 and isinstance(row[0], (int, float)) else None


def build(tasks: dict[str, Task], labels: list[dict], adjudication: dict) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, dict]] = {}
    for r in labels:
        by_id.setdefault(r["id"], {})[judge_name(r["model"])] = r
    chosen = adjudication.get("choose_correction", {})
    contested = set(adjudication.get("contested", []))  # flawed beyond fan-out: left out
    other = set(adjudication.get("other_error", []))
    notes = adjudication.get("notes", {})
    out = []
    for qid in sorted(by_id, key=int):
        judges = by_id[qid]
        task = tasks[qid]
        verdicts = {j: r.get("verdict", "error") for j, r in judges.items()}
        row: dict[str, Any] = {
            "id": qid, "db_id": task.db_id, "question": task.question, "evidence": task.evidence,
            "gold_sql": task.gold_sql, "verdicts": verdicts,
            "reasons": {j: r.get("reason") for j, r in judges.items()},
            "resolvable": all(r.get("resolvable") is True for r in judges.values()),
        }
        values = set(verdicts.values())
        if qid in contested:
            row["category"] = "contested"
        elif values == {"intended"}:
            row["category"] = "intended"
        elif values != {"inflated"}:
            row["category"] = "contested"
        else:
            gold = _result(task, task.gold_sql)
            fixes = {j: (r.get("corrected_sql"), _result(task, r.get("corrected_sql")))
                     for j, r in judges.items()}
            results = {j: res for j, (_, res) in fixes.items()}
            if all(res == gold for res in results.values()):
                row["category"] = "benign"
            elif qid in chosen:
                row.update(category="inflated", corrected_sql=fixes[chosen[qid]][0],
                           corrected_by=chosen[qid])
            elif None not in results.values() and len(
                    {json.dumps(sorted(map(str, r.items()))) for r in results.values()}) == 1:
                judge = sorted(fixes)[0]
                row.update(category="inflated", corrected_sql=fixes[judge][0], corrected_by=judge)
            else:
                row["category"] = "unresolved"  # corrections differ and no adjudication
            if row["category"] == "inflated":
                row["other_error"] = qid in other
                g, c = _scalar(gold), _scalar(_result(task, row["corrected_sql"]))
                if g is not None and c not in (None, 0.0):
                    row["gold_to_corrected"] = round(g / c, 3)
        if qid in notes:
            row["note"] = notes[qid]
        out.append(row)
    return out


def build_fixed(old: dict[str, Task], new: dict[str, Task],
                changes: list[dict]) -> list[dict[str, Any]]:
    """Fan-out fixes made by a later release, kept when both judges agree on the fix."""
    out = []
    for c in changes:
        judges = {judge_name(v["model"]): v for v in c.get("judges", [])}
        if len(judges) < 2 or not all(v.get("change") == "fan_out_fix" and v.get("same_meaning")
                                      for v in judges.values()):
            continue
        o, n = old[c["id"]], new[c["id"]]
        row: dict[str, Any] = {
            "id": c["id"], "db_id": n.db_id, "question": n.question, "evidence": n.evidence,
            "gold_sql": n.gold_sql, "naive_sql": o.gold_sql, "category": "fixed_in_release",
            "found_by_audit": c["candidate"] == "flagged_fixed",
            "resolvable": all(v.get("resolvable") is True for v in judges.values()),
            "reasons": {j: v.get("reason") for j, v in judges.items()}}
        if c["question_changed"]:
            row["old_question"] = o.question
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", type=Path, default=BENCH_ROOTS["bird"])
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--escalated", type=Path,
                   help="labels by stronger judges for the questions the first judges left unsure; "
                        "they replace the first judges' labels for those questions")
    p.add_argument("--adjudication", type=Path, required=True)
    p.add_argument("--changes", type=Path, help="judged release changes (misqal.fanout.bird_changes)")
    p.add_argument("--old-root", type=Path, help="the earlier release, with --changes")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    flagged = {a["id"] for a in map(json.loads, args.audit.read_text().splitlines())
               if a["realized"] or a.get("realized_ambiguous") or a.get("repeated_values")}
    every = {t.id: t for t in load_bird(args.root)}
    tasks = {i: t for i, t in every.items() if i in flagged}
    labels = [r for r in map(json.loads, args.labels.read_text().splitlines()) if r["id"] in tasks]
    escalated: set[str] = set()
    if args.escalated:
        strong = [r for r in map(json.loads, args.escalated.read_text().splitlines())
                  if r["id"] in tasks]
        escalated = {r["id"] for r in strong}
        labels = [r for r in labels if r["id"] not in escalated] + strong
    rows = build(tasks, labels, json.loads(args.adjudication.read_text()))
    for r in rows:
        r["escalated"] = r["id"] in escalated
    if args.changes:
        old = {t.id: t for t in load_bird(args.old_root)}
        changes = list(map(json.loads, args.changes.read_text().splitlines()))
        labelled = {r["id"] for r in rows}  # still flagged in this release: judged above
        fixed = [r for r in build_fixed(old, every, changes) if r["id"] not in labelled]
        rows = sorted(rows + fixed, key=lambda r: int(r["id"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(Counter((r["category"], "resolvable" if r["resolvable"] else "not resolvable")
                  for r in rows))
    ratios = sorted(r["gold_to_corrected"] for r in rows if "gold_to_corrected" in r)
    if ratios:
        print(f"scalar gold / corrected: n={len(ratios)}, below 1: {sum(x < 1 for x in ratios)}, "
              f"median={ratios[len(ratios) // 2]}, max={ratios[-1]}")


if __name__ == "__main__":
    main()
