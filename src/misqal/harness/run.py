"""Run arms × models over a benchmark slice.

    uv run python -m misqal.harness.run --bench traps --repeats 3 \
        --model <model> --arm none-minimal stats-minimal full2-minimal facts2-minimal-tool \
        --out runs/traps

Results append to ``<out>/results.jsonl`` (one line per task × model × arm × repeat; reruns skip
lines already present); full message traces go to ``<out>/traces/``.

Spend: for private gateways that report it, a job is skipped (not recorded, so a rerun picks it
up) once the gateway key's spend in its budget window comes within a small margin of
``--daily-budget`` dollars; ``--budget-start FILE`` counts that spend from a mark set by the first
run in the window instead of from the window's start.
``--serve-via OLD=NEW`` serves ``OLD:<model>`` through gateway ``NEW`` and still records it as
``OLD:<model>``, so runs started on one gateway resume on another; with full model specs
(``mygw:some-model=other:model``) it maps one model to another provider's name for it.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from collections import defaultdict
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from misqal.harness.agent import MATRIX_TOOL, Arm, known_tables, run_episode, usage
from misqal.harness.bench import Task, load_tasks, score
from misqal.harness.models import DAILY_BUDGET, load_dotenv, make_model, over_budget


# Evidence kinds by delivery. Paper names: none = no evidence, stats = Stats, stats2 = Stats+Rep,
# full = FD, facts (tool) = FD (tool), full2 = Grain-FD, facts2 (tool) = Grain-FD (tool).
# labels and the "grain" procedure were explored early and are not used in the paper.
INJECTED = ("none", "stats", "stats2", "full", "labels", "facts", "full2")
AS_TOOL = ("full", "labels", "facts", "facts2")


def parse_arm(spec: str, **kw: Any) -> Arm:
    """``evidence-procedure`` or ``evidence-procedure-tool`` (evidence offered as a mandatory tool
    instead of injected into the prompt), e.g. ``full2-minimal`` or ``facts2-minimal-tool``."""
    parts = spec.split("-")
    tool = len(parts) == 3 and parts[2] == "tool"
    if tool:
        kw["delivery"] = "tool"
        parts = parts[:2]
    if len(parts) != 2 or parts[0] not in (AS_TOOL if tool else INJECTED) or parts[1] not in ("minimal", "grain"):
        raise argparse.ArgumentTypeError(
            f"unknown arm {spec!r}: use <evidence>-<procedure>[-tool] with evidence in {INJECTED} "
            f"(in the prompt) or {AS_TOOL} (as a tool), procedure minimal or grain")
    return Arm(evidence=parts[0], procedure=parts[1], **kw)


def message_dict(m: BaseMessage) -> dict:
    d: dict[str, Any] = {"role": m.type, "content": m.content}
    for attr in ("tool_calls", "tool_call_id", "name", "status", "usage_metadata"):
        if v := getattr(m, attr, None):
            d[attr] = v
    if thinking := (m.additional_kwargs or {}).get("reasoning_content"):
        d["thinking"] = thinking
    return d


def record(task: Task, model_spec: str, arm: Arm, root: Path | None, trace_dir: Path,
           temperature: float | None = 0.0, rep: int = 0, served_by: str | None = None) -> dict:
    model = make_model(served_by or model_spec, temperature=temperature)
    ep = run_episode(model, task, arm)
    tool_calls = [tc for m in ep.messages for tc in (getattr(m, "tool_calls", None) or [])]
    row: dict[str, Any] = {
        "task_id": task.id, "bench": task.bench, "db_id": task.db_id, "category": task.category,
        "model": model_spec, **({"served_by": served_by} if served_by else {}),
        "arm": arm.name, "rep": rep, "evidence": arm.evidence,
        "procedure": arm.procedure,
        "status": ep.status, "error": ep.error, "seconds": round(ep.seconds, 2),
        "pred_sql": ep.pred_sql, "sql_source": ep.sql_source,
        "n_tool_calls": len(tool_calls),
        "n_matrix_calls": sum(tc["name"] == MATRIX_TOOL for tc in tool_calls),
        "matrix_calls": ep.state.matrix_calls,
        "gate_rejections": ep.state.gate_rejections,
        **({"reminders": ep.state.reminders} if ep.state.reminders else {}),
        **({"tables_shown": list(task.tables)} if task.tables else {}),
        **({"measures_excluded": len(task.measures)} if task.measures else {}),
        **usage(ep.messages),
        **score(task, ep.pred_sql, root),
    }
    safe_model = model_spec.replace(":", "_").replace("/", "_")
    path = trace_dir / safe_model / arm.name / f"{task.id}{f'.r{rep}' if rep else ''}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task": task.__dict__ | {"db_path": str(task.db_path)},
                                "result": row,
                                "messages": [message_dict(m) for m in ep.messages]},
                               default=str, indent=1))
    return row


def summarize(rows: list[dict]) -> str:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["model"], r["arm"])].append(r)
    keys = ["exec_ok", "set_ex", "multiset_ex", "original_set_ex", "ex"]
    lines = [f"{'model':38} {'arm':22} {'n':>4} " + " ".join(f"{k:>11}" for k in keys)
             + f" {'matrix/q':>9} {'tok_in/q':>9}"]
    for (model, arm), rs in sorted(by.items()):
        cells = []
        for k in keys:
            vals = [r[k] for r in rs if k in r]
            cells.append(f"{sum(vals) / len(vals):>11.3f}" if vals else f"{'-':>11}")
        n = len(rs)
        lines.append(f"{model[:38]:38} {arm:22} {n:>4} " + " ".join(cells)
                     + f" {sum(r['n_matrix_calls'] for r in rs) / n:>9.2f}"
                     + f" {sum(r['input_tokens'] for r in rs) / n:>9.0f}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bench", default="bird",
                   choices=["bird", "bird-fanout", "bird-fanout-20251106", "bird-fanout-train", "traps",
                            "spider2"])
    p.add_argument("--root", type=Path, help="benchmark root (default: data/...)")
    p.add_argument("--db", nargs="*", help="restrict to these db_ids")
    p.add_argument("--ids", nargs="*", help="restrict to these task ids")
    p.add_argument("--limit", type=int, help="first N tasks (after --sample)")
    p.add_argument("--sample", type=int, help="random N tasks (seeded)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--arm", nargs="+", default=["none-minimal"])
    p.add_argument("--delivery", choices=["inject", "tool"], default="inject",
                   help="evidence arms: inject the profile into the prompt, or offer the tool")
    p.add_argument("--recursion-limit", type=int, default=50)
    p.add_argument("--temperature", default="0",
                   help='sampling temperature; "none" = provider default (required by some models)')
    p.add_argument("--repeats", type=int, default=1, help="runs per task × model × arm")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--daily-budget", type=float, default=DAILY_BUDGET,
                   help="USD per gateway key per budget window (gateways that report spend)")
    p.add_argument("--budget-start", type=Path, metavar="FILE",
                   help="count the budget from the key's spend when a run first checked it in "
                        "this window (kept in FILE), not from the window's start")
    p.add_argument("--serve-via", nargs="*", default=[], metavar="OLD=NEW",
                   help="serve OLD:<model> through gateway prefix NEW (or model spec OLD through "
                        "spec NEW), recorded under the OLD name")
    p.add_argument("--known-tables", action="store_true",
                   help="show the agent only the tables its reference queries read (gold and trap), "
                        "with all their columns, in every arm")
    p.add_argument("--submit-reminder", default="", metavar="TEXT",
                   help="user message sent when a run ends without calling submit_answer (at most "
                        "twice); a model-specific accommodation for models that answer in text")
    p.add_argument("--column-roles", type=Path, metavar="FILE",
                   help="column roles (misqal.harness.column_roles); evidence leaves out measures")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    load_dotenv()
    temperature = None if args.temperature.lower() == "none" else float(args.temperature)
    arms = [parse_arm(a, recursion_limit=args.recursion_limit, delivery=args.delivery,
                      submit_reminder=args.submit_reminder) for a in args.arm]
    tasks = load_tasks(args.bench, args.root, args.db)
    if args.ids:
        tasks = [t for t in tasks if t.id in set(args.ids)]
    if args.sample:
        tasks = random.Random(args.seed).sample(tasks, min(args.sample, len(tasks)))
    if args.limit:
        tasks = tasks[:args.limit]
    if args.known_tables:
        tasks = [replace(t, tables=known_tables(t)) for t in tasks]
    if args.column_roles:
        measures: dict[str, set[str]] = {}
        for role in map(json.loads, args.column_roles.read_text().splitlines()):
            if role["role"] == "measure":
                measures.setdefault(role["db_id"], set()).add(f"{role['table']}.{role['column']}".lower())
        tasks = [replace(t, measures=tuple(sorted(measures.get(t.db_id, ())))) for t in tasks]

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    done: set[tuple[str, str, str, int]] = set()
    previous: list[dict] = []
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            previous.append(r)
            done.add((r["task_id"], r["model"], r["arm"], r.get("rep", 0)))
    jobs = [(t, m, a, k) for k in range(args.repeats) for t in tasks for m in args.model
            for a in arms if (t.id, m, a.name, k) not in done]
    print(f"{len(tasks)} tasks × {len(args.model)} models × {len(arms)} arms × {args.repeats} "
          f"repeats: {len(jobs)} to run, {len(done)} already done")

    via = dict(v.split("=", 1) for v in args.serve_via)

    def serving(m: str) -> str | None:
        if m in via:
            return via[m]
        prefix, _, name = m.partition(":")
        return f"{via[prefix]}:{name}" if prefix in via else None

    def job(t: Task, m: str, a: Arm, k: int) -> dict | None:
        if over_budget(serving(m) or m, args.daily_budget, args.budget_start):
            return None
        return record(t, m, a, args.root, args.out / "traces", temperature, k, serving(m))

    lock = threading.Lock()
    rows: list[dict] = []
    skipped: defaultdict[str, int] = defaultdict(int)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, results_path.open("a") as out:
        futures = {pool.submit(job, t, m, a, k): (t, m, a, k) for t, m, a, k in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            t, m, a, k = futures[fut]
            row = fut.result()
            if row is None:
                if not skipped[m]:
                    print(f"[{i}/{len(jobs)}] {m}: daily budget of ${args.daily_budget:g} reached; "
                          "skipping its remaining jobs")
                skipped[m] += 1
                continue
            with lock:
                out.write(json.dumps(row, default=str) + "\n")
                out.flush()
                rows.append(row)
            ok = row.get("multiset_ex", row.get("ex"))
            print(f"[{i}/{len(jobs)}] {t.id} r{k} {m} {a.name}: {row['status']} "
                  f"ex={ok} matrix={row['n_matrix_calls']} {row['seconds']}s")
    for m, n in skipped.items():
        print(f"{m}: {n} jobs skipped for budget; rerun to resume")
    print(summarize([r for r in previous + rows
                     if r["task_id"] in {t.id for t in tasks}]))


if __name__ == "__main__":
    main()
