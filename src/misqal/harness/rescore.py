"""Re-score saved results against the current answer keys, without calling any model.

    uv run python -m misqal.harness.rescore runs/<traps-run> [--drop <question-id> ...]
    uv run python -m misqal.harness.rescore runs/<train-run> ... --bench bird-fanout-train --to runs/reviewed

Each row's saved ``pred_sql`` is scored again (gold, alternative readings, outcome). ``--drop``
removes the rows of questions whose wording changed, so a rerun can fill them. Traps are re-scored
in place and the previous file kept as ``results.jsonl.bak``.

BIRD-FanOut runs are re-scored against the reviewed golds (``misqal.fanout.gold_review``) into
``--to``/<run name>/results.jsonl, leaving the run untouched; rows of questions the review left out
are dropped, and each row records the question's ``gold_status``.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from misqal.harness.bench import load_tasks, load_traps, score_bird, score_traps

SCORE_FIELDS = ("exec_ok", "set_ex", "multiset_ex", "outcome", "pred_rows", "gold_rows",
                "alt_match", "pred_error", "gold_error", "original_set_ex", "gold_status")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, nargs="+", help="run folders")
    ap.add_argument("--drop", nargs="*", default=[], help="task ids to remove")
    ap.add_argument("--root", type=Path, default=Path("data/traps"))
    ap.add_argument("--bench", default="traps")
    ap.add_argument("--to", type=Path, help="BIRD-FanOut: folder for the re-scored runs")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    traps = args.bench == "traps"
    if not traps and not args.to:
        ap.error("--to is required for BIRD-FanOut runs")
    tasks = {t.id: t for t in (load_traps(args.root) if traps else load_tasks(args.bench))}
    for run in args.out:
        path = run / "results.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not traps:  # score the answers of fixed questions in parallel (queries can be slow)
            todo = [r for r in rows if r["task_id"] in tasks and tasks[r["task_id"]].gold_status == "fixed"
                    and r.get("status") != "error"]
            with ThreadPoolExecutor(args.workers) as pool:
                scores = dict(zip(map(id, todo), pool.map(
                    lambda r: score_bird(tasks[r["task_id"]], r.get("pred_sql")), todo)))
        kept, changed = [], 0
        for r in rows:
            if r["task_id"] in args.drop:
                continue
            t = tasks.get(r["task_id"])
            if traps:
                if r.get("bench") == "traps" and t is not None:
                    new = score_traps(t, r.get("pred_sql"))
                    changed += new["multiset_ex"] != r.get("multiset_ex")
                    r = {k: v for k, v in r.items() if k not in SCORE_FIELDS} | new
            else:
                if t is None:
                    continue
                if t.gold_status == "fixed" and r.get("status") != "error":
                    new = scores[id(r)] | {"gold_status": t.gold_status}
                    changed += new["set_ex"] != r.get("set_ex")
                    r = {k: v for k, v in r.items() if k not in SCORE_FIELDS} | new
                else:
                    r = r | {"gold_status": t.gold_status}
            kept.append(r)
        if traps:
            path.with_suffix(".jsonl.bak").write_text(path.read_text())
            dest = path
        else:
            dest = args.to / run.name / "results.jsonl"
            dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(r) + "\n" for r in kept))
        print(f"{dest}: {len(rows)} rows, {len(rows) - len(kept)} dropped, "
              f"{changed} changed correctness", flush=True)


if __name__ == "__main__":
    main()
