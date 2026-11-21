"""Paired evidence-vs-none statistics with questions, not runs, as the unit of independence.

Every evidence run is paired with the none-arm run of the same (question, model, repeat). Three
views of each comparison:

- **runs**: two-sided exact sign test over discordant run pairs (wins: right with evidence and
  wrong without; losses: the reverse). Treats repeats and models as independent, so it overstates
  significance; reported for reference only.
- **questions**: per question, the net of wins minus losses over all its runs (models × repeats);
  sign test over questions that moved up or down.
- **bootstrap**: 95% interval of the accuracy difference (evidence − none), resampling base
  questions (a question and its vague twin together) with replacement.

Scoring: ``scored`` uses the harness outcome (traps: ``outcome == "correct"``; BIRD: set EX).
``submitted`` counts only answers the model submitted; runs scored on the fallback (the last
query it executed) count as wrong.

    uv run python -m misqal.harness.stats traps runs/<traps-run> [runs/<traps-run> ...]
    uv run python -m misqal.harness.stats bird runs/<bird-run> --category inflated fixed_in_release
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Callable, Iterable

# Larger models; used to split the full-schema runs into larger and smaller models.
STRONG = ("opus", "sonnet", "gpt-5.5", "gpt-5.6-terra", "gpt-5.6-sol", "gemini-3.1-pro")
NONE = "none-minimal"


def load(dirs: Iterable[Path]) -> dict[tuple[str, str, str, int], dict]:
    rows = {}
    for d in dirs:
        for line in (d / "results.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["status"] != "error":
                rows[(r["model"], r["arm"], r["task_id"], r.get("rep", 0))] = r
    return rows


def is_strong(model: str) -> bool:
    return any(s in model.lower() for s in STRONG)


# Model tiers as in the paper, by grain-error rate on GrainTraps without evidence (low: 2.4%, mid: 13.2%,
# high: 40.8%); BIRD-FanOut train gives the same order.
ERROR_TIERS = {"low": ("sonnet-5-5", "haiku-5-5"),
               "mid": ("deepseek-v4.1", "gpt-5.6-luna", "gpt-5.6-terra", "glm-5.3"),
               "high": ("flash-lite", "nemotron")}


def tier(model: str) -> str:
    m = model.lower()
    for t, keys in ERROR_TIERS.items():
        if any(k in m for k in keys):
            return t
    return "other"


_FAMILY = re.compile(r"(claude|gpt|gemini|llama|qwen|deepseek|kimi|glm|mistral|nemotron)[\w.\-]*", re.I)


def short(model: str) -> str:
    """Model family and version only, without provider or deployment prefixes."""
    name = model.split("/")[-1].split(":")[-1]
    m = _FAMILY.search(name)
    return (m.group(0) if m else name)[:18]


def correct_fn(bench: str, scoring: str, metric: str = "set_ex") -> Callable[[dict], bool]:
    def ok(r: dict) -> bool:
        if scoring == "submitted" and r.get("sql_source") != "submitted":
            return False
        return r["outcome"] == "correct" if bench == "traps" else bool(r.get(metric))
    return ok


def p_sign(w: int, l: int) -> float:
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def base(task_id: str) -> str:
    return task_id.removesuffix("-v")


def pairs(rows: dict, arm: str, keep: Callable[[str, dict], bool],
          ok: Callable[[dict], bool], baseline: str = NONE) -> list[tuple[str, int, int]]:
    """(task_id, arm correct, baseline correct) for every paired run."""
    out = []
    for (m, a, t, k), r in rows.items():
        if a == arm and keep(m, r) and (b := rows.get((m, baseline, t, k))) is not None:
            out.append((t, int(ok(r)), int(ok(b))))
    return out


def compare(ps: list[tuple[str, int, int]], n_boot: int = 2000, seed: int = 0) -> dict:
    w = sum(e and not b for _, e, b in ps)
    l = sum(b and not e for _, e, b in ps)
    # The unit is the base question: a question and its vague twin form one cluster, for the
    # sign test, the permutation test and the bootstrap alike.
    clusters: defaultdict[str, list[int]] = defaultdict(list)
    for t, e, b in ps:
        clusters[base(t)].append(e - b)
    sums = [sum(v) for v in clusters.values()]
    up, down = sum(v > 0 for v in sums), sum(v < 0 for v in sums)
    keys, rng, diffs = list(clusters), random.Random(seed), []
    for _ in range(n_boot):
        sample = [d for c in rng.choices(keys, k=len(keys)) for d in clusters[c]]
        diffs.append(sum(sample) / len(sample))
    diffs.sort()
    return {"n": len(ps), "w": w, "l": l, "p_runs": p_sign(w, l), "up": up, "down": down,
            "p_questions": p_sign(up, down), "p_perm": p_flip(sums, seed=seed),
            "diff": sum(e - b for _, e, b in ps) / max(len(ps), 1),
            "lo": diffs[int(0.025 * n_boot)], "hi": diffs[int(0.975 * n_boot) - 1]}


def p_flip(sums: list[int], n: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test on per-cluster net changes (uses their size too)."""
    rng, obs = random.Random(seed), abs(sum(sums))
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in sums)) >= obs for _ in range(n))
    return (hits + 1) / (n + 1)


def fmt(c: dict) -> str:
    return (f"n={c['n']:4d}  runs +{c['w']}/-{c['l']} p={c['p_runs']:.2g}  "
            f"questions {c['up']} up/{c['down']} down p={c['p_questions']:.2g}  "
            f"diff {100 * c['diff']:+.1f}pp [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("bench", choices=["traps", "bird"])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--category", nargs="*", help="keep only these question categories")
    p.add_argument("--metric", default="set_ex", help="BIRD: set_ex or multiset_ex")
    p.add_argument("--baseline", default=NONE, help="arm every other arm is compared with")
    args = p.parse_args()

    rows = load(args.runs)
    if args.category:
        rows = {k: r for k, r in rows.items() if r["category"] in args.category}
    if args.bench == "traps":
        rows = {k: r for k, r in rows.items() if r["category"] != "control"}
    arms = sorted({k[1] for k in rows} - {args.baseline})
    groups = {"low": lambda m, r: tier(m) == "low", "mid": lambda m, r: tier(m) == "mid",
              "high": lambda m, r: tier(m) == "high", "all": lambda m, r: True}
    sens = {"all questions": lambda m, r: True,
            "without NULL split": lambda m, r: r["category"] != "null_split",
            "without gpt-oss": lambda m, r: "gpt-oss" not in m}

    for scoring in ("scored", "submitted"):
        ok = correct_fn(args.bench, scoring, args.metric)
        print(f"\n==== scoring: {scoring}")
        for sname, sfilter in sens.items():
            if sname == "without NULL split" and args.bench != "traps":
                continue
            print(f"\n-- {sname}")
            for g, gfilter in groups.items():
                for a in arms:
                    ps = pairs(rows, a, lambda m, r: gfilter(m, r) and sfilter(m, r), ok,
                               args.baseline)
                    if ps:
                        print(f"{g:6s} {a:20s} {fmt(compare(ps))}")

    ok = correct_fn(args.bench, "scored", args.metric)
    print("\n==== per model × wording × arm (scored): run-level p, Holm over all cells")
    cells = []
    order = {"low": 0, "mid": 1, "high": 2}
    for m in sorted({k[0] for k in rows}, key=lambda m: (order.get(tier(m), 3), m)):
        for vague in ((False, True) if args.bench == "traps" else (None,)):
            for a in arms:
                ps = pairs(rows, a, lambda mm, r: mm == m and (
                    vague is None or r["task_id"].endswith("-v") == vague), ok)
                if ps:
                    c = compare(ps, n_boot=200)
                    cells.append((m, vague, a, c))
    for (m, vague, a, c), padj in zip(cells, holm([c["p_runs"] for *_, c in cells])):
        wording = "" if vague is None else ("vague" if vague else "explicit")
        print(f"{short(m):18s} {wording:8s} {a:20s} +{c['w']}/-{c['l']} p={c['p_runs']:.3f} "
              f"holm={padj:.3f} questions {c['up']}/{c['down']} p={c['p_questions']:.3f}"
              + ("  *" if padj < 0.05 else ""))

    print("\n==== answers not submitted (scored on the last executed query)")
    for m in sorted({k[0] for k in rows}):
        line = []
        for a in [NONE, *arms]:
            rs = [r for k, r in rows.items() if k[0] == m and k[1] == a]
            if rs:
                n = sum(r.get("sql_source") != "submitted" for r in rs)
                line.append(f"{a.replace('-minimal', '')} {n}/{len(rs)}")
        print(f"{short(m):18s} " + " | ".join(line))


if __name__ == "__main__":
    main()
