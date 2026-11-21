"""Did the model check the data model before answering? Trap-suite traces, per model and arm.

A run *checked* if a query it executed before its answer (``execute_sql``, not the profiling tool,
and not the submitted SQL itself) tests how columns relate: a ``COUNT(DISTINCT ...)``, a
``HAVING COUNT(...)``, or an ``IS NULL`` test. For each model and arm: the share of runs that
checked, and the trapped rate among runs that checked vs runs that did not (controls excluded),
with a two-sided Fisher exact test. Per group of models, a Mantel-Haenszel test pools the 2x2
tables of each (model, question), so a model's skill and a question's difficulty are held fixed.

    uv run python -m misqal.harness.checking runs/<traps-run> [runs/<traps-run> ...]
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from math import comb
from pathlib import Path

from misqal.harness.stats import NONE, is_strong, short

CHECK = re.compile(r"count\s*\(\s*distinct|having\s+count|\bis\s+(not\s+)?null\b", re.I)


def _norm(sql: str | None) -> str:
    return " ".join((sql or "").lower().split()).rstrip(";")


def queries(trace: dict) -> list[str]:
    out = []
    for m in trace["messages"]:
        calls = m.get("tool_calls")
        if not calls:
            continue
        for c in ast.literal_eval(calls) if isinstance(calls, str) else calls:
            if c["name"] == "execute_sql":
                out.append(c["args"].get("query", ""))
    return out


def checked(trace: dict, answer_sql: str | None) -> bool:
    final = _norm(answer_sql)
    return any(CHECK.search(q) for q in queries(trace) if _norm(q) != final)


def trace_path(run: Path, r: dict) -> Path:
    model = r["model"].replace(":", "_").replace("/", "_")
    name = r["task_id"] + (f".r{r['rep']}" if r.get("rep") else "") + ".json"
    return run / "traces" / model / r["arm"] / name


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for [[a, b], [c, d]]."""
    r1, c1, n = a + b, a + c, a + b + c + d
    prob = lambda x: comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)
    observed = prob(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return min(1.0, sum(p for x in range(lo, hi + 1) if (p := prob(x)) <= observed * (1 + 1e-9)))


def mantel_haenszel(tables: list[tuple[int, int, int, int]]) -> tuple[float, float]:
    """Common odds ratio and two-sided p (continuity-corrected chi-square, 1 df) over 2x2 tables
    [[a, b], [c, d]]; tables with one empty row or column carry no information and are skipped."""
    num = den = dev = var = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if min(a + b, c + d, a + c, b + d) == 0:
            continue
        num += a * d / n
        den += b * c / n
        dev += a - (a + b) * (a + c) / n
        var += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    if var == 0:
        return float("nan"), 1.0
    chi2 = (abs(dev) - 0.5) ** 2 / var
    return (num / den if den else float("inf")), math.erfc(math.sqrt(max(chi2, 0) / 2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    args = p.parse_args()

    # (model, arm) -> [(task_id, checked, trapped)]
    cells: defaultdict[tuple[str, str], list[tuple[str, bool, bool]]] = defaultdict(list)
    for run in args.runs:
        for line in (run / "results.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "error" or r.get("category") == "control":
                continue
            path = trace_path(run, r)
            if not path.exists():
                continue
            trapped = r.get("outcome") == "trapped"
            cells[(r["model"], r["arm"])].append(
                (r["task_id"], checked(json.loads(path.read_text()), r.get("pred_sql")), trapped))

    print(f"{'model':18s} {'arm':20s} {'checked':>8s}  trapped if checked  trapped if not   Fisher p")
    for (m, a), rs in sorted(cells.items(), key=lambda kv: (not is_strong(kv[0][0]), kv[0][0],
                                                          kv[0][1] != NONE, kv[0][1])):
        ct, cn = sum(t for _, c, t in rs if c), sum(c for _, c, _ in rs)
        nt, nn = sum(t for _, c, t in rs if not c), sum(not c for _, c, _ in rs)
        rate = lambda k, n: f"{k:3d}/{n:3d}" + (f" ({100 * k / n:3.0f}%)" if n else "       ")
        print(f"{short(m):18s} {a:20s} {100 * cn / len(rs):7.0f}%  {rate(ct, cn):>18s}  "
              f"{rate(nt, nn):>14s}   {fisher(ct, cn - ct, nt, nn - nt):.3f}")

    print("\nWithin (model, question), Mantel-Haenszel over strata: trapped ~ checked")
    for group, pick in (("strong", is_strong), ("weak", lambda m: not is_strong(m))):
        for arm in sorted({a for _, a in cells}):
            strata: defaultdict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
            for (m, a), rs in cells.items():
                if a == arm and pick(m):
                    for t, c, trapped in rs:
                        s = strata[(m, t)]
                        s[(0 if c else 2) + (0 if trapped else 1)] += 1
            if strata:
                odds, pval = mantel_haenszel([tuple(s) for s in strata.values()])
                mixed = [s for s in strata.values() if s[0] + s[1] and s[2] + s[3]]
                ct, cn = sum(s[0] for s in mixed), sum(s[0] + s[1] for s in mixed)
                nt, nn = sum(s[2] for s in mixed), sum(s[2] + s[3] for s in mixed)
                print(f"{group:6s} {arm:20s} strata with both: {len(mixed):3d}  trapped "
                      f"{ct}/{cn} checked vs {nt}/{nn} not  MH odds ratio {odds:.2f} p={pval:.3f}")


if __name__ == "__main__":
    main()
