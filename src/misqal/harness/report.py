"""Per model × arm report of a run, split by question category.

    uv run python -m misqal.harness.report runs/<run>/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def rate(rows: list[dict], key: str) -> str:
    vals = [bool(r.get(key)) for r in rows if key in r]
    if not vals:
        return "-"
    k, n = sum(vals), len(vals)
    lo, hi = wilson(k, n)
    return f"{k / n:5.1%} [{lo:4.0%}-{hi:4.0%}]"


def report(rows: list[dict]) -> str:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"].split(":", 1)[-1], r["arm"])].append(r)
    head = (f"{'model':28} {'arm':16} {'n':>3}  {'inflated (corrected gold)':>24}  "
            f"{'controls':>20}  {'inflated vs BIRD gold':>22}  {'evidence':>11}  {'errors':>6}")
    lines = [head, "-" * len(head)]
    for (model, arm), rs in sorted(groups.items()):
        infl = [r for r in rs if r.get("category") == "inflated"]
        ctrl = [r for r in rs if r.get("category") == "intended"]
        used = sum(bool(r.get("matrix_calls")) for r in rs) / len(rs)  # tool calls or injected
        errors = sum(r["status"] != "completed" for r in rs)
        lines.append(f"{model[:28]:28} {arm:16} {len(rs):>3}  {rate(infl, 'multiset_ex'):>24}  "
                     f"{rate(ctrl, 'multiset_ex'):>20}  {rate(infl, 'original_set_ex'):>22}  "
                     f"{used:>11.0%}  {errors:>6}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("results", type=Path)
    args = p.parse_args()
    print(report([json.loads(line) for line in args.results.read_text().splitlines()]))


if __name__ == "__main__":
    main()
