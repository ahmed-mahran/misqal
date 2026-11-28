"""Fan-out in the gold of other benchmarks: golds, the share that join, definite flags and distorted
golds, as % of all golds and of golds that join (paper Figure 6, extended version Appendix D).

    uv run python -m misqal.xbench.table [--audits data/xbench/audits]

A gold is distorted when both agent labels in ``benchmarks/other-benchmarks/labels.jsonl`` say so.
BEAVER's generated warehouse questions were labelled on a random sample of 200 definite flags; their
count is scaled to all definite flags, with a 95% Wilson interval.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LABELS = Path(__file__).parents[3] / "benchmarks" / "other-benchmarks" / "labels.jsonl"
ROWS = [  # name, audit file, databases left out (shipped without rows)
    ("Spider 1.0 dev", "spider1-dev", ()), ("Spider 1.0 test", "spider1-test", ()),
    ("Spider 1.0 train", "spider1-train", ("music_2",)), ("KaggleDBQA", "kaggledbqa", ()),
    ("Spider 2.0-Lite (SQLite)", "spider2-lite", ()), ("ATIS", "atis", ()), ("Advising", "advising", ()),
    ("FIBEN", "fiben", ()), ("BEAVER warehouse, query logs", "beaver-dw_real", ()),
    ("BEAVER warehouse, generated", "beaver-dw", ()), ("BEAVER network, generated", "beaver-neutron", ()),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--audits", type=Path, default=Path("data/xbench/audits"))
    p.add_argument("--labels", type=Path, default=LABELS)
    args = p.parse_args()
    labels = [json.loads(line) for line in args.labels.read_text().splitlines()]
    print("| benchmark | golds | join | definite | labelled | agree | distorted | % golds | % joined |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for name, bench, left_out in ROWS:
        audits = [json.loads(line) for line in (args.audits / f"{bench}.jsonl").read_text().splitlines()]
        audits = [a for a in audits if a.get("db_id") not in left_out]
        n, j = len(audits), sum(bool(a.get("has_join")) for a in audits)
        d = sum(bool(a.get("realized")) for a in audits)
        mine = [r for r in labels if r["bench"] == bench]
        m, k = len(mine), sum(r["distorted"] for r in mine)
        agree = sum(r["first_label"]["label"] == r["second_label"]["label"] for r in mine)
        if m and m < d:
            lo, hi = wilson(k, m)
            est = k / m * d
            dist, pa, pj = f"~{est:.0f} ({lo * d:.0f}–{hi * d:.0f})", f"~{100 * est / n:.1f}", f"~{100 * est / j:.1f}"
        else:
            dist, pa, pj = str(k), f"{100 * k / n:.2f}", (f"{100 * k / j:.2f}" if j else "")
        print(f"| {name} | {n:,} | {100 * j / n:.0f}% | {d:,} | {m}/{d} | {agree}/{m} | {dist} | {pa} | {pj} |")


if __name__ == "__main__":
    main()
