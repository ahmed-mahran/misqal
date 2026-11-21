"""Fan-out audit of BIRD dev gold SQL.

    uv run python -m misqal.fanout.bird_audit --out runs/bird_audit.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from misqal.fanout.audit import QueryAudit, audit_query
from misqal.harness.bench import BENCH_ROOTS, load_bird


def summarize(audits: list[QueryAudit]) -> str:
    by_db: dict[str, list[QueryAudit]] = defaultdict(list)
    for a in audits:
        by_db[a.db_id].append(a)
    cols = ["n", "parse_err", "join", "exposed", "realized", "ambiguous", "rep_vals",
            "guarded", "topN_real"]
    lines = [f"{'db':28}" + "".join(f"{c:>10}" for c in cols)]

    def row(name: str, rs: list[QueryAudit]) -> str:
        vals = [
            len(rs),
            sum(a.parse_error is not None for a in rs),
            sum(a.has_join for a in rs),
            sum(a.exposed for a in rs),
            sum(a.realized for a in rs),
            sum(a.realized_ambiguous for a in rs),
            sum(a.repeated_values for a in rs),
            sum(a.guarded for a in rs),
            sum(any(b.top_n_realized for b in a.blocks) for a in rs),
        ]
        return f"{name:28}" + "".join(f"{v:>10}" for v in vals)

    for db in sorted(by_db):
        lines.append(row(db, by_db[db]))
    lines.append(row("ALL", audits))

    funcs: Counter = Counter()
    for a in audits:
        for b in a.blocks:
            for g in b.aggs:
                if g.realized and not g.ambiguous:
                    funcs["COUNT(*)" if g.star else g.func] += 1
    lines.append("\nrealized exposed aggregates by function: " + ", ".join(
        f"{k} {v}" for k, v in funcs.most_common()))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", type=Path, default=BENCH_ROOTS["bird"])
    p.add_argument("--db", nargs="*")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    tasks = load_bird(args.root, args.db)

    def work(t):
        return audit_query(t.id, t.db_id, t.db_path, t.gold_sql or "", timeout=args.timeout)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        audits = list(pool.map(work, tasks))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for a in audits:
            f.write(json.dumps(a.as_dict()) + "\n")
    print(summarize(audits))


if __name__ == "__main__":
    main()
