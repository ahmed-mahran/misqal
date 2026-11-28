"""ATIS and Advising gold queries (Finegan-Dollak et al., 2018) on SQLite.

    uv run python -m misqal.xbench.mysql_dump <repo>/data/atis-db.sql data/xbench/atis.sqlite
    uv run python -m misqal.xbench.mysql_dump <repo>/data/advising-db.sql data/xbench/advising.sqlite
    uv run python -m misqal.xbench.text2sql --repo <text2sql-data repo> --db-dir data/xbench \\
        --out data/xbench/golds/text2sql.jsonl

One gold per (question, SQL) pair of every sentence, with the entry's example value for each
variable the sentence leaves empty. The MySQL gold is transpiled to SQLite with sqlglot and kept if
it runs within 60 s.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sqlglot

from misqal.xbench._run import run_sql, write_jsonl

BENCHES = ("atis", "advising")


def substitute(entry: dict, sentence: dict) -> tuple[str, str]:
    """The release's own rule (an empty value takes the entry's example value), with longer names
    first so that, e.g., ``number0`` cannot overwrite part of ``flight_number0``."""
    text, sql = sentence["text"], entry["sql"][0]
    vals = dict(sentence["variables"])
    for v in entry["variables"]:
        if vals.get(v["name"], "") == "":
            vals[v["name"]] = v["example"]
    for name in sorted(vals, key=len, reverse=True):
        text, sql = text.replace(name, vals[name]), sql.replace(name, vals[name])
    return text, sql


def _run(args: tuple[str, str]) -> str:
    return run_sql(*args)[0]


def golds(repo: Path, db_dir: Path, bench: str) -> tuple[list[dict], Counter]:
    db = db_dir / f"{bench}.sqlite"
    pairs: dict[tuple[str, str], str] = {}
    for i, e in enumerate(json.loads((repo / "data" / f"{bench}.json").read_text())):
        for j, s in enumerate(e["sentences"]):
            pairs.setdefault(substitute(e, s), f"{bench}-{i:04d}-{j:03d}")
    sqlite_of, stats = {}, Counter()
    for mysql in sorted({sql for _, sql in pairs}):
        try:
            (sqlite_of[mysql],) = sqlglot.transpile(mysql, read="mysql", write="sqlite")
        except Exception:  # noqa: BLE001
            stats["transpile error"] += 1
    distinct = sorted(set(sqlite_of.values()))
    with ProcessPoolExecutor(8) as pool:
        status = dict(zip(distinct, pool.map(_run, [(str(db), s) for s in distinct], chunksize=1)))
    rows = []
    for (q, mysql), qid in sorted(pairs.items(), key=lambda kv: kv[1]):
        sql = sqlite_of.get(mysql)
        if sql is None or status[sql] != "ok":
            stats["dropped"] += 1
            continue
        rows.append({"bench": bench, "id": qid, "db_id": bench, "db": str(db), "question": q, "sql": sql})
    stats["kept"] = len(rows)
    return rows, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--repo", type=Path, required=True, help="clone of jkkummerfeld/text2sql-data")
    p.add_argument("--db-dir", type=Path, default=Path("data/xbench"))
    p.add_argument("--out", type=Path, default=Path("data/xbench/golds/text2sql.jsonl"))
    args = p.parse_args()
    rows = []
    for bench in BENCHES:
        r, stats = golds(args.repo, args.db_dir, bench)
        print(bench, dict(stats))
        rows += r
    write_jsonl(args.out, rows)


if __name__ == "__main__":
    main()
