"""Fan-out audit of the gold SQL of other benchmarks.

    uv run python -m misqal.xbench.audit spider1-dev spider1-test spider1-train kaggledbqa \\
        spider2-lite atis advising fiben beaver-dw beaver-dw_real beaver-neutron

Writes ``<out>/<benchmark>.jsonl``: one audit (``misqal.fanout.audit``) per gold, with the question.
Each distinct (database, SQL) pair is audited once. Data layout under ``--data``:

- ``spider1/spider_data/{dev,test,train_spider}.json`` and ``database/``, ``test_database/``
  (Spider 1.0; ``train_others`` is not audited: its databases ship without rows);
- ``kaggledbqa/``: the release's ``examples/*.json`` and its ``*.sqlite`` databases, anywhere below;
- ``spider2/spider2-lite/``: ``spider2-lite.jsonl``, ``evaluation_suite/gold/sql/local*.sql`` and
  ``resource/databases/spider2-localdb/`` (only the SQLite questions with public gold);
- ``xbench/golds/{text2sql,fiben,beaver}.jsonl``: the converted golds (see ``misqal.xbench``).

A converted gold is kept only if it runs within 60 s, which depends on the machine. By default the
converted benchmarks are therefore restricted to the gold ids the paper audited
(``benchmarks/other-benchmarks/golds.json``); ``--ids ''`` audits every converted gold.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from misqal.fanout.audit import audit_query

Gold = tuple[str, str, Path | None, str, str]  # id, db_id, database, sql, question


def spider1(data: Path, split: str) -> Iterator[Gold]:
    base = data / "spider1" / "spider_data"
    dbdir = base / ("test_database" if split == "test" else "database")
    for i, q in enumerate(json.loads((base / f"{split}.json").read_text())):
        yield f"{split}-{i}", q["db_id"], dbdir / q["db_id"] / f"{q['db_id']}.sqlite", q["query"], q["question"]


def kaggledbqa(data: Path) -> Iterator[Gold]:
    root = data / "kaggledbqa"
    dbs = {p.stem: p for p in root.rglob("*.sqlite")}
    for f in sorted(root.rglob("examples/*.json")):
        if "fewshot" in f.stem or f.stem.endswith("_test"):
            continue
        for i, q in enumerate(json.loads(f.read_text())):
            yield f"{f.stem}-{i}", q["db_id"], dbs.get(q["db_id"]), q["query"], q["question"]


def spider2_lite(data: Path) -> Iterator[Gold]:
    root = data / "spider2" / "spider2-lite"
    for line in (root / "spider2-lite.jsonl").read_text().splitlines():
        t = json.loads(line)
        gold = root / "evaluation_suite" / "gold" / "sql" / f"{t['instance_id']}.sql"
        if t["instance_id"].startswith("local") and gold.exists():
            yield (t["instance_id"], t["db"], root / "resource/databases/spider2-localdb" / f"{t['db']}.sqlite",
                   gold.read_text(), t["question"])


def converted(data: Path, file: str, bench: str) -> Iterator[Gold]:
    for line in (data / "xbench" / "golds" / file).read_text().splitlines():
        r = json.loads(line)
        if r["bench"] == bench:
            yield r["id"], r["db_id"], Path(r["db"]), r["sql"], r["question"]


SOURCES = {
    "spider1-dev": lambda d: spider1(d, "dev"),
    "spider1-test": lambda d: spider1(d, "test"),
    "spider1-train": lambda d: spider1(d, "train_spider"),
    "kaggledbqa": kaggledbqa,
    "spider2-lite": spider2_lite,
    "atis": lambda d: converted(d, "text2sql.jsonl", "atis"),
    "advising": lambda d: converted(d, "text2sql.jsonl", "advising"),
    "fiben": lambda d: converted(d, "fiben.jsonl", "fiben"),
    "beaver-dw": lambda d: converted(d, "beaver.jsonl", "beaver-dw"),
    "beaver-dw_real": lambda d: converted(d, "beaver.jsonl", "beaver-dw_real"),
    "beaver-neutron": lambda d: converted(d, "beaver.jsonl", "beaver-neutron"),
}
FLAGS = ("missing_db", "parse_error", "has_join", "exposed", "realized", "realized_ambiguous", "repeated_values")


def _audit(g: Gold) -> dict:
    qid, db_id, db, sql, question = g
    if db is None or not Path(db).exists():
        return {"id": qid, "db_id": db_id, "missing_db": True}
    return {**audit_query(qid, db_id, Path(db), sql, timeout=60.0).as_dict(), "question": question}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("benchmarks", nargs="+", choices=sorted(SOURCES))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("data/xbench/audits"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ids", default="benchmarks/other-benchmarks/golds.json",
                   help="JSON {benchmark: [gold id]} restricting the converted benchmarks; '' for all")
    args = p.parse_args()
    keep = {k: set(v) for k, v in json.loads(Path(args.ids).read_text()).items()} if args.ids else {}
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.benchmarks:
        golds = list(SOURCES[name](args.data))
        if name in keep:
            have = {g[0] for g in golds}
            golds = [g for g in golds if g[0] in keep[name]]
            if missing := keep[name] - have:
                print(f"{name}: {len(missing)} listed golds did not convert or run, e.g. {sorted(missing)[:5]}")
        uniq = {(str(g[2]), g[3]): g for g in reversed(golds)}  # first gold of each (db, SQL)
        with ProcessPoolExecutor(args.workers) as pool:
            done = dict(zip(uniq, pool.map(_audit, list(uniq.values()), chunksize=4)))
        rows = [{**done[(str(g[2]), g[3])], "id": g[0], "question": g[4]} for g in golds]
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        n = len(rows)
        print(f"{name}: {n} golds, {len(uniq)} distinct; "
              + ", ".join(f"{k} {sum(bool(r.get(k)) for r in rows)}" for k in FLAGS), flush=True)


if __name__ == "__main__":
    main()
