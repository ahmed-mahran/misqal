"""BEAVER (Chen et al., 2024) gold queries on SQLite.

    uv run python -m misqal.xbench.mysql_dump <dw dump>.sql data/xbench/dw.sqlite
    uv run python -m misqal.xbench.mysql_dump <neutron dump>.sql data/xbench/neutron.sqlite
    uv run --extra xbench python -m misqal.xbench.beaver --queries <beaver-query>/data \\
        --db-dir data/xbench --out data/xbench/golds/beaver.jsonl

Reads the dw, dw_real and neutron question files of the ``beaverbench/beaver-query`` release
(parquet), transpiles each MySQL gold to SQLite with sqlglot and keeps it if it runs within 60 s
(timeouts are retried once with fewer workers). MySQL's VARIANCE/STDDEV have no SQLite
counterpart; they are rebuilt from AVG/SUM/COUNT, which keeps their sensitivity to repeated rows.
ASCII becomes UNICODE, and identifiers spelled like SQLite keywords are quoted.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sqlglot
from sqlglot import exp

from misqal.xbench._run import run_sql, write_jsonl

logging.getLogger("sqlglot").setLevel(logging.ERROR)  # unsupported-syntax warnings are expected

BENCHES = [("dw", "beaver-dw"), ("dw_real", "beaver-dw_real"), ("neutron", "beaver-neutron")]

# SQLite keywords (https://sqlite.org/lang_keywords.html); identifiers spelled like one get quoted.
SQLITE_KW = set("""ABORT ACTION ADD AFTER ALL ALTER ALWAYS ANALYZE AND AS ASC ATTACH AUTOINCREMENT BEFORE BEGIN
BETWEEN BY CASCADE CASE CAST CHECK COLLATE COLUMN COMMIT CONFLICT CONSTRAINT CREATE CROSS CURRENT CURRENT_DATE
CURRENT_TIME CURRENT_TIMESTAMP DATABASE DEFAULT DEFERRABLE DEFERRED DELETE DESC DETACH DISTINCT DO DROP EACH ELSE
END ESCAPE EXCEPT EXCLUDE EXCLUSIVE EXISTS EXPLAIN FAIL FILTER FIRST FOLLOWING FOR FOREIGN FROM FULL GENERATED GLOB
GROUP GROUPS HAVING IF IGNORE IMMEDIATE IN INDEX INDEXED INITIALLY INNER INSERT INSTEAD INTERSECT INTO IS ISNULL
JOIN KEY LAST LEFT LIKE LIMIT MATCH MATERIALIZED NATURAL NO NOT NOTHING NOTNULL NULL NULLS OF OFFSET ON OR ORDER
OTHERS OUTER OVER PARTITION PLAN PRAGMA PRECEDING PRIMARY QUERY RAISE RANGE RECURSIVE REFERENCES REGEXP REINDEX
RELEASE RENAME REPLACE RESTRICT RETURNING RIGHT ROLLBACK ROW ROWS SAVEPOINT SELECT SET TABLE TEMP TEMPORARY THEN TIES
TO TRANSACTION TRIGGER UNBOUNDED UNION UNIQUE UPDATE USING VACUUM VALUES VIEW VIRTUAL WHEN WHERE WINDOW WITH
WITHOUT""".split())


def _R(x):  # noqa: N802
    return exp.Cast(this=x.copy(), to=exp.DataType.build("REAL"))


def _moments(kind, x):
    """Rebuild VAR/STDDEV from AVG/SUM/COUNT, which SQLite has (MySQL VARIANCE/STDDEV are population)."""
    sq = lambda: exp.Mul(this=_R(x), expression=_R(x))
    if kind in ("var_pop", "std_pop"):
        v = exp.Sub(this=exp.Avg(this=sq()),
                    expression=exp.Mul(this=exp.Avg(this=_R(x)), expression=exp.Avg(this=_R(x))))
    else:
        num = exp.Sub(this=exp.Sum(this=sq()),
                      expression=exp.Div(this=exp.Mul(this=exp.Sum(this=_R(x)), expression=exp.Sum(this=_R(x))),
                                         expression=exp.Count(this=x.copy())))
        den = exp.Nullif(this=exp.Sub(this=exp.Count(this=x.copy()), expression=exp.Literal.number(1)),
                         expression=exp.Literal.number(0))
        v = exp.Div(this=exp.Paren(this=num), expression=den)
    if kind.startswith("std"):
        return exp.Anonymous(this="SQRT", expressions=[exp.Abs(this=v)])
    return exp.Paren(this=v)


KIND = {exp.VariancePop: "var_pop", exp.Variance: "var_samp", exp.StddevPop: "std_pop", exp.StddevSamp: "std_samp"}


def rewrite(tree, tags):
    for node in list(tree.find_all(*KIND)):
        x = node.this
        if isinstance(x, exp.Distinct) or node.expressions:
            continue  # leave unsupported form untouched
        new = _moments(KIND[type(node)], x)
        win = node.parent if isinstance(node.parent, exp.Window) and node.parent.this is node else None
        if win is not None:
            for agg in list(new.find_all(exp.Avg, exp.Sum, exp.Count)):
                w = win.copy(); w.set("this", agg.copy()); agg.replace(w)
            win.replace(new)
        else:
            node.replace(new)
        tags.add("moments")
    for node in list(tree.find_all(exp.Func)):
        if type(node).__name__ == "Ascii" or (isinstance(node, exp.Anonymous) and str(node.this).upper() == "ASCII"):
            arg = node.this if not isinstance(node, exp.Anonymous) else node.expressions[0]
            node.replace(exp.Anonymous(this="UNICODE", expressions=[arg.copy()]))
            tags.add("ascii")
    for ident in tree.find_all(exp.Identifier):
        if not ident.quoted and ident.name.upper() in SQLITE_KW:
            ident.set("quoted", True)
            tags.add("kw_quote")
    return tree


def transpile(sql: str) -> tuple[list[str], set[str]]:
    tags: set[str] = set()
    trees = sqlglot.parse(sql, read="mysql")
    out = [rewrite(t, tags).sql(dialect="sqlite") for t in trees if t is not None]
    return [s for s in out if s.strip()], tags


def _run(task: tuple[int, str, str]) -> tuple[int, str]:
    key, db, sql = task
    return key, run_sql(db, sql)[0]


def golds(query_dir: Path, db_dir: Path, workers: int = 8) -> tuple[list[dict], Counter]:
    import pyarrow.parquet as pq

    recs, tasks, stats = [], [], Counter()
    for fname, bench in BENCHES:
        for r in pq.read_table(query_dir / f"{fname}-00000-of-00001.parquet").to_pylist():
            if r["db"] not in ("dw", "neutron"):
                stats["other database"] += 1
                continue
            try:
                out, tags = transpile(r["sql"])
            except Exception:  # noqa: BLE001
                out, tags = [], set()
            if len(out) != 1:
                stats["transpile error"] += 1
                continue
            stats.update(f"rewrite: {t}" for t in tags)
            db = str(db_dir / f"{r['db']}.sqlite")
            recs.append({"bench": bench, "id": r["id"], "db_id": r["db"], "db": db, "question": r["question"],
                         "sql": out[0]})
            tasks.append((len(recs) - 1, db, out[0]))
    status = {}
    with ProcessPoolExecutor(workers) as pool:
        status.update(pool.map(_run, tasks, chunksize=4))
    retry = [t for t in tasks if status[t[0]] == "timeout"]
    with ProcessPoolExecutor(3) as pool:  # less contention for the slow ones
        status.update(pool.map(_run, retry))
    rows = [r for k, r in enumerate(recs) if status[k] == "ok"]
    stats["dropped"] = len(recs) - len(rows)
    stats["kept"] = len(rows)
    return rows, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--queries", type=Path, required=True, help="the release's data/ directory (parquet files)")
    p.add_argument("--db-dir", type=Path, default=Path("data/xbench"))
    p.add_argument("--out", type=Path, default=Path("data/xbench/golds/beaver.jsonl"))
    p.add_argument("--workers", type=int, default=int(os.environ.get("NPROC", "8")))
    args = p.parse_args()
    rows, stats = golds(args.queries, args.db_dir, args.workers)
    write_jsonl(args.out, rows)
    print(dict(stats))


if __name__ == "__main__":
    main()
