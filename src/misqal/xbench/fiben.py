"""FIBEN (Sen et al., PVLDB 2020) on SQLite: the database from its Db2 DDL and CSV files, and the
gold queries.

    uv run python -m misqal.xbench.fiben db --src <FIBEN repo> --csv <csv dir> --out data/xbench/fiben.sqlite
    uv run python -m misqal.xbench.fiben golds --queries <queries.json> --db data/xbench/fiben.sqlite \\
        --out data/xbench/golds/fiben.jsonl

The CSVs follow Db2's ``import ... of del`` defaults: comma-separated, double-quoted strings, no
header, an empty unquoted field is NULL. Db2 timestamps ('YYYY-MM-DD-HH.MM.SS.ffffff') are stored as
ISO text and YYYYMMDD dates as 'YYYY-MM-DD', so SQLite's date functions and the gold queries'
literals behave as in Db2.

Gold queries are transpiled with sqlglot from its PostgreSQL dialect (the closest to Db2), with
MONTH/YEAR/DAY rewritten to STRFTIME and the FIBEN schema qualifier dropped. SQLite cannot resolve
outer columns in a subquery's GROUP BY; a subquery grouped only by outer columns, with only
SUM/AVG/MIN/MAX outputs, returns the same value without it, so that GROUP BY is dropped (four
golds). Paraphrases share a query id and gold; the first question of each id is kept.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import sqlglot
from sqlglot import exp

from misqal.xbench._run import run_sql, write_jsonl

TS = re.compile(r"^(\d{4}-\d{2}-\d{2})-(\d{2})\.(\d{2})\.(\d{2})(\.\d+)?$")


def _value(v: str, typ: str):
    if v == "":
        return None  # the CSVs have no quoted empty strings
    try:
        if typ in ("BIGINT", "INTEGER", "SMALLINT"):
            return int(v)
        if typ in ("DOUBLE", "DECIMAL", "REAL", "FLOAT"):
            return float(v)
    except ValueError:
        return v
    if typ == "TIMESTAMP" and (m := TS.match(v)):
        return f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}{m.group(5) or ''}"
    if typ == "DATE" and re.fullmatch(r"\d{8}", v):
        return f"{v[:4]}-{v[4:6]}-{v[6:]}"
    return v


def build_db(src: Path, csv_dir: Path, out: Path) -> dict[str, int]:
    ddl = (src / "FIBEN.sql").read_text()
    tables = [t.strip() for t in (src / "tablelist.txt").read_text().splitlines() if t.strip()]
    creates, fks = {}, defaultdict(list)
    q = lambda cols: ", ".join(f'"{c.strip()}"' for c in cols.split(","))  # noqa: E731
    for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
        if m := re.match(r"CREATE TABLE (\w+) \((.*)\)$", stmt, re.S):
            cols, cons = [], []
            for part in (p.strip() for p in re.split(r",\s*(?![^()]*\))", m.group(2))):
                if pk := re.match(r"PRIMARY KEY \((.*)\)$", part):
                    cons.append(f"PRIMARY KEY ({q(pk.group(1))})")
                elif uq := re.match(r"UNIQUE \((.*)\)$", part):
                    cons.append(f"UNIQUE ({q(uq.group(1))})")
                else:
                    cols.append(re.match(r"(\w+) (.*)$", part).groups())
            creates[m.group(1)] = (cols, cons)
        elif m := re.match(r"ALTER TABLE (\w+) ADD CONSTRAINT (\w+) FOREIGN KEY \((.*?)\) REFERENCES (\w+) \((.*?)\)$",
                           stmt, re.S):
            t, name, c, rt, rc = m.groups()
            fks[t].append(f'CONSTRAINT "{name}" FOREIGN KEY ({q(c)}) REFERENCES "{rt}" ({q(rc)})')
        else:
            raise ValueError(f"unparsed DDL: {stmt[:200]}")
    assert set(creates) == set(tables), set(creates) ^ set(tables)
    out.unlink(missing_ok=True)
    con = sqlite3.connect(out)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    rows_per_table = {}
    for name, (cols, cons) in creates.items():
        con.execute(f'CREATE TABLE "{name}" (\n  ' + ",\n  ".join([f'"{c}" {t}' for c, t in cols] + cons + fks[name])
                    + "\n)")
        types = [t.split()[0].upper() for _, t in cols]
        with open(csv_dir / f"{name}.csv", newline="", encoding="utf-8") as f:
            rows = [tuple(_value(v, t) for v, t in zip(rec, types)) for rec in csv.reader(f) if len(rec) == len(types)]
        con.executemany(f'INSERT INTO "{name}" VALUES ({",".join("?" * len(types))})', rows)
        rows_per_table[name] = len(rows)
    con.execute("ANALYZE")
    con.commit()
    con.close()
    return rows_per_table


DATEPART = {"MONTH": "%m", "YEAR": "%Y", "DAY": "%d"}


def _rewrite(node):
    name = None
    if isinstance(node, exp.Anonymous) and str(node.this).upper() in DATEPART:
        name, arg = str(node.this).upper(), node.expressions[0]
    elif isinstance(node, (exp.Month, exp.Year, exp.Day)):
        name, arg = type(node).__name__.upper(), node.this
    if name:
        return exp.Cast(this=exp.Anonymous(this="STRFTIME", expressions=[exp.Literal.string(DATEPART[name]),
                                                                         arg.copy()]),
                        to=exp.DataType.build("INTEGER"))
    if isinstance(node, exp.Table) and node.args.get("db") is not None and node.db.upper() == "FIBEN":
        node.set("db", None)
        node.set("catalog", None)
    return node


def _drop_outer_group_by(node):
    if not isinstance(node, exp.Select) or node.parent is None or not node.args.get("group"):
        return node
    local = {(s.this.alias_or_name or "").upper()
             for s in [node.args.get("from")] + list(node.args.get("joins") or []) if s is not None}
    keys = node.args["group"].expressions
    if (keys and all(isinstance(k, exp.Column) and k.table and k.table.upper() not in local for k in keys)
            and all(isinstance(e.unalias(), (exp.Sum, exp.Avg, exp.Min, exp.Max)) for e in node.expressions)
            and not node.args.get("having")):
        node.set("group", None)
    return node


def transpile(sql: str) -> str:
    (tree,) = sqlglot.parse(sql, read="postgres")
    return tree.transform(_rewrite).transform(_drop_outer_group_by).sql(dialect="sqlite")


def golds(queries: Path, db: Path) -> tuple[list[dict], Counter]:
    rows, seen, stats = [], set(), Counter()
    for q in json.loads(queries.read_text()):
        qid = str(q["uniqueQueryID"])
        if qid in seen:
            continue
        seen.add(qid)
        try:
            sql = transpile(q["SQL"].strip())
        except Exception:  # noqa: BLE001
            stats["transpile error"] += 1
            continue
        status, _, _ = run_sql(db, sql)
        if status != "ok":
            stats["dropped"] += 1
            continue
        rows.append({"bench": "fiben", "id": qid, "db_id": "fiben", "db": str(db),
                     "question": q["question"].strip(), "sql": sql})
    stats["kept"] = len(rows)
    return rows, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("db")
    d.add_argument("--src", type=Path, required=True, help="directory with FIBEN.sql and tablelist.txt")
    d.add_argument("--csv", type=Path, required=True, help="directory with one CSV per table")
    d.add_argument("--out", type=Path, default=Path("data/xbench/fiben.sqlite"))
    g = sub.add_parser("golds")
    g.add_argument("--queries", type=Path, required=True, help="FIBEN's JSON file of questions and SQL")
    g.add_argument("--db", type=Path, default=Path("data/xbench/fiben.sqlite"))
    g.add_argument("--out", type=Path, default=Path("data/xbench/golds/fiben.jsonl"))
    args = p.parse_args()
    if args.cmd == "db":
        n = build_db(args.src, args.csv, args.out)
        print(json.dumps({"tables": len(n), "rows": sum(n.values())}))
    else:
        rows, stats = golds(args.queries, args.db)
        write_jsonl(args.out, rows)
        print(dict(stats))


if __name__ == "__main__":
    main()
