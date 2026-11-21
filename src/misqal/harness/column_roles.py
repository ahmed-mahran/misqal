"""Column roles: identifier, dimension or measure, for every column of the tables a benchmark uses.

A data-preparation step, done once per table and independent of any question (like a data
catalog). Evidence arms profile only identifiers and dimensions: measures (numeric quantities one
sums or averages) carry no grain and add noise such as accidental keys.

1. **Features** per column, from the data: declared type, primary / foreign key, whether its name
   is a key column elsewhere in the database, rows, distinct and NULL counts, share of numeric
   values, a few sample values, and the benchmark's column description when it ships one (BIRD's
   ``database_description/<table>.csv``).
2. **Heuristic**: keys are identifiers, non-numeric columns and dates are dimensions, non-integer
   numbers are measures; other integer columns are left open.
3. **Judges**: two LLMs label every column of a table in one call, without seeing the heuristic.
   Where they disagree, a third judge decides.

    uv run python -m misqal.harness.column_roles --bench traps bird-fanout-20251106 \
        bird-fanout-train --judges <a> <b> --tiebreak <c> --out benchmarks/column-roles.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from misqal.harness.agent import known_tables
from misqal.harness.bench import load_tasks
from misqal.fanout.build_subset import judge_name
from misqal.harness.models import load_dotenv, make_model, over_budget

ROLES = ("identifier", "dimension", "measure")
NUMERIC_TYPES = ("INT", "REAL", "NUM", "DEC", "FLOA", "DOUB")
DATE_TYPES = ("DATE", "TIME")

SYSTEM = """You label the columns of a database table by role, as a data modeller would:
- "identifier": identifies a row of this table or refers to a row of another table (keys, codes or
  numbers used to join or look up), even when numeric.
- "dimension": describes, groups or filters entities: names, categories, flags, dates and years,
  codes with a fixed meaning, free text.
- "measure": a numeric quantity it makes sense to add up or average: amounts, prices, counts,
  scores, durations, sizes.
Judge from the column's meaning, its description, and its values; not from its type alone."""

JSON_INSTRUCTION = ('\n\nReply with one JSON object and nothing else, mapping every column name to '
                    '"identifier", "dimension" or "measure".')


def _description(db: Path, table: str) -> dict[str, str]:
    path = db.parent / "database_description" / f"{table}.csv"
    if not path.exists():
        return {}
    out = {}
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with path.open(encoding=enc, newline="") as f:
                for r in csv.DictReader(f):
                    name = (r.get("original_column_name") or "").strip()
                    text = " ".join(x for x in ((r.get("column_description") or "").strip(),
                                                (r.get("value_description") or "").strip()) if x)
                    if name:
                        out[name.lower()] = text[:300]
            return out
        except UnicodeDecodeError:
            continue
    return out


def table_features(db: Path, table: str) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        keys_elsewhere = set()
        for t in tables:
            for c in con.execute(f'PRAGMA table_info("{t}")'):
                if c[5] and t != table:
                    keys_elsewhere.add(c[1].lower())
            for fk in con.execute(f'PRAGMA foreign_key_list("{t}")'):
                keys_elsewhere.add(fk[3].lower())
                keys_elsewhere.add((fk[4] or "").lower())
        fks = {fk[3].lower() for fk in con.execute(f'PRAGMA foreign_key_list("{table}")')}
        rows = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        desc = _description(db, table)
        out = []
        for c in con.execute(f'PRAGMA table_info("{table}")').fetchall():
            name, decl = c[1], (c[2] or "").upper()
            q = f'"{name}"'
            distinct, nulls, numeric, fractional = con.execute(
                f"SELECT COUNT(DISTINCT {q}), SUM({q} IS NULL), "
                f"SUM(typeof({q}) IN ('integer','real')), "
                f"SUM(typeof({q}) = 'real' AND {q} <> CAST({q} AS INTEGER)) FROM \"{table}\"").fetchone()
            sample = [str(v)[:40] for (v,) in con.execute(
                f'SELECT DISTINCT {q} FROM "{table}" WHERE {q} IS NOT NULL LIMIT 5')]
            out.append({"column": name, "type": decl, "pk": bool(c[5]), "fk": name.lower() in fks,
                        "key_elsewhere": name.lower() in keys_elsewhere, "rows": rows,
                        "distinct": distinct, "nulls": nulls or 0,
                        "numeric_share": round((numeric or 0) / max(rows - (nulls or 0), 1), 3),
                        "fractional": bool(fractional), "sample": sample,
                        "description": desc.get(name.lower(), "")})
        return out


def heuristic(f: dict[str, Any]) -> str | None:
    if f["pk"] or f["fk"] or f["key_elsewhere"]:
        return "identifier"
    if any(t in f["type"] for t in DATE_TYPES):
        return "dimension"
    if f["numeric_share"] < 0.5:
        return "dimension"
    if f["fractional"] or any(t in f["type"] for t in ("REAL", "FLOA", "DOUB", "DEC")):
        return "measure"
    return None  # integers that are not keys: years, codes, counts, ratings ... left to the judges


def user_prompt(table: str, feats: list[dict[str, Any]]) -> str:
    lines = [f"Table `{table}` ({feats[0]['rows'] if feats else 0} rows). Columns:"]
    for f in feats:
        key = "primary key" if f["pk"] else "foreign key" if f["fk"] else ""
        lines.append(f"- `{f['column']}` {f['type'] or 'untyped'}{', ' + key if key else ''}; "
                     f"{f['distinct']} distinct, {f['nulls']} NULL; samples: {f['sample']}"
                     + (f"; description: {f['description']}" if f["description"] else ""))
    return "\n".join(lines)


def judge(spec: str, table: str, feats: list[dict[str, Any]]) -> dict[str, str]:
    if over_budget(spec):
        raise RuntimeError("daily budget reached")
    model = make_model(spec, max_tokens=4096, temperature=None)
    messages = [("system", SYSTEM + JSON_INSTRUCTION), ("user", user_prompt(table, feats))]
    for attempt in range(2):  # one retry on malformed JSON
        reply = model.invoke(messages)
        text = reply.text if hasattr(reply, "text") else str(reply.content)
        try:
            raw = json.loads(text[text.find("{"):text.rfind("}") + 1])
            names = {f["column"].lower(): f["column"] for f in feats}
            return {names[k.lower()]: v for k, v in raw.items() if k.lower() in names and v in ROLES}
        except (ValueError, AttributeError):
            if attempt:
                raise
    return {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bench", nargs="+", required=True)
    p.add_argument("--judges", nargs=2, required=True)
    p.add_argument("--tiebreak", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    load_dotenv()

    tables: dict[tuple[str, str], Path] = {}
    for bench in args.bench:
        for t in load_tasks(bench):
            for table in known_tables(t):
                tables[(t.db_id, table)] = t.db_path
    done = set()
    if args.out.exists():
        done = {(r["db_id"], r["table"]) for r in map(json.loads, args.out.read_text().splitlines())}
    todo = [k for k in sorted(tables) if k not in done]
    print(f"{len(tables)} tables, {len(todo)} to label")

    def work(key: tuple[str, str]) -> list[dict[str, Any]]:
        db_id, table = key
        feats = table_features(tables[key], table)
        votes = {}
        for spec in args.judges:
            try:
                votes[spec] = judge(spec, table, feats)
            except Exception as e:  # recorded; the tiebreak or heuristic decides
                votes[spec] = {"_error": f"{type(e).__name__}: {e}"[:200]}
        a, b = (votes[s] for s in args.judges)
        split = [f["column"] for f in feats if a.get(f["column"]) != b.get(f["column"])
                 or a.get(f["column"]) is None]
        tie = {}
        if split:
            try:
                tie = judge(args.tiebreak, table, feats)
            except Exception as e:
                tie = {"_error": f"{type(e).__name__}: {e}"[:200]}
        rows = []
        for f in feats:
            c = f["column"]
            if c in split:
                role, source = tie.get(c) or heuristic(f) or "dimension", "tiebreak"
            else:
                role, source = a[c], "judges"
            rows.append({"db_id": db_id, "table": table, **f, "heuristic": heuristic(f),
                         "judges": {judge_name(s): votes[s].get(c) for s in args.judges},
                         "tiebreak": tie.get(c), "role": role, "source": source})
        return rows

    with ThreadPoolExecutor(args.workers) as pool, args.out.open("a") as out:
        for rows in pool.map(work, todo):
            out.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            out.flush()

    rows = list(map(json.loads, args.out.read_text().splitlines()))
    print("roles:", dict(Counter(r["role"] for r in rows)), "| source:", dict(Counter(r["source"] for r in rows)))
    agree = [r for r in rows if r["heuristic"]]
    print(f"heuristic decided {len(agree)}/{len(rows)}; final role equals it for "
          f"{sum(r['heuristic'] == r['role'] for r in agree)}")
    j = [tuple(r["judges"].values()) for r in rows]
    print(f"judges agree on {sum(len(set(v)) == 1 and None not in v for v in j)}/{len(rows)}")


if __name__ == "__main__":
    main()
