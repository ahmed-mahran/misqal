"""Grain-aware FD evidence: the paper's Grain-FD arms (code: ``full2``, in the prompt; ``facts2``, as a
tool).

Added to the FD profile and facts (``full`` / ``facts``), for the shown tables and computed from the
data:

- **Blanks apart from values.** The FD profile counts NULL as a value (the semantics of GROUP BY), so
  an entity whose attribute is sometimes missing looks as if it had two values. Here every pair
  whose target has NULLs is re-checked on the filled-in rows only, with where the blanks fall.
- **Rows per value** for identifier columns, with the consequence for a join on them: each matching
  row of another table is repeated that many times.
- **Measure grain.** The FD profile leaves measures out. Here each measure gets the coarsest identifier
  that determines it and how many rows share each value: a sum over rows counts it that often.
- **Keys always present.** The tool adds the table's identifier columns to the ones the model asks
  for, so a fact never goes missing because the model omitted a key.

Roles (identifier / dimension / measure) come from `benchmarks/column-roles.jsonl`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from misqal.cardinality.models import CardinalityProfile
from misqal.harness.profiles import _db_key, _FileLock, table_profile
from misqal.paths import shipped

ROLES_FILE = shipped("benchmarks/column-roles.jsonl")
V2_CACHE_DIR = Path("runs/cache/evidence-v2")
MAX_COLUMNS = 25


@lru_cache(maxsize=1)
def _roles() -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    if ROLES_FILE.exists():
        for r in map(json.loads, ROLES_FILE.read_text().splitlines()):
            out.setdefault((r["db_id"], r["table"].lower()), {})[r["column"]] = r["role"]
    return out


def roles_for(db_id: str, table: str) -> dict[str, str]:
    return _roles().get((db_id, table.lower()), {})


QUERY_SECONDS = 60.0  # per heavy scan; a fact whose query runs longer is left out and said so
NEAR_UNIQUE = 0.9  # columns with distinct values >= this share of filled rows are not grain candidates


def _q(con: sqlite3.Connection, sql: str, seconds: float | None = None) -> tuple | None:
    """One row, or None when the query runs longer than ``seconds``."""
    if seconds is None:
        return con.execute(sql).fetchone()
    deadline = time.monotonic() + seconds
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 100_000)
    try:
        return con.execute(sql).fetchone()
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            return None
        raise
    finally:
        con.set_progress_handler(None, 0)


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:,.1f}".rstrip("0").rstrip(".")


def table_facts(db: Path, db_id: str, table: str, seconds: float | None = None) -> dict[str, Any]:
    """Facts over every column of ``table``: computed once (precompute before runs) and cached;
    each tool call then renders the part for the columns it asks about."""
    return compute(db, db_id, table, table_columns(db, table), seconds)


def compute(db: Path, db_id: str, table: str, columns: list[str],
            seconds: float | None = None) -> dict[str, Any]:
    """Facts for ``table`` over ``columns`` (non-measures profiled pairwise; measures get a grain).
    Cached per table and column list."""
    db = db.resolve()
    key = hashlib.sha1(json.dumps([table, columns]).encode()).hexdigest()[:16]
    path = V2_CACHE_DIR / f"{db.stem}-{_db_key(db)}" / f"{key}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("columns") == columns:
            return cached
    with _FileLock(path):
        if path.exists() and json.loads(path.read_text()).get("columns") == columns:
            return json.loads(path.read_text())
        out = _compute(db, db_id, table, columns, QUERY_SECONDS if seconds is None else seconds)
        path.write_text(json.dumps(out))
        return out


def _compute(db: Path, db_id: str, table: str, columns: list[str], seconds: float) -> dict[str, Any]:
    roles = roles_for(db_id, table)
    measures = [c for c in columns if roles.get(c) == "measure"]
    profiled = [c for c in columns if c not in measures]
    profile = table_profile(db, table, profiled) if len(profiled) >= 2 else None
    t = f'"{table}"'
    sep = "char(31)"  # joins two values into one distinct key
    pair = lambda a, b: f'"{a}" || {sep} || "{b}"'  # noqa: E731
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        rows = _q(con, f"SELECT COUNT(*) FROM {t}")[0]
        res = _q(con, "SELECT " + ", ".join(f'SUM("{c}" IS NULL), COUNT(DISTINCT "{c}")' for c in columns)
                 + f" FROM {t}")
        nulls = {c: res[2 * i] or 0 for i, c in enumerate(columns)}
        distinct = {c: res[2 * i + 1] for i, c in enumerate(columns)}
        keys = [c for c in profiled if profile and c in profile.primary_keys]
        identifiers = [c for c in profiled if roles.get(c) == "identifier" and c not in keys]
        per_value = {}
        for c in identifiers:
            lo, mean, hi, n = _q(con, f'SELECT MIN(n), AVG(n), MAX(n), COUNT(*) FROM '
                                      f'(SELECT COUNT(*) n FROM {t} WHERE "{c}" IS NOT NULL GROUP BY "{c}")')
            if n and hi > 1:
                per_value[c] = {"min": lo, "mean": mean, "max": hi, "values": n}
        # pairs whose target b has blanks, re-checked on filled-in rows: one scan per b for all a.
        # a -> b on filled rows iff distinct a (where b filled) == distinct (a, b) pairs (where b filled);
        # a values with both blank and filled b = filled_a + blank_a - distinct a.
        # Only pairs the FD profile says do NOT hold can change (blanks can only split groups), and
        # near-unique columns are skipped as sources (they determine almost everything).
        near_unique = {c for c in columns if distinct[c] >= NEAR_UNIQUE * max(rows - nulls[c], 1)}
        filled, skipped = {}, []
        for b in profiled:
            others = [a for a in profiled if a != b and a not in keys and a not in near_unique
                      and profile is not None and profile.determines(a, b) is False]
            if not nulls.get(b) or b in keys or not others:
                continue
            res = _q(con, "SELECT " + ", ".join(
                f'COUNT(DISTINCT CASE WHEN "{b}" IS NOT NULL THEN "{a}" END), '
                f'COUNT(DISTINCT CASE WHEN "{b}" IS NOT NULL THEN {pair(a, b)} END), '
                f'COUNT(DISTINCT CASE WHEN "{b}" IS NULL THEN "{a}" END)' for a in others) + f" FROM {t}",
                seconds)
            if res is None:
                skipped.append(["blanks", b])
                continue
            for i, a in enumerate(others):
                fa, fpairs, blank_a = res[3 * i: 3 * i + 3]
                filled[f"{a}::{b}"] = {"determines": fa == fpairs, "values": distinct[a],
                                       "mixed": max(fa + blank_a - distinct[a], 0)}
        # measure grain: the coarsest non-measure column that determines the measure; one scan per measure
        candidates = [c for c in columns if roles.get(c, "dimension") != "measure" and c not in near_unique]
        grains = {}
        for m in measures:
            res = _q(con, "SELECT " + ", ".join(
                f'COUNT(DISTINCT CASE WHEN "{m}" IS NOT NULL THEN "{k}" END), '
                f'COUNT(DISTINCT CASE WHEN "{m}" IS NOT NULL THEN {pair(k, m)} END)' for k in candidates)
                + f" FROM {t}", seconds) if candidates else ()
            if res is None:
                skipped.append(["grain", m])
                grains[m] = None
                continue
            best = None
            for i, k in enumerate(candidates):
                dk, dkm = res[2 * i], res[2 * i + 1]
                if dk and dk == dkm and dk < rows and (best is None or dk < best[1]):
                    best = (k, dk)
            grains[m] = {"per": best[0], "values": best[1], "rows_per": rows / best[1]} if best else None
    return {"columns": columns, "table": table, "rows": rows, "nulls": nulls, "keys": keys,
            "per_value": per_value, "filled": filled, "grains": grains, "measures": measures,
            "skipped": skipped, "profile": profile.model_dump(mode="json") if profile else None}


def render_facts(f: dict[str, Any], columns: list[str] | None = None) -> str:
    """Plain sentences (Grain-FD (tool)), for ``columns`` (default: all) of the table."""
    table, rows = f["table"], f["rows"]
    show = set(columns) if columns else set(f["columns"])
    prof = CardinalityProfile.model_validate(f["profile"]) if f["profile"] else None
    keys = f["keys"]
    head = f"`{table}` ({rows:,} rows). " + ("Each row has its own " + " / ".join(f"`{k}`" for k in keys) + "."
                                            if keys else "No single column identifies a row.")
    lines = [head]
    for c, v in f["per_value"].items():
        if c in show:
            lines.append(f"- Each `{c}` appears on {v['min']}–{v['max']} rows ({_fmt(v['mean'])} on average): "
                         f"joining another table on `{c}` repeats each of its matching rows that many times.")
    if prof:
        cols = [c for c in prof.used_columns or sorted(prof.column_counts) if c not in keys and c in show]
        for a in cols:
            one, many, gaps = [], [], []
            for b in cols:
                if b == a:
                    continue
                fl = f["filled"].get(f"{a}::{b}")
                d = fl["determines"] if fl else prof.determines(a, b)
                if d is None:
                    continue
                if d and fl and fl["mixed"]:
                    gaps.append(f"`{b}` where filled in")
                elif d:
                    one.append(f"`{b}`")
                else:
                    many.append(f"`{b}`")
            parts = []
            if one:
                parts.append("has exactly one " + ", ".join(one))
            if gaps:
                parts.append("has one " + ", ".join(gaps))
            if many and len(many) == len(cols) - 1 - len(one) - len(gaps) and len(many) > 3:
                parts.append("can have many of each other column" if not (one or gaps)
                             else "can have many of each of the others")
            elif many:
                parts.append("can have many " + ", ".join(many))
            if parts:
                lines.append(f"- Each `{a}` " + "; ".join(parts) + ".")
    for b in dict.fromkeys(k.split("::")[1] for k in f["filled"]):
        if b not in show:
            continue
        splits = [(k.split("::")[0], fl) for k, fl in f["filled"].items()
                  if k.endswith(f"::{b}") and k.split("::")[0] in show and fl["determines"] and fl["mixed"]]
        line = f"- `{b}` is blank on {f['nulls'][b]:,} of {rows:,} rows"
        if splits:
            line += "; " + "; ".join(f"{fl['mixed']} of {fl['values']} `{a}` values have both blank and filled `{b}`"
                                     for a, fl in splits)
        lines.append(line + ".")
    for kind, c in f.get("skipped", []):
        if c in show:
            lines.append(f"- (Not computed in time: {'where blanks fall in' if kind == 'blanks' else 'the grain of'} `{c}`.)")
    for m in f["measures"]:
        if m not in show or ["grain", m] in f.get("skipped", []):
            continue
        g = f["grains"].get(m)
        if g:
            lines.append(f"- Measure `{m}` has one value per `{g['per']}`: each value appears on the "
                         f"{_fmt(g['rows_per'])} rows of its `{g['per']}` on average, so a sum over rows "
                         f"counts it that many times.")
        else:
            lines.append(f"- Measure `{m}` varies from row to row.")
    return "\n".join(lines)


def render_full(f: dict[str, Any]) -> str:
    """The FD profile plus rows per value, blanks and measure grain (Grain-FD)."""
    from misqal.harness import prompts
    prof = CardinalityProfile.model_validate(f["profile"]) if f["profile"] else None
    out = [prompts.render_profile(prof, "full") if prof else f"**Table:** `{f['table']}` ({f['rows']} total rows)"]
    if f["per_value"]:
        out.append("**Rows per value (identifiers):**\n" + "\n".join(
            f"  - `{c}`: {v['min']}–{v['max']} rows (mean {_fmt(v['mean'])}); a join on it repeats each "
            f"matching row that many times" for c, v in f["per_value"].items()))
    blanks = [(k.split("::"), fl) for k, fl in f["filled"].items() if fl["mixed"] or fl["determines"]]
    if blanks:
        out.append("**Ignoring blanks (NULL):**\n" + "\n".join(
            f"  - `{a}` → `{b}` holds on filled-in rows; `{b}` is blank on {f['nulls'][b]:,} rows"
            + (f", in {fl['mixed']} `{a}` values that also have a filled `{b}`" if fl["mixed"] else "")
            for (a, b), fl in blanks if fl["determines"]))
    if f["measures"]:
        out.append("**Measures (grain):**\n" + "\n".join(
            (f"  - `{m}`: one value per `{g['per']}` ({_fmt(g['rows_per'])} rows each on average; a sum "
             f"over rows counts it that many times)" if (g := f["grains"].get(m)) else f"  - `{m}`: varies by row")
            for m in f["measures"]))
    return "\n\n".join(x for x in out if x)


def table_columns(db: Path, table: str) -> list[str]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]


def tool_columns(db: Path, db_id: str, table: str, requested: list[str]) -> list[str]:
    """The model's columns plus the table's identifier columns (keys always present)."""
    roles = roles_for(db_id, table)
    ids = [c for c in table_columns(db, table) if roles.get(c) == "identifier"]
    cols = list(dict.fromkeys([*requested, *ids]))
    return cols[:MAX_COLUMNS]


def main() -> None:
    """Precompute every table's facts for the given benchmarks (run before the experiment)."""
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    from misqal.harness.agent import known_tables
    from misqal.harness.bench import load_tasks
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--bench", nargs="+", required=True)
    p.add_argument("--seconds", type=float, default=1800.0, help="time limit per heavy scan")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    tables = {}
    for bench in args.bench:
        for t in load_tasks(bench):
            for table in known_tables(t):
                tables[(t.db_id, table)] = t.db_path

    def work(key: tuple[str, str]) -> str:
        t0 = time.monotonic()
        f = table_facts(tables[key], key[0], key[1], args.seconds)
        return f"{key[0]}.{key[1]}: {time.monotonic() - t0:.0f}s, skipped {f.get('skipped')}"

    with ThreadPoolExecutor(args.workers) as pool:
        for line in pool.map(work, sorted(tables, key=lambda k: tables[k].stat().st_size)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
