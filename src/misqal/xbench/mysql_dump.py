"""Convert a mysqldump file into a SQLite database.

    uv run python -m misqal.xbench.mysql_dump dump.sql out.sqlite

Prints the number of rows per table found in the dump against the rows in SQLite.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

TYPE_MAP = [
    (re.compile(r"^(tinyint|smallint|mediumint|int|integer|bigint)\b(\(\d+\))?( unsigned)?", re.I), "INTEGER"),
    (re.compile(r"^(float|double|decimal|real)\b(\([\d,]+\))?( unsigned)?", re.I), "REAL"),
    (re.compile(r"^(varchar|char)\(\d+\)", re.I), None),  # kept as is (TEXT affinity)
    (re.compile(r"^(text|mediumtext|longtext|tinytext)\b", re.I), "TEXT"),
    (re.compile(r"^(datetime|timestamp|date|time)\b", re.I), "TEXT"),
    (re.compile(r"^enum\((?:'[^']*'(?:,)?)+\)", re.I), "TEXT"),
]
TOK = re.compile(r"""'((?:[^'\\]|\\.)*)'|(NULL)|(-?\d+)(?![\d.eE])|(-?[\d.]+(?:[eE][-+]?\d+)?)|(\()|(\))|(,)|(;)|(\s+)""",
                 re.S)
ESC = {"0": "\0", "'": "'", '"': '"', "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a", "\\": "\\",
       "%": "\\%", "_": "\\_"}
ESC_RX = re.compile(r"\\(.)", re.S)


def _quote(s: str) -> str:
    return re.sub(r"`([^`]*)`", r'"\1"', s)


def _create(lines: list[str]) -> tuple[str, str, list[tuple[str, str]]]:
    tname = re.match(r"CREATE TABLE `([^`]+)` \(", lines[0]).group(1)
    cols, cons, idx = [], [], []
    for ln in lines[1:-1]:
        ln = ln.strip().rstrip(",")
        if ln.startswith("`"):
            cname, rest = re.match(r"`([^`]+)` (.*)$", ln).groups()
            for rx, rep in TYPE_MAP:
                tm = rx.match(rest)
                if tm:
                    newtype = rep or tm.group(0)
                    rest = rest[tm.end():]
                    break
            else:
                raise ValueError(f"unknown type in {tname}: {ln}")
            for pat in (r"CHARACTER SET \w+", r"COLLATE \w+", r"AUTO_INCREMENT", r"COMMENT '(?:[^'\\]|\\.)*'"):
                rest = re.sub(pat, "", rest)
            cols.append(f'"{cname}" {newtype} {re.sub(r"\s+", " ", rest).strip()}'.strip())
        elif ln.startswith(("PRIMARY KEY", "CONSTRAINT")):
            cons.append(_quote(ln))
        elif ln.startswith("UNIQUE KEY"):
            cons.append("UNIQUE " + _quote(re.match(r"UNIQUE KEY `[^`]+` (\(.*\))", ln).group(1)))
        elif ln.startswith(("KEY", "INDEX")):
            name, cols_ = re.match(r"(?:KEY|INDEX) `([^`]+)` (\(.*\))", ln).groups()
            idx.append((name, _quote(re.sub(r"\(\d+\)", "", cols_))))
        elif not ln.startswith("FULLTEXT"):
            raise ValueError(f"unknown DDL line in {tname}: {ln}")
    return tname, f'CREATE TABLE "{tname}" (\n  ' + ",\n  ".join(cols + cons) + "\n)", idx


def _values(s: str, pos: int) -> list[tuple]:
    rows, row, depth = [], None, 0
    while pos < len(s):
        m = TOK.match(s, pos)
        if not m:
            raise ValueError(f"cannot tokenize at {pos}: {s[pos:pos + 80]!r}")
        pos, g = m.end(), m.lastindex
        if g == 1:
            row.append(ESC_RX.sub(lambda e: ESC.get(e.group(1), e.group(1)), m.group(1)))
        elif g == 2:
            row.append(None)
        elif g == 3:
            row.append(int(m.group(3)))
        elif g == 4:
            row.append(float(m.group(4)))
        elif g == 5:
            assert depth == 0
            depth, row = 1, []
        elif g == 6:
            assert depth == 1
            depth = 0
            rows.append(tuple(row))
        elif g == 8:
            assert depth == 0 and s[pos:].strip() == "", "trailing content after INSERT"
            return rows
    raise ValueError("unterminated INSERT")


def convert(dump: Path, out: Path) -> dict[str, tuple[int, int]]:
    """Returns, per table, (rows in the dump, rows in SQLite)."""
    out.unlink(missing_ok=True)
    con = sqlite3.connect(out)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    counts: dict[str, int] = {}
    indexes = []
    with open(dump, encoding="utf-8") as f:
        buf = None
        for line in f:
            if buf is not None:
                buf.append(line.rstrip("\n"))
                if line.startswith(")"):
                    tname, ddl, idx = _create(buf)
                    con.execute(ddl)
                    counts.setdefault(tname, 0)
                    indexes += [(tname, i, c) for i, c in idx]
                    buf = None
            elif line.startswith("CREATE TABLE"):
                buf = [line.rstrip("\n")]
            elif line.startswith("INSERT INTO"):
                im = re.match(r"INSERT INTO `([^`]+)` VALUES ", line)
                rows = _values(line, im.end())
                if rows:
                    con.executemany(f'INSERT INTO "{im.group(1)}" VALUES ({",".join("?" * len(rows[0]))})', rows)
                counts[im.group(1)] = counts.get(im.group(1), 0) + len(rows)
    con.commit()
    used = set()
    for tname, iname, cols in indexes:
        name = iname if iname.lower() not in used else f"{tname}__{iname}"
        used.add(name.lower())
        con.execute(f'CREATE INDEX "{name}" ON "{tname}" {cols}')
    con.execute("ANALYZE")
    con.commit()
    report = {t: (n, con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]) for t, n in counts.items()}
    con.close()
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("dump", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    report = convert(args.dump, args.out)
    bad = {t: v for t, v in report.items() if v[0] != v[1]}
    print(json.dumps({"tables": len(report), "rows": sum(v[1] for v in report.values()),
                      "empty_tables": sum(v[1] == 0 for v in report.values()), "mismatches": bad}))


if __name__ == "__main__":
    main()
