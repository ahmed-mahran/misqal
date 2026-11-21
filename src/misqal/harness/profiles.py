"""Cardinality profiles of a database's tables.

Injected evidence covers every table the model is shown: the whole database, or with
``--known-tables`` the tables of the question's reference queries. Profiles are computed once per
database and cached as JSON.
Tool calls (one table, chosen columns) are cached the same way, per exact column list.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
import threading
from collections import defaultdict
from pathlib import Path
from typing import Callable

from misqal.cardinality.dialect import SQLiteDialect
from misqal.cardinality.matrix import MatrixStats, SQLiteExecutor, run_cardinality_matrix
from misqal.cardinality.models import CardinalityProfile

CACHE_DIR = Path("runs/cache/profiles")
CALL_CACHE_DIR = Path("runs/cache/matrix-calls")
STATS_CACHE_DIR = Path("runs/cache/column-stats")
_locks: dict[Path, threading.Lock] = defaultdict(threading.Lock)
_memory: dict[Path, dict[str, CardinalityProfile]] = {}


def _tables(db: Path) -> dict[str, list[str]]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {t: [c[1] for c in con.execute(f'PRAGMA table_info("{t}")')] for t in tables}


def compute_db_profiles(db: Path) -> tuple[dict[str, CardinalityProfile], dict[str, dict]]:
    """Profile every table over all its columns (tables with fewer than two columns are skipped)."""
    profiles, stats = {}, {}
    for table, cols in _tables(db).items():
        if len(cols) < 2:
            continue
        s = MatrixStats()
        profiles[table] = run_cardinality_matrix(table, cols, SQLiteExecutor(db), SQLiteDialect(),
                                                 stats=s, max_columns=None)
        stats[table] = s.as_dict()
    return profiles, stats


class _FileLock:
    """Exclusive lock on ``<path>.lock`` across threads and processes, so that one computes a cache
    entry while the others wait and then read it."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path.with_name(path.name + ".lock")

    def __enter__(self) -> "_FileLock":
        self.f = self.path.open("w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: object) -> None:
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _db_key(db: Path) -> str:
    st = db.stat()
    return hashlib.sha1(f"{db}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:10]


def cached_call_profile(db: Path, table: str, columns: list[str],
                        compute: Callable[[], CardinalityProfile],
                        cache_dir: Path = CALL_CACHE_DIR) -> tuple[CardinalityProfile, bool]:
    """The profile of one tool call (a table and an exact column list, computed with no earlier
    profile of the table), from disk when an earlier episode computed it. The data does not change,
    so the result is the same as computing it again. Returns (profile, from cache)."""
    db = db.resolve()
    name = hashlib.sha1(json.dumps([table, columns]).encode()).hexdigest()[:16]
    path = cache_dir / f"{db.stem}-{_db_key(db)}" / f"{name}.json"
    if path.exists():
        return CardinalityProfile.model_validate_json(path.read_text()), True
    with _FileLock(path):  # concurrent runs wait for the first to compute it
        if path.exists():
            return CardinalityProfile.model_validate_json(path.read_text()), True
        profile = compute()
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(profile.model_dump_json())
        tmp.replace(path)  # atomic: concurrent runs never read a partial file
    return profile, False


def db_profiles(db: Path, cache_dir: Path = CACHE_DIR) -> dict[str, CardinalityProfile]:
    db = db.resolve()
    with _locks[db]:
        if db in _memory:
            return _memory[db]
        path = cache_dir / f"{db.stem}-{_db_key(db)}.json"
        if path.exists():
            raw = json.loads(path.read_text())
            profiles = {t: CardinalityProfile.model_validate(p) for t, p in raw["profiles"].items()}
        else:
            profiles, stats = compute_db_profiles(db)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"db": db.name, "stats": stats, "profiles": {
                t: p.model_dump(mode="json") for t, p in profiles.items()}}))
        _memory[db] = profiles
        return profiles


def table_profile(db: Path, table: str, columns: list[str]) -> CardinalityProfile:
    """Profile of ``table`` over exactly ``columns`` (all pairs), cached like a first tool call."""
    def compute() -> CardinalityProfile:
        return run_cardinality_matrix(table, columns, SQLiteExecutor(db), SQLiteDialect(),
                                      max_columns=None)
    return cached_call_profile(db, table, columns, compute)[0]


def column_stats(db: Path, table: str, columns: list[str],
                 cache_dir: Path = STATS_CACHE_DIR) -> dict:
    """Rows of ``table`` and, per column, distinct values and NULLs. Cached per table."""
    db = db.resolve()
    path = cache_dir / f"{db.stem}-{_db_key(db)}" / f"{table}.json"
    def hit() -> dict | None:
        if path.exists():
            cached = json.loads(path.read_text())
            if all(c in cached["columns"] for c in columns):
                return {"rows": cached["rows"], "columns": {c: cached["columns"][c] for c in columns}}
        return None
    if (found := hit()) is not None:
        return found
    with _FileLock(path):  # concurrent runs wait for the first to compute it
        if (found := hit()) is not None:
            return found
        return _compute_column_stats(db, table, columns, path)


def _compute_column_stats(db: Path, table: str, columns: list[str], path: Path) -> dict:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        rows = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        cols = {}
        for c in columns:
            d, n = con.execute(f'SELECT COUNT(DISTINCT "{c}"), SUM("{c}" IS NULL) FROM "{table}"').fetchone()
            cols[c] = {"distinct": d, "nulls": n or 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"rows": rows, "columns": cols}))
    tmp.replace(path)
    return {"rows": rows, "columns": cols}
