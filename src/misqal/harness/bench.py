"""Benchmarks: task loading, time-limited read-only SQL, and scoring.

Scoring reports BIRD's official set-based execution match (``set_ex``) next to the multiset match
(``multiset_ex``; duplicate rows count). Set-EX cannot see row multiplication, which is the
error class this project studies (cf. ModularSQL's Multiset-EX). The agent never sees scores.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from misqal.harness import spider2
from misqal.paths import shipped


@dataclass(frozen=True)
class Task:
    id: str
    bench: str
    db_id: str
    db_path: Path
    question: str
    evidence: str = ""  # benchmark-provided hint / external knowledge
    gold_sql: str | None = None  # BIRD; Spider2 scores against gold CSVs instead
    original_gold_sql: str | None = None  # BIRD-FanOut: the uncorrected BIRD gold
    category: str = ""  # BIRD-FanOut: inflated | intended; traps: trap kind
    alt_gold_sqls: tuple[str, ...] = ()  # traps: other readings of the question that are also right
    naive_sql: str | None = None  # traps: the query that falls into the trap
    tables: tuple[str, ...] = ()  # tables shown to the agent (schema and profiles); empty: all
    measures: tuple[str, ...] = ()  # "table.column" (lower case) of measure columns: no evidence on them
    former_sqls: tuple[str, ...] = ()  # BIRD-FanOut: golds the gold review replaced; their tables stay shown
    gold_status: str = ""  # BIRD-FanOut gold review: ok | fixed | ambiguous (flawed, disputed: left out)


class QueryTimeout(Exception):
    pass


def run_query(db: Path, sql: str, timeout: float = 30.0,
              max_rows: int | None = None) -> tuple[list[str], list[tuple]]:
    """Read-only query with a wall-clock limit. Raises ``sqlite3.Error`` or ``QueryTimeout``."""
    deadline = time.monotonic() + timeout
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e):
                raise QueryTimeout(f"query exceeded {timeout:.0f}s") from e
            raise
    return cols, rows


# --- BIRD -------------------------------------------------------------------------------

def load_bird(root: Path, dbs: list[str] | None = None) -> list[Task]:
    """BIRD layout: ``<split>.json`` + ``<split>_databases/<db>/<db>.sqlite``, split ``dev`` or
    ``train``. Train questions carry no id; their position in ``train.json`` is used."""
    split = "dev" if (root / "dev.json").exists() else "train"
    out = []
    for i, r in enumerate(json.loads((root / f"{split}.json").read_text())):
        if dbs and r["db_id"] not in dbs:
            continue
        out.append(Task(
            id=str(r.get("question_id", i)), bench="bird", db_id=r["db_id"],
            db_path=root / f"{split}_databases" / r["db_id"] / f"{r['db_id']}.sqlite",
            question=r["question"], evidence=r.get("evidence", ""), gold_sql=r["SQL"],
        ))
    return out


FANOUT_SUBSET = shipped("benchmarks/bird-fanout/subset.jsonl")
FANOUT_SUBSET_1106 = shipped("benchmarks/bird-fanout-20251106/subset.jsonl")
FANOUT_SUBSET_TRAIN = shipped("benchmarks/bird-fanout-train/subset.jsonl")


def load_bird_fanout(root: Path, dbs: list[str] | None = None,
                     subset: Path = FANOUT_SUBSET, bench: str = "bird-fanout",
                     unreviewed: bool = False) -> list[Task]:
    """BIRD-FanOut: inflated questions scored against the corrected gold, questions a later release
    fixed against the release's gold, intended ones (controls) against the BIRD gold. Contested and
    benign questions, and (where judged) those whose reading the data model does not settle, are
    left out.

    Where the benchmark has a ``gold_review.jsonl`` (``misqal.fanout.gold_review``), golds with a
    non-fan-out error are replaced by the fix and the original by the fix's trap (no trap: no
    trapped score), and questions the review found flawed or disputed, or whose agreed fix returns
    no answer and awaits a data check, are left out.
    ``unreviewed`` ignores the review."""
    bird = {t.id: t for t in load_bird(root, dbs)}
    review_file = subset.parent / "gold_review.jsonl"
    reviews = ({d["task_id"]: d for d in map(json.loads, review_file.read_text().splitlines())}
               if review_file.exists() and not unreviewed else {})
    out = []
    for r in map(json.loads, subset.read_text().splitlines()):
        keep = r["category"] in ("intended", "inflated", "fixed_in_release")
        if not keep or not r.get("resolvable", True) or r["id"] not in bird:
            continue
        t = bird[r["id"]]
        gold = r["corrected_sql"] if r["category"] == "inflated" else t.gold_sql
        original = r.get("naive_sql") or t.gold_sql
        d = reviews.get(r["id"], {})
        status, former = d.get("status", ""), ()
        if status in ("flawed", "disputed", "check"):
            continue
        if status == "fixed":
            former = (gold, original)
            gold, original = d["fixed_sql"], d.get("trap_sql") or (None if original != gold else d["fixed_sql"])
        out.append(Task(id=t.id, bench=bench, db_id=t.db_id, db_path=t.db_path,
                        question=t.question, evidence=t.evidence, gold_sql=gold,
                        original_gold_sql=original, category=r["category"],
                        former_sqls=former, gold_status=status))
    return out


def load_bird_fanout_1106(root: Path, dbs: list[str] | None = None) -> list[Task]:
    return load_bird_fanout(root, dbs, FANOUT_SUBSET_1106, "bird-fanout-20251106")


def load_bird_fanout_train(root: Path, dbs: list[str] | None = None) -> list[Task]:
    return load_bird_fanout(root, dbs, FANOUT_SUBSET_TRAIN, "bird-fanout-train")


@functools.lru_cache(maxsize=4096)
def _reference_rows(db: Path, sql: str, timeout: float) -> list[tuple]:
    """Results of a gold or original query, computed once per process (databases are read-only)."""
    return run_query(db, sql, timeout)[1]


def score_bird(task: Task, pred_sql: str | None, timeout: float = 60.0) -> dict[str, Any]:
    res: dict[str, Any] = {"exec_ok": False, "set_ex": False, "multiset_ex": False}
    if not pred_sql:
        return res
    try:
        _, pred = run_query(task.db_path, pred_sql, timeout)
    except (sqlite3.Error, QueryTimeout) as e:
        res["pred_error"] = str(e)[:300]
        return res
    res["exec_ok"] = True
    try:
        gold = _reference_rows(task.db_path, task.gold_sql or "", timeout)
    except (sqlite3.Error, QueryTimeout) as e:
        res["gold_error"] = str(e)[:300]
        return res
    res.update(
        set_ex=set(pred) == set(gold),
        multiset_ex=Counter(pred) == Counter(gold),
        pred_rows=len(pred),
        gold_rows=len(gold),
    )
    if task.original_gold_sql and task.original_gold_sql != task.gold_sql:
        try:
            original = _reference_rows(task.db_path, task.original_gold_sql, timeout)
            res["original_set_ex"] = set(pred) == set(original)
        except (sqlite3.Error, QueryTimeout):
            pass
    return res


# --- Spider 2.0-Lite (local SQLite subset) -----------------------------------------------

def load_spider2(root: Path, dbs: list[str] | None = None) -> list[Task]:
    return [
        Task(id=t.id, bench="spider2", db_id=t.db_id,
             db_path=spider2.spider2_db_path(root, t.db_id), question=t.question,
             evidence=t.evidence)
        for t in spider2.load_spider2_local(root)
        if not dbs or t.db_id in dbs
    ]


def score_spider2(task: Task, pred_sql: str | None, root: Path,
                  timeout: float = 120.0) -> dict[str, Any]:
    res: dict[str, Any] = {"exec_ok": False, "ex": False}
    if not pred_sql:
        return res
    try:
        _, rows = run_query(task.db_path, pred_sql, timeout)
    except (sqlite3.Error, QueryTimeout) as e:
        res["pred_error"] = str(e)[:300]
        return res
    res.update(exec_ok=True, pred_rows=len(rows),
               ex=spider2.spider2_match_table(root, task.id, [list(r) for r in rows]))
    return res


# --- Fan-out trap suite (synthetic, generated deterministically) -------------------------

TRAP_QUESTIONS = shipped("benchmarks/graintraps/questions.json")


def load_traps(root: Path, dbs: list[str] | None = None,
               questions: Path = TRAP_QUESTIONS) -> list[Task]:
    from misqal.traps.generate import ensure_databases

    paths = ensure_databases(root)
    return [Task(id=q["id"], bench="traps", db_id=q["db"], db_path=paths[q["db"]],
                 question=q["question"], gold_sql=q["gold_sql"], category=q["trap"],
                 alt_gold_sqls=tuple(q.get("alt_gold_sql", [])), naive_sql=q["naive_sql"])
            for q in json.loads(questions.read_text()) if not dbs or q["db"] in dbs]


def _normalize(rows: list[tuple]) -> Counter:
    return Counter(tuple(round(v, 2) if isinstance(v, float) else v for v in r) for r in rows)


def score_traps(task: Task, pred_sql: str | None, timeout: float = 60.0) -> dict[str, Any]:
    """Multiset match after rounding floats to 2 decimals; row order is ignored. A match with any
    accepted reading (gold or alternative) is correct.

    ``outcome`` sorts every answer into one of: ``correct``; ``trapped`` (same rows as the naive
    query, or the audit sees joins actually repeating rows under a duplicate-sensitive aggregate);
    ``other`` (wrong for some other reason); ``failed`` (no SQL or it did not run)."""
    res: dict[str, Any] = {"exec_ok": False, "set_ex": False, "multiset_ex": False,
                           "outcome": "failed"}
    if not pred_sql:
        return res
    try:
        _, pred = run_query(task.db_path, pred_sql, timeout)
    except (sqlite3.Error, QueryTimeout) as e:
        res["pred_error"] = str(e)[:300]
        return res
    p = _normalize(pred)
    golds = [_normalize(run_query(task.db_path, g, timeout)[1])
             for g in (task.gold_sql or "", *task.alt_gold_sqls)]
    match = next((i for i, g in enumerate(golds) if p == g), None)
    res.update(exec_ok=True, set_ex=any(set(p) == set(g) for g in golds),
               multiset_ex=match is not None, pred_rows=len(pred), gold_rows=len(golds[0]))
    if match:  # index 0 is the gold itself
        res["alt_match"] = match
    if match is not None:
        res["outcome"] = "correct"
        return res
    naive = task.naive_sql and p == _normalize(run_query(task.db_path, task.naive_sql, timeout)[1])
    from misqal.fanout.audit import audit_query
    res["outcome"] = ("trapped" if naive or audit_query(task.id, task.db_id, task.db_path,
                                                        pred_sql, timeout).realized else "other")
    return res


BENCH_ROOTS = {"bird": Path("data/bird/dev"), "bird-fanout": Path("data/bird/dev"),
               "bird-fanout-20251106": Path("data/bird-dev-20251106"),
               "bird-fanout-train": Path("data/bird-train/train"),
               "spider2": Path("data/spider2/spider2-lite"), "traps": Path("data/traps")}


def load_tasks(bench: str, root: Path | None = None, dbs: list[str] | None = None) -> list[Task]:
    root = root or BENCH_ROOTS[bench]
    return {"bird": load_bird, "bird-fanout": load_bird_fanout,
            "bird-fanout-20251106": load_bird_fanout_1106,
            "bird-fanout-train": load_bird_fanout_train, "traps": load_traps,
            "spider2": load_spider2}[bench](root, dbs)


def score(task: Task, pred_sql: str | None, root: Path | None = None) -> dict[str, Any]:
    if task.bench in ("bird", "bird-fanout", "bird-fanout-20251106", "bird-fanout-train"):
        return score_bird(task, pred_sql)
    if task.bench == "traps":
        return score_traps(task, pred_sql)
    return score_spider2(task, pred_sql, root or BENCH_ROOTS["spider2"])
