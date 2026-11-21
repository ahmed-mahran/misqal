"""End-to-end matrix on SQLite, checked against a brute-force FD oracle.

Every edge the matrix emits — measured, sampled, rule-seeded or inferred by
closure — must agree with the instance's true unary FDs (NULL = NULL, i.e. GROUP BY semantics).
"""

from __future__ import annotations

import random
import sqlite3
from itertools import combinations
from pathlib import Path

import pytest

from misqal.cardinality.dialect import SQLiteDialect, quote_sqlite_identifier
from misqal.cardinality.matrix import MatrixStats, SQLiteExecutor, run_cardinality_matrix
from misqal.cardinality.models import CardinalityKind


def fd_holds(rows: list[dict], a: str, b: str) -> bool:
    seen: dict = {}
    for r in rows:
        if seen.setdefault(r[a], r[b]) != r[b]:
            return False
    return True


def oracle(rows: list[dict], a: str, b: str) -> CardinalityKind:
    ab, ba = fd_holds(rows, a, b), fd_holds(rows, b, a)
    if ab and ba:
        return CardinalityKind.ONE_ONE
    if ab:
        return CardinalityKind.N_ONE
    if ba:
        return CardinalityKind.ONE_N
    return CardinalityKind.M_N


def maybe_null(rng: random.Random, v, p: float):
    return None if rng.random() < p else v


def random_table(rng: random.Random, n: int) -> list[dict]:
    """Hierarchy region -> country -> city, an id key, a 1:1 alias of city, an independent
    dimension, a constant, and NULLs sprinkled in (which can break or create FDs)."""
    p = rng.choice([0.0, 0.02, 0.1])
    rows = []
    for i in range(n):
        city = rng.randrange(rng.choice([5, 20, 60]))
        country = city // 4
        region = country // 3
        rows.append({
            "id": i,
            "city": maybe_null(rng, city, p),
            "city_code": f"C{city}",
            "country": maybe_null(rng, country, p),
            "region": region,
            "channel": maybe_null(rng, rng.randrange(3), p),
            "tenant": "acme",
            "note": maybe_null(rng, "x", 0.5),  # one value plus NULLs: two groups, not a constant
        })
    return rows


def write_sqlite(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0])
    with sqlite3.connect(path) as con:
        con.execute(f"CREATE TABLE t ({', '.join(quote_sqlite_identifier(c) for c in cols)})")
        con.executemany(f"INSERT INTO t VALUES ({', '.join('?' * len(cols))})",
                        [tuple(r[c] for c in cols) for r in rows])


@pytest.mark.parametrize("seed", range(12))
def test_every_emitted_edge_matches_oracle(tmp_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    rows = random_table(rng, rng.choice([60, 400, 3000]))  # below/around/above sampling cutoffs
    db = tmp_path / "t.sqlite"
    write_sqlite(db, rows)
    columns = list(rows[0])
    rng.shuffle(columns)

    stats = MatrixStats()
    profile = run_cardinality_matrix("t", columns, SQLiteExecutor(db), SQLiteDialect(),
                                     stats=stats)

    for a, b in combinations(columns, 2):
        rel = profile.relationships.get(f"{a}::{b}")
        assert rel is not None, f"missing edge {a}::{b}"
        assert rel.cardinality == oracle(rows, a, b), f"{a}::{b} ({rel.proof})"
        assert profile.relationships[f"{b}::{a}"].cardinality == oracle(rows, b, a)

    n = len(rows)
    for c in columns:
        d = len({r[c] for r in rows})
        assert (c in profile.primary_keys) == (d == n)
        assert (c in profile.constants) == (d <= 1)
    grain = profile.tested_grains[0]
    assert grain.is_unique == (len({tuple(r[c] for c in columns) for r in rows}) == n)


def _assert_matches_oracle(profile, rows: list[dict], columns: list[str]) -> None:
    for a, b in combinations(columns, 2):
        rel = profile.relationships[f"{a}::{b}"]
        assert rel.cardinality == oracle(rows, a, b), f"{a}::{b} ({rel.proof})"


def test_midflight_pruning_wide_table(tmp_path: Path) -> None:
    """1:1 aliases found in the first batch let later (alias, x) pairs be inferred, not scanned."""
    rows = []
    for i in range(3000):
        a = i % 300
        rows.append({"a": a, **{f"alias{k}": f"A{k}_{a}" for k in range(5)},
                     **{f"x{k}": a // (k + 2) for k in range(6)}})
    db = tmp_path / "t.sqlite"
    write_sqlite(db, rows)
    columns = list(rows[0])
    stats = MatrixStats()
    profile = run_cardinality_matrix("t", columns, SQLiteExecutor(db), SQLiteDialect(),
                                     stats=stats)
    _assert_matches_oracle(profile, rows, columns)
    assert stats.pairs_pruned_midflight > 0
    assert stats.pairs_full_scan + stats.pairs_pruned_midflight + stats.pairs_sample_mn \
        + stats.pairs_pigeonhole_mn == stats.pairs_delta


def test_pigeonhole_rule(tmp_path: Path) -> None:
    """d(p) == d(q) and the sample shows p -> q violated (p=0 maps to 0 and 'z') but rarely
    q -> p violated (q=98 has two p's, one row each): only the pigeonhole rule can prove M:N."""
    rows = [{"p": 0, "q": "0" if i % 2 else "z"} for i in range(4000)]
    rows += [{"p": k, "q": str(k)} for k in range(1, 98) for _ in range(30)]
    rows += [{"p": 98, "q": "98"}, {"p": 99, "q": "98"}]
    db = tmp_path / "t.sqlite"
    write_sqlite(db, rows)
    fired = 0
    for _ in range(5):  # sampling is random; the label is not
        stats = MatrixStats()
        profile = run_cardinality_matrix("t", ["p", "q"], SQLiteExecutor(db), SQLiteDialect(),
                                         stats=stats)
        _assert_matches_oracle(profile, rows, ["p", "q"])
        fired += stats.pairs_pigeonhole_mn
    assert fired >= 1


def test_incremental_call_reuses_profile(tmp_path: Path) -> None:
    rng = random.Random(7)
    rows = random_table(rng, 500)
    db = tmp_path / "t.sqlite"
    write_sqlite(db, rows)
    ex, dialect = SQLiteExecutor(db), SQLiteDialect()
    first = run_cardinality_matrix("t", ["city", "country", "region"], ex, dialect)
    stats = MatrixStats()
    second = run_cardinality_matrix("t", ["city", "country", "region", "channel"], ex, dialect,
                                    existing_profile=first, stats=stats)
    assert stats.pairs_delta <= 3  # only pairs with the new column can be unknown
    for a, b in combinations(["city", "country", "region", "channel"], 2):
        assert second.relationships[f"{a}::{b}"].cardinality == oracle(rows, a, b)


def test_null_fanout_is_reported(tmp_path: Path) -> None:
    """p1 maps to c1 and to NULL: GROUP BY product, corp gives two rows for p1, so product
    fans out to corp even though every non-NULL corp has a single product."""
    rows = [{"product": "p1", "corp": "c1"}, {"product": "p1", "corp": None},
            {"product": "p2", "corp": "c2"}, {"product": "p3", "corp": "c3"}]
    db = tmp_path / "t.sqlite"
    write_sqlite(db, rows)
    profile = run_cardinality_matrix("t", ["product", "corp"], SQLiteExecutor(db),
                                     SQLiteDialect())
    assert profile.relationships["product::corp"].cardinality == CardinalityKind.ONE_N
    assert profile.relationships["corp::product"].cardinality == CardinalityKind.N_ONE


def test_determines_matches_oracle(tmp_path: Path) -> None:
    rows = random_table(random.Random(5), 300)
    db = tmp_path / "t.sqlite"
    write_sqlite(db, rows)
    columns = list(rows[0])
    profile = run_cardinality_matrix("t", columns, SQLiteExecutor(db), SQLiteDialect())
    for a in columns:
        for b in columns:
            if a != b:
                assert profile.determines(a, b) == fd_holds(rows, a, b), (a, b)


def test_sentences_state_null_fanout(tmp_path: Path) -> None:
    rows = [{"id": 1, "product": "p1", "corp": "c1", "region": "r1"},
            {"id": 2, "product": "p1", "corp": None, "region": "r1"},
            {"id": 3, "product": "p2", "corp": "c2", "region": "r1"},
            {"id": 4, "product": "p2", "corp": "c2", "region": "r1"}]
    db = tmp_path / "t.sqlite"
    write_sqlite(db, rows)
    profile = run_cardinality_matrix("t", ["id", "product", "corp", "region"],
                                     SQLiteExecutor(db), SQLiteDialect())
    text = profile.to_sentences()
    assert "Each row has its own `id`." in text
    assert "Same value on every row: `region`." in text
    assert "- Each `product` can have many `corp`." in text
    assert "- Each `corp` has exactly one `product`." in text


def test_null_roles_follow_group_by(tmp_path: Path) -> None:
    """GROUP BY puts NULLs in one group: a key with a single NULL is still a key, one value plus
    NULLs is two groups (not a constant), and an all-NULL column is a constant."""
    rows = [{"k": None if i == 0 else i, "half": "v" if i % 2 else None, "nothing": None,
             "grp": i % 3} for i in range(9)]
    db = tmp_path / "t.sqlite"
    write_sqlite(db, rows)
    columns = ["k", "half", "nothing", "grp"]
    profile = run_cardinality_matrix("t", columns, SQLiteExecutor(db), SQLiteDialect())
    assert profile.primary_keys == ["k"]
    assert profile.constants == ["nothing"]
    _assert_matches_oracle(profile, rows, columns)


def test_argument_limits(tmp_path: Path) -> None:
    db = tmp_path / "t.sqlite"
    write_sqlite(db, [{"a": 1, "b": 2}])
    with pytest.raises(ValueError):
        run_cardinality_matrix("t", ["a"], SQLiteExecutor(db), SQLiteDialect())
    with pytest.raises(ValueError):
        run_cardinality_matrix("t", [f"c{i}" for i in range(26)], SQLiteExecutor(db),
                               SQLiteDialect())
