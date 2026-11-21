"""SQL builders: result aliases, identifier quoting, sampling, and NULL-inclusive counting."""

from __future__ import annotations

import sqlite3

import pytest

from misqal.cardinality.core import _parse_row
from misqal.cardinality.dialect import (
    SparkDialect,
    SQLiteDialect,
    quote_spark_identifier,
    quote_sqlite_identifier,
)

DIALECTS = [SparkDialect(), SQLiteDialect()]


@pytest.mark.parametrize("d", DIALECTS, ids=lambda d: d.name)
def test_result_names(d) -> None:
    assert d.query_1d("t", ["a", "b"])[1] == ["total_rows", "distinct_0", "distinct_1"]
    assert d.query_composite_grain("t", ["a", "b"])[1] == ["distinct_grain"]
    assert d.query_pairwise_batch("t", [("a", "b"), ("c", "d")])[1] == ["combos_0", "combos_1"]


@pytest.mark.parametrize("d", DIALECTS, ids=lambda d: d.name)
def test_sample_dedups_1d_columns(d) -> None:
    # pairs share 'a': unique sorted cols a, b, c -> dist_1d_0..2, then one combos per pair
    _, names = d.query_sample_pairwise_batch("t", [("a", "b"), ("a", "c")], 5.0)
    assert names == ["dist_1d_0", "dist_1d_1", "dist_1d_2", "combos_0", "combos_1"]


def test_spark_shapes() -> None:
    d = SparkDialect()
    sql, _ = d.query_1d("db.t", ["my col"])
    assert "COUNT(1) AS total_rows" in sql and "FROM db.t" in sql
    assert "COUNT(DISTINCT named_struct('c0', `my col`)) AS distinct_0" in sql
    sql, _ = d.query_pairwise_batch("t", [("a", "b")])
    assert "COUNT(DISTINCT named_struct('c0', `a`, 'c1', `b`)) AS combos_0" in sql
    assert "TABLESAMPLE (5.0 PERCENT)" in d.query_sample_pairwise_batch("t", [("a", "b")], 5.0)[0]
    assert "TABLESAMPLE (0.1 PERCENT)" in d.query_sample_pairwise_batch("t", [("a", "b")], 0.0)[0]


def test_identifier_quoting() -> None:
    assert quote_spark_identifier("col`a") == "`col``a`"
    assert quote_spark_identifier("عمود") == "`عمود`"
    assert quote_sqlite_identifier('my "col"') == '"my ""col"""'


@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE "odd table" ("my col" TEXT, b TEXT);
        INSERT INTO "odd table" VALUES ('x', '1'), ('x', NULL), (NULL, '1'), (NULL, NULL),
                                       (NULL, NULL), ('y', '2');
    """)
    yield c
    c.close()


def _run(con, built) -> dict[str, int]:
    sql, names = built
    return _parse_row(con.execute(sql).fetchone(), names)


def test_sqlite_counts_include_null(con) -> None:
    d = SQLiteDialect()
    one = _run(con, d.query_1d("odd table", ["my col", "b"]))
    assert one == {"total_rows": 6, "distinct_0": 3, "distinct_1": 3}  # {x, y, NULL}, {1, 2, NULL}
    # distinct (my col, b) tuples with NULL = NULL: (x,1) (x,N) (N,1) (N,N) (y,2)
    assert _run(con, d.query_pairwise_batch("odd table", [("my col", "b")]))["combos_0"] == 5
    assert _run(con, d.query_composite_grain("odd table", ["my col", "b"]))["distinct_grain"] == 5


def test_sqlite_sample_uses_one_consistent_sample(con) -> None:
    """At 100% the sample is the whole table, so sampled counts equal the full counts."""
    d = SQLiteDialect()
    if not d.supports_sample:
        pytest.skip("SQLite < 3.35")
    raw = _run(con, d.query_sample_pairwise_batch("odd table", [("my col", "b")], 100.0))
    assert raw == {"dist_1d_0": 3, "dist_1d_1": 3, "combos_0": 5}  # b < my col when sorted
