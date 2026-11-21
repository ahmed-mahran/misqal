"""
SQL dialects for the cardinality matrix queries.

**NULL semantics.** NULL counts as one value, as in GROUP BY and DISTINCT. Then
``d(A) = d(AB)`` holds iff ``A → B`` on the instance (null = null), which is exactly the condition
for ``GROUP BY A`` not to fan out when ``B`` is attached. Plain ``COUNT(DISTINCT a)`` skips NULLs,
and Spark's ``COUNT(DISTINCT a, b)`` skips any row where an argument is NULL; mixing the two
compares counts over different rows and can report 1:1 for a pair that fans out. Every count
below therefore includes NULL.

Every builder returns ``(sql, result_column_names)`` in the shape ``core._parse_row`` expects:
``total_rows`` / ``distinct_<i>`` (1D), ``distinct_grain`` (composite), ``combos_<i>`` (pairs), and
``dist_1d_<n>`` + ``combos_<i>`` (sample; ``n`` indexes the sorted unique columns of the batch).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Protocol


class Dialect(Protocol):
    name: str
    supports_sample: bool

    def query_1d(self, table: str, columns: list[str]) -> tuple[str, list[str]]: ...

    def query_composite_grain(self, table: str, columns: list[str]) -> tuple[str, list[str]]: ...

    def query_pairwise_batch(
        self, table: str, pairs: list[tuple[str, str]]
    ) -> tuple[str, list[str]]: ...

    def query_sample_pairwise_batch(
        self, table: str, pairs: list[tuple[str, str]], sample_percent: float
    ) -> tuple[str, list[str]]: ...


def _sample_aliases(pairs: list[tuple[str, str]]) -> tuple[list[str], list[str], list[str]]:
    """Sorted unique columns, their ``dist_1d_<n>`` aliases, and ``combos_<i>`` aliases."""
    unique_cols = sorted({c for pair in pairs for c in pair})
    return (
        unique_cols,
        [f"dist_1d_{i}" for i in range(len(unique_cols))],
        [f"combos_{i}" for i in range(len(pairs))],
    )


def _clamp_percent(sample_percent: float) -> float:
    return max(0.1, min(100.0, sample_percent))


def quote_spark_identifier(identifier: str) -> str:
    """Backtick-quote for any column name (spaces, non-ASCII); backticks inside are doubled."""
    return "`" + identifier.replace("`", "``") + "`"


def quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


@dataclass(frozen=True)
class SparkDialect:
    """Spark SQL. ``COUNT(DISTINCT named_struct(...))`` keeps rows with NULL fields and compares
    NULL fields as equal. The table name is used as given (it may be qualified)."""

    name: str = "spark"
    supports_sample: bool = True

    @staticmethod
    def _struct(columns: list[str]) -> str:
        fields = ", ".join(f"'c{i}', {quote_spark_identifier(c)}" for i, c in enumerate(columns))
        return f"named_struct({fields})"

    def query_1d(self, table, columns):
        select = ["COUNT(1) AS total_rows"] + [
            f"COUNT(DISTINCT {self._struct([c])}) AS distinct_{i}" for i, c in enumerate(columns)
        ]
        names = ["total_rows"] + [f"distinct_{i}" for i in range(len(columns))]
        return "SELECT\n  " + ",\n  ".join(select) + f"\nFROM {table}", names

    def query_composite_grain(self, table, columns):
        return (
            f"SELECT COUNT(DISTINCT {self._struct(columns)}) AS distinct_grain FROM {table}",
            ["distinct_grain"],
        )

    def query_pairwise_batch(self, table, pairs):
        select = [
            f"COUNT(DISTINCT {self._struct([a, b])}) AS combos_{i}" for i, (a, b) in enumerate(pairs)
        ]
        names = [f"combos_{i}" for i in range(len(pairs))]
        return "SELECT\n  " + ",\n  ".join(select) + f"\nFROM {table}", names

    def query_sample_pairwise_batch(self, table, pairs, sample_percent):
        # One TABLESAMPLE scan feeds every aggregate, so all counts see the same sample.
        unique_cols, dist_names, combo_names = _sample_aliases(pairs)
        select = [
            f"COUNT(DISTINCT {self._struct([c])}) AS {alias}"
            for c, alias in zip(unique_cols, dist_names)
        ] + [
            f"COUNT(DISTINCT {self._struct([a, b])}) AS {alias}"
            for (a, b), alias in zip(pairs, combo_names)
        ]
        sql = (
            "SELECT\n  " + ",\n  ".join(select)
            + f"\nFROM {table} TABLESAMPLE ({_clamp_percent(sample_percent)} PERCENT)"
        )
        return sql, dist_names + combo_names


# MATERIALIZED (SQLite >= 3.35) pins one random sample for every aggregate in the batch; without
# it SQLite may inline the CTE and draw a different sample per subquery, which would void the
# "sample violation ⇒ global violation" argument.
_SQLITE_MATERIALIZED = sqlite3.sqlite_version_info >= (3, 35, 0)


@dataclass(frozen=True)
class SQLiteDialect:
    """SQLite has no multi-argument COUNT(DISTINCT) and no TABLESAMPLE: distinct tuples are
    counted with ``SELECT COUNT(*) FROM (SELECT DISTINCT ...)`` (DISTINCT treats NULLs as equal),
    and the sample is a materialized ``random()`` filter."""

    name: str = "sqlite"
    supports_sample: bool = _SQLITE_MATERIALIZED

    @staticmethod
    def _t(table: str) -> str:
        return quote_sqlite_identifier(table)

    @staticmethod
    def _distinct_1d(col: str) -> str:
        q = quote_sqlite_identifier(col)
        return f"(COUNT(DISTINCT {q}) + COALESCE(MAX({q} IS NULL), 0))"

    @staticmethod
    def _distinct_tuples(source: str, columns: list[str]) -> str:
        cols = ", ".join(quote_sqlite_identifier(c) for c in columns)
        return f"(SELECT COUNT(*) FROM (SELECT DISTINCT {cols} FROM {source}))"

    def query_1d(self, table, columns):
        select = ["COUNT(1) AS total_rows"] + [
            f"{self._distinct_1d(c)} AS distinct_{i}" for i, c in enumerate(columns)
        ]
        names = ["total_rows"] + [f"distinct_{i}" for i in range(len(columns))]
        return "SELECT\n  " + ",\n  ".join(select) + f"\nFROM {self._t(table)}", names

    def query_composite_grain(self, table, columns):
        return (
            f"SELECT {self._distinct_tuples(self._t(table), columns)} AS distinct_grain",
            ["distinct_grain"],
        )

    def query_pairwise_batch(self, table, pairs):
        select = [
            f"{self._distinct_tuples(self._t(table), [a, b])} AS combos_{i}"
            for i, (a, b) in enumerate(pairs)
        ]
        return "SELECT\n  " + ",\n  ".join(select), [f"combos_{i}" for i in range(len(pairs))]

    def query_sample_pairwise_batch(self, table, pairs, sample_percent):
        if not self.supports_sample:
            raise NotImplementedError("SQLite < 3.35 cannot pin a sample (no MATERIALIZED CTE)")
        unique_cols, dist_names, combo_names = _sample_aliases(pairs)
        threshold = int(_clamp_percent(sample_percent) * 10_000)  # out of 1,000,000
        select = [
            f"(SELECT {self._distinct_1d(c)} FROM s) AS {alias}"
            for c, alias in zip(unique_cols, dist_names)
        ] + [
            f"{self._distinct_tuples('s', [a, b])} AS {alias}"
            for (a, b), alias in zip(pairs, combo_names)
        ]
        sql = (
            f"WITH s AS MATERIALIZED (SELECT * FROM {self._t(table)} "
            f"WHERE abs(random()) % 1000000 < {threshold})\n"
            "SELECT\n  " + ",\n  ".join(select)
        )
        return sql, dist_names + combo_names
