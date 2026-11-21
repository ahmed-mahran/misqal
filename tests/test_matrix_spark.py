"""Spark dialect against the same oracle (needs the ``spark`` extra and a JVM; skipped otherwise)."""

from __future__ import annotations

import random
from itertools import combinations

import pytest

pyspark = pytest.importorskip("pyspark")

from misqal.cardinality.dialect import SparkDialect  # noqa: E402
from misqal.cardinality.matrix import MatrixStats, SparkExecutor, run_cardinality_matrix  # noqa: E402
from misqal.cardinality.models import CardinalityKind  # noqa: E402
from test_matrix_sqlite import oracle, random_table  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    s = SparkSession.builder.master("local[1]").appName("misqal-test").getOrCreate()
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def _view(spark, name: str, rows: list[dict]) -> None:
    cols = list(rows[0])
    schema = ", ".join(f"{c} string" for c in cols)
    data = [tuple(None if r[c] is None else str(r[c]) for c in cols) for r in rows]
    spark.createDataFrame(data, schema).createOrReplaceTempView(name)


@pytest.mark.parametrize("seed", range(3))
def test_spark_eq_matches_oracle(spark, seed: int) -> None:
    rng = random.Random(seed)
    rows = [{k: (None if v is None else str(v)) for k, v in r.items()}
            for r in random_table(rng, 3000)]
    _view(spark, "t", rows)
    columns = list(rows[0])
    stats = MatrixStats()
    profile = run_cardinality_matrix("t", columns, SparkExecutor(spark), SparkDialect(),
                                     stats=stats)
    for a, b in combinations(columns, 2):
        assert profile.relationships[f"{a}::{b}"].cardinality == oracle(rows, a, b), (a, b)
    assert stats.queries["sample_pairwise"] >= 1


def test_spark_null_fanout_is_reported(spark) -> None:
    """p1 maps to c1 and to NULL: GROUP BY product, corp gives two rows for p1, so product
    fans out to corp even though every non-NULL corp has a single product."""
    rows = [{"product": "p1", "corp": "c1"}, {"product": "p1", "corp": None},
            {"product": "p2", "corp": "c2"}, {"product": "p3", "corp": "c3"}]
    _view(spark, "t2", rows)
    profile = run_cardinality_matrix("t2", ["product", "corp"], SparkExecutor(spark),
                                     SparkDialect())
    assert profile.relationships["product::corp"].cardinality == CardinalityKind.ONE_N
