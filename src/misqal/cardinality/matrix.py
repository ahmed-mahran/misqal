"""
Cardinality matrix orchestration: a cache-aware, delta-completion pipeline that issues as little
SQL as possible and deduces the rest.

1. 1D baseline (delta against ``existing_profile``).
2. Composite grain: deduce without SQL when possible, else one scan.
3. Classify PK / constants / dimensions; seed edges from rules and the cached profile.
4. Closure (1:1 equivalence + transitivity) on the seeded graph; only missing pairs proceed.
5. Sample phase: sampled M:N ⇒ global M:N; 1D pigeonhole.
6. Full-table pairwise for survivors, in batches, with mid-flight deduction pruning.
7. Final closure, assembly, hierarchy detection.

SQL text comes from a ``Dialect`` and runs through an ``Executor``; ``MatrixStats`` counts queries
and pair outcomes per phase.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol, Sequence

from misqal.cardinality.core import (
    MAX_PAIRS_PER_QUERY,
    _apply_1d_pigeonhole,
    _apply_1v1_equivalence,
    _assemble_profile,
    _classify_columns,
    _deduce_cardinality,
    _evaluate_sample_pairs,
    _infer_constant_relationships,
    _infer_pk_relationships,
    _infer_transitive_relationships,
    _pairwise_prune_allowed,
    _parse_row,
    _try_deduce_composite_grain,
    detect_hierarchies,
)
from misqal.cardinality.dialect import Dialect
from misqal.cardinality.models import (
    CardinalityKind,
    CardinalityProfile,
    CardinalityRelation,
)

logger = logging.getLogger(__name__)

# Sample fraction for fail-fast M:N detection (e.g. 0.05 = 5%). M:N in sample => globally M:N.
CARDINALITY_SAMPLE_FRACTION = 0.05
# If table has fewer rows than this, skip sample phase (run full-table for all pairs).
# When sampling, use at least this many rows by raising sample % for small tables.
MIN_SAMPLE_ROWS = 100
# Maximum number of columns per call: wider lists make heavy queries; split into groups instead.
MAX_COLUMNS_PER_CARDINALITY_MATRIX_CALL = 25


class Executor(Protocol):
    def query(self, sql: str) -> list[Sequence[Any]]: ...


@dataclass
class SQLiteExecutor:
    """Read-only SQLite executor."""

    db_path: Path
    timeout: float = 60.0

    def query(self, sql: str) -> list[Sequence[Any]]:
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=self.timeout) as con:
            return con.execute(sql).fetchall()


@dataclass
class SparkExecutor:
    """Runs queries on a ``pyspark.sql.SparkSession`` (optional ``spark`` extra)."""

    spark: Any

    def query(self, sql: str) -> list[Sequence[Any]]:
        return [tuple(r) for r in self.spark.sql(sql).collect()]


@dataclass
class MatrixStats:
    """Query and pair accounting for one matrix call."""

    queries: Counter = field(default_factory=Counter)  # phase -> number of SQL statements
    pairs_requested: int = 0  # C(|dimensions|, 2)
    pairs_delta: int = 0  # still unknown after rules + cached profile + closure
    pairs_sample_mn: int = 0  # proved M:N by a sampled double violation
    pairs_pigeonhole_mn: int = 0  # proved M:N by 1D bounds + sampled violation
    pairs_full_scan: int = 0  # resolved by exact COUNT(DISTINCT a, b)
    pairs_pruned_midflight: int = 0  # dropped from the queue by mid-flight deduction
    grain_deduced: bool = False
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "queries": dict(self.queries)}


def _run(executor: Executor, stats: MatrixStats, phase: str, sql: str) -> list[Sequence[Any]]:
    stats.queries[phase] += 1
    logger.debug("cardinality_matrix SQL phase=%s\n%s", phase, sql)
    return executor.query(sql)


def _mn_pair(a: str, b: str, proof: str) -> dict[str, CardinalityRelation]:
    return {
        f"{a}::{b}": CardinalityRelation(
            source_column=a, target_column=b, cardinality=CardinalityKind.M_N, proof=proof
        ),
        f"{b}::{a}": CardinalityRelation(
            source_column=b, target_column=a, cardinality=CardinalityKind.M_N, proof=proof
        ),
    }


def _run_sample_pairwise_phase(
    table_name: str,
    col_counts: dict[str, int],
    executor: Executor,
    dialect: Dialect,
    stats: MatrixStats,
    total_rows: int,
    pairs: list[tuple[str, str]],
    sample_percent: float = CARDINALITY_SAMPLE_FRACTION * 100.0,
) -> tuple[dict[str, CardinalityRelation], list[tuple[str, str]]]:
    """Mark M:N from a sample or the 1D pigeonhole; return them and the pairs that still need a
    full-table count. Skips sampling for small tables or dialects that cannot sample."""
    if not pairs:
        return {}, []
    if (total_rows > 0 and total_rows < MIN_SAMPLE_ROWS) or not dialect.supports_sample:
        return {}, list(pairs)

    effective_pct = sample_percent
    if total_rows > 0 and total_rows < (MIN_SAMPLE_ROWS / (sample_percent / 100.0)):
        effective_pct = min(100.0, (MIN_SAMPLE_ROWS / total_rows) * 100.0)

    relationships: dict[str, CardinalityRelation] = {}
    remaining_pairs: list[tuple[str, str]] = []
    for i in range(0, len(pairs), MAX_PAIRS_PER_QUERY):
        batch = pairs[i : i + MAX_PAIRS_PER_QUERY]
        query, col_names = dialect.query_sample_pairwise_batch(table_name, batch, effective_pct)
        try:
            rows = _run(executor, stats, "sample_pairwise", query)
        except Exception:
            # e.g. TABLESAMPLE unsupported on a view: fall back to full-table for all pairs.
            logger.debug("sample phase failed; falling back to full table", exc_info=True)
            return relationships, remaining_pairs + list(pairs[i:])
        if not rows:
            remaining_pairs.extend(batch)
            continue
        raw = _parse_row(rows[0], col_names)
        unique_col_index = {c: j for j, c in enumerate(sorted({x for p in batch for x in p}))}
        for j, (a, b) in enumerate(batch):
            sc = raw.get(f"combos_{j}", 0)
            sa = raw.get(f"dist_1d_{unique_col_index[a]}", 0)
            sb = raw.get(f"dist_1d_{unique_col_index[b]}", 0)
            fa = col_counts.get(a, 0)
            fb = col_counts.get(b, 0)
            if _evaluate_sample_pairs(sc, sa, sb) == CardinalityKind.M_N:
                relationships.update(
                    _mn_pair(a, b, f"Sample ({effective_pct:.1f}%) proved M:N; holds globally.")
                )
                stats.pairs_sample_mn += 1
            elif _apply_1d_pigeonhole(fa, fb, sc, sa, sb):
                relationships.update(
                    _mn_pair(
                        a, b, "1D pigeonhole (full-table bounds + sample violation) → globally M:N."
                    )
                )
                stats.pairs_pigeonhole_mn += 1
            else:
                remaining_pairs.append((a, b))
    return relationships, remaining_pairs


def _run_pairwise_batches(
    table_name: str,
    col_counts: dict[str, int],
    executor: Executor,
    dialect: Dialect,
    stats: MatrixStats,
    pairs: list[tuple[str, str]],
    relationships: dict[str, CardinalityRelation],
    transitivity_columns: list[str],
) -> None:
    """Exact pairwise counts in batches; after each batch run 1:1-equivalence and transitivity
    and drop pending pairs that are now inferred. Mutates ``relationships``."""
    pending: list[tuple[str, str]] = list(pairs)
    while pending:
        batch = pending[:MAX_PAIRS_PER_QUERY]
        batch_set = set(batch)
        query, col_names = dialect.query_pairwise_batch(table_name, batch)
        rows = _run(executor, stats, "pairwise", query)
        pending = [p for p in pending if p not in batch_set]
        if not rows:
            continue
        raw = _parse_row(rows[0], col_names)
        for i, (a, b) in enumerate(batch):
            combos = raw.get(f"combos_{i}", 0)
            dist_a = col_counts.get(a, 0)
            dist_b = col_counts.get(b, 0)
            card_ab, card_ba = _deduce_cardinality(dist_a, dist_b, combos)
            proof = f"dist_src={dist_a}, dist_tgt={dist_b}, combos={combos}"
            relationships[f"{a}::{b}"] = CardinalityRelation(
                source_column=a, target_column=b, cardinality=card_ab, proof=proof
            )
            relationships[f"{b}::{a}"] = CardinalityRelation(
                source_column=b, target_column=a, cardinality=card_ba, proof=proof
            )
            stats.pairs_full_scan += 1

        # Mid-flight deduction: 1:1-equivalence first, then transitivity (so 1:1 chains stay 1:1)
        _apply_1v1_equivalence(relationships)
        _infer_transitive_relationships(transitivity_columns, relationships)
        inferred = {(a, b) for (a, b) in pending if _pairwise_prune_allowed(relationships, a, b)}
        if inferred:
            pending = [p for p in pending if p not in inferred]
            stats.pairs_pruned_midflight += len(inferred)


def run_cardinality_matrix(
    table_name: str,
    columns: list[str],
    executor: Executor,
    dialect: Dialect,
    existing_profile: CardinalityProfile | None = None,
    stats: MatrixStats | None = None,
    max_columns: int | None = MAX_COLUMNS_PER_CARDINALITY_MATRIX_CALL,
) -> CardinalityProfile:
    """Compute (or complete, given ``existing_profile``) the cardinality profile of ``columns``
    in ``table_name``. ``max_columns=None`` lifts the per-call column limit (offline profiling of
    whole tables). Raises ``ValueError`` on invalid arguments or an empty result."""
    stats = stats if stats is not None else MatrixStats()
    started = time.monotonic()
    if len(columns) < 2:
        raise ValueError(
            "The cardinality matrix computes pairwise relationships between columns; "
            "at least two columns are required."
        )
    if max_columns is not None and len(columns) > max_columns:
        raise ValueError(
            f"At most {max_columns} columns per call; got "
            f"{len(columns)}. Split the dimensions into groups and run one call per group."
        )

    # ----- Phase 1: 1D baseline (delta: reuse cached counts from existing_profile) -----
    column_counts: dict[str, int] = {}
    total_rows = 0
    if existing_profile:
        column_counts = {
            k: v for k, v in (existing_profile.column_counts or {}).items() if k in set(columns)
        }
        total_rows = int(existing_profile.total_rows or 0)
    missing_1d = [c for c in columns if c not in column_counts]
    if missing_1d or total_rows == 0:
        query_1d, names_1d = dialect.query_1d(table_name, missing_1d)
        rows = _run(executor, stats, "1d", query_1d)
        if not rows:
            raise ValueError("No rows were returned for this table.")
        raw_1d = _parse_row(rows[0], names_1d)
        total_rows = int(raw_1d.get("total_rows", total_rows))
        for i, col in enumerate(missing_1d):
            column_counts[col] = raw_1d.get(f"distinct_{i}", 0)
    for c in columns:
        column_counts.setdefault(c, 0)

    # ----- Phase 1b: composite grain (deduce when possible to avoid a scan) -----
    distinct_grain = 0
    composite_grain_unique: bool | None = None
    grain_distinct_exact = True
    deduced = _try_deduce_composite_grain(columns, column_counts, total_rows, existing_profile)
    if deduced is not None:
        d_g, composite_grain_unique, grain_distinct_exact = deduced
        distinct_grain = 0 if d_g is None else d_g
        stats.grain_deduced = True
    else:
        query_grain, names_grain = dialect.query_composite_grain(table_name, columns)
        rows = _run(executor, stats, "composite_grain", query_grain)
        if rows:
            distinct_grain = _parse_row(rows[0], names_grain).get("distinct_grain", 0)

    # ----- Phase 2: classify, seed cached graph, deduce, then delta pairs -----
    primary_keys, constants, dimensions = _classify_columns(columns, column_counts, total_rows)
    relationships: dict[str, CardinalityRelation] = {}
    all_known_cols: set[str] = set(columns)
    if existing_profile and existing_profile.relationships:
        relationships.update(existing_profile.relationships)
        all_known_cols.update(existing_profile.used_columns)
    relationships.update(_infer_pk_relationships(primary_keys, columns))
    relationships.update(_infer_constant_relationships(constants, columns, primary_keys))

    all_known_dims = sorted(all_known_cols)
    _apply_1v1_equivalence(relationships)
    _infer_transitive_relationships(all_known_dims, relationships)

    # Before any SQL on this call, transitive edges only chain rule-seeded or
    # cached edges, so any existing relationship is enough to skip the pair here.
    stats.pairs_requested = len(dimensions) * (len(dimensions) - 1) // 2
    delta_pairs = [
        (a, b) for a, b in combinations(dimensions, 2) if f"{a}::{b}" not in relationships
    ]
    stats.pairs_delta = len(delta_pairs)

    # ----- Phase 2b/2c: sample (fail-fast M:N) + 1D pigeonhole; Phase 3: full-table -----
    if delta_pairs:
        sample_rels, remaining = _run_sample_pairwise_phase(
            table_name, column_counts, executor, dialect, stats, total_rows, delta_pairs
        )
        relationships.update(sample_rels)
        if remaining:
            _run_pairwise_batches(
                table_name, column_counts, executor, dialect, stats, remaining,
                relationships, all_known_dims,
            )
        _apply_1v1_equivalence(relationships)
        _infer_transitive_relationships(all_known_dims, relationships)

    # ----- Phase 4: assembly -----
    profile = _assemble_profile(
        table_name=table_name,
        columns=columns,
        total_rows=total_rows,
        column_counts=column_counts,
        distinct_grain=distinct_grain,
        relationships=relationships,
        primary_keys=primary_keys,
        constants=constants,
        composite_grain_unique=composite_grain_unique,
        grain_distinct_exact=grain_distinct_exact,
    )
    profile = profile.model_copy(update={"hierarchies": detect_hierarchies(profile)})
    stats.seconds = time.monotonic() - started
    return profile
