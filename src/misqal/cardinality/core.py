"""
Pure cardinality logic: no SQL text, no execution.

Used by ``misqal.cardinality.matrix`` and unit-tested in isolation.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

from misqal.cardinality.models import (
    CardinalityKind,
    CardinalityProfile,
    CardinalityRelation,
    GrainObservation,
    HierarchyChain,
)

# Safe limit for Spark: avoid StackOverflowError / Catalyst blow-up from too many projections
MAX_PAIRS_PER_QUERY = 25

# Proofs containing this substring are transitive FD closure only — do not skip full-table
# pairwise work based on them alone (can incorrectly prune before an empirical 1:1 is observed).
_TRANSITIVE_INF_PROOF_MARK = "Inferred via transitive functional dependency"


def _try_deduce_composite_grain(
    columns: list[str],
    column_counts: dict[str, int],
    total_rows: int,
    existing_profile: CardinalityProfile | None,
) -> tuple[int | None, bool, bool] | None:
    """
    Infer composite-grain distinct count / uniqueness without COUNT(DISTINCT a,b,...) when possible.

    - Exact match on ``existing_profile.tested_grains`` reuses prior evidence.
    - If any requested column is a row-unique key (distinct == total_rows), the tuple is unique.
    - If ∏_i COUNT(DISTINCT col_i) < total_rows, the tuple cannot be unique (too few joint values).

    Returns ``(distinct_grain, is_unique, distinct_is_exact)`` when resolved without SQL.
    ``distinct_grain`` is None when non-uniqueness is proved by the pigeonhole bound only (no exact
    COUNT(DISTINCT tuple)). Returns ``None`` if a full SQL grain scan is still needed.
    """
    requested_cols = frozenset(columns)
    if (
        existing_profile
        and existing_profile.tested_grains
        and (
            grain := next(
                (
                    g
                    for g in existing_profile.tested_grains
                    if frozenset(g.columns) == requested_cols
                ),
                None,
            )
        )
    ):
        return (
            grain.distinct_count,
            grain.is_unique,
            grain.distinct_count_is_exact,
        )

    if total_rows <= 0:
        return (0, False, True)

    if any(column_counts.get(c, 0) == total_rows for c in columns):
        return (total_rows, True, True)

    upper_bound = 1
    for n in (column_counts.get(c, 0) for c in columns):
        if n == 0:
            return (0, False, True)
        # upper_bound * n >= total_rows => cannot prove joint distinct < total_rows via ∏ bound
        if upper_bound > (total_rows - 1) // n:
            return None
        upper_bound *= n
    # Product bound < total_rows ⇒ not unique; exact composite distinct requires a scan (omitted).
    return (None, False, False)


def _classify_columns(
    columns: list[str],
    col_counts: dict[str, int],
    total_rows: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Classify columns into primary_keys, constants, dimensions.
    - Primary Key: COUNT(DISTINCT X) == total_rows and total_rows > 0 → X determines everything.
    - Constant: COUNT(DISTINCT Z) <= 1 → determined by everything.
    - Dimension: everything else (need pairwise empirical checks).
    """
    primary_keys: list[str] = []
    constants: list[str] = []
    dimensions: list[str] = []
    for col in columns:
        cnt = col_counts.get(col, 0)
        if total_rows > 0 and cnt == total_rows:
            primary_keys.append(col)
        elif cnt <= 1:
            constants.append(col)
        else:
            dimensions.append(col)
    return primary_keys, constants, dimensions


def _infer_pk_relationships(
    primary_keys: list[str],
    all_columns: list[str],
) -> dict[str, CardinalityRelation]:
    """
    Law of Absolute Identity: PK determines every other column (1:1 to other PKs, N:1 to rest).
    Inserts both P::D and D::P so downstream lookups by subject::pass_through find fan-out (1:N).
    """
    relationships: dict[str, CardinalityRelation] = {}
    pk_set = set(primary_keys)
    proof = "Inferred: source is a Primary Key (distinct count = total rows)."
    rev_proof = (
        "Inferred: target is a Primary Key (distinct count = total rows); reverse edge."
    )
    for pk in primary_keys:
        for other in all_columns:
            if pk == other:
                continue
            card = CardinalityKind.ONE_ONE if other in pk_set else CardinalityKind.N_ONE
            rev_card = (
                CardinalityKind.ONE_ONE if other in pk_set else CardinalityKind.ONE_N
            )
            relationships[f"{pk}::{other}"] = CardinalityRelation(
                source_column=pk,
                target_column=other,
                cardinality=card,
                proof=proof,
            )
            relationships[f"{other}::{pk}"] = CardinalityRelation(
                source_column=other,
                target_column=pk,
                cardinality=rev_card,
                proof=rev_proof,
            )
    return relationships


def _infer_constant_relationships(
    constants: list[str],
    all_columns: list[str],
    primary_keys: list[str],
) -> dict[str, CardinalityRelation]:
    """
    Law of Constants: every non-PK column determines each constant (N:1).
    Inserts both D::C (N:1) and C::D (1:N) so lookups by subject::pass_through see the reverse edge.
    """
    relationships: dict[str, CardinalityRelation] = {}
    pk_set = set(primary_keys)
    proof = "Inferred: target is a Constant (distinct count ≤ 1)."
    rev_proof = "Inferred: source is a Constant (distinct count ≤ 1); reverse edge."
    for const in constants:
        for other in all_columns:
            if const == other or other in pk_set:
                continue
            relationships[f"{other}::{const}"] = CardinalityRelation(
                source_column=other,
                target_column=const,
                cardinality=CardinalityKind.N_ONE,
                proof=proof,
            )
            relationships[f"{const}::{other}"] = CardinalityRelation(
                source_column=const,
                target_column=other,
                cardinality=CardinalityKind.ONE_N,
                proof=rev_proof,
            )
    return relationships


def _deduce_cardinality(
    dist_a: int,
    dist_b: int,
    combos: int,
) -> tuple[CardinalityKind, CardinalityKind]:
    """From 1D counts and pair combo count, deduce A→B and B→A cardinality."""
    if combos <= 0:
        return CardinalityKind.M_N, CardinalityKind.M_N
    if dist_a == combos and dist_b == combos:
        return CardinalityKind.ONE_ONE, CardinalityKind.ONE_ONE
    if dist_a == combos and dist_b < combos:
        return CardinalityKind.N_ONE, CardinalityKind.ONE_N
    if dist_a < combos and dist_b == combos:
        return CardinalityKind.ONE_N, CardinalityKind.N_ONE
    return CardinalityKind.M_N, CardinalityKind.M_N


def _evaluate_sample_pairs(
    sample_combos: int,
    sample_dist_a: int,
    sample_dist_b: int,
) -> CardinalityKind | None:
    """
    Fail-fast from sample: if M:N in sample, then globally M:N (adding rows cannot undo M:N).
    Returns M_N if proven, else None (inconclusive; needs full-table or deduction).
    """
    if sample_combos > sample_dist_a and sample_combos > sample_dist_b:
        return CardinalityKind.M_N
    return None


def _apply_1d_pigeonhole(
    full_dist_a: int,
    full_dist_b: int,
    sample_combos: int,
    sample_dist_a: int,
    sample_dist_b: int,
) -> bool:
    """
    Returns True if the global relationship is mathematically forced to be M:N
    based on the combination of full-table 1D boundaries and sample violations.

    Directional limits: D(A) > D(B) => only N:1 or M:N possible; if sample shows
    A mapping to multiple B's, N:1 is violated => M:N. Symmetric for D(B) > D(A).
    """
    # A violates 1:1 or N:1 if a single A maps to multiple B's in the sample
    a_has_duplicates = sample_combos > sample_dist_a
    # B violates 1:1 or 1:N if a single B maps to multiple A's in the sample
    b_has_duplicates = sample_combos > sample_dist_b

    if full_dist_a == full_dist_b:
        # Globally must be 1:1 or M:N. If either side has duplicates in sample, 1:1 is dead.
        if a_has_duplicates or b_has_duplicates:
            return True
    elif full_dist_a > full_dist_b:
        # Globally must be N:1 or M:N (many A's determine one B). If A maps to multiple B's, N:1 is dead.
        if a_has_duplicates:
            return True
    elif full_dist_b > full_dist_a:
        # Globally must be 1:N or M:N (many B's determine one A). If B maps to multiple A's, 1:N is dead.
        if b_has_duplicates:
            return True
    return False


def _infer_transitive_relationships(
    dimensions: list[str],
    relationships: dict[str, CardinalityRelation],
) -> None:
    """
    Apply Armstrong-style deduction: if A→B and B→C are N:1 or 1:1, infer A→C as N:1 (and C→A as 1:N).
    Only adds entries for pairs not already present. Mutates relationships in place.
    """
    # A determines B (A→B is N:1 or 1:1) => edge A -> B
    determines: dict[str, set[str]] = {col: set() for col in dimensions}
    for a in dimensions:
        for b in dimensions:
            if a == b:
                continue
            rel = relationships.get(f"{a}::{b}")
            if rel is None:
                continue
            if rel.cardinality in (CardinalityKind.N_ONE, CardinalityKind.ONE_ONE):
                determines[a].add(b)

    for start in dimensions:
        queue = list(determines[start])
        visited = set(queue)
        while queue:
            current = queue.pop(0)
            rel_key = f"{start}::{current}"
            rev_key = f"{current}::{start}"
            if rel_key not in relationships:
                relationships[rel_key] = CardinalityRelation(
                    source_column=start,
                    target_column=current,
                    cardinality=CardinalityKind.N_ONE,
                    proof=_TRANSITIVE_INF_PROOF_MARK,
                )
            if rev_key not in relationships:
                relationships[rev_key] = CardinalityRelation(
                    source_column=current,
                    target_column=start,
                    cardinality=CardinalityKind.ONE_N,
                    proof=f"{_TRANSITIVE_INF_PROOF_MARK} (reverse)",
                )
            for neighbor in determines[current]:
                if neighbor not in visited and neighbor != start:
                    visited.add(neighbor)
                    queue.append(neighbor)


def _pairwise_prune_allowed(
    relationships: dict[str, CardinalityRelation], a: str, b: str
) -> bool:
    """True if pending (a,b) may be dropped — never for transitive-inference-only edges."""
    rel = relationships.get(f"{a}::{b}")
    if rel is None:
        return False
    if _TRANSITIVE_INF_PROOF_MARK in rel.proof:
        return False
    return True


def _apply_1v1_equivalence(relationships: dict[str, CardinalityRelation]) -> None:
    """
    If A↔B is 1:1, then for any X, A::X equals B::X.

    Repeats until a full pass adds no new edges. Each pass may need to run again only when
    new 1:1 structure appears mid-pass; the iteration cap is ``n * (n - 1) + 1`` for
    ``n`` distinct column names in the graph — a directed graph has at most ``n * (n-1)``
    non-loop edges, so no more productive passes are possible beyond that.
    """
    all_cols_init: set[str] = set()
    for key in relationships:
        a, b = key.split("::", 1)
        all_cols_init.update((a, b))
    n = len(all_cols_init)
    max_rounds = max(1, n * (n - 1) + 1)

    for _round in range(max_rounds):
        size_before = len(relationships)
        one_one_pairs: list[tuple[str, str]] = []
        all_cols: set[str] = set()
        for key in relationships:
            a, b = key.split("::", 1)
            all_cols.add(a)
            all_cols.add(b)
            if a >= b:
                continue
            rel = relationships[key]
            if rel.cardinality == CardinalityKind.ONE_ONE:
                one_one_pairs.append((a, b))

        for a, b in one_one_pairs:
            for x in all_cols:
                if x == a or x == b:
                    continue

                ax, bx = f"{a}::{x}", f"{b}::{x}"
                xa, xb = f"{x}::{a}", f"{x}::{b}"

                # 1. Forward propagation (A -> B)
                if ax in relationships and bx not in relationships:
                    relationships[bx] = CardinalityRelation(
                        source_column=b,
                        target_column=x,
                        cardinality=relationships[ax].cardinality,
                        proof=f"1:1 equivalence with {a}",
                    )
                if xa in relationships and xb not in relationships:
                    relationships[xb] = CardinalityRelation(
                        source_column=x,
                        target_column=b,
                        cardinality=relationships[xa].cardinality,
                        proof=f"1:1 equivalence with {a}",
                    )

                # 2. Backward propagation (B -> A)
                if bx in relationships and ax not in relationships:
                    relationships[ax] = CardinalityRelation(
                        source_column=a,
                        target_column=x,
                        cardinality=relationships[bx].cardinality,
                        proof=f"1:1 equivalence with {b}",
                    )
                if xb in relationships and xa not in relationships:
                    relationships[xa] = CardinalityRelation(
                        source_column=x,
                        target_column=a,
                        cardinality=relationships[xb].cardinality,
                        proof=f"1:1 equivalence with {b}",
                    )

        if len(relationships) == size_before:
            return
    logging.getLogger(__name__).debug(
        "_apply_1v1_equivalence stopped after iteration cap (graph may be incomplete)"
    )


def existing_profile_covers_columns(
    profile: CardinalityProfile, columns: list[str]
) -> bool:
    """
    Return True only if the profile has explicit relationship entries for every requested pair.

    Merged profiles can list many columns in ``used_columns`` while missing edges between some
    pairs (e.g. incremental tool calls); subset-of-columns is not sufficient for a cache hit.
    """
    if len(columns) < 2:
        return False
    if not set(columns).issubset(profile.used_columns):
        return False
    for a, b in combinations(columns, 2):
        if f"{a}::{b}" not in profile.relationships:
            return False
    return True


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _extract_value(v: Any) -> int:
    """Normalize cell value to int (handles {'value': ...} or raw)."""
    if isinstance(v, dict) and "value" in v:
        v = v["value"]
    if isinstance(v, dict):
        return 0
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def _parse_row(row: Any, expected_col_names: list[str]) -> dict[str, int]:
    """Turn executor row (list or dict col_0, col_1, ...) into {name: int}."""
    if isinstance(row, dict):
        values = [row.get(f"col_{i}") for i in range(len(expected_col_names))]
    else:
        values = list(row)
    return {
        name: _extract_value(values[i]) if i < len(values) else 0
        for i, name in enumerate(expected_col_names)
    }


# ---------------------------------------------------------------------------
# Profile assembly and column subsetting
# ---------------------------------------------------------------------------


def _assemble_profile(
    table_name: str,
    columns: list[str],
    total_rows: int,
    column_counts: dict[str, int],
    distinct_grain: int,
    relationships: dict[str, CardinalityRelation],
    primary_keys: list[str],
    constants: list[str],
    composite_grain_unique: bool | None = None,
    grain_distinct_exact: bool = True,
) -> CardinalityProfile:
    """Build the unified CardinalityProfile for downstream agents."""
    if composite_grain_unique is None:
        grain_unique = (total_rows == distinct_grain) if total_rows else False
    else:
        grain_unique = composite_grain_unique
    observation = GrainObservation(
        columns=columns,
        distinct_count=distinct_grain,
        is_unique=grain_unique,
        distinct_count_is_exact=grain_distinct_exact,
    )
    return CardinalityProfile(
        table_name=table_name,
        total_rows=total_rows,
        column_counts=column_counts,
        primary_keys=primary_keys,
        constants=constants,
        tested_grains=[observation],
        relationships=relationships,
    )


def profile_subset_for_columns(
    profile: CardinalityProfile, columns: list[str]
) -> CardinalityProfile:
    """
    Return a new CardinalityProfile containing only the requested columns:
    - column_counts, primary_keys, constants restricted to those columns
    - relationships where both source and target are in columns
    - tested_grains that are subsets of columns
    - hierarchies filtered to chains whose columns are in the requested set (≥2 columns)
    """
    col_set = set(columns)
    column_counts = {k: v for k, v in profile.column_counts.items() if k in col_set}
    primary_keys = [c for c in profile.primary_keys if c in col_set]
    constants = [c for c in profile.constants if c in col_set]
    tested_grains = [
        g for g in profile.tested_grains if set(g.columns).issubset(col_set)
    ]
    relationships = {
        k: v
        for k, v in profile.relationships.items()
        if v.source_column in col_set and v.target_column in col_set
    }
    hierarchies_raw = [
        [c for c in h.columns if c in col_set]
        for h in profile.hierarchies
        if len([c for c in h.columns if c in col_set]) >= 2
    ]
    seen: set[tuple[str, ...]] = set()
    hierarchies = []
    for cols in hierarchies_raw:
        key = tuple(cols)
        if key not in seen:
            seen.add(key)
            hierarchies.append(HierarchyChain(columns=cols))
    return CardinalityProfile(
        table_name=profile.table_name,
        total_rows=profile.total_rows,
        column_counts=column_counts,
        primary_keys=primary_keys,
        constants=constants,
        tested_grains=tested_grains,
        relationships=relationships,
        hierarchies=hierarchies,
    )


def try_existing_cardinality_profile(
    table_name: str,
    columns: list[str],
    existing_profiles: dict[str, CardinalityProfile] | None,
) -> CardinalityProfile | None:
    """
    If an existing profile for the table covers the requested columns, return a subset profile.
    Otherwise return None. Call this before run_cardinality_matrix to avoid redundant runs.
    """
    if not columns or len(columns) < 2 or not existing_profiles:
        return None
    profile = existing_profiles.get(table_name)
    if profile is None:
        return None
    if not existing_profile_covers_columns(profile, columns):
        return None
    return profile_subset_for_columns(profile, columns)


# ---------------------------------------------------------------------------
# Hierarchy detection (used by to_markdown)
# ---------------------------------------------------------------------------


def _extract_1n_and_equivalences(
    profile: CardinalityProfile,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Extract 1:N edges (parent, child) and 1:1 equivalences from profile.relationships.
    Parent 1:N child means parent is coarser (one parent has many children).
    """
    true_1_to_n: list[tuple[str, str]] = []
    equivalences: list[tuple[str, str]] = []
    for a, b in combinations(profile.used_columns, 2):
        rel_ab = profile.relationships.get(f"{a}::{b}")
        rel_ba = profile.relationships.get(f"{b}::{a}")
        if not rel_ab or not rel_ba:
            continue
        if (
            rel_ab.cardinality == CardinalityKind.ONE_ONE
            and rel_ba.cardinality == CardinalityKind.ONE_ONE
        ):
            equivalences.append((a, b))
        elif (
            rel_ab.cardinality == CardinalityKind.ONE_N
            and rel_ba.cardinality == CardinalityKind.N_ONE
        ):
            true_1_to_n.append((a, b))  # a is parent (1), b is child (N)
        elif (
            rel_ab.cardinality == CardinalityKind.N_ONE
            and rel_ba.cardinality == CardinalityKind.ONE_N
        ):
            true_1_to_n.append((b, a))  # b is parent, a is child
    return true_1_to_n, equivalences


def _transitive_reduce_1n(
    used_columns: list[str],
    true_1_to_n: list[tuple[str, str]],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """
    Keep only direct parent→child edges (remove bypass edges).
    Returns (reduced_adj, in_degree): reduced_adj[parent] = list of direct children.
    """
    adj: dict[str, list[str]] = {c: [] for c in used_columns}
    for p, c in true_1_to_n:
        adj[p].append(c)
    reduced_adj: dict[str, list[str]] = {c: [] for c in used_columns}
    in_degree: dict[str, int] = {c: 0 for c in used_columns}
    for p in used_columns:
        for c in adj[p]:
            queue = [m for m in adj[p] if m != c]
            visited = set(queue)
            found_bypass = False
            while queue:
                curr = queue.pop(0)
                if curr == c:
                    found_bypass = True
                    break
                for nxt in adj.get(curr, []):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            if not found_bypass:
                reduced_adj[p].append(c)
                in_degree[c] += 1
    return reduced_adj, in_degree


def _hierarchy_chains_from_reduced(
    used_columns: list[str],
    reduced_adj: dict[str, list[str]],
    in_degree: dict[str, int],
) -> list[list[str]]:
    """Enumerate root-to-leaf paths (each path = one hierarchy chain, coarsest → finest)."""
    roots = [c for c in used_columns if in_degree.get(c, 0) == 0]
    chains: list[list[str]] = []

    def dfs(path: list[str], node: str) -> None:
        children = reduced_adj.get(node, [])
        if not children:
            if len(path) >= 2:
                chains.append(path[:])
            return
        for child in children:
            path.append(child)
            dfs(path, child)
            path.pop()

    for root in roots:
        dfs([root], root)
    return chains


def detect_hierarchies(profile: CardinalityProfile) -> list[HierarchyChain]:
    """
    Detect column hierarchies from 1:N relationships (with transitive reduction).
    Returns list of chains ordered coarsest (root) → finest (leaf); chains have length >= 2.
    """
    if not profile.used_columns or not profile.relationships:
        return []
    true_1_to_n, _ = _extract_1n_and_equivalences(profile)
    if not true_1_to_n:
        return []
    reduced_adj, in_degree = _transitive_reduce_1n(profile.used_columns, true_1_to_n)
    chains = _hierarchy_chains_from_reduced(
        profile.used_columns, reduced_adj, in_degree
    )
    return [HierarchyChain(columns=ch) for ch in chains]


def get_hierarchy_graph(
    profile: CardinalityProfile,
) -> tuple[dict[str, list[str]], dict[str, int], list[tuple[str, str]]]:
    """
    Return (reduced_adj, in_degree, equivalences) for visualization.
    reduced_adj[parent] = direct children; in_degree[col] = number of parents; equivalences = 1:1 pairs.
    """
    used = profile.used_columns
    if not used:
        return ({}, {}, [])
    true_1_to_n, equivalences = _extract_1n_and_equivalences(profile)
    reduced_adj, in_degree = _transitive_reduce_1n(used, true_1_to_n)
    return reduced_adj, in_degree, equivalences
