"""Unit tests for cardinality matrix pure logic and CardinalityProfile models (no SQL)."""

from __future__ import annotations

import copy
import json
import unittest
from itertools import combinations

from misqal.cardinality.core import (
    _TRANSITIVE_INF_PROOF_MARK,
    _apply_1d_pigeonhole,
    _apply_1v1_equivalence,
    _assemble_profile,
    _classify_columns,
    _deduce_cardinality,
    _evaluate_sample_pairs,
    _extract_1n_and_equivalences,
    _extract_value,
    _hierarchy_chains_from_reduced,
    _infer_constant_relationships,
    _infer_pk_relationships,
    _infer_transitive_relationships,
    _pairwise_prune_allowed,
    _parse_row,
    _transitive_reduce_1n,
    _try_deduce_composite_grain,
    detect_hierarchies,
    existing_profile_covers_columns,
    profile_subset_for_columns,
    try_existing_cardinality_profile,
)
from misqal.cardinality.models import (
    TO_MARKDOWN_RISKY_ONLY_COLUMNS_THRESHOLD,
    TO_MARKDOWN_RISKY_ONLY_RELATIONS_THRESHOLD,
    CardinalityKind,
    CardinalityProfile,
    CardinalityRelation,
    GrainObservation,
    HierarchyChain,
    merge_cardinality_profiles,
)


class TestDeduceCardinality(unittest.TestCase):
    def test_combos_zero_returns_m_n(self) -> None:
        ab, ba = _deduce_cardinality(5, 5, 0)
        self.assertEqual(ab, CardinalityKind.M_N)
        self.assertEqual(ba, CardinalityKind.M_N)

    def test_combos_negative_returns_m_n(self) -> None:
        ab, ba = _deduce_cardinality(3, 3, -1)
        self.assertEqual(ab, CardinalityKind.M_N)
        self.assertEqual(ba, CardinalityKind.M_N)

    def test_one_one(self) -> None:
        ab, ba = _deduce_cardinality(10, 10, 10)
        self.assertEqual(ab, CardinalityKind.ONE_ONE)
        self.assertEqual(ba, CardinalityKind.ONE_ONE)

    def test_n_one_a_determines_b(self) -> None:
        # dist_a=10, dist_b=5, combos=10: each A maps to one B (A determines B).
        # combos == dist_a and dist_b < combos — invariant combos >= max(dist_a, dist_b) holds.
        ab, ba = _deduce_cardinality(10, 5, 10)
        self.assertEqual(ab, CardinalityKind.N_ONE)
        self.assertEqual(ba, CardinalityKind.ONE_N)

    def test_one_n_b_determines_a(self) -> None:
        # dist_a=5, dist_b=10, combos=10: each B maps to one A (B determines A).
        # combos == dist_b and dist_a < combos — invariant holds.
        ab, ba = _deduce_cardinality(5, 10, 10)
        self.assertEqual(ab, CardinalityKind.ONE_N)
        self.assertEqual(ba, CardinalityKind.N_ONE)

    def test_m_n(self) -> None:
        ab, ba = _deduce_cardinality(20, 30, 50)
        self.assertEqual(ab, CardinalityKind.M_N)
        self.assertEqual(ba, CardinalityKind.M_N)


class TestTryDeduceCompositeGrain(unittest.TestCase):
    def test_total_rows_zero(self) -> None:
        r = _try_deduce_composite_grain(["a", "b"], {"a": 1, "b": 1}, 0, None)
        self.assertEqual(r, (0, False, True))

    def test_column_zero_distinct(self) -> None:
        r = _try_deduce_composite_grain(["a", "b"], {"a": 0, "b": 5}, 100, None)
        self.assertEqual(r, (0, False, True))

    def test_single_column_pk_in_tuple(self) -> None:
        r = _try_deduce_composite_grain(
            ["id", "dim"], {"id": 100, "dim": 10}, 100, None
        )
        self.assertEqual(r, (100, True, True))

    def test_pigeonhole_product_strictly_below_total(self) -> None:
        # 5 * 10 = 50 < 100 ⇒ not unique; no exact distinct without scan
        r = _try_deduce_composite_grain(["a", "b"], {"a": 5, "b": 10}, 100, None)
        self.assertEqual(r, (None, False, False))

    def test_pigeonhole_product_equals_total_needs_sql(self) -> None:
        # 3 * 3 = 9 == total_rows ⇒ bound does not prove duplication; not < total_rows
        r = _try_deduce_composite_grain(["a", "b"], {"a": 3, "b": 3}, 9, None)
        self.assertIsNone(r)

    def test_bound_exceeds_early_returns_none(self) -> None:
        # After first column ub=6; 6 > (10-1)//3 = 3 ⇒ cannot multiply further → SQL
        r = _try_deduce_composite_grain(["a", "b"], {"a": 6, "b": 3}, 10, None)
        self.assertIsNone(r)

    def test_cache_hit_reuses_grain(self) -> None:
        profile = CardinalityProfile(
            table_name="t",
            total_rows=1,
            column_counts={},
            tested_grains=[
                GrainObservation(
                    columns=["a", "b"],
                    distinct_count=42,
                    is_unique=False,
                    distinct_count_is_exact=True,
                )
            ],
            relationships={},
        )
        r = _try_deduce_composite_grain(["a", "b"], {"a": 5, "b": 5}, 100, profile)
        self.assertEqual(r, (42, False, True))

    def test_cache_hit_forwards_inexact_flag(self) -> None:
        profile = CardinalityProfile(
            table_name="t",
            total_rows=100,
            column_counts={},
            tested_grains=[
                GrainObservation(
                    columns=["a", "b"],
                    distinct_count=0,
                    is_unique=False,
                    distinct_count_is_exact=False,
                )
            ],
            relationships={},
        )
        r = _try_deduce_composite_grain(["a", "b"], {"a": 5, "b": 10}, 100, profile)
        self.assertEqual(r, (0, False, False))


class TestClassifyColumns(unittest.TestCase):
    def test_total_rows_zero_no_pk(self) -> None:
        pks, consts, dims = _classify_columns(["x"], {"x": 0}, 0)
        self.assertEqual(pks, [])
        self.assertEqual(consts, ["x"])
        self.assertEqual(dims, [])

    def test_distinct_one_is_constant_not_pk_in_multi_row_table(self) -> None:
        # cnt=1 < total_rows=2 → not a PK; cnt<=1 → constant.
        pks, consts, dims = _classify_columns(["x", "y"], {"x": 1, "y": 1}, 2)
        self.assertEqual(pks, [])
        self.assertEqual(sorted(consts), ["x", "y"])
        self.assertEqual(dims, [])

    def test_pk_constant_dimension_mixed(self) -> None:
        pks, consts, dims = _classify_columns(
            ["id", "country", "flag"],
            {"id": 100, "country": 50, "flag": 1},
            100,
        )
        self.assertEqual(pks, ["id"])
        self.assertEqual(consts, ["flag"])
        self.assertEqual(dims, ["country"])


class TestInferPkRelationships(unittest.TestCase):
    def test_no_pk_empty(self) -> None:
        self.assertEqual(_infer_pk_relationships([], ["a", "b"]), {})

    def test_single_pk_to_dimensions(self) -> None:
        rel = _infer_pk_relationships(["id"], ["id", "dim"])
        self.assertEqual(rel["id::dim"].cardinality, CardinalityKind.N_ONE)
        self.assertEqual(rel["dim::id"].cardinality, CardinalityKind.ONE_N)

    def test_two_pks_mutual_one_one(self) -> None:
        rel = _infer_pk_relationships(["id1", "id2"], ["id1", "id2", "dim"])
        self.assertEqual(rel["id1::id2"].cardinality, CardinalityKind.ONE_ONE)
        self.assertEqual(rel["id2::id1"].cardinality, CardinalityKind.ONE_ONE)
        self.assertEqual(rel["id1::dim"].cardinality, CardinalityKind.N_ONE)


class TestInferConstantRelationships(unittest.TestCase):
    def test_constant_dimension_edges(self) -> None:
        rel = _infer_constant_relationships(
            ["flag"], ["flag", "dim"], primary_keys=["id"]
        )
        self.assertEqual(rel["dim::flag"].cardinality, CardinalityKind.N_ONE)
        self.assertEqual(rel["flag::dim"].cardinality, CardinalityKind.ONE_N)

    def test_skips_pk_constant_pair(self) -> None:
        rel = _infer_constant_relationships(
            ["flag"], ["flag", "id"], primary_keys=["id"]
        )
        self.assertNotIn("id::flag", rel)
        self.assertNotIn("flag::id", rel)


class TestApply1v1Equivalence(unittest.TestCase):
    def _pair(self, a: str, b: str, kind: CardinalityKind) -> None:
        self.relationships[f"{a}::{b}"] = CardinalityRelation(
            source_column=a, target_column=b, cardinality=kind, proof="test"
        )

    def setUp(self) -> None:
        self.relationships: dict[str, CardinalityRelation] = {}

    def test_forward_propagation(self) -> None:
        self._pair("a", "b", CardinalityKind.ONE_ONE)
        self._pair("b", "a", CardinalityKind.ONE_ONE)
        self._pair("a", "x", CardinalityKind.N_ONE)
        self._pair("x", "a", CardinalityKind.ONE_N)
        _apply_1v1_equivalence(self.relationships)
        self.assertEqual(self.relationships["b::x"].cardinality, CardinalityKind.N_ONE)
        self.assertEqual(self.relationships["x::b"].cardinality, CardinalityKind.ONE_N)

    def test_backward_propagation(self) -> None:
        self._pair("a", "b", CardinalityKind.ONE_ONE)
        self._pair("b", "a", CardinalityKind.ONE_ONE)
        self._pair("b", "x", CardinalityKind.N_ONE)
        self._pair("x", "b", CardinalityKind.ONE_N)
        _apply_1v1_equivalence(self.relationships)
        self.assertEqual(self.relationships["a::x"].cardinality, CardinalityKind.N_ONE)

    def test_multi_pass_chain(self) -> None:
        # a<b<c<x lexicographically for stable one_one_pairs extraction
        self._pair("a", "b", CardinalityKind.ONE_ONE)
        self._pair("b", "a", CardinalityKind.ONE_ONE)
        self._pair("b", "c", CardinalityKind.ONE_ONE)
        self._pair("c", "b", CardinalityKind.ONE_ONE)
        self._pair("a", "x", CardinalityKind.N_ONE)
        self._pair("x", "a", CardinalityKind.ONE_N)
        _apply_1v1_equivalence(self.relationships)
        self.assertIn("c::x", self.relationships)
        self.assertEqual(self.relationships["c::x"].cardinality, CardinalityKind.N_ONE)

    def test_second_call_is_idempotent(self) -> None:
        self._pair("a", "b", CardinalityKind.ONE_ONE)
        self._pair("b", "a", CardinalityKind.ONE_ONE)
        self._pair("a", "x", CardinalityKind.N_ONE)
        self._pair("x", "a", CardinalityKind.ONE_N)
        _apply_1v1_equivalence(self.relationships)
        snap = copy.deepcopy(self.relationships)
        _apply_1v1_equivalence(self.relationships)
        self.assertEqual(self.relationships, snap)

    def test_does_not_overwrite_different_cardinality(self) -> None:
        self._pair("a", "b", CardinalityKind.ONE_ONE)
        self._pair("b", "a", CardinalityKind.ONE_ONE)
        self._pair("a", "x", CardinalityKind.M_N)
        self._pair("x", "a", CardinalityKind.M_N)
        _apply_1v1_equivalence(self.relationships)
        self.assertEqual(self.relationships["b::x"].cardinality, CardinalityKind.M_N)


class TestInferTransitiveRelationships(unittest.TestCase):
    def test_chain_n_one(self) -> None:
        rel: dict[str, CardinalityRelation] = {
            "a::b": CardinalityRelation(
                source_column="a",
                target_column="b",
                cardinality=CardinalityKind.N_ONE,
                proof="empirical",
            ),
            "b::c": CardinalityRelation(
                source_column="b",
                target_column="c",
                cardinality=CardinalityKind.N_ONE,
                proof="empirical",
            ),
        }
        _infer_transitive_relationships(["a", "b", "c"], rel)
        self.assertEqual(rel["a::c"].cardinality, CardinalityKind.N_ONE)
        self.assertEqual(rel["c::a"].cardinality, CardinalityKind.ONE_N)
        self.assertIn(_TRANSITIVE_INF_PROOF_MARK, rel["a::c"].proof)
        self.assertIn(_TRANSITIVE_INF_PROOF_MARK, rel["c::a"].proof)

    def test_one_one_bridge(self) -> None:
        rel = {
            "a::b": CardinalityRelation(
                source_column="a",
                target_column="b",
                cardinality=CardinalityKind.ONE_ONE,
                proof="e",
            ),
            "b::a": CardinalityRelation(
                source_column="b",
                target_column="a",
                cardinality=CardinalityKind.ONE_ONE,
                proof="e",
            ),
            "b::c": CardinalityRelation(
                source_column="b",
                target_column="c",
                cardinality=CardinalityKind.N_ONE,
                proof="e",
            ),
        }
        _infer_transitive_relationships(["a", "b", "c"], rel)
        self.assertEqual(rel["a::c"].cardinality, CardinalityKind.N_ONE)

    def test_does_not_overwrite_existing(self) -> None:
        rel: dict[str, CardinalityRelation] = {
            "a::b": CardinalityRelation(
                source_column="a",
                target_column="b",
                cardinality=CardinalityKind.N_ONE,
                proof="e",
            ),
            "b::c": CardinalityRelation(
                source_column="b",
                target_column="c",
                cardinality=CardinalityKind.N_ONE,
                proof="e",
            ),
            "a::c": CardinalityRelation(
                source_column="a",
                target_column="c",
                cardinality=CardinalityKind.M_N,
                proof="empirical A::C",
            ),
        }
        _infer_transitive_relationships(["a", "b", "c"], rel)
        self.assertEqual(rel["a::c"].cardinality, CardinalityKind.M_N)
        self.assertEqual(rel["a::c"].proof, "empirical A::C")


class TestPairwisePruneAllowed(unittest.TestCase):
    def setUp(self) -> None:
        self.rel: dict[str, CardinalityRelation] = {}

    def test_missing_false(self) -> None:
        self.assertFalse(_pairwise_prune_allowed(self.rel, "a", "b"))

    def test_transitive_forward_false(self) -> None:
        self.rel["a::b"] = CardinalityRelation(
            source_column="a",
            target_column="b",
            cardinality=CardinalityKind.N_ONE,
            proof=_TRANSITIVE_INF_PROOF_MARK,
        )
        self.assertFalse(_pairwise_prune_allowed(self.rel, "a", "b"))

    def test_transitive_reverse_false(self) -> None:
        self.rel["b::a"] = CardinalityRelation(
            source_column="b",
            target_column="a",
            cardinality=CardinalityKind.ONE_N,
            proof=f"{_TRANSITIVE_INF_PROOF_MARK} (reverse)",
        )
        self.assertFalse(_pairwise_prune_allowed(self.rel, "b", "a"))

    def test_empirical_true(self) -> None:
        self.rel["a::b"] = CardinalityRelation(
            source_column="a",
            target_column="b",
            cardinality=CardinalityKind.N_ONE,
            proof="dist_src=5, dist_tgt=10, combos=10",
        )
        self.assertTrue(_pairwise_prune_allowed(self.rel, "a", "b"))

    def test_pk_proof_true(self) -> None:
        self.rel["id::dim"] = CardinalityRelation(
            source_column="id",
            target_column="dim",
            cardinality=CardinalityKind.N_ONE,
            proof="Inferred: source is a Primary Key (distinct count = total rows).",
        )
        self.assertTrue(_pairwise_prune_allowed(self.rel, "id", "dim"))

    def test_one_one_equiv_proof_true(self) -> None:
        self.rel["b::x"] = CardinalityRelation(
            source_column="b",
            target_column="x",
            cardinality=CardinalityKind.N_ONE,
            proof="1:1 equivalence with a",
        )
        self.assertTrue(_pairwise_prune_allowed(self.rel, "b", "x"))


class TestSampleAndPigeonholeHelpers(unittest.TestCase):
    def test_evaluate_sample_pairs_only_double_excess(self) -> None:
        # combos == dist → no excess on either side
        self.assertIsNone(_evaluate_sample_pairs(5, 5, 5))
        # only A side exceeds (combos > dist_a but not dist_b)
        self.assertIsNone(_evaluate_sample_pairs(6, 5, 7))
        # only B side exceeds (combos > dist_b but not dist_a)
        self.assertIsNone(_evaluate_sample_pairs(6, 7, 5))
        # both sides exceed → M:N
        self.assertEqual(_evaluate_sample_pairs(6, 5, 5), CardinalityKind.M_N)

    def test_apply_1d_pigeonhole_equal_dist_either_duplicate(self) -> None:
        self.assertTrue(_apply_1d_pigeonhole(10, 10, 5, 4, 5))
        self.assertTrue(_apply_1d_pigeonhole(10, 10, 5, 5, 4))

    def test_apply_1d_pigeonhole_a_larger(self) -> None:
        # dist_a > dist_b → only N:1 or M:N possible.
        # a_has_duplicates (combos > sample_dist_a) kills N:1 → M:N.
        self.assertTrue(_apply_1d_pigeonhole(20, 10, 12, 10, 10))
        # b_has_duplicates alone (a_has_duplicates=False) does not kill N:1 when dist_a > dist_b.
        # sample_combos=10 == sample_dist_a=10 → a_has_duplicates=False; sample_dist_b=9 < 10 → b_has_duplicates=True.
        self.assertFalse(_apply_1d_pigeonhole(20, 10, 10, 10, 9))

    def test_apply_1d_pigeonhole_b_larger(self) -> None:
        # dist_b > dist_a → only 1:N or M:N possible.
        # b_has_duplicates (combos > sample_dist_b) kills 1:N → M:N.
        self.assertTrue(_apply_1d_pigeonhole(10, 20, 12, 10, 10))
        # a_has_duplicates alone does not kill 1:N when dist_b > dist_a.
        # sample_combos=12 > sample_dist_a=11 → a_has_duplicates=True; sample_dist_b=12 == combos → b_has_duplicates=False.
        self.assertFalse(_apply_1d_pigeonhole(10, 20, 12, 11, 12))


class TestCardinalityModels(unittest.TestCase):
    def test_grain_observation_default_exact(self) -> None:
        g = GrainObservation(columns=["a"], distinct_count=1, is_unique=True)
        self.assertTrue(g.distinct_count_is_exact)

    def test_grain_round_trip_json(self) -> None:
        g = GrainObservation(
            columns=["a", "b"],
            distinct_count=0,
            is_unique=False,
            distinct_count_is_exact=False,
        )
        raw = json.dumps(g.model_dump())
        g2 = GrainObservation.model_validate_json(raw)
        self.assertFalse(g2.distinct_count_is_exact)

    def test_merge_grains_exact_self_kept_over_inexact_other(self) -> None:
        self_p = CardinalityProfile(
            table_name="t",
            total_rows=10,
            column_counts={},
            tested_grains=[
                GrainObservation(
                    columns=["a", "b"],
                    distinct_count=99,
                    is_unique=False,
                    distinct_count_is_exact=True,
                ),
            ],
            relationships={},
        )
        other_p = CardinalityProfile(
            table_name="t",
            total_rows=10,
            column_counts={},
            tested_grains=[
                GrainObservation(
                    columns=["a", "b"],
                    distinct_count=0,
                    is_unique=False,
                    distinct_count_is_exact=False,
                ),
            ],
            relationships={},
        )
        merged = self_p.merge(other_p)
        self.assertTrue(merged.tested_grains[0].distinct_count_is_exact)
        self.assertEqual(merged.tested_grains[0].distinct_count, 99)

    def test_merge_grains_inexact_self_upgraded_by_exact_other(self) -> None:
        self_p = CardinalityProfile(
            table_name="t",
            total_rows=10,
            column_counts={},
            tested_grains=[
                GrainObservation(
                    columns=["a", "b"],
                    distinct_count=0,
                    is_unique=False,
                    distinct_count_is_exact=False,
                ),
            ],
            relationships={},
        )
        other_p = CardinalityProfile(
            table_name="t",
            total_rows=10,
            column_counts={},
            tested_grains=[
                GrainObservation(
                    columns=["a", "b"],
                    distinct_count=50,
                    is_unique=False,
                    distinct_count_is_exact=True,
                ),
            ],
            relationships={},
        )
        merged = self_p.merge(other_p)
        self.assertTrue(merged.tested_grains[0].distinct_count_is_exact)
        self.assertEqual(merged.tested_grains[0].distinct_count, 50)

    def test_merge_grains_both_inexact_keeps_self(self) -> None:
        self_p = CardinalityProfile(
            table_name="t",
            total_rows=10,
            column_counts={},
            tested_grains=[
                GrainObservation(
                    columns=["a", "b"],
                    distinct_count=0,
                    is_unique=False,
                    distinct_count_is_exact=False,
                ),
            ],
            relationships={},
        )
        other_p = CardinalityProfile(
            table_name="t",
            total_rows=10,
            column_counts={},
            tested_grains=[
                GrainObservation(
                    columns=["a", "b"],
                    distinct_count=0,
                    is_unique=False,
                    distinct_count_is_exact=False,
                ),
            ],
            relationships={},
        )
        merged = self_p.merge(other_p)
        self.assertFalse(merged.tested_grains[0].distinct_count_is_exact)

    def test_merge_new_column_set_from_other_added(self) -> None:
        self_p = CardinalityProfile(
            table_name="t",
            total_rows=5,
            column_counts={},
            tested_grains=[
                GrainObservation(columns=["a"], distinct_count=3, is_unique=False),
            ],
            relationships={},
        )
        other_p = CardinalityProfile(
            table_name="t",
            total_rows=7,
            column_counts={},
            tested_grains=[
                GrainObservation(columns=["b"], distinct_count=4, is_unique=False),
            ],
            relationships={},
        )
        merged = self_p.merge(other_p)
        keys = {frozenset(g.columns) for g in merged.tested_grains}
        self.assertEqual(keys, {frozenset({"a"}), frozenset({"b"})})

    def test_merge_total_rows_max(self) -> None:
        a = CardinalityProfile(
            table_name="t", total_rows=3, column_counts={}, relationships={}
        )
        b = CardinalityProfile(
            table_name="t", total_rows=7, column_counts={}, relationships={}
        )
        self.assertEqual(a.merge(b).total_rows, 7)

    def test_merge_relationships_other_wins(self) -> None:
        a = CardinalityProfile(
            table_name="t",
            total_rows=1,
            column_counts={},
            relationships={
                "x::y": CardinalityRelation(
                    source_column="x",
                    target_column="y",
                    cardinality=CardinalityKind.M_N,
                    proof="first",
                ),
            },
        )
        b = CardinalityProfile(
            table_name="t",
            total_rows=1,
            column_counts={},
            relationships={
                "x::y": CardinalityRelation(
                    source_column="x",
                    target_column="y",
                    cardinality=CardinalityKind.N_ONE,
                    proof="second",
                ),
            },
        )
        self.assertEqual(a.merge(b).relationships["x::y"].proof, "second")

    def test_merge_hierarchies_cleared(self) -> None:
        a = CardinalityProfile(
            table_name="t",
            total_rows=1,
            column_counts={},
            hierarchies=[HierarchyChain(columns=["p", "c"])],
            relationships={},
        )
        b = CardinalityProfile(
            table_name="t", total_rows=1, column_counts={}, relationships={}
        )
        self.assertEqual(a.merge(b).hierarchies, [])


class TestExistingProfileCoversColumns(unittest.TestCase):
    def _profile_with_edges(self, cols: list[str]) -> CardinalityProfile:
        rel = {}
        for x, y in combinations(cols, 2):
            rel[f"{x}::{y}"] = CardinalityRelation(
                source_column=x,
                target_column=y,
                cardinality=CardinalityKind.N_ONE,
                proof="e",
            )
        return CardinalityProfile(
            table_name="t",
            total_rows=1,
            column_counts={},
            relationships=rel,
            tested_grains=[
                GrainObservation(columns=cols, distinct_count=1, is_unique=True)
            ],
        )

    def test_single_column_false(self) -> None:
        p = self._profile_with_edges(["a", "b"])
        self.assertFalse(existing_profile_covers_columns(p, ["a"]))

    def test_all_pairs_true(self) -> None:
        p = self._profile_with_edges(["a", "b", "c"])
        self.assertTrue(existing_profile_covers_columns(p, ["a", "b", "c"]))

    def test_one_edge_missing(self) -> None:
        p = CardinalityProfile(
            table_name="t",
            total_rows=1,
            column_counts={},
            relationships={
                "a::b": CardinalityRelation(
                    source_column="a",
                    target_column="b",
                    cardinality=CardinalityKind.N_ONE,
                    proof="e",
                ),
            },
            tested_grains=[
                GrainObservation(
                    columns=["a", "b", "c"], distinct_count=1, is_unique=True
                )
            ],
        )
        self.assertFalse(existing_profile_covers_columns(p, ["a", "b", "c"]))

    def test_column_not_in_used_columns(self) -> None:
        p = self._profile_with_edges(["a", "b"])
        self.assertFalse(existing_profile_covers_columns(p, ["a", "b", "c"]))


class TestToMarkdownGrainsAndHierarchies(unittest.TestCase):
    def test_grain_exact_line(self) -> None:
        p = CardinalityProfile(
            table_name="t",
            total_rows=10,
            column_counts={},
            tested_grains=[
                GrainObservation(
                    columns=["a"],
                    distinct_count=7,
                    is_unique=False,
                    distinct_count_is_exact=True,
                ),
            ],
            relationships={},
        )
        md = p.to_markdown()
        self.assertIn("7 distinct", md)

    def test_grain_inexact_line(self) -> None:
        p = CardinalityProfile(
            table_name="t",
            total_rows=10,
            column_counts={},
            tested_grains=[
                GrainObservation(
                    columns=["a", "b"],
                    distinct_count=0,
                    is_unique=False,
                    distinct_count_is_exact=False,
                ),
            ],
            relationships={},
        )
        md = p.to_markdown()
        self.assertIn(
            "distinct count not computed (non-unique by per-column bound)", md
        )

    def test_hierarchies_shown_despite_omit_safe(self) -> None:
        n_cols = TO_MARKDOWN_RISKY_ONLY_COLUMNS_THRESHOLD + 5
        cols = [f"c{i}" for i in range(n_cols)]
        rel = {}
        for i in range(TO_MARKDOWN_RISKY_ONLY_RELATIONS_THRESHOLD + 5):
            a, b = f"r{i}_a", f"r{i}_b"
            rel[f"{a}::{b}"] = CardinalityRelation(
                source_column=a,
                target_column=b,
                cardinality=CardinalityKind.N_ONE,
                proof="p",
            )
        p = CardinalityProfile(
            table_name="t",
            total_rows=1,
            column_counts={},
            relationships=rel,
            hierarchies=[HierarchyChain(columns=[cols[0], cols[1]])],
            tested_grains=[
                GrainObservation(columns=cols[:2], distinct_count=1, is_unique=True)
            ],
        )
        md = p.to_markdown()
        self.assertIn("**Hierarchies (1:N chains, coarsest → finest):**", md)
        self.assertIn(cols[0], md)


# ---------------------------------------------------------------------------
# core — result parsing
# ---------------------------------------------------------------------------


class TestExtractValue(unittest.TestCase):
    def test_dict_with_value_key(self) -> None:
        self.assertEqual(_extract_value({"value": 42}), 42)

    def test_raw_int(self) -> None:
        self.assertEqual(_extract_value(7), 7)

    def test_raw_string_int(self) -> None:
        self.assertEqual(_extract_value("15"), 15)

    def test_nested_dict_without_value_key_returns_zero(self) -> None:
        self.assertEqual(_extract_value({"other": 5}), 0)

    def test_non_numeric_string_returns_zero(self) -> None:
        self.assertEqual(_extract_value("abc"), 0)

    def test_none_returns_zero(self) -> None:
        self.assertEqual(_extract_value(None), 0)


class TestParseRow(unittest.TestCase):
    def test_dict_row(self) -> None:
        row = {"col_0": 5, "col_1": 10}
        result = _parse_row(row, ["a", "b"])
        self.assertEqual(result, {"a": 5, "b": 10})

    def test_list_row(self) -> None:
        result = _parse_row([5, 10], ["a", "b"])
        self.assertEqual(result, {"a": 5, "b": 10})

    def test_short_list_pads_missing_with_zero(self) -> None:
        result = _parse_row([5], ["a", "b"])
        self.assertEqual(result["a"], 5)
        self.assertEqual(result["b"], 0)


# ---------------------------------------------------------------------------
# Profile assembly and column subsetting
# ---------------------------------------------------------------------------


class TestAssembleProfile(unittest.TestCase):
    def test_composite_grain_unique_none_derives_from_counts(self) -> None:
        p = _assemble_profile("t", ["a", "b"], 10, {}, 10, {}, [], [])
        self.assertTrue(p.tested_grains[0].is_unique)

    def test_composite_grain_unique_none_not_unique_when_mismatch(self) -> None:
        p = _assemble_profile("t", ["a", "b"], 10, {}, 7, {}, [], [])
        self.assertFalse(p.tested_grains[0].is_unique)

    def test_composite_grain_unique_explicit_true_overrides(self) -> None:
        # distinct_grain=0 but composite_grain_unique=True → trust the explicit flag
        p = _assemble_profile(
            "t", ["a"], 10, {}, 0, {}, [], [], composite_grain_unique=True
        )
        self.assertTrue(p.tested_grains[0].is_unique)

    def test_grain_distinct_exact_false_forwarded(self) -> None:
        p = _assemble_profile(
            "t", ["a"], 10, {}, 0, {}, [], [], grain_distinct_exact=False
        )
        self.assertFalse(p.tested_grains[0].distinct_count_is_exact)

    def test_zero_total_rows_not_unique(self) -> None:
        p = _assemble_profile("t", ["a"], 0, {}, 0, {}, [], [])
        self.assertFalse(p.tested_grains[0].is_unique)


class TestProfileSubsetForColumns(unittest.TestCase):
    def _profile(self) -> CardinalityProfile:
        return CardinalityProfile(
            table_name="t",
            total_rows=100,
            column_counts={"a": 10, "b": 5, "c": 3},
            primary_keys=["a"],
            constants=["c"],
            tested_grains=[
                GrainObservation(columns=["a", "b"], distinct_count=10, is_unique=True),
                GrainObservation(columns=["b", "c"], distinct_count=5, is_unique=False),
            ],
            relationships={
                "a::b": CardinalityRelation(
                    source_column="a",
                    target_column="b",
                    cardinality=CardinalityKind.N_ONE,
                    proof="e",
                ),
                "b::a": CardinalityRelation(
                    source_column="b",
                    target_column="a",
                    cardinality=CardinalityKind.ONE_N,
                    proof="e",
                ),
                "b::c": CardinalityRelation(
                    source_column="b",
                    target_column="c",
                    cardinality=CardinalityKind.N_ONE,
                    proof="e",
                ),
                "c::b": CardinalityRelation(
                    source_column="c",
                    target_column="b",
                    cardinality=CardinalityKind.ONE_N,
                    proof="e",
                ),
            },
            hierarchies=[HierarchyChain(columns=["a", "b", "c"])],
        )

    def test_column_counts_filtered(self) -> None:
        p = profile_subset_for_columns(self._profile(), ["a", "b"])
        self.assertIn("a", p.column_counts)
        self.assertNotIn("c", p.column_counts)

    def test_primary_keys_filtered(self) -> None:
        p = profile_subset_for_columns(self._profile(), ["b", "c"])
        self.assertEqual(p.primary_keys, [])

    def test_relationships_filtered_to_requested_cols(self) -> None:
        p = profile_subset_for_columns(self._profile(), ["a", "b"])
        self.assertIn("a::b", p.relationships)
        self.assertNotIn("b::c", p.relationships)

    def test_grains_subset_filtered(self) -> None:
        p = profile_subset_for_columns(self._profile(), ["a", "b"])
        grain_cols = [frozenset(g.columns) for g in p.tested_grains]
        self.assertIn(frozenset({"a", "b"}), grain_cols)
        self.assertNotIn(frozenset({"b", "c"}), grain_cols)

    def test_hierarchies_filtered_and_deduped(self) -> None:
        # only a,b survive after filtering c out; hierarchy gets trimmed to [a,b]
        p = profile_subset_for_columns(self._profile(), ["a", "b"])
        self.assertEqual(len(p.hierarchies), 1)
        self.assertEqual(p.hierarchies[0].columns, ["a", "b"])


class TestTryExistingCardinalityProfile(unittest.TestCase):
    def _covering_profile(self, cols: list[str]) -> CardinalityProfile:
        rel = {
            f"{a}::{b}": CardinalityRelation(
                source_column=a,
                target_column=b,
                cardinality=CardinalityKind.N_ONE,
                proof="e",
            )
            for a, b in __import__("itertools").combinations(cols, 2)
        }
        return CardinalityProfile(
            table_name="t",
            total_rows=1,
            column_counts={},
            relationships=rel,
            tested_grains=[
                GrainObservation(columns=cols, distinct_count=1, is_unique=True)
            ],
        )

    def test_none_profiles_returns_none(self) -> None:
        self.assertIsNone(try_existing_cardinality_profile("t", ["a", "b"], None))

    def test_table_not_in_profiles_returns_none(self) -> None:
        self.assertIsNone(
            try_existing_cardinality_profile(
                "t", ["a", "b"], {"other": self._covering_profile(["a", "b"])}
            )
        )

    def test_profile_does_not_cover_returns_none(self) -> None:
        profiles = {"t": self._covering_profile(["a", "b"])}
        self.assertIsNone(
            try_existing_cardinality_profile("t", ["a", "b", "c"], profiles)
        )

    def test_profile_covers_returns_subset(self) -> None:
        profiles = {"t": self._covering_profile(["a", "b", "c"])}
        result = try_existing_cardinality_profile("t", ["a", "b"], profiles)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("a::b", result.relationships)
        self.assertNotIn("a::c", result.relationships)


# ---------------------------------------------------------------------------
# Hierarchy detection
# ---------------------------------------------------------------------------


def _rel(src: str, tgt: str, kind: CardinalityKind) -> CardinalityRelation:
    return CardinalityRelation(
        source_column=src, target_column=tgt, cardinality=kind, proof="e"
    )


def _profile_with_rels(rels: dict[str, CardinalityRelation]) -> CardinalityProfile:
    tested_cols = sorted({c for k in rels for c in k.split("::")})
    return CardinalityProfile(
        table_name="t",
        total_rows=10,
        column_counts={},
        relationships=rels,
        tested_grains=[
            GrainObservation(columns=tested_cols, distinct_count=1, is_unique=True)
        ],
    )


class TestExtract1nAndEquivalences(unittest.TestCase):
    def test_1n_edge_parent_child_order(self) -> None:
        # city::country N:1, country::city 1:N → country is parent (coarser)
        rels = {
            "city::country": _rel("city", "country", CardinalityKind.N_ONE),
            "country::city": _rel("country", "city", CardinalityKind.ONE_N),
        }
        true_1n, equivs = _extract_1n_and_equivalences(_profile_with_rels(rels))
        self.assertEqual(len(equivs), 0)
        self.assertEqual(len(true_1n), 1)
        parent, child = true_1n[0]
        self.assertEqual(parent, "country")
        self.assertEqual(child, "city")

    def test_1n_forward_direction(self) -> None:
        # a::b 1:N, b::a N:1 → a is parent
        rels = {
            "a::b": _rel("a", "b", CardinalityKind.ONE_N),
            "b::a": _rel("b", "a", CardinalityKind.N_ONE),
        }
        true_1n, _ = _extract_1n_and_equivalences(_profile_with_rels(rels))
        self.assertEqual(true_1n[0], ("a", "b"))

    def test_equivalence_pair(self) -> None:
        rels = {
            "a::b": _rel("a", "b", CardinalityKind.ONE_ONE),
            "b::a": _rel("b", "a", CardinalityKind.ONE_ONE),
        }
        true_1n, equivs = _extract_1n_and_equivalences(_profile_with_rels(rels))
        self.assertEqual(len(true_1n), 0)
        self.assertEqual(len(equivs), 1)

    def test_missing_reverse_edge_skipped(self) -> None:
        # only one direction present → skip
        rels = {"a::b": _rel("a", "b", CardinalityKind.ONE_N)}
        true_1n, equivs = _extract_1n_and_equivalences(_profile_with_rels(rels))
        self.assertEqual(len(true_1n), 0)


class TestTransitiveReduce1n(unittest.TestCase):
    def test_bypass_edge_removed(self) -> None:
        # A→B, B→C, A→C (bypass). After reduction: A→C removed.
        cols = ["A", "B", "C"]
        reduced, in_deg = _transitive_reduce_1n(
            cols, [("A", "B"), ("B", "C"), ("A", "C")]
        )
        self.assertNotIn("C", reduced["A"])
        self.assertIn("B", reduced["A"])
        self.assertIn("C", reduced["B"])

    def test_no_bypass_all_edges_kept(self) -> None:
        # A→B, A→C but B↛C and C↛B: neither is a bypass
        cols = ["A", "B", "C"]
        reduced, _ = _transitive_reduce_1n(cols, [("A", "B"), ("A", "C")])
        self.assertIn("B", reduced["A"])
        self.assertIn("C", reduced["A"])


class TestHierarchyChains(unittest.TestCase):
    def test_linear_chain(self) -> None:
        cols = ["A", "B", "C"]
        reduced = {"A": ["B"], "B": ["C"], "C": []}
        in_deg = {"A": 0, "B": 1, "C": 1}
        chains = _hierarchy_chains_from_reduced(cols, reduced, in_deg)
        self.assertEqual(chains, [["A", "B", "C"]])

    def test_two_branches_from_root(self) -> None:
        cols = ["A", "B", "C"]
        reduced = {"A": ["B", "C"], "B": [], "C": []}
        in_deg = {"A": 0, "B": 1, "C": 1}
        chains = _hierarchy_chains_from_reduced(cols, reduced, in_deg)
        self.assertEqual(len(chains), 2)
        for ch in chains:
            self.assertEqual(ch[0], "A")
            self.assertGreaterEqual(len(ch), 2)

    def test_isolated_node_not_included(self) -> None:
        # X has no children and no parent → no chain containing only X
        cols = ["X"]
        reduced = {"X": []}
        in_deg = {"X": 0}
        chains = _hierarchy_chains_from_reduced(cols, reduced, in_deg)
        self.assertEqual(chains, [])


class TestDetectHierarchies(unittest.TestCase):
    def test_empty_profile_returns_empty(self) -> None:
        p = CardinalityProfile(
            table_name="t", total_rows=0, column_counts={}, relationships={}
        )
        self.assertEqual(detect_hierarchies(p), [])

    def test_linear_chain_detected(self) -> None:
        # country 1:N city (country is coarser)
        rels = {
            "city::country": _rel("city", "country", CardinalityKind.N_ONE),
            "country::city": _rel("country", "city", CardinalityKind.ONE_N),
            "city::district": _rel("city", "district", CardinalityKind.ONE_N),
            "district::city": _rel("district", "city", CardinalityKind.N_ONE),
        }
        p = _profile_with_rels(rels)
        chains = detect_hierarchies(p)
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].columns, ["country", "city", "district"])

    def test_only_1to1_returns_empty(self) -> None:
        rels = {
            "a::b": _rel("a", "b", CardinalityKind.ONE_ONE),
            "b::a": _rel("b", "a", CardinalityKind.ONE_ONE),
        }
        p = _profile_with_rels(rels)
        self.assertEqual(detect_hierarchies(p), [])

    def test_transitive_bypass_reduced(self) -> None:
        # A→B, B→C, A→C (transitive bypass): final chain should be A→B→C
        rels = {
            "a::b": _rel("a", "b", CardinalityKind.ONE_N),
            "b::a": _rel("b", "a", CardinalityKind.N_ONE),
            "b::c": _rel("b", "c", CardinalityKind.ONE_N),
            "c::b": _rel("c", "b", CardinalityKind.N_ONE),
            "a::c": _rel("a", "c", CardinalityKind.ONE_N),
            "c::a": _rel("c", "a", CardinalityKind.N_ONE),
        }
        p = _profile_with_rels(rels)
        chains = detect_hierarchies(p)
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].columns, ["a", "b", "c"])


# ---------------------------------------------------------------------------
# CardinalityProfile model properties and merge_cardinality_profiles
# ---------------------------------------------------------------------------


class TestCardinalityProfileProperties(unittest.TestCase):
    def _profile(self) -> CardinalityProfile:
        return CardinalityProfile(
            table_name="t",
            total_rows=10,
            column_counts={"a": 5},
            relationships={
                "a::b": CardinalityRelation(
                    source_column="a",
                    target_column="b",
                    cardinality=CardinalityKind.N_ONE,
                    proof="e",
                ),
            },
            tested_grains=[
                GrainObservation(columns=["a", "b"], distinct_count=7, is_unique=False),
            ],
        )

    def test_used_columns_from_grains_and_relationships(self) -> None:
        p = self._profile()
        cols = p.used_columns
        self.assertIn("a", cols)
        self.assertIn("b", cols)

    def test_used_columns_deduplicated_sorted(self) -> None:
        p = self._profile()
        cols = p.used_columns
        self.assertEqual(cols, sorted(set(cols)))

    def test_distinct_grain_returns_first_grain_count(self) -> None:
        self.assertEqual(self._profile().distinct_grain, 7)

    def test_distinct_grain_empty_returns_zero(self) -> None:
        p = CardinalityProfile(
            table_name="t", total_rows=0, column_counts={}, relationships={}
        )
        self.assertEqual(p.distinct_grain, 0)

    def test_is_unique_grain_true(self) -> None:
        p = CardinalityProfile(
            table_name="t",
            total_rows=5,
            column_counts={},
            relationships={},
            tested_grains=[
                GrainObservation(columns=["a"], distinct_count=5, is_unique=True)
            ],
        )
        self.assertTrue(p.is_unique_grain)

    def test_is_unique_grain_empty_returns_false(self) -> None:
        p = CardinalityProfile(
            table_name="t", total_rows=0, column_counts={}, relationships={}
        )
        self.assertFalse(p.is_unique_grain)


class TestMergeCardinalityProfiles(unittest.TestCase):
    def _p(self, name: str, rows: int, rels: dict | None = None) -> CardinalityProfile:
        return CardinalityProfile(
            table_name=name, total_rows=rows, column_counts={}, relationships=rels or {}
        )

    def test_none_left_returns_right(self) -> None:
        right = {"t": self._p("t", 5)}
        result = merge_cardinality_profiles(None, right)
        self.assertEqual(result["t"].total_rows, 5)

    def test_none_right_returns_left(self) -> None:
        left = {"t": self._p("t", 3)}
        result = merge_cardinality_profiles(left, None)
        self.assertEqual(result["t"].total_rows, 3)

    def test_both_none_returns_empty(self) -> None:
        self.assertEqual(merge_cardinality_profiles(None, None), {})

    def test_new_table_in_right_added(self) -> None:
        left = {"t1": self._p("t1", 1)}
        right = {"t2": self._p("t2", 2)}
        result = merge_cardinality_profiles(left, right)
        self.assertIn("t1", result)
        self.assertIn("t2", result)

    def test_same_table_rows_max(self) -> None:
        left = {"t": self._p("t", 3)}
        right = {"t": self._p("t", 9)}
        result = merge_cardinality_profiles(left, right)
        self.assertEqual(result["t"].total_rows, 9)

    def test_dict_input_validates_via_pydantic(self) -> None:
        # Validation fires in the merge loop, not on the None-left fast path.
        # Provide a different table in left so the loop runs and validates right's dict entry.
        raw_profile = {
            "table_name": "t",
            "total_rows": 5,
            "column_counts": {},
            "relationships": {},
        }
        left = {"other": self._p("other", 1)}
        result = merge_cardinality_profiles(left, {"t": raw_profile})  # type: ignore[arg-type]
        self.assertIsInstance(result["t"], CardinalityProfile)
        self.assertEqual(result["t"].total_rows, 5)


# ---------------------------------------------------------------------------
# to_markdown relation sections
# ---------------------------------------------------------------------------


class TestToMarkdownRelationSections(unittest.TestCase):
    def _profile_with_single_rel(self, kind: CardinalityKind) -> CardinalityProfile:
        rels: dict[str, CardinalityRelation] = {}
        rels["a::b"] = CardinalityRelation(
            source_column="a", target_column="b", cardinality=kind, proof="e"
        )
        if kind == CardinalityKind.ONE_ONE:
            rels["b::a"] = CardinalityRelation(
                source_column="b", target_column="a", cardinality=kind, proof="e"
            )
        elif kind == CardinalityKind.M_N:
            rels["b::a"] = CardinalityRelation(
                source_column="b", target_column="a", cardinality=kind, proof="e"
            )
        return CardinalityProfile(
            table_name="t",
            total_rows=10,
            column_counts={},
            relationships=rels,
            tested_grains=[
                GrainObservation(columns=["a", "b"], distinct_count=5, is_unique=False)
            ],
        )

    def test_n_one_section_shown(self) -> None:
        md = self._profile_with_single_rel(CardinalityKind.N_ONE).to_markdown()
        self.assertIn("N:1", md)
        self.assertIn("`a`", md)

    def test_1n_section_shown(self) -> None:
        md = self._profile_with_single_rel(CardinalityKind.ONE_N).to_markdown()
        self.assertIn("1:N", md)

    def test_one_one_equivalences_section(self) -> None:
        md = self._profile_with_single_rel(CardinalityKind.ONE_ONE).to_markdown()
        self.assertIn("1:1", md)
        self.assertIn("↔", md)

    def test_mn_section_shown(self) -> None:
        md = self._profile_with_single_rel(CardinalityKind.M_N).to_markdown()
        self.assertIn("M:N", md)

    def test_risky_only_mode_suppresses_safe_sections(self) -> None:
        # Build a profile that triggers risky-only mode (too many relations)
        n_rels = TO_MARKDOWN_RISKY_ONLY_RELATIONS_THRESHOLD + 5
        rels: dict[str, CardinalityRelation] = {}
        for i in range(n_rels):
            rels[f"r{i}::s{i}"] = CardinalityRelation(
                source_column=f"r{i}",
                target_column=f"s{i}",
                cardinality=CardinalityKind.N_ONE,
                proof="e",
            )
        # Add a 1:N pair so we can check it IS shown
        rels["x::y"] = CardinalityRelation(
            source_column="x",
            target_column="y",
            cardinality=CardinalityKind.ONE_N,
            proof="e",
        )
        all_cols = sorted({c for k in rels for c in k.split("::")})
        p = CardinalityProfile(
            table_name="t",
            total_rows=1,
            column_counts={},
            relationships=rels,
            tested_grains=[
                GrainObservation(columns=all_cols[:2], distinct_count=1, is_unique=True)
            ],
        )
        md = p.to_markdown()
        self.assertIn("1:N", md)
        self.assertNotIn("N:1 (source determines", md)


if __name__ == "__main__":
    unittest.main()
