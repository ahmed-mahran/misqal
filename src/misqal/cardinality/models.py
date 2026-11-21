"""
Structured models for cardinality matrix findings.
Used as first-class agent state, keyed by table name.

Supports accumulative profiling: multiple tool runs for the same table merge
(tested_grains list, relationships dict) instead of overwriting.

Supports the four relational-algebra states per (A, B) pair:
- 1:1 (equivalence): distinct_A == combos and distinct_B == combos
- N:1 (functional dependency): A determines B; safe to join from B to A
- 1:N (fan-out): A fans out to B; requires LOD handling when projecting B with A
- M:N (many-to-many): both sides fan out; high Cartesian risk
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CardinalityKind(StrEnum):
    """Cardinality between two columns: 1:1, N:1, 1:N, or M:N."""

    ONE_ONE = "1:1"
    N_ONE = "N:1"
    ONE_N = "1:N"
    M_N = "M:N"


class CardinalityRelation(BaseModel):
    """Programmatic representation of a column-to-column relationship."""

    source_column: str
    target_column: str
    cardinality: CardinalityKind
    proof: str = Field(
        description="Mathematical proof comparing distinct counts to combinations (e.g. dist_src=15, dist_tgt=5411, combos=5411)."
    )


class GrainObservation(BaseModel):
    """A record of a specific combination of columns tested for uniqueness."""

    columns: list[str]
    distinct_count: int
    is_unique: bool
    distinct_count_is_exact: bool = Field(
        default=True,
        description="False when distinct_count is unknown (e.g. pigeonhole proved non-uniqueness without COUNT(DISTINCT tuple)).",
    )


class HierarchyChain(BaseModel):
    """
    A detected hierarchy: ordered list of columns from coarsest (root) to finest (leaf).
    E.g. ["Family", "SubFamily", "Product"] means Family 1:N SubFamily 1:N Product.
    """

    columns: list[str] = Field(
        description="Columns ordered root (coarsest) to leaf (finest); length >= 2.",
        min_length=2,
    )


# When relations or columns exceed this, to_markdown omits safe relations (1:1, N:1) and keeps risky ones (1:N, M:N) only
TO_MARKDOWN_RISKY_ONLY_RELATIONS_THRESHOLD = 40
TO_MARKDOWN_RISKY_ONLY_COLUMNS_THRESHOLD = 15


class CardinalityProfile(BaseModel):
    """Accumulating cardinality findings for a specific base table. Merge new runs via .merge()."""

    table_name: str
    total_rows: int
    column_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Distinct count per column (key = column name).",
    )
    primary_keys: list[str] = Field(
        default_factory=list,
        description="Columns with COUNT(DISTINCT col) == total_rows (determine everything).",
    )
    constants: list[str] = Field(
        default_factory=list,
        description="Columns with COUNT(DISTINCT col) <= 1 (determined by everything).",
    )
    tested_grains: list[GrainObservation] = Field(
        default_factory=list,
        description="All tested column combinations (one per matrix run).",
    )
    relationships: dict[str, CardinalityRelation] = Field(
        default_factory=dict,
        description='Keyed by "source_column::target_column" for deduplication and merging.',
    )
    hierarchies: list[HierarchyChain] = Field(
        default_factory=list,
        description="Detected column hierarchies (coarse→fine) from 1:N relationships; empty if not computed.",
    )

    def merge(self, other: CardinalityProfile) -> CardinalityProfile:
        """Deep merges another profile into this one (same table)."""
        if self.table_name != other.table_name:
            return self
        merged_rows = max(self.total_rows, other.total_rows)
        merged_counts = {**self.column_counts, **other.column_counts}
        merged_pks = list(dict.fromkeys(self.primary_keys + other.primary_keys))
        merged_constants = list(dict.fromkeys(self.constants + other.constants))
        grains_map: dict[frozenset[str], GrainObservation] = {
            frozenset(g.columns): g for g in self.tested_grains
        }
        for g in other.tested_grains:
            key = frozenset(g.columns)
            existing = grains_map.get(key)
            if existing is None or g.distinct_count_is_exact:
                grains_map[key] = g
        merged_rels = {**self.relationships, **other.relationships}
        # Hierarchies are derived; clear so caller can recompute via detect_hierarchies if needed
        return CardinalityProfile(
            table_name=self.table_name,
            total_rows=merged_rows,
            column_counts=merged_counts,
            primary_keys=merged_pks,
            constants=merged_constants,
            tested_grains=list(grains_map.values()),
            relationships=merged_rels,
            hierarchies=[],
        )

    def to_markdown(self, relations_only: bool = False) -> str:
        """Human-readable summary organized by the four relational states. Above threshold, only risky (1:N, M:N) relations are listed.

        ``relations_only`` keeps just the table header and the pairwise relation lists (no keys,
        constants, hierarchies, reading guide or tested grains)."""
        n_rels = len(self.relationships)
        n_cols = len(self.used_columns)
        omit_safe = (
            n_rels > TO_MARKDOWN_RISKY_ONLY_RELATIONS_THRESHOLD
            or n_cols > TO_MARKDOWN_RISKY_ONLY_COLUMNS_THRESHOLD
        )

        parts = [f"**Table:** `{self.table_name}` ({self.total_rows} total rows)"]
        if not relations_only:
            parts.extend(self._derived_markdown())
        parts.extend(self._relations_markdown(omit_safe))
        return "\n".join(parts)

    def _derived_markdown(self) -> list[str]:
        parts: list[str] = []
        if self.primary_keys:
            parts.append(
                "\n**Primary keys (inferred):** "
                + ", ".join(f"`{c}`" for c in self.primary_keys)
            )
        if self.constants:
            parts.append(
                "\n**Constants (inferred):** "
                + ", ".join(f"`{c}`" for c in self.constants)
            )
        if self.hierarchies:
            parts.append("\n**Hierarchies (1:N chains, coarsest → finest):**")
            for h in self.hierarchies:
                parts.append("  - " + " → ".join(f"`{c}`" for c in h.columns))
        parts.append("\n**Tested Grains:**")
        for g in self.tested_grains:
            status = "✅ Unique" if g.is_unique else "⚠️ Not Unique"
            if g.distinct_count_is_exact:
                dist_part = f"{g.distinct_count} distinct"
            else:
                dist_part = (
                    "distinct count not computed (non-unique by per-column bound)"
                )
            parts.append(f"  - {status}: `[{', '.join(g.columns)}]` ({dist_part})")
        return parts

    def _relations_markdown(self, omit_safe: bool) -> list[str]:
        parts: list[str] = []
        equivalences: list[str] = []
        dependencies: dict[str, list[str]] = {}
        fan_outs: dict[str, list[str]] = {}
        mn_pairs: set[tuple[str, str]] = set()  # unordered (a, b) with a < b

        for r in self.relationships.values():
            src, tgt = r.source_column, r.target_column
            if r.cardinality == "1:1":
                rev = f"`{tgt}` ↔ `{src}`"
                if rev not in equivalences:
                    equivalences.append(f"`{src}` ↔ `{tgt}`")
            elif r.cardinality == "N:1":
                dependencies.setdefault(src, []).append(tgt)
            elif r.cardinality == "1:N":
                fan_outs.setdefault(src, []).append(tgt)
            elif r.cardinality == "M:N":
                mn_pairs.add((min(src, tgt), max(src, tgt)))

        if not omit_safe:
            if equivalences:
                parts.append(
                    "\n**True 1:1 (Equivalences):**\n"
                    + "\n".join(f"  - {e}" for e in equivalences)
                )
            if dependencies:
                parts.append("\n**N:1 (source determines exactly one target):**")
                for src, targets in dependencies.items():
                    parts.append(
                        "  - `" + src + "`: " + ", ".join(f"`{t}`" for t in targets)
                    )
        if fan_outs:
            parts.append("\n**1:N (source fans out to multiple targets):**")
            for src, targets in fan_outs.items():
                parts.append(
                    "  - `" + src + "`: " + ", ".join(f"`{t}`" for t in targets)
                )
        if mn_pairs:
            by_col: dict[str, list[str]] = {}
            for a, b in mn_pairs:
                by_col.setdefault(a, []).append(b)
            parts.append("\n**M:N (many-to-many — high Cartesian risk):**")
            lines = [
                f"  - `{col}` ⤄ " + ", ".join(f"`{p}`" for p in sorted(partners))
                for col in sorted(by_col)
                for partners in [by_col[col]]
            ]
            parts.append("\n".join(lines))
        return parts

    def determines(self, source: str, target: str) -> bool | None:
        """True if each ``source`` value has exactly one ``target`` value; None if untested."""
        if source in self.primary_keys or target in self.constants:
            return True
        fwd = self.relationships.get(f"{source}::{target}")
        if fwd is not None:
            return fwd.cardinality in ("1:1", "N:1")
        rev = self.relationships.get(f"{target}::{source}")
        if rev is not None:
            return rev.cardinality in ("1:1", "1:N")
        return None

    def to_sentences(self) -> str:
        """Plain-language relations: per column, which columns it determines (exactly one value)
        and which can take many values for one of its values. Key columns are stated once."""
        cols = self.used_columns or sorted(self.column_counts)
        keys = [c for c in self.primary_keys if c in cols]
        head = f"`{self.table_name}` ({self.total_rows} rows). "
        head += ("Each row has its own " + " / ".join(f"`{k}`" for k in keys) + "." if keys
                 else "No single column identifies a row.")
        lines = [head]
        if self.constants:
            lines.append("- Same value on every row: "
                         + ", ".join(f"`{c}`" for c in self.constants) + ".")
        for a in cols:
            if a in keys or a in self.constants:
                continue
            one, many = [], []
            for b in cols:
                if b == a or b in keys or b in self.constants:
                    continue
                d = self.determines(a, b)
                if d is not None:
                    (one if d else many).append(f"`{b}`")
            parts = []
            if one:
                parts.append("has exactly one " + ", ".join(one))
            if many:
                parts.append("can have many " + ", ".join(many))
            if parts:
                lines.append(f"- Each `{a}` " + "; and ".join(parts) + ".")
        return "\n".join(lines)

    # Convenience for code that still expects “current grain” and “all columns” from a single run
    @property
    def used_columns(self) -> list[str]:
        """All columns that appear in any tested grain or relationship (deduplicated, stable order)."""
        cols: set[str] = set()
        for g in self.tested_grains:
            cols.update(g.columns)
        for key in self.relationships:
            if "::" in key:
                a, b = key.split("::", 1)
                cols.add(a)
                cols.add(b)
        return sorted(cols)

    @property
    def distinct_grain(self) -> int:
        """Distinct count of the first tested grain, or 0 if none (for backward compatibility)."""
        if not self.tested_grains:
            return 0
        return self.tested_grains[0].distinct_count

    @property
    def is_unique_grain(self) -> bool:
        """True if the first tested grain is unique (for backward compatibility)."""
        if not self.tested_grains:
            return False
        return self.tested_grains[0].is_unique


def merge_cardinality_profiles(
    left: dict[str, CardinalityProfile] | None,
    right: dict[str, CardinalityProfile] | None,
) -> dict[str, CardinalityProfile]:
    """Custom LangGraph reducer: deep-merge profiles per table instead of overwriting."""
    if not left:
        return dict(right) if right else {}
    if not right:
        return dict(left)
    merged = dict(left)
    for table_name, new_profile in right.items():
        profile = (
            CardinalityProfile.model_validate(new_profile)
            if isinstance(new_profile, dict)
            else new_profile
        )
        if table_name in merged:
            existing = merged[table_name]
            existing_profile = (
                CardinalityProfile.model_validate(existing)
                if isinstance(existing, dict)
                else existing
            )
            merged[table_name] = existing_profile.merge(profile)
        else:
            merged[table_name] = profile
    return merged
