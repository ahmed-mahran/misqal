"""Fan-out audit of SQL queries against their database.

For every SELECT block of a query:

- **Join edges** are the equality predicates between columns of two base tables (in ``ON`` or
  ``WHERE``). Each side is labelled unique or not by counting, on the data, whether its join columns
  are a key of its table (NULL counted as a value, as everywhere in ``misqal``).
- A table's rows are **statically multiplied** when a join, followed from that table, reaches a side
  whose join columns are not unique: each row can then meet several rows there.
- An aggregate is **exposed** when it is duplicate-sensitive (``SUM``, ``AVG``, ``COUNT``, ``TOTAL``,
  ``GROUP_CONCAT`` without ``DISTINCT``) and reads a multiplied table; ``COUNT(*)`` is exposed when any
  table of the block is multiplied: when every table is, all of them are over-counted; when only
  some are, ``COUNT(*)`` counts the others' rows exactly, so whether it is wrong depends on which
  entity the question counts and the aggregate is marked **ambiguous** instead (reported apart, not
  in ``exposed`` / ``realized``). ``DISTINCT`` aggregates over a multiplied table are **guarded**.
- **Realized** multiplication runs the block's own ``FROM``/``JOIN``/``WHERE`` and compares
  ``COUNT(*)`` with ``COUNT(DISTINCT alias.rowid)``: did rows actually repeat on this data, under
  these filters?
- ``ORDER BY … LIMIT`` over a multiplied block is flagged as a **top-N** exposure: repeated rows can
  take several of the N slots.
- **Repeated values** (no join needed): ``COUNT(col)`` without ``DISTINCT`` whose column takes the
  same value on several rows under the block's filters, e.g. ``COUNT(ID) FROM Laboratory`` counts
  lab rows, not patients. Ambiguous like a partial ``COUNT(*)``: wrong only if the question counts
  the values (entities) rather than the rows.

Static exposure is conservative (it ignores filters); realized multiplication is the evidence that
the data and filters actually fan out.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from misqal.cardinality.dialect import SQLiteDialect
from misqal.harness.bench import QueryTimeout, run_query

SENSITIVE = (exp.Sum, exp.Avg, exp.Count, exp.GroupConcat)


@dataclass
class Edge:
    left: str  # alias
    right: str
    left_cols: list[str]
    right_cols: list[str]
    left_unique: bool | None = None
    right_unique: bool | None = None

    @property
    def kind(self) -> str:
        if self.left_unique is None or self.right_unique is None:
            return "unknown"
        return {(True, True): "1:1", (False, True): "N:1", (True, False): "1:N",
                (False, False): "M:N"}[(self.left_unique, self.right_unique)]


@dataclass
class Agg:
    func: str
    distinct: bool
    star: bool
    aliases: list[str]
    exposed: bool = False
    guarded: bool = False
    ambiguous: bool = False  # COUNT(*) with some, not all, tables multiplied
    realized: bool | None = None
    repeated_values: bool | None = None  # COUNT(col): col repeats on the block's rows
    sql: str = ""


@dataclass
class BlockAudit:
    tables: dict[str, str]  # alias -> base table
    derived: list[str]  # aliases of subquery / CTE sources
    edges: list[Edge] = field(default_factory=list)
    cross_joined: list[str] = field(default_factory=list)  # aliases with no equi-join edge
    multiplied: list[str] = field(default_factory=list)  # statically
    realized_multiplied: list[str] | None = None  # empirically (None = not checked / failed)
    aggs: list[Agg] = field(default_factory=list)
    top_n: bool = False
    top_n_exposed: bool = False
    top_n_realized: bool | None = None


@dataclass
class QueryAudit:
    id: str
    db_id: str
    sql: str
    parse_error: str | None = None
    blocks: list[BlockAudit] = field(default_factory=list)

    @property
    def has_join(self) -> bool:
        return any(len(b.tables) + len(b.derived) > 1 for b in self.blocks)

    @property
    def exposed(self) -> bool:
        return any(a.exposed for b in self.blocks for a in b.aggs) or \
            any(b.top_n_exposed for b in self.blocks)

    @property
    def realized(self) -> bool:
        return any(a.realized and not a.ambiguous for b in self.blocks for a in b.aggs) or any(
            b.top_n_realized for b in self.blocks)

    @property
    def realized_ambiguous(self) -> bool:
        """Only an ambiguous ``COUNT(*)`` actually repeats rows: wrong only if the question counts
        the entity whose rows repeat."""
        return not self.realized and any(a.realized and a.ambiguous
                                         for b in self.blocks for a in b.aggs)

    @property
    def repeated_values(self) -> bool:
        """A ``COUNT(col)`` counts rows whose ``col`` repeats (no join fan-out needed)."""
        return any(a.repeated_values and not a.realized for b in self.blocks for a in b.aggs)

    @property
    def guarded(self) -> bool:
        return any(a.guarded for b in self.blocks for a in b.aggs)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "has_join": self.has_join, "exposed": self.exposed,
                "realized": self.realized, "realized_ambiguous": self.realized_ambiguous,
                "repeated_values": self.repeated_values, "guarded": self.guarded}


def sqlite_schema(db: Path) -> dict[str, dict[str, str]]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {t: {c[1]: (c[2] or "TEXT") for c in con.execute(f'PRAGMA table_info("{t}")')}
                for t in tables}


@lru_cache(maxsize=4096)
def is_unique(db: Path, table: str, columns: tuple[str, ...], timeout: float = 60.0) -> bool | None:
    """Are ``columns`` a key of ``table`` (NULL as a value)? None if the count fails."""
    d = SQLiteDialect()
    count_sql, _ = d.query_1d(table, [])
    grain_sql, _ = d.query_composite_grain(table, list(columns))
    try:
        (_, [(total,)]) = run_query(db, count_sql, timeout)
        (_, [(distinct,)]) = run_query(db, grain_sql, timeout)
    except (sqlite3.Error, QueryTimeout, ValueError):
        return None
    return distinct == total


def _value_aliases(node: exp.Expression, aliases: set[str]) -> set[str]:
    """Aliases whose values an aggregate actually accumulates: the THEN/ELSE results of CASE and
    IIF, not their conditions (``SUM(CASE WHEN a.x = 1 THEN b.v END)`` measures ``b`` only)."""
    if isinstance(node, exp.Case):
        out: set[str] = set()
        for branch in node.args.get("ifs") or []:
            out |= _value_aliases(branch.args["true"], aliases)
        if default := node.args.get("default"):
            out |= _value_aliases(default, aliases)
        return out
    if isinstance(node, exp.If):
        out = _value_aliases(node.args["true"], aliases)
        if false := node.args.get("false"):
            out |= _value_aliases(false, aliases)
        return out
    if isinstance(node, exp.Column):
        return {node.table} & aliases
    out = set()
    for child in node.iter_expressions():
        out |= _value_aliases(child, aliases)
    return out


def _limit_value(select: exp.Select) -> int | None:
    limit = select.args.get("limit")
    if limit is None:
        return None
    try:
        return int(limit.expression.name)
    except (AttributeError, ValueError, TypeError):
        return 2  # non-literal limit: assume more than one row


def _own(node: exp.Expression, select: exp.Select) -> bool:
    """True if ``node`` belongs to ``select`` itself, not to a nested query."""
    return node.find_ancestor(exp.Select) is select


def _column_alias(col: exp.Column) -> str:
    return col.table


def _join_edges(select: exp.Select, aliases: set[str]) -> list[Edge]:
    predicates = [j.args.get("on") for j in select.args.get("joins") or []]
    if where := select.args.get("where"):
        predicates.append(where.this)
    grouped: dict[tuple[str, str], tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for pred in filter(None, predicates):
        for eq in pred.find_all(exp.EQ):
            if not _own(eq, select):
                continue
            left, right = eq.this, eq.expression
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue
            la, ra = _column_alias(left), _column_alias(right)
            if la == ra or la not in aliases or ra not in aliases:
                continue
            if la > ra:
                (la, left), (ra, right) = (ra, right), (la, left)
            cols = grouped[(la, ra)]
            cols[0].append(left.name)
            cols[1].append(right.name)
    return [Edge(la, ra, lc, rc) for (la, ra), (lc, rc) in grouped.items()]


def _multiplied(aliases: set[str], edges: list[Edge], cross: list[str]) -> set[str]:
    """Aliases whose rows a join can repeat: some edge reachable from them leads to a side whose
    join columns are not unique (unknown uniqueness counts as non-unique)."""
    adjacency: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for e in edges:
        adjacency[e.left].append((e.right, e.right_unique is True))
        adjacency[e.right].append((e.left, e.left_unique is True))
    out = set()
    for start in aliases:
        # Walk away from `start` only: an edge back to a visited alias is the one we came along.
        seen, stack, hit = {start}, [start], False
        while stack and not hit:
            for nxt, far_unique in adjacency[stack.pop()]:
                if nxt in seen:
                    continue
                if not far_unique:
                    hit = True
                    break
                seen.add(nxt)
                stack.append(nxt)
        if hit or (cross and len(aliases) > 1):
            out.add(start)
    return out


def _realized(db: Path, select: exp.Select, aliases: list[str],
              timeout: float) -> list[str] | None:
    """Aliases whose rows actually repeat in this block's FROM/JOIN/WHERE on the data."""
    probe = select.copy()
    for arg in ("group", "having", "order", "limit", "offset", "distinct", "qualify"):
        probe.set(arg, None)
    probe.set("expressions", [exp.Count(this=exp.Star())] + [
        exp.Count(this=exp.Distinct(expressions=[exp.column("rowid", table=a)])) for a in aliases])
    try:
        _, rows = run_query(db, probe.sql(dialect="sqlite"), timeout)
    except (sqlite3.Error, QueryTimeout):
        return None
    total, *distinct = rows[0]
    return [a for a, d in zip(aliases, distinct) if total > d]


def _repeats(db: Path, select: exp.Select, arg: exp.Expression, timeout: float) -> bool | None:
    """Does ``arg`` take the same value on several of the block's rows (FROM/JOIN/WHERE)?"""
    probe = select.copy()
    for key in ("group", "having", "order", "limit", "offset", "distinct", "qualify"):
        probe.set(key, None)
    probe.set("expressions", [exp.Count(this=arg.copy()),
                              exp.Count(this=exp.Distinct(expressions=[arg.copy()]))])
    try:
        _, rows = run_query(db, probe.sql(dialect="sqlite"), timeout)
    except (sqlite3.Error, QueryTimeout):
        return None
    return rows[0][0] > rows[0][1]


def _audit_block(db: Path, scope: Scope, timeout: float, realize: bool) -> BlockAudit | None:
    select = scope.expression
    if not isinstance(select, exp.Select):
        return None
    tables = {a: s.name for a, s in scope.sources.items() if isinstance(s, exp.Table)}
    derived = [a for a, s in scope.sources.items() if not isinstance(s, exp.Table)]
    block = BlockAudit(tables=tables, derived=derived)
    aliases = set(tables) | set(derived)

    block.edges = _join_edges(select, aliases)
    for e in block.edges:
        if e.left in tables:
            e.left_unique = is_unique(db, tables[e.left], tuple(e.left_cols))
        if e.right in tables:
            e.right_unique = is_unique(db, tables[e.right], tuple(e.right_cols))
    connected = {a for e in block.edges for a in (e.left, e.right)}
    block.cross_joined = sorted(aliases - connected) if len(aliases) > 1 else []
    multiplied = _multiplied(aliases, block.edges, block.cross_joined)
    block.multiplied = sorted(multiplied)

    if realize and len(aliases) > 1 and not derived:
        block.realized_multiplied = _realized(db, select, sorted(tables), timeout)

    for agg in select.find_all(exp.AggFunc):
        if not isinstance(agg, SENSITIVE + (exp.Min, exp.Max)) or not _own(agg, select):
            continue
        distinct = isinstance(agg.this, exp.Distinct)
        star = isinstance(agg.this, exp.Star)
        used = sorted(_value_aliases(agg.this, aliases)) if agg.this is not None else []
        a = Agg(func=type(agg).__name__.upper(), distinct=distinct, star=star, aliases=used,
                sql=agg.sql(dialect="sqlite"))
        if isinstance(agg, SENSITIVE):
            hit = multiplied if star else multiplied & set(used)
            a.ambiguous = star and bool(hit) and hit != aliases
            a.exposed = bool(hit) and not distinct and not a.ambiguous
            a.guarded = bool(hit) and distinct
            if block.realized_multiplied is not None and (a.exposed or a.ambiguous):
                real = set(block.realized_multiplied)
                if star:
                    a.realized = bool(real) and (a.ambiguous or real >= set(tables))
                else:
                    a.realized = bool(real & set(used))
        if (realize and isinstance(agg, exp.Count) and not distinct and not star
                and isinstance(agg.this, exp.Column) and not derived and not a.realized):
            a.repeated_values = _repeats(db, select, agg.this, timeout)
        block.aggs.append(a)

    # Top-N: repeated rows can take several of the N slots only when N > 1 and every projected
    # column comes from a multiplied table (the output rows are copies of the same entity).
    limit = _limit_value(select)
    block.top_n = bool(select.args.get("order")) and limit is not None
    projected = {c.table for p in select.selects for c in p.find_all(exp.Column)} & aliases
    block.top_n_exposed = (block.top_n and limit > 1 and bool(projected)
                           and projected <= multiplied and not select.args.get("distinct")
                           and not select.args.get("group"))
    if block.top_n_exposed and block.realized_multiplied is not None:
        block.top_n_realized = projected <= set(block.realized_multiplied)
    return block


def audit_query(qid: str, db_id: str, db: Path, sql: str, timeout: float = 30.0,
                realize: bool = True) -> QueryAudit:
    qa = QueryAudit(id=qid, db_id=db_id, sql=sql)
    try:
        tree = qualify(sqlglot.parse_one(sql, read="sqlite"), schema=sqlite_schema(db),
                       dialect="sqlite", validate_qualify_columns=False)
        scopes = traverse_scope(tree)
    except Exception as e:  # unparsable gold SQL is reported, not fatal
        qa.parse_error = f"{type(e).__name__}: {e}"[:300]
        return qa
    for scope in scopes:
        if (block := _audit_block(db, scope, timeout, realize)) is not None:
            qa.blocks.append(block)
    return qa
