"""The SQL agent: one tool loop per task with ``execute_sql``, ``submit_answer`` and, in evidence
arms, ``cardinality_matrix`` (run on the task's database)."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, ToolException
from langgraph.errors import GraphRecursionError
import sqlglot
from sqlglot import exp

from misqal.cardinality.core import try_existing_cardinality_profile
from misqal.cardinality.dialect import SQLiteDialect
from misqal.cardinality.matrix import MatrixStats, SQLiteExecutor, run_cardinality_matrix
from misqal.cardinality.models import CardinalityProfile, merge_cardinality_profiles
from misqal.harness import evidence_v2, prompts
from misqal.harness.bench import QueryTimeout, Task, run_query
from misqal.harness.profiles import cached_call_profile, column_stats, db_profiles, table_profile

SUBMIT_TOOL = "submit_answer"
MATRIX_TOOL = "cardinality_matrix"
# Forced-tool arms: a submission that reads unprofiled tables is rejected at most this many times.
MAX_GATE_REJECTIONS = 3
SUBMITTED = "\n\nAnswer submitted. Reply with one short final message; do not call more tools."


@dataclass(frozen=True)
class Arm:
    evidence: str = "none"  # none | stats | stats2 | full | labels | facts | full2 | facts2 (v2: evidence_v2)
    procedure: str = "minimal"  # minimal | grain
    delivery: str = "inject"  # inject: evidence in the prompt | tool: agent calls cardinality_matrix
    recursion_limit: int = 50
    # Sent as a user message when a run ends without a submission (for models that answer in text
    # instead of calling the tool); at most ``max_reminders`` times. Not part of the arm's name.
    submit_reminder: str = ""
    max_reminders: int = 2
    sql_timeout: float = 30.0
    max_rows: int = 50

    @property
    def name(self) -> str:
        n = f"{self.evidence}-{self.procedure}"
        return f"{n}-tool" if self.delivery == "tool" and self.evidence != "none" else n


@dataclass
class EpisodeState:
    profiles: dict[str, CardinalityProfile] = field(default_factory=dict)
    matrix_calls: list[dict[str, Any]] = field(default_factory=list)
    gate_rejections: int = 0
    reminders: int = 0


def schema_ddl(db: Path, tables: tuple[str, ...] = ()) -> str:
    """CREATE statements of the database's tables, or only of ``tables`` when given."""
    keep = {t.lower() for t in tables}
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        rows = con.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
                           "AND name NOT LIKE 'sqlite_%'")
        return "\n".join(sql + ";" for name, sql in rows if not keep or name.lower() in keep)


def _format_rows(cols: list[str], rows: list[tuple], limit: int) -> str:
    out = [" | ".join(cols)] + [" | ".join(map(str, r)) for r in rows[:limit]]
    if len(rows) > limit:
        out.append(f"... (showing first {limit} rows)")
    return "\n".join(out)


def _table_columns(db: Path) -> dict[str, list[str]]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {t: [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')] for t in tables}


def sql_tables(sql: str, catalog: dict[str, list[str]]) -> list[str]:
    """Base tables of ``catalog`` that ``sql`` reads (CTE names excluded), canonical names."""
    names = {t.lower(): t for t in catalog}
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return []
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return sorted({names[t.name.lower()] for t in tree.find_all(exp.Table)
                   if t.name.lower() in names and t.name.lower() not in ctes})


def known_tables(task: Task) -> tuple[str, ...]:
    """Tables read by any reference query of the task: the gold, the uncorrected or naive query
    (the trap) and accepted alternatives. Showing only these keeps both readings open."""
    catalog = _table_columns(task.db_path)
    sqls = [task.gold_sql, task.original_gold_sql, task.naive_sql, *task.alt_gold_sqls,
            *task.former_sqls]
    return tuple(sorted({t for q in sqls if q for t in sql_tables(q, catalog)}))


def evidence_columns(task: Task, table: str, columns: list[str]) -> list[str]:
    """The columns evidence may describe: all but the task's measure columns."""
    drop = set(task.measures)
    return [c for c in columns if f"{table}.{c}".lower() not in drop]


def stats_context(task: Task, per_value: bool = False) -> str:
    """Content control: rows and per-column distinct and NULL counts of the shown tables, with no
    relation between columns.

    ``per_value`` (arm stats2): also, for each identifier column, how many rows each value appears on
    and what a join on it does, in v2's words (`evidence_v2.render_full`). These are counts of one
    column; stats2 still says nothing about relations between columns or about measures, so stats2
    against full2 isolates the dependencies from v2's wording."""
    catalog = _table_columns(task.db_path)
    lines = []
    for table in task.tables or tuple(catalog):
        cols = evidence_columns(task, table, catalog[table])
        st = column_stats(task.db_path, table, cols)
        parts = [f"`{c}` {v['distinct']} distinct" + (f", {v['nulls']} NULL" if v["nulls"] else "")
                 for c, v in st["columns"].items()]
        lines.append(f"`{table}` ({st['rows']} rows): " + "; ".join(parts) + ".")
        if per_value:
            pv = evidence_v2.table_facts(task.db_path, task.db_id, table)["per_value"]
            lines += [f"  - `{c}`: {v['min']}–{v['max']} rows per value (mean {evidence_v2._fmt(v['mean'])}); "
                      f"a join on it repeats each matching row that many times"
                      for c, v in pv.items() if c in cols]
    return "## Column statistics\n" + "\n".join(lines)


def make_tools(task: Task, arm: Arm, state: EpisodeState) -> list[StructuredTool]:
    db = task.db_path

    def execute_sql(query: str) -> str:
        """Run a read-only SQL query against the database and return the rows."""
        try:
            cols, rows = run_query(db, query, arm.sql_timeout, max_rows=arm.max_rows + 1)
        except (sqlite3.Error, QueryTimeout) as e:
            raise ToolException(f"SQL error: {e}") from e
        return _format_rows(cols, rows, arm.max_rows)

    catalog = _table_columns(db)
    gated = arm.evidence != "none" and arm.delivery == "tool"

    def submit_answer(sql: str, answer: str) -> str:
        """Submit your final answer: `sql` is the exact query whose result answers the question
        (it is re-run to check it works); `answer` is the short answer in words."""
        tables = sql_tables(sql, catalog)
        missing = [t for t in tables if t not in state.profiles]
        if gated and missing and state.gate_rejections < MAX_GATE_REJECTIONS:
            state.gate_rejections += 1
            raise ToolException(
                f"Not submitted: the query reads {', '.join(tables)}, but `cardinality_matrix` has "
                f"not been run on {', '.join(missing)}. Run it on each of them with the columns "
                "your query relies on, check your query against the evidence, then submit again.")
        return execute_sql(sql) + SUBMITTED

    # submit_answer does not end the loop by itself: a failed or rejected submission returns to
    # the model so it can fix it. The last successful submission is the answer.
    tools = [
        StructuredTool.from_function(execute_sql, handle_tool_error=True),
        StructuredTool.from_function(submit_answer, name=SUBMIT_TOOL, handle_tool_error=True),
    ]
    if not gated:
        return tools

    dialect = SQLiteDialect()

    def cardinality_matrix(table_name: str, columns: list[str]) -> str:
        # Resolve names case-insensitively, as SQLite does, and fail with a usable message.
        tables = {t.lower(): t for t in catalog}
        table = tables.get(table_name.strip().strip('"`[]').lower())
        if table is None:
            raise ToolException(f"no such table: {table_name}. Tables: {', '.join(catalog)}")
        cols_ci = {c.lower(): c for c in catalog[table]}
        resolved, unknown = [], []
        for c in columns:
            (resolved if c.strip('"`[]').lower() in cols_ci else unknown).append(
                cols_ci.get(c.strip('"`[]').lower(), c))
        if unknown:
            raise ToolException(f"unknown columns in {table}: {unknown}. "
                                f"Columns: {', '.join(catalog[table])}")
        resolved = list(dict.fromkeys(resolved))
        if arm.evidence == "facts2":  # v2: keys always added, measures get their grain, blanks apart
            cols = evidence_v2.tool_columns(db, task.db_id, table, resolved)
            facts = evidence_v2.table_facts(db, task.db_id, table)
            state.matrix_calls.append({"table": table, "columns": cols, "requested": resolved,
                                       "skipped": facts.get("skipped", [])})
            prof = (CardinalityProfile.model_validate(facts["profile"]) if facts["profile"]
                    else CardinalityProfile(table_name=table, total_rows=facts["rows"]))
            state.profiles = merge_cardinality_profiles(state.profiles, {table: prof})
            return evidence_v2.render_facts(facts, cols)
        measures = [c for c in resolved if c not in evidence_columns(task, table, resolved)]
        resolved = [c for c in resolved if c not in measures]
        if not resolved:
            raise ToolException(f"{', '.join(measures)}: measure columns (quantities to aggregate) "
                                "are not profiled; pass identifier or dimension columns")
        note = f"(Not profiled, measure columns: {', '.join(measures)}.)\n" if measures else ""
        call: dict[str, Any] = {"table": table, "columns": resolved,
                                **({"measures_dropped": measures} if measures else {})}
        state.matrix_calls.append(call)
        cached = try_existing_cardinality_profile(table, resolved, state.profiles)
        if cached is not None:
            call["cached"] = True
            return note + prompts.render_profile(cached, arm.evidence)
        stats = MatrixStats()
        existing = state.profiles.get(table)

        def compute() -> CardinalityProfile:
            return run_cardinality_matrix(table, resolved, SQLiteExecutor(db), dialect,
                                          existing_profile=existing, stats=stats)
        try:
            if existing is None:  # a first call on this table: same result in every episode
                profile, call["cached"] = cached_call_profile(db, table, resolved, compute)
            else:
                profile = compute()
        except ValueError as e:
            call["error"] = str(e)
            raise ToolException(str(e)) from e
        call["stats"] = stats.as_dict()
        state.profiles = merge_cardinality_profiles(state.profiles, {table: profile})
        return note + prompts.render_profile(profile, arm.evidence)

    doc = {"full": prompts.MATRIX_TOOL_DOC, "labels": prompts.LABELS_TOOL_DOC,
           "facts": prompts.FACTS_TOOL_DOC, "facts2": prompts.FACTS_TOOL_DOC}[arm.evidence]
    tools.append(StructuredTool.from_function(cardinality_matrix, name=MATRIX_TOOL,
                                              description=doc, handle_tool_error=True))
    return tools


def user_message(task: Task, context: str = "") -> str:
    msg = task.question
    if task.evidence:
        msg += f"\n\nHint: {task.evidence}"
    return msg + (f"\n\n{context}" if context else "")


def task_context(task: Task, arm: Arm, state: EpisodeState) -> str:
    """Injected-evidence arms: the cardinality profile of every table and column of the database
    (of the tables shown, when the task restricts them; every arm sees the same tables)."""
    if arm.evidence == "none" or arm.delivery != "inject":
        return ""
    if arm.evidence in ("stats", "stats2"):
        state.matrix_calls.append({"injected": arm.evidence, "tables": len(task.tables)})
        return stats_context(task, per_value=arm.evidence == "stats2")
    if arm.evidence == "full2":  # v2 profile of every shown table, all columns
        catalog = _table_columns(task.db_path)
        blocks = []
        for table in task.tables or tuple(catalog):
            facts = evidence_v2.table_facts(task.db_path, task.db_id, table)
            if facts["profile"]:
                state.profiles[table] = CardinalityProfile.model_validate(facts["profile"])
            blocks.append(evidence_v2.render_full(facts))
        state.matrix_calls.append({"injected": "full2", "tables": len(blocks)})
        return "## Cardinality evidence\n" + "\n\n".join(blocks)
    if task.tables or task.measures:  # profiles of the shown tables over non-measure columns
        catalog = _table_columns(task.db_path)
        profiles = {}
        for table in task.tables or tuple(catalog):
            cols = evidence_columns(task, table, catalog[table])
            if len(cols) >= 2:
                profiles[table] = table_profile(task.db_path, table, cols)
    else:
        profiles = db_profiles(task.db_path)
    state.profiles = profiles
    state.matrix_calls.append({"injected": "database", "tables": len(profiles)})
    return "## Cardinality evidence\n" + "\n\n".join(
        prompts.render_profile(p, arm.evidence) for p in profiles.values())


def final_sql(messages: list[BaseMessage]) -> tuple[str | None, str]:
    """(sql, source): the last successful submission, else the last successful query."""
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    last_query = None
    submitted = None
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for tc in m.tool_calls:
            res = results.get(tc["id"])
            if res is None or res.status == "error":
                continue
            if tc["name"] == SUBMIT_TOOL:
                submitted = tc["args"].get("sql")
            elif tc["name"] == "execute_sql":
                last_query = tc["args"].get("query")
    if submitted:
        return submitted, "submitted"
    return last_query, ("last_query" if last_query else "none")


def usage(messages: list[BaseMessage]) -> dict[str, int]:
    tin = tout = 0
    for m in messages:
        meta = getattr(m, "usage_metadata", None) or {}
        tin += meta.get("input_tokens", 0)
        tout += meta.get("output_tokens", 0)
    return {"input_tokens": tin, "output_tokens": tout}


@dataclass
class Episode:
    messages: list[BaseMessage]
    state: EpisodeState
    pred_sql: str | None
    sql_source: str
    status: str  # completed | recursion_limit | error
    seconds: float
    error: str | None = None


def run_episode(model: BaseChatModel, task: Task, arm: Arm,
                callbacks: list | None = None) -> Episode:
    state = EpisodeState()
    context = task_context(task, arm, state)
    agent = create_agent(model, make_tools(task, arm, state),
                         system_prompt=prompts.system_prompt(schema_ddl(task.db_path, task.tables),
                                                             arm.evidence, arm.procedure, arm.delivery))
    started = time.monotonic()
    last: dict | None = None
    status, error = "completed", None
    try:  # stream so a runaway agent still leaves its partial trace
        for last in agent.stream({"messages": [HumanMessage(user_message(task, context))]},
                                 config={"recursion_limit": arm.recursion_limit,
                                         "callbacks": callbacks or []},
                                 stream_mode="values"):
            pass
    except GraphRecursionError:
        status = "recursion_limit"
    except Exception as e:  # provider/network errors: record, don't kill the run
        status, error = "error", f"{type(e).__name__}: {e}"[:500]
    messages = last["messages"] if last else [HumanMessage(user_message(task, context))]
    sql, source = final_sql(messages)
    while (status == "completed" and arm.submit_reminder and source != "submitted"
           and state.reminders < arm.max_reminders):
        state.reminders += 1
        try:
            for last in agent.stream({"messages": [*messages, HumanMessage(arm.submit_reminder)]},
                                     config={"recursion_limit": arm.recursion_limit,
                                             "callbacks": callbacks or []},
                                     stream_mode="values"):
                pass
        except GraphRecursionError:
            status = "recursion_limit"
        except Exception as e:
            status, error = "error", f"{type(e).__name__}: {e}"[:500]
        messages = last["messages"] if last else messages
        sql, source = final_sql(messages)
    return Episode(messages, state, sql, source, status, time.monotonic() - started, error)
