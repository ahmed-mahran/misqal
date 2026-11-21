"""End-to-end harness run with a scripted model (no API key): tool loop, matrix tool, scoring."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from misqal.harness import prompts
from misqal.harness.agent import Arm, run_episode
from misqal.harness.bench import Task, score_bird


class ScriptedModel(BaseChatModel):
    """Replays a fixed list of tool calls, one per model turn."""

    script: list[dict[str, Any]]
    turn: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.turn += 1
        if self.turn > len(self.script):  # script done: finish with a plain message
            msg = AIMessage(content="done")
        else:
            step = self.script[self.turn - 1]
            msg = AIMessage(content="", tool_calls=[{**step, "id": f"call{self.turn}"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def make_db(path: Path) -> None:
    with sqlite3.connect(path) as con:
        con.executescript("""
            CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer TEXT, total REAL);
            CREATE TABLE lines (line_id INTEGER PRIMARY KEY, order_id INTEGER, product TEXT);
            INSERT INTO orders VALUES (1, 'ann', 100), (2, 'bob', 50), (3, 'ann', 10);
            INSERT INTO lines VALUES (1, 1, 'x'), (2, 1, 'y'), (3, 2, 'x'), (4, 3, 'z');
        """)


FANOUT_SQL = ("SELECT o.customer, SUM(o.total) FROM orders o JOIN lines l "
              "ON o.order_id = l.order_id GROUP BY o.customer")
GOLD_SQL = "SELECT customer, SUM(total) FROM orders GROUP BY customer"


def test_full_grain_episode(tmp_path: Path) -> None:
    db = tmp_path / "shop.sqlite"
    make_db(db)
    task = Task("1", "bird", "shop", db, "Total order value per customer.", gold_sql=GOLD_SQL)
    model = ScriptedModel(script=[
        {"name": "cardinality_matrix",
         "args": {"table_name": "LINES", "columns": ["line_id", "order_id", "product"]}},
        {"name": "submit_answer", "args": {"sql": GOLD_SQL, "answer": "ann 110, bob 50"}},
        {"name": "cardinality_matrix",
         "args": {"table_name": "orders", "columns": ["order_id", "customer"]}},
        {"name": "submit_answer", "args": {"sql": GOLD_SQL, "answer": "ann 110, bob 50"}},
    ])
    ep = run_episode(model, task, Arm(evidence="full", procedure="grain", delivery="tool"))
    assert ep.status == "completed" and ep.pred_sql == GOLD_SQL and ep.sql_source == "submitted"
    assert ep.state.gate_rejections == 1  # GOLD_SQL reads only `orders`, not profiled at first
    call = ep.state.matrix_calls[0]
    assert call["table"] == "lines" and call["stats"]["queries"]["1d"] == 1
    tool_out = next(m.content for m in ep.messages if getattr(m, "name", "") == "cardinality_matrix")
    assert "`order_id` ⤄ `product`" in tool_out or "M:N" in tool_out
    assert score_bird(task, ep.pred_sql)["multiset_ex"]


def test_fanout_is_caught_by_values(tmp_path: Path) -> None:
    db = tmp_path / "shop.sqlite"
    make_db(db)
    task = Task("2", "bird", "shop", db, "Total order value per customer.", gold_sql=GOLD_SQL)
    model = ScriptedModel(script=[{"name": "submit_answer",
                                   "args": {"sql": FANOUT_SQL, "answer": "ann 210"}}])
    ep = run_episode(model, task, Arm())
    s = score_bird(task, ep.pred_sql)
    assert s["exec_ok"] and not s["set_ex"]  # order 1 counted twice: 210 instead of 110


def test_matrix_tool_errors_are_returned_to_the_model(tmp_path: Path) -> None:
    db = tmp_path / "shop.sqlite"
    make_db(db)
    task = Task("3", "bird", "shop", db, "q", gold_sql=GOLD_SQL)
    model = ScriptedModel(script=[
        {"name": "cardinality_matrix", "args": {"table_name": "lines", "columns": ["nope", "x"]}},
        {"name": "submit_answer", "args": {"sql": GOLD_SQL, "answer": "a"}},
    ])
    ep = run_episode(model, task, Arm(evidence="labels", delivery="tool"))
    err = next(m for m in ep.messages if getattr(m, "name", "") == "cardinality_matrix")
    assert err.status == "error" and "unknown columns" in err.content
    assert ep.status == "completed"


def test_prompts_per_arm() -> None:
    assert "Three-Bucket" not in prompts.system_prompt("S", "none", "minimal")
    grain_none = prompts.system_prompt("S", "none", "grain")
    assert "Three-Bucket" in grain_none and "No cardinality tool" in grain_none
    assert "When to run `cardinality_matrix`" in prompts.system_prompt("S", "full", "grain", "tool")
    assert "given with the question" in prompts.system_prompt("S", "full", "grain", "inject")


def test_injected_evidence_covers_whole_database(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # profile cache goes under tmp_path/runs
    db = tmp_path / "shop.sqlite"
    make_db(db)
    task = Task("4", "bird", "shop", db, "Total order value per customer.", gold_sql=FANOUT_SQL)
    submit = [{"name": "submit_answer", "args": {"sql": GOLD_SQL, "answer": "a"}}]
    full = run_episode(ScriptedModel(script=submit), task, Arm(evidence="full"))
    labels = run_episode(ScriptedModel(script=submit), task, Arm(evidence="labels"))
    facts = run_episode(ScriptedModel(script=submit), task, Arm(evidence="facts"))
    none = run_episode(ScriptedModel(script=submit), task, Arm(evidence="none"))
    full_msg, labels_msg, none_msg = (e.messages[0].content for e in (full, labels, none))
    assert "**Table:** `orders`" in full_msg and "**Table:** `lines`" in full_msg
    assert "**Primary keys (inferred):**" in full_msg
    assert "**Table:** `lines`" in labels_msg and "Primary keys" not in labels_msg
    facts_msg = facts.messages[0].content
    assert "`orders` (" in facts_msg and "`lines` (" in facts_msg and "**" not in facts_msg
    assert none_msg == task.question  # no evidence and no table hint
    assert not any(getattr(m, "name", "") == "cardinality_matrix" for m in full.messages)


def test_known_tables_limit_schema_and_injected_evidence(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace

    from misqal.harness.agent import known_tables, schema_ddl
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "shop.sqlite"
    make_db(db)
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE notes (note_id INTEGER PRIMARY KEY, body TEXT)")
    # gold reads one table, the trap two: both stay visible, the unrelated table does not
    task = Task("5", "bird", "shop", db, "Total order value per customer.", gold_sql=GOLD_SQL,
                original_gold_sql=FANOUT_SQL)
    assert known_tables(task) == ("lines", "orders")
    task = replace(task, tables=known_tables(task))
    assert "notes" not in schema_ddl(db, task.tables) and "lines" in schema_ddl(db, task.tables)
    assert "notes" in schema_ddl(db)
    submit = [{"name": "submit_answer", "args": {"sql": GOLD_SQL, "answer": "a"}}]
    msg = run_episode(ScriptedModel(script=submit), task, Arm(evidence="full")).messages[0].content
    assert "**Table:** `orders`" in msg and "**Table:** `lines`" in msg and "`notes`" not in msg


JOIN_SQL = ("SELECT o.customer, SUM(o.total) FROM orders o JOIN lines l ON o.order_id = l.order_id "
            "GROUP BY o.customer")


def test_forced_tool_gate_rejects_unprofiled_tables(tmp_path: Path) -> None:
    db = tmp_path / "shop.sqlite"
    make_db(db)
    task = Task("5", "bird", "shop", db, "Total order value per customer.", gold_sql=GOLD_SQL)
    model = ScriptedModel(script=[
        {"name": "submit_answer", "args": {"sql": JOIN_SQL, "answer": "a"}},
        {"name": "cardinality_matrix", "args": {"table_name": "orders",
                                                "columns": ["order_id", "customer"]}},
        {"name": "cardinality_matrix", "args": {"table_name": "lines",
                                                "columns": ["line_id", "order_id"]}},
        {"name": "submit_answer", "args": {"sql": GOLD_SQL, "answer": "ann 110, bob 50"}},
    ])
    ep = run_episode(model, task, Arm(evidence="full", procedure="grain", delivery="tool"))
    first_submit = next(m for m in ep.messages if getattr(m, "name", "") == "submit_answer")
    assert first_submit.status == "error" and "Not submitted" in first_submit.content
    assert ep.state.gate_rejections == 1 and ep.pred_sql == GOLD_SQL and ep.status == "completed"


def test_gate_gives_up_after_limit_and_failed_submit_can_be_fixed(tmp_path: Path) -> None:
    db = tmp_path / "shop.sqlite"
    make_db(db)
    task = Task("6", "bird", "shop", db, "q", gold_sql=GOLD_SQL)
    submit_join = {"name": "submit_answer", "args": {"sql": JOIN_SQL, "answer": "a"}}
    model = ScriptedModel(script=[submit_join] * 4)
    ep = run_episode(model, task, Arm(evidence="full", delivery="tool"))
    assert ep.state.gate_rejections == 3 and ep.pred_sql == JOIN_SQL  # 4th accepted
    broken = ScriptedModel(script=[
        {"name": "submit_answer", "args": {"sql": "SELECT nope FROM orders", "answer": "a"}},
        {"name": "submit_answer", "args": {"sql": GOLD_SQL, "answer": "a"}}])
    ep = run_episode(broken, task, Arm())
    assert ep.pred_sql == GOLD_SQL and ep.sql_source == "submitted"


def test_budget_check_uses_reported_spend(monkeypatch) -> None:
    from misqal.harness import models
    monkeypatch.setattr(models, "daily_spend", lambda spec: {"gw:a": 30.0, "gw:b": 3.0}.get(spec))
    assert models.over_budget("gw:a") and not models.over_budget("gw:b")
    assert not models.over_budget("cerebras:x")  # no spend reported: never blocked
    assert models.over_budget("gw:b", budget=2.0)
    assert models.over_budget("gw:b", budget=4.0)  # stops BUDGET_MARGIN short of the cap


def test_budget_counts_from_start_mark(monkeypatch, tmp_path) -> None:
    from misqal.harness import models
    state = {"spend": 30.0, "window": "day1"}
    monkeypatch.setattr(models, "_spend_window", lambda spec, max_age=30.0: (state["spend"], state["window"]))
    monkeypatch.setattr(models, "daily_spend", lambda spec, max_age=30.0: state["spend"])
    marks = tmp_path / "start.json"
    assert models.over_budget("gw:a")  # whole window counted
    assert not models.over_budget("gw:a", start_file=marks)  # first check marks 30.0
    state["spend"] = 52.0
    assert not models.over_budget("gw:a", start_file=marks)  # 22 since the mark
    state["spend"] = 53.5
    assert models.over_budget("gw:a", start_file=marks)
    state.update(spend=1.0, window="day2")  # new window: marked again
    assert not models.over_budget("gw:a", start_file=marks)
    assert json.loads(marks.read_text())["gw"] == {"window": "day2", "spend": 1.0}


def test_first_matrix_call_is_cached_across_episodes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # call cache goes under tmp_path/runs
    db = tmp_path / "shop.sqlite"
    make_db(db)
    task = Task("6", "bird", "shop", db, "Total order value per customer.", gold_sql=GOLD_SQL)
    script = [{"name": "cardinality_matrix", "args": {"table_name": "orders",
                                                       "columns": ["order_id", "customer"]}},
              {"name": "submit_answer", "args": {"sql": GOLD_SQL, "answer": "a"}}]
    arm = Arm(evidence="facts", delivery="tool")
    first = run_episode(ScriptedModel(script=script), task, arm)
    second = run_episode(ScriptedModel(script=script), task, arm)
    assert first.state.matrix_calls[0]["cached"] is False
    assert second.state.matrix_calls[0]["cached"] is True
    outputs = [[m.content for m in e.messages if getattr(m, "name", "") == "cardinality_matrix"]
               for e in (first, second)]
    assert outputs[0] == outputs[1]


def test_measures_are_left_out_of_every_evidence_arm(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace

    from misqal.harness.agent import known_tables
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "shop.sqlite"
    make_db(db)
    task = Task("7", "bird", "shop", db, "Total order value per customer.", gold_sql=GOLD_SQL,
                original_gold_sql=FANOUT_SQL)
    task = replace(task, tables=known_tables(task), measures=("orders.total",))
    submit = [{"name": "submit_answer", "args": {"sql": GOLD_SQL, "answer": "a"}}]
    stats = run_episode(ScriptedModel(script=submit), task, Arm(evidence="stats")).messages[0].content
    assert "## Column statistics" in stats and "`customer` 2 distinct" in stats
    assert "`total`" not in stats and "1:N" not in stats  # counts only, no relations, no measures
    from misqal.harness import evidence_v2
    monkeypatch.setattr(evidence_v2, "roles_for", lambda db_id, table: {"order_id": "identifier", "total": "measure"})
    stats2 = run_episode(ScriptedModel(script=submit), task, Arm(evidence="stats2")).messages[0].content
    assert stats2.startswith(stats.split("## Column statistics")[0]) and "rows per value" in stats2
    assert "`total`" not in stats2 and "has exactly one" not in stats2 and "1:N" not in stats2
    full = run_episode(ScriptedModel(script=submit), task, Arm(evidence="full")).messages[0].content
    assert "**Table:** `orders`" in full and "`total`" not in full
    call = {"name": "cardinality_matrix",
            "args": {"table_name": "orders", "columns": ["order_id", "customer", "total"]}}
    ep = run_episode(ScriptedModel(script=[call, *submit]), task, Arm(evidence="facts", delivery="tool"))
    out = next(m.content for m in ep.messages if getattr(m, "name", "") == "cardinality_matrix")
    assert out.startswith("(Not profiled, measure columns: total.)") and ep.state.matrix_calls[0][
        "measures_dropped"] == ["total"]
    only = {"name": "cardinality_matrix", "args": {"table_name": "orders", "columns": ["total"]}}
    ep = run_episode(ScriptedModel(script=[only, *submit]), task, Arm(evidence="facts", delivery="tool"))
    err = next(m.content for m in ep.messages if getattr(m, "name", "") == "cardinality_matrix")
    assert "measure columns" in err


def test_submit_reminder_recovers_a_text_answer(tmp_path: Path) -> None:
    db = tmp_path / "shop.sqlite"
    make_db(db)
    task = Task("8", "bird", "shop", db, "Total order value per customer.", gold_sql=GOLD_SQL)
    submit = {"name": "submit_answer", "args": {"sql": GOLD_SQL, "answer": "a"}}

    class TextFirst(ScriptedModel):  # answers in text first, calls the tool once reminded
        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            self.turn += 1
            reminded = any(getattr(m, "content", "") == "SUBMIT NOW" for m in messages)
            if not reminded or self.turn > 3:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="It is 160."))])
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="", tool_calls=[{**submit, "id": f"call{self.turn}"}]))])

    plain = run_episode(TextFirst(script=[]), task, Arm())
    assert plain.sql_source != "submitted" and plain.state.reminders == 0
    nudged = run_episode(TextFirst(script=[]), task, Arm(submit_reminder="SUBMIT NOW"))
    assert nudged.sql_source == "submitted" and nudged.pred_sql == GOLD_SQL
    assert nudged.state.reminders == 1


def test_gemini_thought_summaries_move_to_reasoning() -> None:
    from misqal.harness.models import _reasoning_chat_openai
    model = _reasoning_chat_openai()(model="m", base_url="http://localhost", api_key="k")
    raw = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
           "choices": [{"index": 0, "finish_reason": "stop", "message": {
               "role": "assistant", "content": "<thought>check the grain</thought>The total is 1,500."}}]}
    msg = model._create_chat_result(raw).generations[0].message
    assert msg.content == "The total is 1,500."
    assert msg.additional_kwargs["reasoning_content"] == "check the grain"


def test_compare_counts_a_question_and_its_twin_once() -> None:
    from misqal.harness.stats import compare
    ps = [("q1", 1, 0), ("q1-v", 1, 0), ("q2", 1, 0), ("q2-v", 0, 0), ("q3", 0, 1)]
    c = compare(ps, n_boot=50)
    assert (c["up"], c["down"]) == (2, 1) and c["w"] == 3 and c["l"] == 1 and 0 < c["p_perm"] <= 1
