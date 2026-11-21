"""Evidence v2 on a hand-built table where every fact is known."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from misqal.harness import evidence_v2 as v2

# lines: one row per order line. order fee repeats per order (measure grain = order_id);
# category is blank on some lines of product A (NULL split), and product C really has two
# categories; unit_price is per product; qty varies by row.
ROWS = [
    # line, order, product, category, fee, unit_price, qty
    (1, 10, "A", "toys", 5.0, 2.0, 1),
    (2, 10, "B", "food", 5.0, 3.0, 2),
    (3, 11, "A", None, 7.0, 2.0, 3),
    (4, 11, "C", "home", 7.0, 4.0, 1),
    (5, 12, "C", "garden", 9.0, 4.0, 2),
    (6, 12, "A", "toys", 9.0, 2.0, 5),
    (7, 13, "B", "food", 4.0, 3.0, 1),
]
ROLES = {"line_id": "identifier", "order_id": "identifier", "product": "dimension",
         "category": "dimension", "fee": "measure", "unit_price": "measure", "qty": "measure"}


@pytest.fixture()
def db(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)  # caches under tmp_path/runs
    path = tmp_path / "shop.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE lines (line_id INTEGER PRIMARY KEY, order_id INTEGER, product TEXT, "
                    "category TEXT, fee REAL, unit_price REAL, qty INTEGER)")
        con.executemany("INSERT INTO lines VALUES (?,?,?,?,?,?,?)", ROWS)
    monkeypatch.setattr(v2, "roles_for", lambda db_id, table: ROLES)
    return path


def facts(db: Path) -> dict:
    return v2.compute(db, "shop", "lines", v2.table_columns(db, "lines"))


def test_blanks_are_kept_apart_from_values(db: Path) -> None:
    f = facts(db)
    a_cat = f["filled"]["product::category"]
    # filled rows: A -> toys only, B -> food, C -> home and garden: product does not determine category
    assert a_cat["determines"] is False
    # product A has both a blank and a filled category; B and C do not
    assert a_cat["mixed"] == 1 and f["nulls"]["category"] == 1
    # order_id -> category on filled rows: order 11 has (A, NULL) and (C, home): one filled value
    assert f["filled"]["order_id::category"]["determines"] is False  # order 10: toys and food


def test_measure_grain(db: Path) -> None:
    g = facts(db)["grains"]
    assert g["fee"]["per"] == "order_id" and g["fee"]["values"] == 4
    assert g["fee"]["rows_per"] == pytest.approx(7 / 4)
    assert g["unit_price"]["per"] == "product"  # coarsest: 3 products < 4 orders
    assert g["qty"] is None  # varies by row


def test_rows_per_value_and_keys(db: Path) -> None:
    f = facts(db)
    assert f["keys"] == ["line_id"]
    assert f["per_value"]["order_id"] == {"min": 1, "mean": pytest.approx(7 / 4), "max": 2, "values": 4}


def test_render_states_the_facts(db: Path) -> None:
    text = v2.render_facts(facts(db))
    assert "Each row has its own `line_id`." in text
    assert "Each `order_id` appears on 1–2 rows" in text
    assert "`category` is blank on 1 of 7 rows" in text
    assert "Measure `fee` has one value per `order_id`" in text
    assert "Measure `qty` varies from row to row." in text
    full = v2.render_full(facts(db))
    assert "**Measures (grain):**" in full and "`fee`: one value per `order_id`" in full


def test_null_split_where_filled(tmp_path: Path, monkeypatch) -> None:
    """The trap case: a product whose only category is sometimes blank."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "t.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, product TEXT, category TEXT)")
        con.executemany("INSERT INTO t VALUES (?,?,?)",
                        [(1, "A", "toys"), (2, "A", None), (3, "B", "food"), (4, "B", "food")])
    monkeypatch.setattr(v2, "roles_for", lambda db_id, table: {"id": "identifier"})
    f = v2.compute(path, "x", "t", ["id", "product", "category"])
    assert f["filled"]["product::category"] == {"determines": True, "values": 2, "mixed": 1}
    text = v2.render_facts(f)
    assert "Each `product` has one `category` where filled in" in text
    assert "1 of 2 `product` values have both blank and filled `category`" in text


def test_tool_columns_always_include_identifiers(db: Path) -> None:
    assert v2.tool_columns(db, "shop", "lines", ["category", "fee"]) == [
        "category", "fee", "line_id", "order_id"]


def test_v2_arms_end_to_end(db: Path) -> None:

    from misqal.harness.agent import Arm, run_episode
    from misqal.harness.bench import Task
    from test_harness_smoke import ScriptedModel

    gold = "SELECT SUM(fee) FROM (SELECT DISTINCT order_id, fee FROM lines)"
    task = Task("9", "bird", "shop", db, "Total shipping fees.", gold_sql=gold, tables=("lines",))
    submit = {"name": "submit_answer", "args": {"sql": gold, "answer": "25"}}
    full2 = run_episode(ScriptedModel(script=[submit]), task, Arm(evidence="full2")).messages[0].content
    assert "## Cardinality evidence" in full2 and "**Measures (grain):**" in full2
    assert "`fee`: one value per `order_id`" in full2 and "**Rows per value (identifiers):**" in full2

    call = {"name": "cardinality_matrix", "args": {"table_name": "lines", "columns": ["category", "fee"]}}
    ep = run_episode(ScriptedModel(script=[call, submit]), task, Arm(evidence="facts2", delivery="tool"))
    out = next(m.content for m in ep.messages if getattr(m, "name", "") == "cardinality_matrix")
    assert "Each row has its own `line_id`." in out  # key added although the model did not ask
    assert "Measure `fee` has one value per `order_id`" in out
    assert ep.state.matrix_calls[0]["requested"] == ["category", "fee"]
    assert ep.sql_source == "submitted" and ep.state.gate_rejections == 0


def test_facts2_profiles_a_one_column_table(tmp_path: Path, monkeypatch) -> None:
    from misqal.harness.agent import Arm, run_episode
    from misqal.harness.bench import Task
    from test_harness_smoke import ScriptedModel
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "p.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE male (name TEXT PRIMARY KEY)")
        con.executemany("INSERT INTO male VALUES (?)", [("a",), ("b",)])
    monkeypatch.setattr(v2, "roles_for", lambda db_id, table: {"name": "identifier"})
    task = Task("10", "bird", "p", path, "How many?", gold_sql="SELECT COUNT(*) FROM male", tables=("male",))
    call = {"name": "cardinality_matrix", "args": {"table_name": "male", "columns": ["name"]}}
    submit = {"name": "submit_answer", "args": {"sql": "SELECT COUNT(*) FROM male", "answer": "2"}}
    ep = run_episode(ScriptedModel(script=[call, submit]), task, Arm(evidence="facts2", delivery="tool"))
    out = next(m.content for m in ep.messages if getattr(m, "name", "") == "cardinality_matrix")
    assert "`male` (2 rows)" in out and ep.sql_source == "submitted"
