from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from misqal.fanout.audit import audit_query


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "shop.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript("""
            CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer TEXT, total REAL);
            CREATE TABLE lines (line_id INTEGER PRIMARY KEY, order_id INTEGER, qty INTEGER);
            CREATE TABLE notes (order_id INTEGER, note TEXT);
            INSERT INTO orders VALUES (1, 'ann', 100), (2, 'bob', 50), (3, 'ann', 10);
            INSERT INTO lines VALUES (1, 1, 2), (2, 1, 1), (3, 2, 5), (4, 3, 1);
            INSERT INTO notes VALUES (1, 'a'), (1, 'c'), (2, 'b');
        """)
    return path


def run(db: Path, sql: str):
    return audit_query("q", "shop", db, sql)


def test_parent_measure_over_child_join_is_exposed_and_realized(db: Path) -> None:
    qa = run(db, "SELECT o.customer, SUM(o.total) FROM orders o JOIN lines l "
                 "ON o.order_id = l.order_id GROUP BY o.customer")
    (block,) = qa.blocks
    (edge,) = block.edges
    assert {edge.left_unique, edge.right_unique} == {True, False}
    assert block.multiplied == ["o"] and block.realized_multiplied == ["o"]
    assert qa.exposed and qa.realized and not qa.guarded


def test_child_measure_through_many_to_one_is_safe(db: Path) -> None:
    qa = run(db, "SELECT o.customer, SUM(l.qty) FROM lines l JOIN orders o "
                 "ON l.order_id = o.order_id GROUP BY o.customer")
    assert qa.has_join and not qa.exposed and not qa.realized


def test_distinct_aggregate_is_guarded(db: Path) -> None:
    qa = run(db, "SELECT COUNT(DISTINCT o.order_id) FROM orders o JOIN lines l "
                 "ON o.order_id = l.order_id")
    assert qa.guarded and not qa.exposed


def test_count_star_is_ambiguous_when_one_side_is_exact(db: Path) -> None:
    # lines are counted exactly, orders repeat: wrong only if the question counts orders
    qa = run(db, "SELECT COUNT(*) FROM orders o JOIN lines l ON o.order_id = l.order_id "
                 "WHERE o.customer = 'ann'")
    assert not qa.exposed and not qa.realized and qa.realized_ambiguous


def test_count_star_over_many_to_many_is_exposed(db: Path) -> None:
    qa = run(db, "SELECT COUNT(*) FROM notes n JOIN lines l ON n.order_id = l.order_id")
    assert qa.exposed and qa.realized and not qa.realized_ambiguous


def test_filter_can_cancel_realized_fanout(db: Path) -> None:
    qa = run(db, "SELECT SUM(o.total) FROM orders o JOIN lines l ON o.order_id = l.order_id "
                 "WHERE l.line_id = 1")
    assert qa.exposed and not qa.realized  # statically possible, not on these rows


def test_join_without_key_on_either_side(db: Path) -> None:
    qa = run(db, "SELECT SUM(n.order_id) FROM notes n JOIN lines l ON n.order_id = l.order_id")
    (edge,) = qa.blocks[0].edges
    assert edge.kind == "M:N" and qa.exposed and qa.realized


def test_top_n_over_fanout(db: Path) -> None:
    qa = run(db, "SELECT o.customer FROM orders o JOIN lines l ON o.order_id = l.order_id "
                 "ORDER BY o.total DESC LIMIT 2")
    assert qa.blocks[0].top_n_exposed and qa.exposed and qa.realized


def test_condition_columns_are_not_measures(db: Path) -> None:
    """``o`` only filters inside CASE; the summed values come from ``l``, which is not repeated."""
    qa = run(db, "SELECT SUM(CASE WHEN o.customer = 'ann' THEN l.qty ELSE 0 END), "
                 "COUNT(IIF(o.total > 20, l.line_id, NULL)) "
                 "FROM orders o JOIN lines l ON o.order_id = l.order_id")
    assert not qa.exposed and not qa.realized


def test_top_1_and_mixed_projection_are_not_exposed(db: Path) -> None:
    top1 = run(db, "SELECT o.customer FROM orders o JOIN lines l ON o.order_id = l.order_id "
                   "ORDER BY o.total DESC LIMIT 1")
    mixed = run(db, "SELECT o.customer, l.qty FROM orders o JOIN lines l "
                    "ON o.order_id = l.order_id ORDER BY l.qty DESC LIMIT 3")
    assert not top1.exposed and not mixed.exposed


def test_subquery_blocks_are_audited_separately(db: Path) -> None:
    qa = run(db, "SELECT customer FROM orders WHERE total > (SELECT AVG(o.total) FROM orders o "
                 "JOIN lines l ON o.order_id = l.order_id)")
    assert len(qa.blocks) == 2 and qa.exposed


def test_unparsable_sql_is_reported(db: Path) -> None:
    qa = run(db, "SELEC nonsense FROM")
    assert qa.parse_error and not qa.blocks


def test_count_of_a_repeated_value_without_join(db: Path) -> None:
    qa = run(db, "SELECT COUNT(order_id) FROM lines")
    assert not qa.has_join and not qa.realized and qa.repeated_values
    assert not run(db, "SELECT COUNT(line_id) FROM lines").repeated_values
    assert not run(db, "SELECT COUNT(order_id) FROM lines WHERE line_id = 1").repeated_values
