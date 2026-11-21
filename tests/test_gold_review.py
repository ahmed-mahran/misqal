import sqlite3
from pathlib import Path

from misqal.fanout import gold_review as gr
from misqal.harness.bench import Task


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "s.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, c TEXT)")
        con.execute("CREATE TABLE l (o_id INTEGER, qty INTEGER)")
        con.executemany("INSERT INTO o VALUES (?,?)", [(1, "a"), (2, "b"), (3, "a")])
        con.executemany("INSERT INTO l VALUES (?,?)", [(1, 1), (1, 2), (2, 3), (3, 1)])
    return path


def _task(db: Path) -> Task:
    # Question: orders of customer 'a' with lines; the gold forgot the customer filter.
    return Task("7", "bird", "s", db, "How many orders of customer a have lines?",
                gold_sql="SELECT COUNT(DISTINCT o.id) FROM o JOIN l ON l.o_id = o.id",
                original_gold_sql="SELECT COUNT(o.id) FROM o JOIN l ON l.o_id = o.id", category="inflated")


FIX = "SELECT COUNT(DISTINCT o.id) FROM o JOIN l ON l.o_id = o.id WHERE o.c = 'a'"
TRAP = "SELECT COUNT(o.id) FROM o JOIN l ON l.o_id = o.id WHERE o.c = 'a'"


def _review(task: Task, model: str, **kw) -> dict:
    r = gr.Review(**kw)
    return {"task_id": task.id, "model": model, **r.model_dump(), **gr.checks(task, r)}


def test_prompt_shows_samples_and_original(tmp_path: Path) -> None:
    text = gr.user_prompt(_task(_db(tmp_path)))
    assert "Sample rows:" in text and "Original gold SQL (counts repeated rows)" in text


def test_checks_fix_and_trap(tmp_path: Path) -> None:
    t = _task(_db(tmp_path))
    r = _review(t, "a", verdict="error", fixed_sql=FIX, trap_sql=TRAP)
    assert r["fixed_runs"] and r["fixed_changes_gold"] and r["trap_differs"]
    assert r["fixed_repeats_rows"] is False and gr.usable_fix(r)


def test_decide(tmp_path: Path) -> None:
    t = _task(_db(tmp_path))
    a = _review(t, "a", verdict="error", fixed_sql=FIX, trap_sql=TRAP)
    b = _review(t, "b", verdict="error", fixed_sql=FIX.replace("o.c = 'a'", "c = 'a'"), trap_sql=TRAP)
    d = gr.decide(t, a, b)
    assert d["status"] == "fixed" and d["fixed_sql"] == FIX and d["trap_sql"] == TRAP
    other = _review(t, "b", verdict="error", fixed_sql=FIX.replace("'a'", "'b'"), trap_sql=TRAP)
    assert gr.decide(t, a, other)["status"] == "disputed"
    ok = _review(t, "b", verdict="correct")
    assert gr.decide(t, ok, {**ok, "model": "c"})["status"] == "ok"
    assert gr.decide(t, a, ok)["status"] == "disputed"
    # A fix that still counts repeated rows is not usable.
    bad = _review(t, "b", verdict="error", fixed_sql=TRAP)
    assert not gr.usable_fix(bad)


def test_prior_reviews_hide_judges(tmp_path: Path) -> None:
    t = _task(_db(tmp_path))
    a = _review(t, "secret-a", verdict="error", problem="missing filter", fixed_sql=FIX, trap_sql=TRAP)
    b = _review(t, "secret-b", verdict="correct")
    text = gr.prior_reviews([a, b])
    assert "Reviewer A: error. missing filter" in text and "Reviewer B: correct" in text
    assert "secret" not in text and FIX in text


def test_fix_that_keeps_result_counts_as_correct(tmp_path: Path) -> None:
    t = _task(_db(tmp_path))
    same = _review(t, "a", verdict="error", fixed_sql="SELECT COUNT(DISTINCT l.o_id) FROM l")
    ok = _review(t, "b", verdict="correct")
    assert same["fixed_changes_gold"] is False and gr.decide(t, same, ok)["status"] == "ok"


def test_fixes_agreeing_on_nothing_need_a_data_check(tmp_path: Path) -> None:
    t = _task(_db(tmp_path))
    empty = FIX.replace("o.c = 'a'", "o.c = 'zzz'")
    a = _review(t, "a", verdict="error", fixed_sql=empty)
    b = _review(t, "b", verdict="error", fixed_sql=empty.replace("o.c", "c"))
    d = gr.decide(t, a, b)
    assert d["status"] == "check" and gr.degenerate([(0,)]) and gr.degenerate([]) and not gr.degenerate([(2,)])
