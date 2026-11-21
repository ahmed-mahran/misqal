"""Every trap question: gold runs and is non-empty, and the naive (fan-out) SQL gives a different
answer on the generated data, so the trap actually bites. Controls have no naive SQL. Every question
carries judge-facing prose that never reaches the prompt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from misqal.harness.bench import TRAP_QUESTIONS, _normalize, load_traps, run_query, score_traps
from misqal.traps.generate import ensure_databases

QUESTIONS = json.loads(TRAP_QUESTIONS.read_text())


@pytest.fixture(scope="module")
def dbs(tmp_path_factory) -> dict[str, Path]:
    return ensure_databases(tmp_path_factory.mktemp("traps"))


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q["id"])
def test_trap_bites(q, dbs) -> None:
    _, gold = run_query(dbs[q["db"]], q["gold_sql"])
    assert gold, "gold answer is empty"
    if q["trap"] == "control":  # no fan-out risk: the straightforward query is the gold
        assert q["naive_sql"] is None
        return
    _, naive = run_query(dbs[q["db"]], q["naive_sql"])
    assert _normalize(gold) != _normalize(naive), "naive SQL reproduces the gold answer"
    for alt in q.get("alt_gold_sql", []):
        _, rows = run_query(dbs[q["db"]], alt)
        assert rows and _normalize(rows) not in (_normalize(gold), _normalize(naive))


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q["id"])
def test_question_is_documented(q) -> None:
    assert q["trap_explanation"].strip() and q["solution"].strip()
    assert bool(q.get("vague_reading", "").strip()) == q["id"].endswith("-v")


def test_docs_stay_out_of_the_prompt(tmp_path: Path) -> None:
    tasks = {t.id: t for t in load_traps(tmp_path)}
    for q in QUESTIONS:
        assert tasks[q["id"]].question == q["question"]
        assert q["solution"] not in repr(tasks[q["id"]])


def test_loader_and_scoring(tmp_path: Path) -> None:
    tasks = {t.id: t for t in load_traps(tmp_path)}
    assert len(tasks) == len(QUESTIONS)
    q = QUESTIONS[0]
    gold = score_traps(tasks[q["id"]], q["gold_sql"])
    assert gold["multiset_ex"] and gold["outcome"] == "correct" and "alt_match" not in gold
    naive = score_traps(tasks[q["id"]], q["naive_sql"])
    assert not naive["multiset_ex"] and naive["outcome"] == "trapped"
    assert score_traps(tasks[q["id"]], "SELECT 1")["outcome"] == "other"
    assert score_traps(tasks[q["id"]], "SELECT nope")["outcome"] == "failed"


def test_alternative_reading_scores_correct(tmp_path: Path) -> None:
    tasks = {t.id: t for t in load_traps(tmp_path)}
    q = next(q for q in QUESTIONS if q.get("alt_gold_sql"))
    res = score_traps(tasks[q["id"]], q["alt_gold_sql"][0])
    assert res["multiset_ex"] and res["alt_match"] == 1 and res["outcome"] == "correct"


def test_generation_is_deterministic(tmp_path: Path) -> None:
    a = ensure_databases(tmp_path / "a")
    b = ensure_databases(tmp_path / "b")
    for name in a:
        assert a[name].read_bytes() == b[name].read_bytes()
