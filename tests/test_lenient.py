from misqal.harness.lenient import normalize


def test_formatting_and_column_order_are_ignored() -> None:
    assert normalize([(1, "a")]) == normalize([("a", 1.0)])
    assert normalize([(0.333333,)]) == normalize([(0.33,)])


def test_bird_compares_sets_traps_multisets() -> None:
    assert normalize([(1,), (1,)], as_set=True) == normalize([(1,)], as_set=True)
    assert normalize([(1,), (1,)]) != normalize([(1,)])
