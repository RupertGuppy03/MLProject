import pandas as pd

from src.ingest.build_canonical import compute_result


def test_home_win_label():
    assert compute_result(2, 1) == "HW"


def test_draw_label():
    assert compute_result(1, 1) == "D"


def test_away_win_label():
    assert compute_result(0, 3) == "AW"


def test_missing_goals_is_none():
    assert compute_result(pd.NA, 1) is None
    assert compute_result(1, pd.NA) is None
