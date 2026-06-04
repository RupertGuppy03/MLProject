import pandas as pd

from src.features.build_features import build_features


def make_canonical() -> pd.DataFrame:
    """A small canonical-shaped dataset spanning two seasons with one unplayed fixture."""
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4", "m5", "m6"],
            "date": [
                "2024-08-10",
                "2024-08-10",
                "2024-08-17",
                "2024-08-17",
                "2025-08-09",
                "2025-08-16",
            ],
            "season": [2024, 2024, 2024, 2024, 2025, 2025],
            "home_team": ["A", "C", "A", "D", "A", "B"],
            "away_team": ["B", "D", "C", "B", "B", "A"],
            "home_goals": [2, 1, 1, 0, 3, None],
            "away_goals": [0, 0, 0, 0, 1, None],
            "result": ["HW", "HW", "HW", "D", "HW", None],
        }
    )


class TestBuildFeatures:
    """Tests for the (X, y, meta) feature pipeline contract."""

    def test_returns_x_y_meta(self):
        """build_features returns three aligned outputs with required meta columns."""
        X, y, meta = build_features(make_canonical(), save_schema=False)

        assert len(X) == len(y), "X and y row counts must match"
        assert len(meta) == len(X), "meta and X row counts must match"
        assert "match_id" in meta.columns
        assert "date" in meta.columns

    def test_meta_aligned_with_x(self):
        """meta is aligned row-for-row with X (same index)."""
        X, y, meta = build_features(make_canonical(), save_schema=False)

        assert list(X.index) == list(meta.index)
        assert list(X.index) == list(y.index)

    def test_schema_stable(self):
        """Running twice on identical data yields identical columns in name and order."""
        X1, _, _ = build_features(make_canonical(), save_schema=False)
        X2, _, _ = build_features(make_canonical(), save_schema=False)

        assert list(X1.columns) == list(X2.columns)

    def test_no_leakage_columns(self):
        """result, home_goals, away_goals must never appear in X."""
        X, _, _ = build_features(make_canonical(), save_schema=False)

        for col in ["result", "home_goals", "away_goals"]:
            assert col not in X.columns, f"Leakage column present in X: {col}"

    def test_unplayed_dropped(self):
        """Fixtures with no result are excluded from X/y."""
        X, y, meta = build_features(make_canonical(), save_schema=False)

        # m6 has no result and must not appear.
        assert "m6" not in set(meta["match_id"])
        # 5 played matches out of 6 input rows.
        assert len(X) == 5
        assert not y.isna().any()

    def test_feature_columns_present(self):
        """Key features from each source (rolling, league position, Elo) appear in X."""
        X, _, _ = build_features(make_canonical(), save_schema=False)

        expected = [
            "home_rolling_win_rate",
            "away_rolling_win_rate",
            "home_league_position",
            "away_league_position",
            "home_position_diff",
            "elo_home_pre",
            "elo_away_pre",
            "elo_diff",
        ]
        for col in expected:
            assert col in X.columns, f"Missing expected feature: {col}"
