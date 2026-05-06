import pandas as pd
from src.features.rolling import (
    build_team_match_history,
    add_rolling_features,
    add_venue_splits,
    add_extended_rolling_features,
    add_days_rest,
    add_new_team_imputation,
    build_match_level_features,
)

matches = pd.DataFrame(
    {
        "match_id": [1, 2, 3, 4, 5],
        "date": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"],
        "home_team": ["TeamA", "TeamB", "TeamA", "TeamC", "TeamA"],
        "away_team": ["TeamB", "TeamA", "TeamC", "TeamA", "TeamD"],
        "home_goals": [2, 1, 3, 2, 1],
        "away_goals": [0, 0, 1, 0, 1],
    }
)

for window_size in [2, 3, 5]:
    history = build_team_match_history(matches)
    history = add_rolling_features(history, window_size=window_size)
    history = add_venue_splits(history, window_size=window_size)
    history = add_extended_rolling_features(history, window_size=window_size)
    history = add_days_rest(history)
    history = add_new_team_imputation(history)
    features = build_match_level_features(history)

    print(f"\n--- window_size={window_size} ---")
    print(
        features[
            [
                "match_id",
                "home_team",
                "away_team",
                "home_rolling_win_rate",
                "away_rolling_win_rate",
            ]
        ].to_string(index=False)
    )
