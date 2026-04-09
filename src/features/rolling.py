from __future__ import annotations

import pandas as pd


def build_team_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Expected input columns:
        - match_id
        - date
        - home_team
        - away_team
        - home_goals
        - away_goals

    Output columns:
        - match_id
        - date
        - team
        - opponent
        - is_home
        - goals_for
        - goals_against
        - win
        - draw
        - loss
    """
    # required input columns
    required_columns = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]
    missing = [col for col in required_columns if col not in matches.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = matches.copy()

    # Ensure date is in datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Ensure they are ordred by date
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    # Create home team records
    home_rows = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "is_home": 1,
            "goals_for": df["home_goals"],
            "goals_against": df["away_goals"],
        }
    )

    # Create away team records
    away_rows = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "is_home": 0,
            "goals_for": df["away_goals"],
            "goals_against": df["home_goals"],
        }
    )

    # Combine home and away records
    history = pd.concat([home_rows, away_rows], ignore_index=True)

    # Calculate win/draw/loss
    history["win"] = (history["goals_for"] > history["goals_against"]).astype(int)
    history["draw"] = (history["goals_for"] == history["goals_against"]).astype(int)
    history["loss"] = (history["goals_for"] < history["goals_against"]).astype(int)

    # final ordering of columns
    history = history.sort_values(
        ["date", "match_id", "is_home"], ascending=[True, True, False]
    ).reset_index(drop=True)

    return history

def add_rolling_features(history: pd.DataFrame, window_size: int = 5,) -> pd.DataFrame:
    
    
    """
    Rolling features are computed using only PRIOR matches:
    - current match is excluded via shift(1)
    - only last `window_size` prior matches are used
    """
    
    df = history.copy()
    
    #ensure correct orderting before rolling calculations
    df = df.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    
    grouped = df.groupby("team", group_keys=False)
    
    # Exclude current match and use only last `window` matches for rolling calculations
    df["rolling_win_rate"] = grouped["win"].transform(
        lambda s: s.shift(1).rolling(window=window_size, min_periods=1).mean()
    )

    df["rolling_goals_for_avg"] = grouped["goals_for"].transform(
        lambda s: s.shift(1).rolling(window=window_size, min_periods=1).mean()
    )

    df["rolling_goals_against_avg"] = grouped["goals_against"].transform(
        lambda s: s.shift(1).rolling(window=window_size, min_periods=1).mean()
    )
    
    #restore original order
    df = df.sort_values(["date", "match_id", "is_home"], ascending=[True, True, False])
    df = df.reset_index(drop=True)
    
    return df

def add_venue_splits(history: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    
    """
    Add leakage-safe venue-split rolling features per team.

    These are computed separately from:
    - prior home matches only
    - prior away matches only

    Current match is excluded.
    """
    
    df = history.copy()
    
    # stable ordering for rolling calculations
    df = df.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    
    # doing the calculations separately for home and away matches to avoid leakage
    def compute_split(series_df: pd.DataFrame, home_flag: int) -> pd.Series:
        mask = series_df["is_home"] == home_flag
        values = series_df["win"].where(mask)
        
        return values.shift(1).rolling(window=window_size, min_periods=1).mean()
    
    df["rolling_win_rate_home"] = (
        df.groupby("team", group_keys=False)
        .apply(lambda g: compute_split(g, home_flag=1))
        .reset_index(level=0, drop=True)
    )

    df["rolling_win_rate_away"] = (
        df.groupby("team", group_keys=False)
        .apply(lambda g: compute_split(g, home_flag=0))
        .reset_index(level=0, drop=True)
    )
    
    # restore original order
    df = df.sort_values(["date", "match_id", "is_home"], ascending=[True, True, False])
    df = df.reset_index(drop=True)
    
    return df
    
def add_days_rest(history: pd.DataFrame) -> pd.DataFrame:
    
    """
    Add leakage-safe days_rest per team.

    days_rest = number of days since the team's previous match.
    First match for a team will have NaN for now.
    """
    
    df = history.copy()
    
    df = df.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    
    df["prev_match_date"] = df.groupby("team")["date"].shift(1)
    df["days_rest"] = (df["date"] - df["prev_match_date"]).dt.days
    
    df = df.drop(columns=["prev_match_date"])
    
    df = df.sort_values(["date", "match_id", "is_home"], ascending=[True, True, False]).reset_index(drop=True)
    df = df.reset_index(drop=True)
    
    return df


def add_new_team_impution(history: pd.DataFrame) -> pd.DataFrame:
    """
    Apply promoted/new-team imputation policy.

    - If a team has no prior history, mark is_new_team = 1
    - Fill rolling numeric NaNs with league-average for that column
    - Fill days_rest NaN with 7
    """
    
    df = history.copy()
    
    rolling_numeric_cols = [
        "rolling_win_rate",
        "rolling_goals_for_avg",
        "rolling_goals_against_avg",
        "rolling_win_rate_home",
        "rolling_win_rate_away",
    ]
    
    # a team is new if this row has no prior match 
    df["is_new_team"] = df["days_rest"].isna().astype(int)
    # fill rolling numeric features with league average for that column
    for col in rolling_numeric_cols:
        league_avg = df[col].mean()
        df[col] = df[col].fillna(league_avg)
        
        #fill missing days_rest with 7(assumption)
    df["days_rest"] = df["days_rest"].fillna(7)        
                     
    return df

def build_match_level_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-form team history back into one row per match,
    with separate home_ and away_ feature columns.
    """
    df = history.copy()

    feature_cols = [
        "match_id",
        "date",
        "team",
        "opponent",
        "is_home",
        "rolling_win_rate",
        "rolling_goals_for_avg",
        "rolling_goals_against_avg",
        "rolling_win_rate_home",
        "rolling_win_rate_away",
        "days_rest",
        "is_new_team",
    ]

    df = df[feature_cols].copy()

    home_df = df[df["is_home"] == 1].copy()
    away_df = df[df["is_home"] == 0].copy()

    home_df = home_df.rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "rolling_win_rate": "home_rolling_win_rate",
            "rolling_goals_for_avg": "home_rolling_goals_for_avg",
            "rolling_goals_against_avg": "home_rolling_goals_against_avg",
            "rolling_win_rate_home": "home_rolling_win_rate_home",
            "rolling_win_rate_away": "home_rolling_win_rate_away",
            "days_rest": "home_days_rest",
            "is_new_team": "home_is_new_team",
        }
    )

    away_df = away_df.rename(
        columns={
            "team": "away_team",
            "opponent": "home_team",
            "rolling_win_rate": "away_rolling_win_rate",
            "rolling_goals_for_avg": "away_rolling_goals_for_avg",
            "rolling_goals_against_avg": "away_rolling_goals_against_avg",
            "rolling_win_rate_home": "away_rolling_win_rate_home",
            "rolling_win_rate_away": "away_rolling_win_rate_away",
            "days_rest": "away_days_rest",
            "is_new_team": "away_is_new_team",
        }
    )

    home_keep = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_rolling_win_rate",
        "home_rolling_goals_for_avg",
        "home_rolling_goals_against_avg",
        "home_rolling_win_rate_home",
        "home_rolling_win_rate_away",
        "home_days_rest",
        "home_is_new_team",
    ]

    away_keep = [
        "match_id",
        "away_rolling_win_rate",
        "away_rolling_goals_for_avg",
        "away_rolling_goals_against_avg",
        "away_rolling_win_rate_home",
        "away_rolling_win_rate_away",
        "away_days_rest",
        "away_is_new_team",
    ]

    match_features = home_df[home_keep].merge(
        away_df[away_keep],
        on="match_id",
        how="inner",
    )

    match_features = match_features.sort_values(["date", "match_id"]).reset_index(drop=True)
    return match_features


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "date": ["2024-03-01", "2024-03-08", "2024-03-15", "2024-03-22"],
            "home_team": ["Arsenal", "Chelsea", "Arsenal", "Liverpool"],
            "away_team": ["Liverpool", "Arsenal", "Chelsea", "Arsenal"],
            "home_goals": [2, 1, 0, 2],
            "away_goals": [1, 3, 0, 1],
        }
    )

    history = build_team_match_history(sample)
    history = add_rolling_features(history, window_size=2)
    history = add_venue_splits(history, window_size=2)
    history = add_days_rest(history)
    history = add_new_team_impution(history)

    match_features = build_match_level_features(history)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(match_features.to_string(index=False))