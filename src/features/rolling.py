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
