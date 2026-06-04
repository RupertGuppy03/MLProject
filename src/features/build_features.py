from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, PROCESSED_DIR
from src.features.elo import compute_elo_features
from src.features.rolling import (
    add_days_rest,
    add_extended_rolling_features,
    add_league_position,
    add_new_team_imputation,
    add_rolling_features,
    add_venue_splits,
    build_match_level_features,
    build_team_match_history,
)

# Canonical dataset produced by src/ingest/build_canonical.py
CANONICAL_PATH = PROCESSED_DIR / "matches_canonical.parquet"

# Where the locked feature schema is snapshotted
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"

# Columns that are identifiers, the label, or raw outcome data — never features.
# Keeping result/home_goals/away_goals out of X is what makes the matrix leakage-safe.
NON_FEATURE_COLUMNS = [
    "match_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "result",
    "home_goals",
    "away_goals",
]

# Columns carried alongside X to identify each row.
META_COLUMNS = ["match_id", "date", "season", "home_team", "away_team"]


def _build_rolling_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Run the full leakage-safe rolling chain and return one row per match."""
    history = build_team_match_history(matches)
    history = add_rolling_features(history)
    history = add_venue_splits(history)
    history = add_extended_rolling_features(history)
    history = add_days_rest(history)
    history = add_new_team_imputation(history)
    return build_match_level_features(history)


def _save_schema(X: pd.DataFrame, path: Path = SCHEMA_PATH) -> None:
    """Atomically snapshot the locked feature schema (columns + dtypes + count)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_columns": list(X.columns),
        "dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()},
        "n_features": X.shape[1],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def build_features(
    matches: pd.DataFrame | None = None,
    save_schema: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Assemble the leakage-safe (X, y, meta) matrix from the canonical dataset.

    Merges every feature source on match_id:
      - rolling team form / venue splits / extended rolling / days rest (src/features/rolling.py)
      - league position (add_league_position)
      - sequential Elo (compute_elo_features)

    Only played matches (known result) are kept, so y has no missing labels.
    When `matches` is None the canonical parquet is read from disk. Pass
    `save_schema=False` in tests to avoid writing over the artifacts schema.

    Returns:
        X    — feature matrix (no identifiers, label, or raw goals)
        y    — match result label (HW / D / AW)
        meta — match_id, date, season, home_team, away_team aligned row-for-row with X
    """
    if matches is None:
        if not CANONICAL_PATH.exists():
            raise FileNotFoundError(
                f"Canonical dataset not found at {CANONICAL_PATH}. Run build_canonical first."
            )
        matches = pd.read_parquet(CANONICAL_PATH)

    # Drop unplayed fixtures (no result) up front: they are future matches that never
    # contribute to any played match's history, and their NA goals would break the
    # rolling builders. This leaves only fully-played matches for every feature source.
    matches = matches[matches["result"].notna()].reset_index(drop=True)

    # Each feature source keys on match_id; build them independently then merge.
    base = _build_rolling_match_features(matches)

    position = add_league_position(matches)[
        [
            "match_id",
            "home_league_position",
            "away_league_position",
            "home_position_diff",
        ]
    ]

    elo, _ = compute_elo_features(matches)
    elo = elo[["match_id", "elo_home_pre", "elo_away_pre", "elo_diff"]]

    # Pull season + label back in from the canonical frame.
    labels = matches[["match_id", "season", "result"]]

    merged = (
        base.merge(position, on="match_id", how="inner")
        .merge(elo, on="match_id", how="inner")
        .merge(labels, on="match_id", how="inner")
    ).reset_index(drop=True)

    # Stable, deterministic feature ordering: everything that isn't an id/label/raw outcome.
    feature_columns = [c for c in merged.columns if c not in NON_FEATURE_COLUMNS]

    X = merged[feature_columns].copy()
    y = merged["result"].copy()
    meta = merged[META_COLUMNS].copy()

    # Leakage guard: outcome columns must never reach the model.
    leak = {"result", "home_goals", "away_goals"} & set(X.columns)
    assert not leak, f"Leakage columns present in X: {sorted(leak)}"

    if save_schema:
        _save_schema(X)

    return X, y, meta


def main() -> None:
    """Build features from the canonical dataset and write the schema snapshot."""
    X, y, meta = build_features()
    print(
        f"Built features: X={X.shape}, y={len(y)}, meta={meta.shape}. "
        f"Schema saved to {SCHEMA_PATH}."
    )


if __name__ == "__main__":
    main()
