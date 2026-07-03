"""Inference-time feature generation.

Turns a hypothetical fixture ``(home_team, away_team, as_of_date)`` into the exact
one-row feature matrix the trained model expects. The row is built by running the
*same* leakage-safe training pipeline (`build_features`) over the match history up to
the as-of date plus a single synthetic fixture row, then extracting that row. Reusing
the training chain is what guarantees the inference vector matches the locked schema
(`artifacts/feature_schema.json`) and stays leakage-safe — no feature math is
re-implemented here.

This module is pure (no web/HTTP). The `/predict` and `/teams` endpoints sit on top of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.build_features import CANONICAL_PATH, SCHEMA_PATH, build_features

# Sentinel id for the one synthetic fixture we append to the history slice. It is unique,
# so it never collides with a real match and is trivial to locate in the output.
_SYNTHETIC_MATCH_ID = "__inference_fixture__"


def _load_canonical(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the canonical dataset, reading from disk unless one is injected (tests)."""
    if matches is not None:
        return matches.copy()
    if not CANONICAL_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found at {CANONICAL_PATH}. Run build_canonical first."
        )
    return pd.read_parquet(CANONICAL_PATH)


def get_valid_teams(matches: pd.DataFrame | None = None) -> list[str]:
    """Sorted, de-duplicated list of every team in the dataset.

    Single source of truth for team names, shared by `/predict` validation and the
    `/teams` endpoint so they can never drift apart.
    """
    df = _load_canonical(matches)
    teams = pd.concat([df["home_team"], df["away_team"]]).dropna().astype(str).unique()
    return sorted(teams)


def _load_schema() -> tuple[list[str], dict[str, str]]:
    """Load the locked feature column order and dtypes from the schema artifact."""
    payload = json.loads(Path(SCHEMA_PATH).read_text())
    return payload["feature_columns"], payload["dtypes"]


def _align_to_schema(x_row: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns and cast dtypes to match the locked schema exactly.

    Raises if any schema column is missing or an unexpected column is present — a hard
    guard against the training/inference feature vectors ever drifting.
    """
    columns, dtypes = _load_schema()

    missing = [c for c in columns if c not in x_row.columns]
    extra = [c for c in x_row.columns if c not in columns]
    if missing or extra:
        raise ValueError(
            f"Inference features do not match schema. Missing: {missing}, Extra: {extra}"
        )

    x_row = x_row[columns].copy()
    for col, dtype in dtypes.items():
        x_row[col] = x_row[col].astype(dtype)  # type:ignore
    return x_row


def build_inference_features(
    home_team: str,
    away_team: str,
    as_of_date: str | pd.Timestamp | None = None,
    matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the one-row feature matrix for a home/away fixture as of a given date.

    Args:
        home_team: home side (must be a known team).
        away_team: away side (must be a known team).
        as_of_date: only matches on or before this date inform the features. Defaults to
            the latest match date in the dataset (i.e. "use everything we know").
        matches: optional canonical DataFrame to use instead of reading from disk (tests).

    Returns:
        A single-row DataFrame whose columns and dtypes match `feature_schema.json`.
    """
    df = _load_canonical(matches)

    # Validate teams against the shared source of truth before doing any work.
    valid_teams = get_valid_teams(df)
    unknown = [t for t in (home_team, away_team) if t not in valid_teams]
    if unknown:
        raise ValueError(f"Unknown team(s): {unknown}. Valid teams: {valid_teams}")
    if home_team == away_team:
        raise ValueError("home_team and away_team must be different.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Resolve the as-of date (default: latest available match date).
    as_of = df["date"].max() if as_of_date is None else pd.to_datetime(as_of_date)

    # History = every match on or before the as-of date. These are the only matches
    # allowed to inform the prediction (leakage-safe).
    history = df[df["date"] <= as_of].reset_index(drop=True)
    if history.empty:
        raise ValueError(f"No matches on or before as_of_date {as_of.date()}.")

    # Append one synthetic fixture, dated strictly after all history so it sorts last.
    # Its placeholder result/goals never feed its own features (rolling shift-1, Elo and
    # league position are snapshotted pre-match) and no real match follows it, so the
    # placeholder cannot leak into anything.
    synthetic = pd.DataFrame(
        [
            {
                "match_id": _SYNTHETIC_MATCH_ID,
                "date": history["date"].max() + pd.Timedelta(days=1),
                "season": int(history["season"].max()),
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": 0,
                "away_goals": 0,
                "result": "D",
            }
        ]
    )
    combined = pd.concat([history, synthetic], ignore_index=True)

    # Run the identical training pipeline; save_schema=False so we never touch the artifact.
    X, _, meta = build_features(combined, save_schema=False)

    mask = (meta["match_id"] == _SYNTHETIC_MATCH_ID).to_numpy()
    if mask.sum() != 1:
        raise RuntimeError(
            f"Expected exactly one synthetic inference row, found {int(mask.sum())}."
        )

    x_row = X[mask].reset_index(drop=True)
    return _align_to_schema(x_row)
