"""Sanity-check a single prediction straight from the model, bypassing the API/UI.

Rebuilds the inference feature row for a fixture and runs the served model directly, so the
numbers here should match what the Streamlit dashboard shows. Usage:

    python scripts/verify_prediction.py "Liverpool FC" "Arsenal FC"
"""

from __future__ import annotations

import sys

from src.api.inference_features import build_inference_features, get_valid_teams
from src.models.rf import load_model, predict_proba
from src.models.save_artifacts import CHOSEN_MODEL_PATH


def main() -> None:
    home = sys.argv[1] if len(sys.argv) > 1 else "Liverpool FC"
    away = sys.argv[2] if len(sys.argv) > 2 else "Arsenal FC"

    # Confirm both teams are valid inference teams (same source of truth the API uses).
    valid = get_valid_teams()
    assert home in valid, f"{home!r} not in valid teams"
    assert away in valid, f"{away!r} not in valid teams"

    # Same path the API's /predict takes: build the one-row feature vector, load the served
    # model, predict, and derive implied odds as 1/probability.
    X = build_inference_features(home, away)
    model = load_model(CHOSEN_MODEL_PATH)
    p_home, p_draw, p_away = (float(p) for p in predict_proba(model, X)[0])

    print(f"Model file : {CHOSEN_MODEL_PATH.name}")
    print(f"Feature row: {X.shape[0]} row x {X.shape[1]} cols")
    print(f"Fixture    : {home} (home) vs {away} (away)\n")
    for label, p in [(f"{home} win", p_home), ("Draw", p_draw), (f"{away} win", p_away)]:
        odds = round(1.0 / p, 2) if p > 0 else None
        print(f"  {label:<22} {p * 100:5.1f}%   implied odds {odds}")
    print(f"\n  sum of probabilities = {p_home + p_draw + p_away:.6f}")


if __name__ == "__main__":
    main()
