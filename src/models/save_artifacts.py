from __future__ import annotations

# One command to (re)generate every artifact the API needs to serve predictions:
#   - artifacts/chosen_model.pkl     <- the model we ship (tuned Random Forest = rf_tuned)
#   - artifacts/feature_schema.json  <- locked feature contract (columns, dtypes, preprocessing)
#   - artifacts/feature_importances.json
#   - artifacts/current_elo.json     <- latest Elo ratings, used to build inference features
#
# Run:  python -m src.models.save_artifacts
# This trains on the full leakage-safe feature matrix and overwrites the artifacts above.

from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR
from src.features.build_features import CANONICAL_PATH, _save_schema, build_features
from src.features.elo import compute_elo_features, save_state
from src.models.rf import (
    load_best_params,
    save_feature_importances,
    save_model,
    train_rf,
)

# The single, named served-model artifact the inference path loads.
CHOSEN_MODEL_PATH = ARTIFACTS_DIR / "chosen_model.pkl"


def save_final_artifacts(
    artifacts_dir: Path = ARTIFACTS_DIR, params: dict | None = None
) -> dict:
    """Train the chosen model on all data and write every served artifact under artifacts_dir.

    The chosen model is the tuned Random Forest (rf_tuned): same leakage-safe full-data fit
    used downstream, persisted under the canonical served name. `params` lets tests inject a
    small/fast forest; by default the tuned hyperparameters from best_params_rf.json are used.
    Returns a summary dict of the paths written and the training shape.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Full leakage-safe feature matrix (no schema side effect — we write it explicitly below).
    X, y, _ = build_features(save_schema=False)
    assert not X.isna().any().any(), "Feature matrix X contains NaN values."

    # Schema contract (columns + dtypes + preprocessing details) for the inference path.
    schema_path = artifacts_dir / "feature_schema.json"
    _save_schema(X, path=schema_path)

    # Chosen model = tuned RF on the full dataset.
    params = params if params is not None else load_best_params()
    model = train_rf(X, y, params=params)
    model_path = artifacts_dir / "chosen_model.pkl"
    save_model(model, path=model_path)

    importances_path = artifacts_dir / "feature_importances.json"
    save_feature_importances(model, list(X.columns), path=importances_path)

    # Latest Elo state — same walk + as_of_date logic as src/features/elo.py main().
    matches = pd.read_parquet(CANONICAL_PATH)
    elo_df, state = compute_elo_features(matches)
    played = elo_df[elo_df["result"].notna()]
    if played.empty:
        raise RuntimeError("No played matches in canonical dataset; cannot set as_of_date.")
    as_of_date = pd.to_datetime(played["date"]).max().date().isoformat()
    elo_path = artifacts_dir / "current_elo.json"
    save_state(state, elo_path, as_of_date=as_of_date)

    return {
        "chosen_model": model_path,
        "feature_schema": schema_path,
        "feature_importances": importances_path,
        "current_elo": elo_path,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "as_of_date": as_of_date,
    }


def main() -> None:
    """Regenerate all served artifacts and print a short summary."""
    summary = save_final_artifacts()
    print(
        f"Saved chosen model (tuned RF) on X=({summary['n_rows']}, {summary['n_features']}). "
        f"Artifacts written to {ARTIFACTS_DIR}/:"
    )
    for key in ("chosen_model", "feature_schema", "feature_importances", "current_elo"):
        print(f"  - {Path(summary[key]).name}")
    print(f"Elo as_of {summary['as_of_date']}.")


if __name__ == "__main__":
    main()
