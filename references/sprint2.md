# Sprint 2 — Model Training and Evaluation

**Goal:** Train baseline and main model using leakage-safe features, evaluate with walk-forward backtesting, and produce log loss + Brier metrics plus calibration evidence. End sprint with saved artifacts. ML model is Random Forest.

---

## Stories

| Story | Status | Key files |
|---|---|---|
| Elo-only baseline probabilities | Done | `src/models/baseline_elo.py` |
| Logistic Regression baseline | Done | `src/models/log_reg.py` |
| Main model training (Random Forest) | To Do | `src/models/rf.py` |
| Walk-forward backtesting engine | Done | `src/backtest/walk_forward.py` |
| Probability calibration | To Do | `artifacts/model_calibrated.pkl` |
| Save final artifacts and schema contract | To Do | `artifacts/` |
| Hyperparameter tuning | To Do | `src/models/tune_rf.py` |
| Model comparison and selection | To Do | `reports/model_comparison.md` |
| Full pipeline integration test | To Do | `tests/test_pipeline_integration.py` |

---

## Elo-only baseline probabilities

**Status:** Done
**Labels:** Must Have, Sprint 2

**User Story:**
As a user, I want an Elo-only baseline so I can benchmark the ML model against a strong simple approach.

**Acceptance Tests:**

- **Acc Test 1: Elo baseline produces probabilities**
  - Given Elo features exist for a match
  - When Elo baseline predicts outcome probabilities
  - Then it outputs `p_home`, `p_draw`, `p_away`
  - And probabilities sum to 1 within tolerance
- **Acc Test 2: Elo baseline runs across dataset**
  - Given the canonical dataset exists
  - When I run the Elo baseline on all matches
  - Then predictions are produced for all rows without crashing

**Definition of Done:**

- Implemented: `src/models/baseline_elo.py`
- Produces predictions file compatible with the backtest pipeline
- Included in model comparison report

---

## Logistic Regression baseline (multinomial)

**Status:** Done
**Labels:** Must Have, Sprint 2

**User Story:**
As a user, I want a multinomial Logistic Regression baseline so I can establish a fast, interpretable ML benchmark.

**Acceptance Tests:**

- **Acc Test 1: Logistic regression trains successfully**
  - Given `X` and `y` from build_features
  - When I train multinomial logistic regression
  - Then training completes without errors
- **Acc Test 2: Model predicts probabilities**
  - Given a trained logistic regression model
  - When I call predict_proba on a sample
  - Then it returns probabilities for 3 classes summing to 1

**Definition of Done:**

- Implemented: `src/models/log_reg.py`
- Included in backtest comparison
- Model can be saved and loaded with joblib

---

## Main model training (Random Forest Classifier)

**Status:** To Do
**Labels:** Must Have, Sprint 2

**User Story:**
As a user, I want to train a Random Forest classifier as the main model so I can generate competitive, calibrated match outcome probabilities with explainability via built-in feature importances.

**Acceptance Tests:**

- **Acc Test 1: Random Forest trains successfully**
  - Given `X` and `y` from `build_features()`
  - When I train a `RandomForestClassifier` with `n_estimators=200`
  - Then training completes without errors and the model exposes `.feature_importances_`
- **Acc Test 2: Model produces valid probabilities**
  - Given a trained `RandomForestClassifier`
  - When I call `predict_proba` on a sample row
  - Then it returns an array with 3 columns (H, D, A) summing to 1.0 within 1e-6
- **Acc Test 3: No NaN values in model input**
  - Given the imputation policy is applied
  - When `X` is passed to the model
  - Then `X` contains no NaN values
- **Acc Test 4: Feature importances are saved**
  - Given the model is trained
  - Then `artifacts/feature_importances.json` contains feature names and scores

**Definition of Done:**

- Implemented: `src/models/rf.py` with `train_rf(X, y, params=None)`
- Default params: `n_estimators=200`, `max_depth=None`, `min_samples_leaf=5`, `random_state=42`, `n_jobs=-1`
- Model saved: `artifacts/model_rf.pkl`
- Feature importances saved: `artifacts/feature_importances.json`
- Included in walk-forward backtest comparison

---

## Walk-forward backtesting engine

**Status:** Done
**Labels:** Must Have, Sprint 2

**User Story:**
As a user, I want walk-forward backtesting so model performance reflects real-time prediction conditions.

**Acceptance Tests:**

- **Acc Test 1: Backtest uses only past data**
  - Given matches are ordered by date
  - When a fold trains on data up to time T
  - Then the test set includes only matches after T and no test rows appear in training
- **Acc Test 2: Backtest generates metrics**
  - Given predictions exist for each fold
  - When metrics are computed
  - Then log loss and Brier score are reported per fold and overall
- **Acc Test 3: Backtest artifacts saved**
  - Given backtest completes
  - Then `reports/predictions.parquet` and `reports/metrics.csv` are written

**Definition of Done:**

- Implemented: `src/backtest/walk_forward.py`
- Produces saved metrics and predictions files
- Summary table included in `reports/README.md`

---

## Probability calibration

**Status:** To Do
**Labels:** Must Have, Sprint 2

**User Story:**
As a user, I want calibrated probabilities so the implied odds derived from model output are meaningful and stable.

**Acceptance Tests:**

- **Acc Test 1: Calibration improves Brier score**
  - Given an uncalibrated model and a calibration method
  - When calibration is applied without leakage
  - Then Brier score improves or does not degrade beyond 2%
- **Acc Test 2: Calibration curve is produced**
  - Given calibrated predictions
  - Then a calibration plot is saved to `reports/calibration_curve.png`

**Definition of Done:**

- Calibrated artifact saved: `artifacts/model_calibrated.pkl`
- Calibration plot saved to `reports/calibration_curve.png`
- Calibration method documented in `reports/README.md`

---

## Save final artifacts and schema contract

**Status:** To Do
**Labels:** Must Have, Sprint 2

**User Story:**
As a user, I want saved model artifacts and a feature schema so API inference is consistent and reproducible.

**Acceptance Tests:**

- **Acc Test 1: Artifacts saved**
  - Given a best model is selected
  - When I run the training pipeline
  - Then `artifacts/model.pkl` and `artifacts/feature_schema.json` exist
- **Acc Test 2: Artifacts can be loaded and used**
  - Given the artifact files exist
  - When I load them and run a sample prediction
  - Then I get valid probability outputs

**Definition of Done:**

- Artifacts saved and loadable
- Feature schema includes column names and preprocessing details
- One command documented to retrain and regenerate all artifacts

---

## Hyperparameter tuning for Random Forest

**Status:** To Do
**Labels:** Must Have, Sprint 2

**User Story:**
As a user, I want to tune the Random Forest hyperparameters using time-series-safe cross-validation so I can improve probabilistic performance without data leakage.

**Acceptance Tests:**

- **Acc Test 1: Tuning uses TimeSeriesSplit**
  - Given `X` and `y` are time-ordered
  - When `RandomizedSearchCV` is initialized
  - Then its `cv` parameter is `TimeSeriesSplit(n_splits=5)` with no future data in any fold
- **Acc Test 2: Search space is defined**
  - Given the tuning script runs
  - When I inspect `param_distributions`
  - Then it includes `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- **Acc Test 3: Best parameters are saved**
  - Given `RandomizedSearchCV` completes
  - Then `artifacts/best_params_rf.json` is written and the tuned model is used downstream
- **Acc Test 4: Tuning improves or maintains log loss**
  - Given default RF walk-forward log loss is available
  - Then tuned log loss is equal or better, or within 2% tolerance

**Definition of Done:**

- Script exists: `src/models/tune_rf.py`
- Uses `RandomizedSearchCV` with `TimeSeriesSplit(n_splits=5)`
- Search space: `n_estimators [100,200,300,500]`, `max_depth [None,10,20,30]`, `min_samples_leaf [1,3,5,10]`, `max_features ['sqrt','log2',0.5]`, `n_iter=50`, `scoring='neg_log_loss'`, `random_state=42`
- Best params saved: `artifacts/best_params_rf.json`
- Tuned model becomes the default used downstream

---

## Model comparison and selection

**Status:** To Do
**Labels:** Must Have, Sprint 2

**User Story:**
As a user, I want a clear comparison of all models so I can justify why the Random Forest was selected as the final model.

**Acceptance Tests:**

- **Acc Test 1: All models are compared on the same backtest**
  - Given walk-forward predictions exist for Elo baseline, LogReg, Default RF, and Tuned RF
  - When the comparison report is produced
  - Then log loss and Brier score are shown for all four models side by side
- **Acc Test 2: A winner is selected and justified**
  - Given the comparison report exists
  - When I read `reports/model_comparison.md`
  - Then it names the selected model and gives a written justification

**Definition of Done:**

- `reports/model_comparison.md` includes all four models with log loss and Brier score
- A written justification explains why the final model was chosen
- The selected model artifact is clearly identified

---

## Full pipeline integration test

**Status:** To Do
**Labels:** Must Have, Sprint 2

**User Story:**
As a developer, I want a single end-to-end integration test so I can catch bugs that only appear when the full pipeline composes together.

**Acceptance Tests:**

- **Acc Test 1: Pipeline runs end to end without error**
  - Given raw season files exist in `data/raw/`
  - When I run the full pipeline (ingest → canonical → features → build_features)
  - Then it completes without errors
- **Acc Test 2: Output shape and columns are correct**
  - Given the pipeline has completed
  - When I inspect `X`, `y`, and `meta`
  - Then all expected feature columns are present
  - And row counts are consistent across `X`, `y`, and `meta`
  - And `X` contains no NaN values
  - And `result`, `home_goals`, `away_goals` are not in `X`

**Definition of Done:**

- Test exists: `tests/test_pipeline_integration.py`
- Runs against real data files from `data/raw/`
- Covers ingest → canonical → features → X/y/meta in one test
- Passes cleanly with `pytest tests/test_pipeline_integration.py`