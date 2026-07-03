# ML Project — User Story Library

> Master source of truth for all user stories, acceptance tests, and Definitions of Done.
> Per-sprint files are in references/sprint1..4/ for quick lookup during a session.

## MoSCoW Labels
- **Must Have**, **Should Have**, **Could Have**, **Won't Have**

---

# Sprint 1 — Data Pipelines

**Goal:** Build a reproducible, leakage-safe data pipeline from Football-Data → canonical dataset → feature matrix (`X, y`) that can train models reliably. This sprint ends with proof that no future games leak into any row's features.

---

## Implement sequential Elo features (pre-match only)

**Labels:** Must Have, Sprint 1 — **DONE**

**User Story:**
As a user, I want Elo ratings updated sequentially so the model has a robust strength signal that is naturally leakage-safe.

**Acceptance Tests:**

- **Acc Test 1: Elo pre-match ratings use only prior matches**
  - Given matches are processed in chronological order
  - When Elo for match M is computed
  - Then `elo_home_pre` and `elo_away_pre` do not depend on the outcome of match M
  - And Elo updates occur only after computing features for match M
- **Acc Test 2: Elo difference feature is present**
  - Given Elo ratings are computed
  - When features are returned
  - Then `elo_diff = elo_home_pre - elo_away_pre` exists

**Definition of Done:**

- Module exists: `src/features/elo.py`
- Elo logic is exposed as a reusable function usable at both training and inference time — no duplicate implementation anywhere in the codebase
- Elo parameters locked:
  - K-factor: **20**
  - Home advantage: **+60 Elo points** added to home rating before expected score
  - Starting Elo for new/promoted teams: **1450**
  - Between-season regression: all teams regress **25% toward 1500** at season start
- At the end of the Elo walk, final ratings per team are written to `artifacts/current_elo.json` with an `as_of_date` field
- Unit test passes: `tests/test_elo.py` covers leakage safety and parameter behavior (K-factor, home advantage offset, promoted-team starting value, season regression)

**Refinement (post-Sprint-1):** added margin-of-victory K-scaling (World Football Elo goal-difference index) so bigger wins move ratings more. Locked params unchanged; draws/1-goal wins behave as before.

---

## Feature pipeline outputs X, y, meta with schema contract

**Labels:** Must Have, Sprint 1 — **DONE**

**User Story:**
As a user, I want a consistent `(X, y, meta)` output so modeling and backtesting can run without manual fixes.

**Acceptance Tests:**

- **Acc Test 1: build_features returns required outputs**
  - Given the canonical dataset exists
  - When I call build_features
  - Then it returns `X`, `y`, and `meta`
  - And `X` row count equals `y` row count
  - And `meta` includes `match_id` and `date` aligned with `X`
- **Acc Test 2: Feature schema is stable**
  - Given I run build_features twice on identical data
  - When I compare `X.columns`
  - Then the columns are identical in name and order
- **Acc Test 3: No leakage in X**
  - Given build_features has run
  - When I inspect the columns of `X`
  - Then `result`, `home_goals`, and `away_goals` are not present in `X`

**Definition of Done:**

- `src/features/build_features.py` exists with a `build_features()` function
- Outputs `(X, y, meta)` with stable schema
- Schema snapshot saved: `artifacts/feature_schema.json`
- `X` explicitly excludes `result`, `home_goals`, and `away_goals` — enforced by an assert

---

## League position rolling feature

**Labels:** Should Have, Sprint 1 — **DONE**

**User Story:**
As a developer, I want a rolling league position feature computed from season-start so the model has a dynamic strength signal that updates after every match.

**Acceptance Tests:**

- **Acc Test 1: League position computed from season start**
  - Given matches for a season exist in the canonical dataset
  - When league position is computed for match M
  - Then the position reflects cumulative points up to but NOT including match M
  - And GW1 position defaults to a mid-table starting value (e.g. 10th) for all teams
- **Acc Test 2: Position resets each season**
  - Given the dataset spans multiple seasons
  - When league position is computed
  - Then positions reset to default at the start of each new season
- **Acc Test 3: Feature appears for both teams**
  - When features are returned
  - Then `home_league_position` and `away_league_position` both exist
  - And `home_position_diff = away_league_position - home_league_position` also exists

**Definition of Done:**

- Implemented in `src/features/rolling.py` alongside other rolling stats
- Columns added: `home_league_position`, `away_league_position`, `home_position_diff`
- Season-boundary reset tested: `tests/test_league_position.py`
- Columns added to `artifacts/feature_schema.json`

---

## Data quality validation on ingest

**Labels:** Must Have, Sprint 1 — **DONE**

**User Story:**
As a user, I want raw match data validated on ingest so postponed, abandoned, or malformed matches don't silently corrupt training data, and known anomalies (e.g. the 2019/20 COVID gap) are surfaced early.

**Acceptance Tests:**

- **Acc Test 1: Schema validation runs on ingest**
  - Given a raw season pull is loaded
  - When the validation step runs
  - Then required columns (`date`, `home_team`, `away_team`, `home_goals`, `away_goals`) are checked for presence and dtype
  - And rows missing required values are flagged
- **Acc Test 2: Malformed rows are quarantined, not silently dropped**
  - Given a raw season contains rows with NaN goals
  - When validation runs
  - Then those rows are removed from the canonical dataset
  - And a count is logged: `"Dropped N rows due to missing goals"`
  - And dropped rows are saved to `data/quarantine/dropped_<season>.parquet`
- **Acc Test 3: Validation report is generated**
  - Given the dataset spans multiple seasons including 2019/20
  - When ingestion completes
  - Then a validation report is written to `data/processed/validation_report.json` containing: rows-per-season counts, dropped-row counts, date gaps > 14 days, and unknown team-name flags

**Definition of Done:**

- Module exists: `src/ingest/validate.py`
- Called automatically as part of `build_canonical.py`
- Validation report saved: `data/processed/validation_report.json`
- Quarantine directory created with dropped rows for inspection
- Test exists: `tests/test_ingest_validation.py` covering NaN goal rows, missing columns, and the COVID gap

---

# Sprint 2 — Model Training and Evaluation

**Goal:** Train baseline and main model using leakage-safe features, evaluate with walk-forward backtesting, and produce metrics for probabilities (log loss + Brier) plus calibration evidence. End sprint with saved artifacts. ML model is Random Forest.

---

## Elo-only baseline probabilities

**Labels:** Must Have, Sprint 2 — **DONE**

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

- Baseline implemented: `src/models/baseline_elo.py`
- Produces a predictions file compatible with the backtest pipeline
- Included in model comparison report

---

## Logistic Regression baseline (multinomial)

**Labels:** Must Have, Sprint 2 — **DONE**

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
  - Then it returns probabilities for 3 classes
  - And each row sums to 1 within tolerance

**Definition of Done:**

- Implemented: `src/models/log_reg.py`
- Included in backtest comparison
- Model can be saved and loaded with joblib

---

## Main model training (Random Forest Classifier)

**Labels:** Must Have, Sprint 2 — **DONE**

**User Story:**
As a user, I want to train a Random Forest classifier as the main model so I can generate competitive, calibrated match outcome probabilities with explainability via built-in feature importances.

**Acceptance Tests:**

- **Acc Test 1: Random Forest trains successfully**
  - Given `X` and `y` from `build_features()`
  - When I train a `RandomForestClassifier` with `n_estimators=200`
  - Then training completes without errors
  - And the trained model exposes `.feature_importances_`
- **Acc Test 2: Model produces valid probabilities**
  - Given a trained `RandomForestClassifier`
  - When I call `predict_proba` on a sample row
  - Then it returns an array with 3 columns corresponding to (H, D, A)
  - And each row sums to 1.0 within a tolerance of 1e-6
- **Acc Test 3: No NaN values in model input**
  - Given the imputation policy is applied
  - When `X` is passed to the model
  - Then `X` contains no NaN values
- **Acc Test 4: Feature importances are saved**
  - Given the model is trained
  - When artifacts are saved
  - Then `artifacts/feature_importances.json` contains feature column names and importance scores

**Definition of Done:**

- Implemented: `src/models/rf.py` with a `train_rf(X, y, params=None)` function
- Default params: `n_estimators=200`, `max_depth=None`, `min_samples_leaf=5`, `random_state=42`, `n_jobs=-1`
- Model saved: `artifacts/model_rf.pkl`
- Feature importances saved: `artifacts/feature_importances.json`
- Included in walk-forward backtest comparison

---

## Walk-forward backtesting engine

**Labels:** Must Have, Sprint 2 — **DONE**

**User Story:**
As a user, I want walk-forward backtesting so model performance reflects real-time prediction conditions.

**Acceptance Tests:**

- **Acc Test 1: Backtest uses only past data**
  - Given matches are ordered by date
  - When a fold trains on data up to time T
  - Then the test set includes only matches after time T
  - And no test rows appear in the training set
- **Acc Test 2: Backtest generates metrics**
  - Given predictions exist for each fold
  - When metrics are computed
  - Then log loss and Brier score are reported per fold and overall
- **Acc Test 3: Backtest artifacts saved**
  - Given backtest completes
  - When outputs are written
  - Then predictions are saved to `reports/predictions.parquet`
  - And metrics are saved to `reports/metrics.csv`

**Definition of Done:**

- Implemented: `src/backtest/walk_forward.py`
- Produces saved metrics and predictions files
- Summary table included in `reports/README.md`

---

## Probability calibration

**Labels:** Must Have, Sprint 2 — **DONE**

**User Story:**
As a user, I want calibrated probabilities so the implied odds derived from model output are meaningful and stable.

**Acceptance Tests:**

- **Acc Test 1: Calibration improves Brier score**
  - Given an uncalibrated model and a calibration method
  - When calibration is applied without leakage
  - Then Brier score improves or does not degrade beyond 2%
- **Acc Test 2: Calibration curve is produced**
  - Given calibrated predictions
  - When I generate calibration plots
  - Then a calibration plot is saved to `reports/calibration_curve.png`

**Definition of Done:**

- Calibrated artifact saved: `artifacts/model_calibrated.pkl`
- Calibration plot saved to `reports/calibration_curve.png`
- Calibration method documented in `reports/README.md`

---

## Save final artifacts and schema contract

**Labels:** Must Have, Sprint 2 — **DONE**

**User Story:**
As a user, I want saved model artifacts and a feature schema so API inference is consistent and reproducible.

**Acceptance Tests:**

- **Acc Test 1: Artifacts saved**
  - Given a best model is selected
  - When I run the training pipeline
  - Then `artifacts/chosen_model.pkl` exists
  - And `artifacts/feature_schema.json` exists
- **Acc Test 2: Artifacts can be loaded and used**
  - Given the artifact files exist
  - When I load them and run a sample prediction
  - Then I get valid probability outputs

**Definition of Done:**

- Artifacts saved and loadable
- Feature schema includes column names and any preprocessing details
- One command documented to retrain and regenerate all artifacts

---

## Hyperparameter tuning for Random Forest

**Labels:** Must Have, Sprint 2 — **DONE**

**User Story:**
As a user, I want to tune the Random Forest hyperparameters using time-series-safe cross-validation so I can improve probabilistic performance without data leakage or overfitting.

**Acceptance Tests:**

- **Acc Test 1: Tuning uses TimeSeriesSplit**
  - Given `X` and `y` are time-ordered by match date
  - When `RandomizedSearchCV` is initialized
  - Then its `cv` parameter is `TimeSeriesSplit(n_splits=5)`
  - And no future data appears in any fold's training set
- **Acc Test 2: Search space is defined**
  - Given the tuning script is executed
  - When I inspect `param_distributions`
  - Then it includes at minimum: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- **Acc Test 3: Best parameters are saved**
  - Given `RandomizedSearchCV` completes
  - When `best_params_` is produced
  - Then it is saved to `artifacts/best_params_rf.json`
  - And the tuned model is used for all downstream artifacts
- **Acc Test 4: Tuning improves or maintains log loss**
  - Given the default RF walk-forward log loss is available
  - When the tuned RF log loss is computed
  - Then tuned log loss is less than or equal to default, or within 2% tolerance

**Definition of Done:**

- Script exists: `src/models/tune_rf.py`
- Uses `RandomizedSearchCV` with `TimeSeriesSplit(n_splits=5)`
- Search space: `n_estimators: [100, 200, 300, 500]`, `max_depth: [None, 10, 20, 30]`, `min_samples_leaf: [1, 3, 5, 10]`, `max_features: ['sqrt', 'log2', 0.5]`, `n_iter=50`, `scoring='neg_log_loss'`, `random_state=42`
- Best params saved: `artifacts/best_params_rf.json`
- Tuned model becomes the default used downstream

---

## Model comparison and selection

**Labels:** Must Have, Sprint 2

**User Story:**
As a user, I want a clear comparison of all models so I can justify why the Random Forest was selected as the final model.

**Acceptance Tests:**

- **Acc Test 1: All models are compared on the same backtest**
  - Given walk-forward predictions exist for Elo baseline, Logistic Regression, Default RF, and Tuned RF
  - When the comparison report is produced
  - Then log loss and Brier score are reported for all four models side by side
- **Acc Test 2: A winner is selected and justified**
  - Given the comparison report exists
  - When I read `reports/model_comparison.md`
  - Then it names the selected model and gives a written justification for the choice

**Definition of Done:**

- `reports/model_comparison.md` exists and includes all four models: Elo baseline, LogReg, Default RF, Tuned RF
- Each model has log loss and Brier score from the walk-forward backtest
- A written justification explains why the final model was selected
- The selected model artifact is clearly identified in the report

---

## Full pipeline integration test

**Labels:** Must Have, Sprint 2 — **DONE**

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
- Runs against real data files (not mocks) from `data/raw/`
- Covers ingest → canonical → features → X/y/meta in one test
- Passes cleanly with `pytest tests/test_pipeline_integration.py`

---

# Sprint 3 — Web App and UI

**Goal:** Deliver a working product. User selects home/away teams, gets probabilities, implied odds, and match context charts. Streamlit calls FastAPI and works from another device on the local network.

---

## FastAPI predict endpoint

**Labels:** Must Have, Sprint 3 — **DONE**

**User Story:**
As a user, I want a `/predict` endpoint so the system can return match outcome probabilities and implied odds for selected teams.

**Acceptance Tests:**

- **Acc Test 1: Predict returns probabilities**
  - Given the API is running
  - When I POST `/predict` with a valid `home_team` and `away_team`
  - Then the response includes `p_home`, `p_draw`, `p_away`
  - And probabilities sum to 1 within tolerance
- **Acc Test 2: Predict returns implied odds**
  - Given valid probabilities are produced
  - When the response is returned
  - Then it includes implied odds for each outcome
  - And odds equal approximately `1 / probability`
- **Acc Test 3: Invalid team is rejected**
  - Given the API is running
  - When I POST `/predict` with an unknown team name
  - Then the API returns status code 400
  - And returns a helpful error message listing valid teams

**Definition of Done:**

- Endpoint implemented in `src/api/main.py`
- Pydantic request/response models defined
- `/docs` shows correct schema
- `GET /health` returns 200 with a simple status payload (used by deployment health checks)
- `GET /metadata` returns `last_updated_date` and `data_through_date` from `artifacts/last_updated.json`
- Smoke test script exists: `scripts/smoke_api.sh`

---

## Inference feature builder

**Labels:** Must Have, Sprint 3

**User Story:**
As a user, I want inference-time feature generation that uses only historical data up to an as-of date so predictions are consistent with training.

**Acceptance Tests:**

- **Acc Test 1: Default as-of date uses latest available**
  - Given the canonical dataset exists
  - When I call predict without providing `as_of_date`
  - Then the system uses the latest match date in the dataset
- **Acc Test 2: Inference features match schema**
  - Given `feature_schema.json` exists
  - When inference features are built for a home/away team
  - Then the resulting feature vector matches the schema exactly

**Definition of Done:**

- `src/api/inference_features.py` exists
- Builds exactly one-row `X` using canonical dataset and locked schema
- Unit tests verify schema alignment: `tests/test_inference_schema.py`

---

## Streamlit dashboard

**Labels:** Must Have, Sprint 3 — **DONE**

**User Story:**
As a user, I want a Streamlit dashboard so I can select teams and instantly view probabilities, implied odds, and the predicted outcome.

**Acceptance Tests:**

- **Acc Test 1: UI prevents invalid selection**
  - Given the Streamlit app is open
  - When I select the same team for home and away
  - Then the app prevents prediction and shows an error message
- **Acc Test 2: UI displays prediction outputs**
  - Given the API is running and reachable
  - When I choose valid home and away teams and click Predict
  - Then the app displays `p_home`, `p_draw`, `p_away`
  - And displays implied odds
  - And displays the predicted outcome label

**Definition of Done:**

- `src/ui/app.py` exists
- UI works against local FastAPI
- README has screenshots of the dashboard

---

## Match context and SHAP explanation panel

**Labels:** Should Have, Sprint 3

**User Story:**
As a user, I want a rich explanation panel showing match context (Elo trajectory, rolling form, head-to-head, home/away splits) plus per-prediction SHAP values, so I can understand why the model favoured this specific outcome. This story also ensures every chart in the UI mock has a backend data source — nothing is faked or hardcoded.

**Acceptance Tests:**

- **Acc Test 1: Per-match SHAP values are computed**
  - Given a trained Random Forest model and a single prediction request
  - When the explanation logic runs
  - Then SHAP values are computed for that specific prediction
  - And the top 5 features driving this prediction are returned with signed contribution values
- **Acc Test 2: API returns rich match context**
  - Given a `/predict` request for two teams
  - When the response is built
  - Then it includes: current Elo for both teams, last-10 GW Elo trajectory, last-5 form sequence, rolling 5-match goals scored and conceded, home/away venue splits, head-to-head record, PPG and league position, and SHAP top contributors
- **Acc Test 3: Streamlit renders all context fields**
  - Given the API returns rich context
  - When the dashboard receives a prediction
  - Then it renders: probability cards, Elo trajectory chart, rolling goals chart, home/away split bars, H2H record, team profile radar, model confidence bar, and SHAP top-5 panel
  - And no chart pulls from hardcoded data

**Definition of Done:**

- SHAP integration: `src/api/explain.py` uses `shap.TreeExplainer` on the underlying RF — not the calibrated wrapper. The API serves calibrated probabilities to the user and pre-calibration SHAP values to the explainer. These must not be conflated.
- `src/api/match_context.py` builds H2H, Elo trajectory, rolling goals, and venue splits at inference time using only data prior to the as-of date
- Elo trajectory and current Elo are read from `src/features/elo.py` — not recomputed independently in the API
- API endpoint returns the full context dictionary
- Streamlit dashboard matches the UI mock in `references/project_UI_example.html`
- Integration test verifies the API response contains every field the UI consumes: `tests/test_api_context_schema.py`

---

## Local network access (CORS + configurable API URL)

**Labels:** Must Have, Sprint 3

**User Story:**
As a user, I want the FastAPI service and Streamlit dashboard reachable from another device on the same local network, so I can test the app from my phone or a second laptop.

**Acceptance Tests:**

- **Acc Test 1: API reachable over the LAN**
  - Given FastAPI is bound to `0.0.0.0`
  - When another device on the network requests `/predict` at the host's LAN IP
  - Then it returns a valid response
- **Acc Test 2: CORS allows the Streamlit origin**
  - Given Streamlit is served from a different origin
  - When it calls the API from the browser
  - Then the response carries `Access-Control-Allow-Origin` and the call is not blocked
- **Acc Test 3: API base URL is configurable**
  - Given an `API_BASE_URL` env var/secret
  - When Streamlit starts
  - Then it calls that URL (default `http://localhost:8000`) with no hardcoded host

**Definition of Done:**

- FastAPI launched on `0.0.0.0` with a documented `uvicorn` command
- CORS middleware configured in `src/api/main.py` (allowed origins via env; permissive default for LAN/dev)
- Streamlit (`src/ui/app.py`) reads `API_BASE_URL` from env/secrets, defaulting to localhost
- README documents running both on the LAN (find host IP, set `API_BASE_URL`)

---

# Sprint 4 — Deployment

**Goal:** Deploy a public, device-accessible web app. Streamlit UI hosted publicly, FastAPI model API deployed on AWS using Docker. End result is a public Streamlit URL calling a public AWS API URL reliably.

---

## Dockerize the FastAPI prediction service

**Labels:** Must Have, Sprint 4

**User Story:**
As a user, I want the FastAPI prediction service containerized so the API runs consistently locally and in AWS.

**Acceptance Tests:**

- **Acc Test 1: API Docker image builds**
  - Given a Dockerfile exists for the API service
  - When I run `docker build`
  - Then the image builds successfully without errors
- **Acc Test 2: API container starts and serves endpoints**
  - Given the API Docker image is built
  - When I run the container
  - Then `/docs` loads successfully
  - And `/health` returns status code 200
- **Acc Test 3: Model artifacts load successfully inside container**
  - Given artifacts are included in the container
  - When the API starts
  - Then POST `/predict` returns probabilities without errors
- **Acc Test 4: Missing artifacts fail fast with a clear error**
  - Given the container is started without required artifacts
  - When the API boots
  - Then it exits with a clear message naming the missing artifact

**Definition of Done:**

- API Dockerfile exists: `src/api/Dockerfile`
- Container starts with uvicorn binding to `0.0.0.0`
- `/health` endpoint implemented
- README includes local run commands: `docker build`, `docker run`, example `curl` to `/predict`

---

## Push the API Docker image to AWS ECR

**Labels:** Must Have, Sprint 4

**User Story:**
As a user, I want to push the API Docker image to AWS ECR with a minimal image size and lifecycle policy so storage costs stay low and old images don't accumulate.

**Acceptance Tests:**

- **Acc Test 1: ECR repository exists**
  - Given AWS credentials are configured
  - When I run the ECR setup commands
  - Then an ECR repository for the API exists
- **Acc Test 2: Docker image uses a slim base**
  - Given the Dockerfile is implemented
  - When I inspect the FROM statement
  - Then it uses `python:3.11-slim`
- **Acc Test 3: Final image size is under threshold**
  - Given the API Docker image is built
  - When I check the image size
  - Then it is less than 600 MB
- **Acc Test 4: ECR lifecycle policy is applied**
  - Given the ECR repository is created
  - When I apply the lifecycle policy
  - Then only the 3 most recent tagged images are retained
  - And untagged images expire after 1 day
- **Acc Test 5: Image is tagged and pushed**
  - Given the image is built locally
  - When I tag and push it
  - Then it appears in ECR with the correct tag

**Definition of Done:**

- Dockerfile uses `python:3.11-slim`
- `.dockerignore` excludes: `data/`, `notebooks/`, `reports/`, `.git/`, `tests/`, `__pycache__/`, `*.parquet`
- Final image size verified under 600 MB
- ECR repo created with lifecycle policy: keep 3 most recent tagged, expire untagged after 1 day
- README includes ECR auth and push steps

---

## Deploy the API to AWS App Runner

**Labels:** Must Have, Sprint 4

**User Story:**
As a user, I want to deploy the FastAPI container to AWS App Runner with auto-pause so the API stays publicly accessible at minimal cost when idle.

**Acceptance Tests:**

- **Acc Test 1: App Runner uses minimum instance size**
  - Given the App Runner service is configured
  - When I inspect the configuration
  - Then instance CPU is 0.25 vCPU and memory is 0.5 GB
- **Acc Test 2: Auto-pause is enabled**
  - Given the service is deployed
  - When no requests are made for 5 minutes
  - Then the service enters a paused state
  - And the next request responds within 30 seconds
- **Acc Test 3: Public endpoint responds**
  - Given the service is running
  - When I call the public App Runner URL `/health`
  - Then it returns status code 200
- **Acc Test 4: Predict endpoint works via public URL**
  - Given the service is running
  - When I POST `/predict` with valid teams
  - Then it returns probabilities and implied odds

**Definition of Done:**

- App Runner service deployed with 0.25 vCPU and 0.5 GB RAM
- Auto-pause enabled
- Service runs uvicorn on port 8080
- Public HTTPS endpoint documented in README as `API_BASE_URL`
- `/health` and `/predict` both work from an off-network device
- README includes a "Cost and cold starts" section with instance size, estimated cost under $2/month, and cold-start warning
- Streamlit handles cold-start delays with a loading spinner

---

## Deploy Streamlit UI on Streamlit Community Cloud

**Labels:** Must Have, Sprint 4

**User Story:**
As a user, I want the Streamlit dashboard deployed on Streamlit Community Cloud so the UI has a public URL accessible from any device.

**Acceptance Tests:**

- **Acc Test 1: Streamlit Cloud deployment succeeds**
  - Given the UI code is pushed to a public GitHub repository
  - When I deploy on Streamlit Community Cloud
  - Then the app builds without errors and provides a public URL
- **Acc Test 2: UI loads on any device**
  - Given the Streamlit Cloud URL exists
  - When I open it on my phone on mobile data
  - Then the app loads successfully
- **Acc Test 3: UI fails gracefully if API is unreachable**
  - Given the API is down
  - When I attempt a prediction
  - Then the UI shows a clear error message and does not crash

**Definition of Done:**

- Streamlit app deployed with a public URL
- README includes UI URL and deployment steps
- Basic error handling added for API call failures

---

## Configure Streamlit to call the AWS API via API_BASE_URL

**Labels:** Must Have, Sprint 4

**User Story:**
As a user, I want Streamlit to use an `API_BASE_URL` environment variable so the UI calls the deployed AWS API without hardcoded URLs.

**Acceptance Tests:**

- **Acc Test 1: Streamlit reads API_BASE_URL**
  - Given `API_BASE_URL` is set in Streamlit Cloud secrets
  - When the Streamlit app starts
  - Then it uses that value as the base URL for all API requests
- **Acc Test 2: Prediction call succeeds end-to-end**
  - Given Streamlit is deployed and API is on App Runner
  - When I select valid teams and click Predict
  - Then Streamlit receives a 200 response and displays results
- **Acc Test 3: Missing API_BASE_URL uses a safe fallback**
  - Given `API_BASE_URL` is not set
  - When the app starts
  - Then it defaults to a documented local URL or shows a clear configuration error

**Definition of Done:**

- Streamlit reads `API_BASE_URL` via env var or Streamlit secrets
- No hardcoded production API URLs in code
- End-to-end demo works publicly from a phone
- README includes "How to set API_BASE_URL in Streamlit Cloud"

---

## Public end-to-end smoke test checklist

**Labels:** Could Have, Sprint 4

**User Story:**
As a user, I want a simple smoke test script so I can verify the public deployment works after any change.

**Acceptance Tests:**

- **Acc Test 1: Smoke tests verify API and UI**
  - Given the deployment is live
  - When I run the smoke test script
  - Then it verifies `/health` returns 200
  - And `/predict` returns valid probabilities
  - And the Streamlit URL loads

**Definition of Done:**

- `scripts/smoke_test_public.sh` exists
- Checklist included in README under "Post-deploy verification"

---

## Data freshness: scheduled refresh script with data as-of timestamp

**Labels:** Should Have, Sprint 4

**User Story:**
As a user, I want the app to display when its data was last updated and provide a documented refresh process so predictions don't silently go stale after new matches are played.

**Acceptance Tests:**

- **Acc Test 1: Streamlit UI shows data as-of date**
  - Given the API is running
  - When the Streamlit app loads
  - Then it displays `Predictions based on data as of: YYYY-MM-DD`
  - And this date is fetched from the API, not hardcoded
- **Acc Test 2: Refresh script runs the full pipeline**
  - Given new EPL matches have been played
  - When I run `scripts/refresh_and_retrain.sh`
  - Then it pulls the latest data from Football-Data API
  - And runs the full pipeline end to end
  - And writes updated artifacts including `artifacts/model_rf.pkl`, `artifacts/feature_schema.json`, `artifacts/current_elo.json`, and `artifacts/last_updated.json`
- **Acc Test 3: API exposes the last-updated timestamp**
  - Given `artifacts/last_updated.json` exists
  - When I call `GET /metadata`
  - Then it returns `last_updated_date` and `data_through_date`

**Definition of Done:**

- Refresh script exists: `scripts/refresh_and_retrain.sh` and is documented in README
- `artifacts/last_updated.json` written after every successful run with `last_updated_date` and `data_through_date`
- `artifacts/current_elo.json` overwritten as part of every pipeline run
- FastAPI exposes `GET /metadata` returning `last_updated_date` and `data_through_date`
- Streamlit displays the data as-of label using the `/metadata` response
- README includes: "Run this script after each gameweek to keep predictions current"

---

# Done

## Save raw pulls by season (reproducible)

**Labels:** Must Have, Sprint 1 — **DONE**

**User Story:**
As a user, I want to save raw match pulls by season from Football-Data so I can reproduce training datasets even if the API changes.

**Acceptance Tests:**

- **Acc Test 1: Raw season file is created**
  - Given I provide a valid season identifier
  - When I run the ingestion script for that season
  - Then a file is saved to `data/raw/matches_<season>.parquet`
  - And the file contains match rows with `date, home_team, away_team, home_goals, away_goals`
- **Acc Test 2: Raw pull is idempotent**
  - Given `data/raw/matches_<season>.parquet` already exists
  - When I re-run the ingestion script for the same season
  - Then the output schema remains the same
  - And the file is overwritten safely
- **Acc Test 3: Metadata is included**
  - Given I run a raw pull
  - When I inspect the saved file
  - Then it contains `pull_timestamp` and `data_source` columns

**Definition of Done:**

- Script exists: `src/ingest/pull_matches.py`
- Produces `data/raw/matches_<season>.parquet` for at least 3 seasons
- Schema is stable
- Re-run behavior is documented and works

---

## Build unified raw dataset

**Labels:** Must Have, Sprint 1 — **DONE**

**User Story:**
As a user, I want to combine all saved raw season pulls into one unified dataset so downstream processing uses a single consistent input.

**Acceptance Tests:**

- **Acc Test 1: Unified raw dataset created**
  - Given multiple season raw files exist in `data/raw/`
  - When I run the unify script
  - Then it creates `data/raw/matches_all.parquet`
  - And the unified file includes all rows across those seasons
- **Acc Test 2: Unified dataset is sorted and consistent**
  - Given the unified raw dataset exists
  - When I load it
  - Then it is sorted by `date` ascending
  - And required columns exist with consistent datatypes

**Definition of Done:**

- Script exists: `src/ingest/unify_raw.py`
- Outputs `data/raw/matches_all.parquet`
- Basic validation checks pass: row counts, required columns, date ordering

---

## Create canonical match table with locked schema

**Labels:** Must Have, Sprint 1 — **DONE**

**User Story:**
As a user, I want a canonical match table with a locked schema so feature engineering and modeling won't break as the project evolves.

**Acceptance Tests:**

- **Acc Test 1: Canonical dataset has required columns**
  - Given `data/raw/matches_all.parquet` exists
  - When I run the canonical builder
  - Then it creates `data/processed/matches_canonical.parquet`
  - And columns include `match_id, date, season, home_team, away_team, home_goals, away_goals`
- **Acc Test 2: match_id is stable**
  - Given the same canonical builder is run twice on identical raw data
  - When I compare `match_id` values
  - Then each match has the same `match_id` across runs
- **Acc Test 3: Result label is correct**
  - Given a match where `home_goals > away_goals`
  - When canonical labels are computed
  - Then `result` equals `HW`
  - And if `home_goals == away_goals` then `result` equals `D`
  - And if `home_goals < away_goals` then `result` equals `AW`

**Definition of Done:**

- Script exists: `src/ingest/build_canonical.py`
- Output exists: `data/processed/matches_canonical.parquet`
- `result` label passes unit tests in `tests/test_labels.py`

---

## Implement leakage-safe rolling feature builder

**Labels:** Must Have, Sprint 1 — **DONE**

**User Story:**
As a user, I want leakage-safe rolling statistics that include home/away venue splits, days-rest features, and a promoted-team NaN imputation policy so the model has richer predictive signals without leaking future information.

**Acceptance Tests:**

- **Acc Test 1: Rolling features exclude the current match**
  - Given a team has played matches before match M
  - When rolling stats are computed for match M
  - Then the stats do not include match M itself
- **Acc Test 2: Rolling window size is respected**
  - Given rolling window size N
  - When features are computed
  - Then rolling aggregates only consider the last N prior matches
- **Acc Test 3: Home and away venue split features are computed separately**
  - Given a team has played both home and away matches before match M
  - When rolling stats are computed
  - Then `home_rolling_win_rate_home` and `home_rolling_win_rate_away` exist and can differ
- **Acc Test 4: Days-rest feature is computed correctly**
  - Given the home team last played on 2024-03-01 and match M is on 2024-03-08
  - When features are computed
  - Then `home_days_rest` equals 7
- **Acc Test 5: Promoted/new team imputation is applied**
  - Given a team has no prior match history
  - When rolling features are computed for their first match
  - Then rolling numeric features are filled with league-average values
  - And `days_rest` is set to 7
  - And `home_is_new_team` or `away_is_new_team` is set to 1
- **Acc Test 6: Features are generated for both teams**
  - Given a canonical match row
  - When features are computed
  - Then home team features exist with prefix `home_` and away with prefix `away_`

**Definition of Done:**

- Module: `src/features/rolling.py`
- Tests passing: `tests/test_leakage_rolling.py`, `tests/test_promoted_team_imputation.py`