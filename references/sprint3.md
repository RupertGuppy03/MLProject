# Sprint 3 — Web App and UI

**Goal:** Deliver a working product. User selects home/away teams and gets probabilities, implied odds, and match context charts. Streamlit calls FastAPI and works from another device on the local network.

---

## Stories

| Story | Status | Key files |
|---|---|---|
| FastAPI predict endpoint | To Do | `src/api/main.py` |
| Inference feature builder | To Do | `src/api/inference_features.py` |
| Streamlit dashboard | To Do | `src/ui/app.py` |
| Match context and SHAP explanation panel | To Do | `src/api/explain.py`, `src/api/match_context.py` |

---

## FastAPI predict endpoint

**Status:** To Do
**Labels:** Must Have, Sprint 3

**User Story:**
As a user, I want a `/predict` endpoint so the system can return match outcome probabilities and implied odds for selected teams.

**Acceptance Tests:**

- **Acc Test 1: Predict returns probabilities**
  - Given the API is running
  - When I POST `/predict` with a valid `home_team` and `away_team`
  - Then the response includes `p_home`, `p_draw`, `p_away` summing to 1
- **Acc Test 2: Predict returns implied odds**
  - Given valid probabilities are produced
  - When the response is returned
  - Then it includes implied odds equal to approximately `1 / probability`
- **Acc Test 3: Invalid team is rejected**
  - Given the API is running
  - When I POST `/predict` with an unknown team name
  - Then the API returns status 400 with a helpful error message

**Definition of Done:**

- Endpoint implemented in `src/api/main.py`
- Pydantic request/response models defined
- `/docs` shows correct schema
- Smoke test script exists: `scripts/smoke_api.sh`

---

## Inference feature builder

**Status:** To Do
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

**Status:** To Do
**Labels:** Must Have, Sprint 3

**User Story:**
As a user, I want a Streamlit dashboard so I can select teams and instantly view probabilities, implied odds, and the predicted outcome.

**Acceptance Tests:**

- **Acc Test 1: UI prevents invalid selection**
  - Given the Streamlit app is open
  - When I select the same team for home and away
  - Then the app prevents prediction and shows an error message
- **Acc Test 2: UI displays prediction outputs**
  - Given the API is running and reachable
  - When I choose valid teams and click Predict
  - Then the app displays `p_home`, `p_draw`, `p_away`, implied odds, and the predicted outcome label

**Definition of Done:**

- `src/ui/app.py` exists
- UI works against local FastAPI
- README has screenshots of the dashboard

---

## Match context and SHAP explanation panel

**Status:** To Do
**Labels:** Should Have, Sprint 3

**User Story:**
As a user, I want a rich explanation panel showing match context (Elo trajectory, rolling form, head-to-head, home/away splits) plus per-prediction SHAP values, so I can understand why the model favoured this specific outcome. Every chart in the UI mock must have a real backend data source — nothing faked or hardcoded.

**Acceptance Tests:**

- **Acc Test 1: Per-match SHAP values are computed**
  - Given a trained Random Forest model and a single prediction request
  - When the explanation logic runs
  - Then SHAP values are computed for that specific prediction
  - And the top 5 features with signed contribution values are returned
- **Acc Test 2: API returns rich match context**
  - Given a `/predict` request for two teams
  - When the response is built
  - Then it includes: current Elo for both teams, last-10 GW Elo trajectory, last-5 form sequence, rolling 5-match goals scored and conceded, home/away venue splits, head-to-head record, PPG and league position, and SHAP top contributors
- **Acc Test 3: Streamlit renders all context fields**
  - Given the API returns rich context
  - When the dashboard receives a prediction
  - Then it renders all charts from the UI mock: probability cards, Elo trajectory, rolling goals, splits, H2H, radar, model confidence, and SHAP panel
  - And no chart uses hardcoded data

**Definition of Done:**

- `src/api/explain.py` uses `shap.TreeExplainer` on the underlying RF — not the calibrated wrapper. The API serves calibrated probabilities to the user and pre-calibration SHAP values to the explainer. These must not be conflated.
- `src/api/match_context.py` builds H2H, Elo trajectory, rolling goals, and venue splits at inference time using only data prior to the as-of date
- Elo trajectory and current Elo are read from `src/features/elo.py` — not recomputed independently in the API
- API endpoint returns the full context dictionary
- Streamlit dashboard matches the UI mock in `references/project_UI_example.html`
- Integration test: `tests/test_api_context_schema.py`