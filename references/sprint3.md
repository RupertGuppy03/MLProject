# Sprint 3 — Web App and UI

**Goal:** Deliver a working product. User selects home/away teams and gets probabilities, implied odds, and match context charts. Streamlit calls FastAPI and works from another device on the local network.

---

## Stories

| Story | Status | Key files |
|---|---|---|
| FastAPI predict endpoint | Done | `src/api/main.py` |
| Inference feature builder | Done | `src/api/inference_features.py` |
| Streamlit dashboard | Done | `src/ui/app.py` |
| Match context and SHAP explanation panel | Done | `src/api/explain.py`, `src/api/match_context.py` |
| Local network access (CORS + configurable API URL) | To Do | `src/api/main.py`, `src/ui/app.py` |
| Teams list endpoint | Done | `src/api/main.py`, `src/ui/app.py` |

---

## FastAPI predict endpoint

**Status:** Done
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
- `GET /health` returns 200 with a simple status payload (used by deployment health checks)
- `GET /metadata` returns `last_updated_date` and `data_through_date` from `artifacts/last_updated.json`
- Smoke test script exists: `scripts/smoke_api.sh`

---

## Inference feature builder

**Status:** Done
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

**Status:** Done
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

**Status:** Done
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

note when building this part, look at the example html webapp to see how the ideal structure and graphs included should look like.

---

## Local network access (CORS + configurable API URL)

**Status:** To Do
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

## Teams list endpoint

**Status:** Done
**Labels:** Must Have, Sprint 3

**User Story:**
As a user, I want a `/teams` endpoint that returns the list of selectable teams, so the dashboard can populate its home/away dropdowns from a single source of truth instead of hardcoding team names.

**Acceptance Tests:**

- **Acc Test 1: Teams endpoint returns the valid team list**
  - Given the API is running and the canonical dataset exists
  - When I GET `/teams`
  - Then the response is a list of team names drawn from the canonical dataset
  - And the list is non-empty, sorted, and free of duplicates
- **Acc Test 2: Team list is consistent with prediction validation**
  - Given `/teams` returns a set of team names
  - When I POST `/predict` with any team from that list
  - Then the team is accepted (not rejected as unknown)
- **Acc Test 3: Dashboard populates dropdowns from the endpoint**
  - Given the Streamlit app is open and the API is reachable
  - When the app loads
  - Then the home and away dropdowns are populated from `/teams`
  - And no team names are hardcoded in the UI

**Definition of Done:**

- `GET /teams` implemented in `src/api/main.py`
- Team list derived from the canonical dataset (single source of truth shared with `/predict` team validation)
- Pydantic response model defined; `/docs` shows correct schema
- Streamlit dashboard (`src/ui/app.py`) reads its dropdown options from `/teams`
- Unit test verifies the endpoint returns a sorted, de-duplicated, non-empty list: `tests/test_teams_endpoint.py`

## Interactive stadium map for selected fixture

**Labels:** Could Have, Sprint 3

**User Story:**
As a user, I want an interactive map showing the selected fixture's stadium with a 3D bar representing league points, so I get a visually engaging way to see team strength alongside the other charts.

**Acceptance Tests:**

- **Acc Test 1: Map centres on selected match**
  - Given a user selects a home and away team
  - When the prediction is run
  - Then the map automatically zooms to the home team's stadium location
  - And the user does not need to manually pan or search

- **Acc Test 2: Bar height reflects league points**
  - Given the selected teams have current league points
  - When the map renders
  - Then each team's bar height is proportional to their points total

- **Acc Test 3: Other fixtures shown in greyscale**
  - Given other Premier League fixtures are upcoming
  - When the map renders
  - Then those fixtures' home stadiums appear as greyscale bars
  - And the selected fixture's bars are shown in full colour

**Definition of Done:**

- Map renders using an interactive 3D map component
- Selected fixture auto-focuses without manual navigation
- Bar height driven by real league points data
- Non-selected fixtures rendered in greyscale for contrast
- Stadium locations are accurate for all Premier League teams