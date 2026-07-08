# Sprint 4 — Deployment

**Goal:** Deploy a public, device-accessible web app. Streamlit UI hosted publicly, FastAPI model API deployed on Azure using Docker. End result is a public Streamlit URL calling a public Azure API URL reliably.

---

## Stories

| Story | Status | Key files |
|---|---|---|
| Dockerize the FastAPI prediction service | To Do | `src/api/Dockerfile` |
| Push the API Docker image to Azure Container Registry (ACR) | To Do | `.dockerignore`, ACR config |
| Deploy the API to Azure Container Apps | To Do | Container Apps config |
| Deploy Streamlit UI on Streamlit Community Cloud | To Do | `src/ui/app.py` |
| Configure Streamlit to call Azure API via API_BASE_URL | To Do | Streamlit secrets |
| Public end-to-end smoke test checklist | To Do | `scripts/smoke_test_public.sh` |
| Data freshness: scheduled refresh script | To Do | `scripts/refresh_and_retrain.sh` |

---

## Dockerize the FastAPI prediction service

**Status:** To Do
**Labels:** Must Have, Sprint 4

**User Story:**
As a user, I want the FastAPI prediction service containerized so the API runs consistently locally and in Azure.

**Acceptance Tests:**

- **Acc Test 1: API Docker image builds**
  - Given a Dockerfile exists
  - When I run `docker build`
  - Then the image builds successfully without errors
- **Acc Test 2: API container starts and serves endpoints**
  - Given the image is built
  - When I run the container
  - Then `/docs` loads and `/health` returns 200
- **Acc Test 3: Model artifacts load inside container**
  - Given artifacts are included in the container
  - When the API starts
  - Then POST `/predict` returns probabilities without errors
- **Acc Test 4: Missing artifacts fail fast**
  - Given the container is started without required artifacts
  - When the API boots
  - Then it exits with a clear message naming the missing artifact

**Definition of Done:**

- API Dockerfile exists: `src/api/Dockerfile`
- Container starts with uvicorn binding to `0.0.0.0`
- `/health` endpoint implemented
- README includes: `docker build`, `docker run`, example `curl` to `/predict`

---

## Push the API Docker image to Azure Container Registry (ACR)

**Status:** To Do
**Labels:** Must Have, Sprint 4

**User Story:**
As a user, I want to push the API Docker image to Azure Container Registry (ACR) with a minimal image size and automatic cleanup of old images so storage costs stay low and images don't accumulate.

**Acceptance Tests:**

- **Acc Test 1: ACR registry exists**
  - Given Azure credentials are configured (`az login`)
  - When I run the ACR setup commands (`az acr create`)
  - Then an ACR registry exists
- **Acc Test 2: Docker image uses a slim base**
  - Given the Dockerfile is implemented
  - When I inspect the FROM statement
  - Then it uses `python:3.11-slim`
- **Acc Test 3: Final image size under 600 MB**
  - Given the image is built locally
  - When I check the size
  - Then it is less than 600 MB
- **Acc Test 4: Old images are cleaned up**
  - Given the ACR registry is created (Basic SKU)
  - When the scheduled `acr purge` task runs
  - Then untagged manifests older than 1 day are deleted and only the 3 most recent tagged images are kept
- **Acc Test 5: Image is pushed successfully**
  - Given the image is built and tagged
  - When I push it
  - Then it appears in ACR with the correct tag

**Definition of Done:**

- Dockerfile uses `python:3.11-slim`
- `.dockerignore` excludes: `data/`, `notebooks/`, `reports/`, `.git/`, `tests/`, `__pycache__/`, `*.parquet`
- Final image size verified under 600 MB
- Image cleanup configured via a scheduled `acr purge` task (works on Basic SKU; native retention policies require Premium): keep 3 most recent tagged, delete untagged older than 1 day
- README includes ACR auth (`az acr login`) and push steps

---

## Deploy the API to Azure Container Apps

**Status:** To Do
**Labels:** Must Have, Sprint 4

**User Story:**
As a user, I want to deploy the FastAPI container to Azure Container Apps with scale-to-zero so the API stays publicly accessible at minimal cost when idle.

**Acceptance Tests:**

- **Acc Test 1: Container App uses minimum instance size**
  - Given the container app is configured
  - Then instance CPU is 0.25 vCPU and memory is 0.5 Gi
- **Acc Test 2: Scale-to-zero is enabled**
  - Given the container app is deployed with min replicas = 0
  - When no requests are made for ~5 minutes
  - Then it scales to zero and the next request responds within ~30 seconds (cold start)
- **Acc Test 3: Public endpoint responds**
  - Given the container app is running
  - When I call the public URL `/health`
  - Then it returns 200
- **Acc Test 4: Predict endpoint works via public URL**
  - Given the container app is running
  - When I POST `/predict` with valid teams
  - Then it returns probabilities and implied odds

**Definition of Done:**

- Container App deployed with 0.25 vCPU and 0.5 Gi RAM, scale-to-zero (min replicas = 0)
- External ingress enabled, target port = uvicorn port (8080)
- Public HTTPS endpoint (the `*.azurecontainerapps.io` FQDN) documented in README as `API_BASE_URL`
- `/health` and `/predict` work from an off-network device
- README includes "Cost and cold starts" section: instance size, free monthly grant, estimated cost ~$0–$2/month, cold-start warning
- Streamlit handles cold starts with a loading spinner

---

## Deploy Streamlit UI on Streamlit Community Cloud

**Status:** To Do
**Labels:** Must Have, Sprint 4

**User Story:**
As a user, I want the Streamlit dashboard deployed on Streamlit Community Cloud so the UI has a public URL accessible from any device.

**Acceptance Tests:**

- **Acc Test 1: Deployment succeeds**
  - Given the UI code is pushed to a public GitHub repo
  - When I deploy on Streamlit Community Cloud
  - Then the app builds without errors and provides a public URL
- **Acc Test 2: UI loads on any device**
  - Given the URL exists
  - When I open it on my phone on mobile data
  - Then the app loads successfully
- **Acc Test 3: UI fails gracefully if API is unreachable**
  - Given the API is down
  - When I attempt a prediction
  - Then the UI shows a clear error and does not crash

**Definition of Done:**

- Streamlit app deployed with a public URL
- README includes UI URL and deployment steps
- Basic error handling added for API call failures

---

## Configure Streamlit to call the Azure API via API_BASE_URL

**Status:** To Do
**Labels:** Must Have, Sprint 4

**User Story:**
As a user, I want Streamlit to use an `API_BASE_URL` environment variable so the UI calls the deployed Azure API without hardcoded URLs.

**Acceptance Tests:**

- **Acc Test 1: Streamlit reads API_BASE_URL**
  - Given `API_BASE_URL` is set in Streamlit Cloud secrets
  - When the app starts
  - Then it uses that value as the base URL for all API requests
- **Acc Test 2: Prediction call succeeds end-to-end**
  - Given Streamlit is deployed and the API is on Azure Container Apps
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

**Status:** To Do
**Labels:** Could Have, Sprint 4

**User Story:**
As a user, I want a simple smoke test script so I can verify the public deployment works after any change.

**Acceptance Tests:**

- **Acc Test 1: Smoke tests verify API and UI**
  - Given the deployment is live
  - When I run the smoke test script
  - Then it verifies `/health` returns 200, `/predict` returns valid probabilities, and the Streamlit URL loads

**Definition of Done:**

- `scripts/smoke_test_public.sh` exists
- Checklist included in README under "Post-deploy verification"

---

## Data freshness: scheduled refresh script with data as-of timestamp

**Status:** To Do
**Labels:** Should Have, Sprint 4

**User Story:**
As a user, I want the app to display when its data was last updated and provide a documented refresh process so predictions don't silently go stale after new matches are played.

**Acceptance Tests:**

- **Acc Test 1: Streamlit UI shows data as-of date**
  - Given the API is running
  - When the Streamlit app loads
  - Then it displays `Predictions based on data as of: YYYY-MM-DD` fetched from the API
- **Acc Test 2: Refresh script runs the full pipeline**
  - Given new EPL matches have been played
  - When I run `scripts/refresh_and_retrain.sh`
  - Then it pulls latest data and runs the full pipeline end to end
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