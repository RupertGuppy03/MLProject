# CLAUDE.md: Premier League Match Predictor

## Project Overview

An end-to-end machine learning system that predicts Premier League match outcomes. A user picks a home and away team and gets win/draw/win probabilities, implied odds, charts explaining the prediction (Elo trajectory, rolling form, head-to-head, home/away splits), and a model confidence indicator.

This is a personal portfolio project for my final year of a Computer Science degree, built to demonstrate end-to-end engineering: data pipeline, ML modelling, API, dashboard, and public deployment. The reference UI is in `references/project_UI_example.html`.

Data is pulled from Football-Data.org, processed into a leakage-safe feature matrix, and used to train a Random Forest classifier with calibrated probabilities. It is served via FastAPI and displayed in a Streamlit dashboard, deployed publicly (Streamlit Community Cloud frontend calling a FastAPI backend on AWS App Runner).

---

## Repo Structure

```
pl-match-predictor/
│
├── CLAUDE.md          <- This file: project context + file map
├── README.md          <- Public README: UI URL, stack, run/deploy steps
├── requirements.txt   <- Pinned dependencies
│
├── data
│   ├── raw            <- Immutable match pulls from Football-Data.org
│   ├── processed      <- Canonical dataset + validation reports
│   └── quarantine     <- Dropped/malformed rows from ingest validation
│
├── artifacts          <- Trained models + locked feature schema + tuning/freshness JSONs
│
├── docs               <- Design docs required by the DoDs (schemas, Elo config, etc.)
│
├── reports            <- Backtest outputs: metrics, predictions, model comparison, plots
│
├── references         <- Planning docs
│   ├── Project_User_Stories.md   <- MASTER source of truth (all user stories)
│   └── sprint1..4/               <- Per-story files grouped by sprint
│
├── notebooks          <- Personal scratch / exploration (not part of the pipeline)
│
├── scripts            <- Operational scripts: smoke tests, scheduled refresh+retrain
│
├── tests              <- Pytest suite, mirrors src/
│
└── src                <- Source code
    ├── config.py      <- Paths, PL competition code, Football-Data settings
    ├── ingest         <- Data pipeline: pull, unify, build canonical, validate
    ├── features       <- Leakage-safe feature engineering: rolling, Elo, build_features
    ├── models         <- Training + baselines: Elo, logistic regression, Random Forest, tuning
    ├── backtest       <- Walk-forward backtesting engine
    ├── api            <- FastAPI service: predict, inference features, match context, SHAP
    ├── ui             <- Streamlit dashboard
    └── services       <- External API clients (Football-Data.org)
```

General rule for where files go: code that builds a feature column lives in `src/features`; code that pulls or cleans raw data lives in `src/ingest`; code that trains a model lives in `src/models`; the trained model file that pops out lives in `artifacts`. Every feature gets a matching test in `tests`. For the exact filename of any given story, check the user story itself, it names the path.

---

## Tech Stack

- **Language:** Python 3.11
- **Data:** pandas, numpy, pyarrow (parquet)
- **ML:** scikit-learn (RandomForestClassifier is the main model; Elo and Logistic Regression baselines; CalibratedClassifierCV, TimeSeriesSplit, RandomizedSearchCV), SHAP
- **Persistence:** joblib
- **Testing:** pytest
- **API:** FastAPI, uvicorn, Pydantic
- **UI:** Streamlit
- **Infra:** Docker, AWS ECR, AWS App Runner, Streamlit Community Cloud
- **Data source:** Football-Data.org API

---

## Build Plan

The project is split into 4 sprints. We work **sprint by sprint, one user story at a time.** We only move on to the next story once the current one is done.

- **Sprint 1 — Data pipelines:** leakage-safe pipeline from raw data to feature matrix. *(Partially done.)*
- **Sprint 2 — Model training and evaluation:** baselines, Random Forest, walk-forward backtesting, calibration, tuning.
- **Sprint 3 — Web app:** FastAPI `/predict`, inference features, Streamlit dashboard, SHAP explanations, match context.
- **Sprint 4 — Deployment:** Dockerise, push to ECR, deploy to App Runner, deploy Streamlit, smoke tests, data refresh.

### Workflow for each user story

1. **Pick one user story** from the current sprint.
2. **Plan first.** Start in plan mode and agree the approach with me before writing any code.
3. **Write the feature**, with brief comments and consistent style.
4. **Write the tests** for the acceptance criteria.
5. **I run the tests.** Running them and confirming they pass is my job. You can help me debug failures and fix the code or tests, but do not claim the tests pass on your own.
6. **Check against the Definition of Done** together.
7. **Ask me to commit** the feature to GitHub.
8. Once I confirm I'm happy, **ask before marking the story done** in its user story file.

---

## Where to find the user stories

The user stories are the source of truth for what to build. Before starting any story, read it.

- Master list: `references/Project_User_Stories.md`
- Grouped by sprint: `references/sprint1/`, `sprint2/`, `sprint3/`, `sprint4/`

Each story has a User Story, Gherkin acceptance tests, and a Definition of Done. Build exactly what those require. Do not invent requirements that aren't there.

---

## Rules

- **I control git, you do not.** Read-only git commands (`status`, `log`, `diff`) are fine for context. Never commit, push, or otherwise change repo state. When a feature is ready, summarise what changed and tell me it's ready to commit.
- **The user stories are the source of truth.** Build what the acceptance tests and DoD ask for, nothing more.
- **No data leakage.** No future data leaks into any feature. Rolling features exclude the current match. Elo updates happen after computing features for that match. Backtest training data is strictly before test data. If a request of mine would break this, tell me directly.
- **Schema is locked.** Any change to feature columns must update `artifacts/feature_schema.json` and the inference path together. The training-time and inference-time feature vectors must match exactly.
- **Random Forest is the main model.** Not HistGradientBoosting, not XGBoost.
- **Tests are mine to run.** Write features and their tests, but I confirm they pass. A story isn't done until I've personally verified it.
- **Comments and style.** All code gets brief, high-level comments so a marker can follow it. Use snake_case for Python and match the style of the existing codebase.
- **One story at a time.** Don't start the next story until the current one is done and committed.