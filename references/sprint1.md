# Sprint 1 — Data Pipelines

**Goal:** Build a reproducible, leakage-safe data pipeline from Football-Data → canonical dataset → feature matrix (`X, y`). Sprint ends with proof that no future games leak into any row's features.

---

## Stories

| Story | Status | Key files |
|---|---|---|
| Save raw pulls by season | Done | `src/ingest/pull_matches.py` |
| Build unified raw dataset | Done | `src/ingest/unify_raw.py` |
| Create canonical match table | Done | `src/ingest/build_canonical.py` |
| Leakage-safe rolling feature builder | Done | `src/features/rolling.py` |
| Implement sequential Elo features | Done | `src/features/elo.py` |
| Feature pipeline X/y/meta contract | Done | `src/features/build_features.py` |
| League position rolling feature | Done | `src/features/rolling.py` |
| Data quality validation on ingest | Done | `src/ingest/validate.py` |

---

## Implement sequential Elo features (pre-match only)

**Status:** Done
**Labels:** Must Have, Sprint 1

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
- Elo logic is exposed as a reusable function usable at both training and inference time — no duplicate implementation
- Elo parameters locked: K-factor **20**, home advantage **+60 points**, starting Elo **1450**, season regression **25% toward 1500**
- At the end of the Elo walk, final ratings per team are written to `artifacts/current_elo.json` with an `as_of_date` field
- Unit test passes: `tests/test_elo.py` covers leakage safety and all parameter behaviors

**Refinement (post-Sprint-1):** added margin-of-victory K-scaling (World Football Elo goal-difference index) so bigger wins move ratings more. Locked params unchanged; draws/1-goal wins behave as before.

---

## Feature pipeline outputs X, y, meta with schema contract

**Status:** Done
**Labels:** Must Have, Sprint 1

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
- `X` explicitly excludes `result`, `home_goals`, `away_goals` — enforced by an assert

---

## League position rolling feature

**Status:** Done
**Labels:** Should Have, Sprint 1

**User Story:**
As a developer, I want a rolling league position feature computed from season-start so the model has a dynamic strength signal that updates after every match.

**Acceptance Tests:**

- **Acc Test 1: League position computed from season start**
  - Given matches for a season exist
  - When league position is computed for match M
  - Then the position reflects cumulative points up to but NOT including match M
  - And GW1 defaults to a mid-table value (e.g. 10th) for all teams
- **Acc Test 2: Position resets each season**
  - Given the dataset spans multiple seasons
  - When league position is computed
  - Then positions reset to default at the start of each new season
- **Acc Test 3: Feature appears for both teams**
  - When features are returned
  - Then `home_league_position` and `away_league_position` both exist
  - And `home_position_diff = away_league_position - home_league_position` also exists

**Definition of Done:**

- Implemented in `src/features/rolling.py`
- Columns added: `home_league_position`, `away_league_position`, `home_position_diff`
- Season-boundary reset tested: `tests/test_league_position.py`
- Columns added to `artifacts/feature_schema.json`

---

## Data quality validation on ingest

**Status:** Done
**Labels:** Must Have, Sprint 1

**User Story:**
As a user, I want raw match data validated on ingest so malformed matches don't silently corrupt training data and known anomalies (e.g. the 2019/20 COVID gap) are surfaced early.

**Acceptance Tests:**

- **Acc Test 1: Schema validation runs on ingest**
  - Given a raw season pull is loaded
  - When the validation step runs
  - Then required columns are checked for presence and dtype
  - And rows missing required values are flagged
- **Acc Test 2: Malformed rows are quarantined, not silently dropped**
  - Given a raw season contains rows with NaN goals
  - When validation runs
  - Then those rows are removed and a count is logged
  - And dropped rows are saved to `data/quarantine/dropped_<season>.parquet`
- **Acc Test 3: Validation report is generated**
  - Given the dataset spans multiple seasons including 2019/20
  - When ingestion completes
  - Then a report is written to `data/processed/validation_report.json`

**Definition of Done:**

- Module exists: `src/ingest/validate.py`
- Called automatically as part of `build_canonical.py`
- Validation report saved: `data/processed/validation_report.json`
- Quarantine directory created with dropped rows
- Test exists: `tests/test_ingest_validation.py`