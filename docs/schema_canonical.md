# Canonical Match Table Schema

**File:** `data/processed/matches_canonical.parquet`

This dataset is the single source of truth for match-level modelling and feature engineering.
The schema is locked so downstream code does not break when the project evolves.

## Columns (locked order)

1. `match_id` (string)  
   Stable deterministic ID created from: `season`, UTC `date`, `home_team`, `away_team` (hashed).

2. `date` (datetime, UTC)  
   Match kickoff timestamp in UTC.

3. `season` (Int64)  
   Season start-year (Premier League style):  
   - Aug–Dec → season = same year  
   - Jan–May → season = previous year

4. `home_team` (string)

5. `away_team` (string)

6. `home_goals` (Int64, nullable)

7. `away_goals` (Int64, nullable)

8. `result` (string, nullable)  
   - `HW` = Home win (home_goals > away_goals)  
   - `D`  = Draw (home_goals == away_goals)  
   - `AW` = Away win (home_goals < away_goals)  
   If goals are missing (future fixtures), `result` is null.

## Determinism

Running `src/ingest/build_canonical.py` multiple times on identical raw input produces identical `match_id` values.
