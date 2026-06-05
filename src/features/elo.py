from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, PROCESSED_DIR

# Locked Elo parameters (per Sprint 1 DoD).
DEFAULT_STARTING_ELO = 1450.0
DEFAULT_K_FACTOR = 20.0
DEFAULT_HOME_ADVANTAGE = 60.0
DEFAULT_REGRESSION_TARGET = 1500.0
DEFAULT_REGRESSION_FACTOR = 0.25

_RESULT_TO_ACTUAL_HOME = {"HW": 1.0, "D": 0.5, "AW": 0.0}


@dataclass
class EloState:
    """Sequential Elo state for one league.

    The state holds each team's current rating and the last season we saw them
    in, so we can apply season regression on a team's first match of any new
    season. Parameters are stored on the state so saved files are self-describing.
    """

    starting_elo: float = DEFAULT_STARTING_ELO
    k_factor: float = DEFAULT_K_FACTOR
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    regression_target: float = DEFAULT_REGRESSION_TARGET
    regression_factor: float = DEFAULT_REGRESSION_FACTOR
    ratings: dict[str, float] = field(default_factory=dict)
    last_season: dict[str, int] = field(default_factory=dict)

    def get(self, team: str) -> float:
        """Return current rating; first-ever teams start at starting_elo."""
        if team not in self.ratings:
            self.ratings[team] = self.starting_elo
        return self.ratings[team]

    def expected_home(self, elo_home: float, elo_away: float) -> float:
        """P(home wins) via the standard Elo formula with home advantage on the home side."""
        return 1.0 / (1.0 + 10 ** ((elo_away - (elo_home + self.home_advantage)) / 400.0))

    def regress_for_season(self, team: str, season: int) -> None:
        """Pull a team's rating toward the regression target at the start of a new season.

        Skipped on a team's first-ever appearance — there is no prior season to regress from.
        """
        prior_season = self.last_season.get(team)
        if prior_season is not None and season > prior_season:
            current = self.get(team)
            self.ratings[team] = current + self.regression_factor * (
                self.regression_target - current
            )
        self.last_season[team] = season

    def update(self, home: str, away: str, result: str) -> None:
        """Apply the post-match Elo update for both teams."""
        if result not in _RESULT_TO_ACTUAL_HOME:
            raise ValueError(f"EloState.update: unknown result {result!r}")

        elo_home = self.get(home)
        elo_away = self.get(away)
        exp_home = self.expected_home(elo_home, elo_away)
        actual_home = _RESULT_TO_ACTUAL_HOME[result]

        self.ratings[home] = elo_home + self.k_factor * (actual_home - exp_home)
        self.ratings[away] = elo_away + self.k_factor * ((1.0 - actual_home) - (1.0 - exp_home))


def compute_elo_features(
    matches: pd.DataFrame,
    state: EloState | None = None,
) -> tuple[pd.DataFrame, EloState]:
    """Walk matches chronologically and emit pre-match Elo features.

    For each match we:
      1. Apply season regression for both teams (if this is their first match of a new season).
      2. Snapshot pre-match Elo BEFORE updating — this is what makes features leakage-safe.
      3. Update state only if the match has a result (future fixtures don't mutate state).

    Adds columns: elo_home_pre, elo_away_pre, elo_diff.
    """
    required = ["match_id", "date", "season", "home_team", "away_team", "result"]
    missing = [c for c in required if c not in matches.columns]
    if missing:
        raise ValueError(f"compute_elo_features: missing columns: {missing}")

    if state is None:
        state = EloState()

    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    elo_home_pre: list[float] = []
    elo_away_pre: list[float] = []

    for row in df.to_dict("records"):
        home = str(row["home_team"])
        away = str(row["away_team"])
        season = int(row["season"])

        state.regress_for_season(home, season)
        state.regress_for_season(away, season)

        eh = state.get(home)
        ea = state.get(away)
        elo_home_pre.append(eh)
        elo_away_pre.append(ea)

        if pd.notna(row["result"]):
            state.update(home, away, str(row["result"]))

    df["elo_home_pre"] = elo_home_pre
    df["elo_away_pre"] = elo_away_pre
    df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]

    return df, state


def save_state(state: EloState, path: Path, as_of_date: str) -> None:
    """Atomically write the Elo state (parameters + ratings + last seasons) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of_date": as_of_date,
        "parameters": {
            "starting_elo": state.starting_elo,
            "k_factor": state.k_factor,
            "home_advantage": state.home_advantage,
            "regression_target": state.regression_target,
            "regression_factor": state.regression_factor,
        },
        "ratings": state.ratings,
        "last_season": state.last_season,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def load_state(path: Path) -> EloState:
    """Load an Elo state previously written with save_state."""
    payload = json.loads(Path(path).read_text())
    params = payload["parameters"]
    state = EloState(
        starting_elo=float(params["starting_elo"]),
        k_factor=float(params["k_factor"]),
        home_advantage=float(params["home_advantage"]),
        regression_target=float(params["regression_target"]),
        regression_factor=float(params["regression_factor"]),
    )
    state.ratings = {team: float(rating) for team, rating in payload["ratings"].items()}
    state.last_season = {team: int(season) for team, season in payload["last_season"].items()}
    return state


def main() -> None:
    """Run the Elo walk on the canonical dataset and save final state to artifacts/."""
    canonical_path = PROCESSED_DIR / "matches_canonical.parquet"
    output_path = ARTIFACTS_DIR / "current_elo.json"

    matches = pd.read_parquet(canonical_path)
    df, state = compute_elo_features(matches)

    played = df[df["result"].notna()]
    if played.empty:
        raise RuntimeError("No played matches found in canonical dataset; cannot set as_of_date.")
    as_of_date = pd.to_datetime(played["date"]).max().date().isoformat()

    save_state(state, output_path, as_of_date=as_of_date)
    print(f"Saved Elo state for {len(state.ratings)} teams to {output_path} (as_of {as_of_date}).")


if __name__ == "__main__":
    main()
