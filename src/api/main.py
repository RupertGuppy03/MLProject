"""FastAPI service for the Premier League match predictor.

Exposes the trained model over HTTP:
  - POST /predict  — win/draw/win probabilities + implied odds for a chosen fixture
  - GET  /teams    — selectable teams (source of truth for the dashboard dropdowns)
  - GET  /elos     — current Elo rating for every current-season team (drives the stadium map)
  - GET  /health   — liveness probe for deployment health checks
  - GET  /metadata — data/artifact freshness dates

Run locally:  uvicorn src.api.main:app --reload
(LAN binding on 0.0.0.0 + CORS is a separate story — Local network access.)
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.api.explain import top_feature_contributions
from src.api.inference_features import build_inference_features, get_selectable_teams
from src.api.match_context import build_match_context
from src.api.metadata import CURRENT_ELO_PATH, _as_of_from_elo, read_last_updated
from src.config import CURRENT_SEASON_TEAMS
from src.features.elo import load_state
from src.models.rf import LABEL_ORDER, load_model, predict_proba
from src.models.save_artifacts import CHOSEN_MODEL_PATH

app = FastAPI(
    title="Premier League Match Predictor",
    description="Predict match outcome probabilities and implied odds for a chosen fixture.",
    version="1.0.0",
)

# Human-readable outcome labels aligned with LABEL_ORDER (["HW", "D", "AW"]).
_OUTCOME_LABELS = ["home_win", "draw", "away_win"]


@lru_cache(maxsize=1)
def _get_model():
    """Load the served model once and cache it for the process lifetime."""
    return load_model(CHOSEN_MODEL_PATH)


@lru_cache(maxsize=1)
def _get_elo_state():
    """Load the current Elo state (all teams) once and cache it for the process lifetime."""
    return load_state(CURRENT_ELO_PATH)


# --- Pydantic request/response models (also drive the /docs schema) ---


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    as_of_date: str | None = None  # defaults to the latest match date in the dataset


class Probabilities(BaseModel):
    p_home: float
    p_draw: float
    p_away: float


class ImpliedOdds(BaseModel):
    home: float | None
    draw: float | None
    away: float | None


class VenueSplits(BaseModel):
    home_goals_for: float
    home_goals_against: float
    away_goals_for: float
    away_goals_against: float


class TeamContext(BaseModel):
    team: str
    current_elo: float
    elo_trajectory: list[float]
    form: list[str]
    rolling_goals_scored: list[float]
    rolling_goals_conceded: list[float]
    venue_splits: VenueSplits
    ppg: float
    clean_sheets: int
    league_position: int


class HeadToHead(BaseModel):
    home_wins: int
    draws: int
    away_wins: int
    total: int


class MatchContext(BaseModel):
    as_of_date: str
    home: TeamContext
    away: TeamContext
    head_to_head: HeadToHead


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class Explanation(BaseModel):
    predicted_class: str
    top_features: list[FeatureContribution]


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    as_of_date: str | None
    probabilities: Probabilities
    implied_odds: ImpliedOdds
    predicted_outcome: str
    context: MatchContext
    explanation: Explanation


class TeamsResponse(BaseModel):
    teams: list[str]


class ElosResponse(BaseModel):
    as_of_date: str
    elos: dict[str, float]


class HealthResponse(BaseModel):
    status: str


class MetadataResponse(BaseModel):
    last_updated_date: str | None
    data_through_date: str | None


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Simple liveness payload used by deployment health checks."""
    return HealthResponse(status="ok")


@app.get("/teams", response_model=TeamsResponse)
def teams() -> TeamsResponse:
    """Sorted roster of the current season — the teams a user can select."""
    return TeamsResponse(teams=get_selectable_teams())


@app.get("/elos", response_model=ElosResponse)
def elos() -> ElosResponse:
    """Current Elo rating for every current-season team (drives the dashboard stadium map)."""
    state = _get_elo_state()
    current = {team: state.ratings[team] for team in CURRENT_SEASON_TEAMS if team in state.ratings}
    return ElosResponse(as_of_date=_as_of_from_elo(CURRENT_ELO_PATH), elos=current)


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Freshness dates read from artifacts/last_updated.json."""
    payload = read_last_updated()
    if payload is None:
        raise HTTPException(
            status_code=503, detail="Freshness metadata not available (last_updated.json missing)."
        )
    return MetadataResponse(**payload)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Return outcome probabilities and implied odds for a home/away fixture.

    Builds the one-row inference feature vector, runs the served model, and derives implied
    odds as 1/probability. Unknown or duplicate teams surface as a 400 whose message lists the
    valid teams (raised by build_inference_features).
    """
    try:
        X = build_inference_features(
            request.home_team, request.away_team, as_of_date=request.as_of_date
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # predict_proba returns [p_home, p_draw, p_away] (reordered from the model's classes_).
    model = _get_model()
    proba = predict_proba(model, X)[0]
    p_home, p_draw, p_away = (float(p) for p in proba)

    # Implied (fair) odds are the reciprocal of each probability; guard against zero.
    def to_odds(p: float) -> float | None:
        return round(1.0 / p, 2) if p > 0 else None

    # Highest-probability outcome. LABEL_ORDER (["HW","D","AW"]) is the model-label form of the
    # same index; _OUTCOME_LABELS is the human-readable form.
    best = max(range(3), key=lambda i: proba[i])
    predicted_outcome = _OUTCOME_LABELS[best]

    # Rich match context (real data behind every chart) + per-prediction SHAP on the same X.
    context = build_match_context(
        request.home_team, request.away_team, as_of_date=request.as_of_date
    )
    explanation = {
        "predicted_class": predicted_outcome,
        "top_features": top_feature_contributions(model, X, LABEL_ORDER[best]),
    }

    return PredictResponse(
        home_team=request.home_team,
        away_team=request.away_team,
        as_of_date=request.as_of_date,
        probabilities=Probabilities(p_home=p_home, p_draw=p_draw, p_away=p_away),
        implied_odds=ImpliedOdds(
            home=to_odds(p_home), draw=to_odds(p_draw), away=to_odds(p_away)
        ),
        predicted_outcome=predicted_outcome,
        context=MatchContext.model_validate(context),
        explanation=Explanation.model_validate(explanation),
    )
