"""Dashboard page: pick two teams, run the prediction, and see the full explanation panel.

Rendered by the multi-page controller in ``src/ui/app.py`` (which has already put the project root
on sys.path and called set_page_config). All data comes from the FastAPI backend via ``api_client``.
"""

from __future__ import annotations

import base64
from pathlib import Path

import requests
import streamlit as st

from src.ui.api_client import get_metadata, get_teams, predict, selection_error
from src.ui.charts import (
    elo_history_chart,
    radar_figure,
    rolling_goals_chart,
    shap_chart,
    venue_splits_chart,
)
from src.ui.team_colors import get_team_color
from src.ui.team_display import display_name
from src.ui.team_logos import get_team_logo

DRAW_COLOR = "#EAB308"
FORM_COLORS = {"W": "#22C55E", "D": "#EAB308", "L": "#EF4444"}
_API_DOWN = "Cannot reach the API. Start it with: `uvicorn src.api.main:app --reload`"


# --- small HTML helpers ---


def _logo_img(team: str, size: int) -> str:
    """Inline base64 <img> for a team crest, or empty string if there's no logo."""
    path = get_team_logo(team)
    if not path:
        return ""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return (
        f"<img src='data:image/png;base64,{data}' width='{size}' "
        f"style='vertical-align:middle;margin-right:8px'/>"
    )


def _colored_span(text: str, color: str) -> str:
    return f"<span style='color:{color};font-weight:700;vertical-align:middle'>{text}</span>"


def _team_label(team: str) -> str:
    """A team's short display name in its brand colour."""
    return _colored_span(display_name(team), get_team_color(team))


def _team_badge(team: str) -> None:
    """Mini logo + coloured team name on one line (selection feedback)."""
    st.markdown(_logo_img(team, 26) + _team_label(team), unsafe_allow_html=True)


def _is_dark_theme() -> bool:
    """Whether the active Streamlit theme is dark (drives the matplotlib radar colours)."""
    theme = getattr(st.context, "theme", None)
    theme_type = getattr(theme, "type", None) if theme is not None else None
    if theme_type is None:
        try:
            theme_type = st.get_option("theme.base")
        except Exception:
            theme_type = "dark"
    return str(theme_type).lower() == "dark"


def _form_chips(sequence: list[str]) -> str:
    chips = [
        f"<span style='background:{FORM_COLORS[r]}22;color:{FORM_COLORS[r]};"
        f"padding:4px 10px;border-radius:6px;font-weight:700;margin-right:4px'>{r}</span>"
        for r in sequence
    ]
    return "".join(chips) or "—"


# --- page ---


def render() -> None:
    st.title("Premier League Match Predictor")
    st.caption(
        "Pick two teams for win / draw / win probabilities, implied odds, and a full explanation."
    )

    try:
        teams = get_teams()
    except requests.RequestException:
        st.error(_API_DOWN)
        st.stop()

    # Seed the selectors; keep them in session_state so the swap button can exchange them.
    st.session_state.setdefault("home_sel", teams[0])
    st.session_state.setdefault("away_sel", teams[1] if len(teams) > 1 else teams[0])

    def _swap() -> None:
        st.session_state.home_sel, st.session_state.away_sel = (
            st.session_state.away_sel,
            st.session_state.home_sel,
        )
        st.session_state.pop("result", None)  # stale after a swap

    with st.container(border=True):
        c_home, c_mid, c_away = st.columns([5, 1, 5])
        with c_home:
            st.selectbox("Home team", teams, key="home_sel", format_func=display_name)
            _team_badge(st.session_state.home_sel)
        with c_mid:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.button(
                "⇄", on_click=_swap, help="Swap home and away", use_container_width=True
            )
        with c_away:
            st.selectbox("Away team", teams, key="away_sel", format_func=display_name)
            _team_badge(st.session_state.away_sel)

    home = st.session_state.home_sel
    away = st.session_state.away_sel

    if st.button("Run Prediction", type="primary"):
        error = selection_error(home, away)
        if error:
            st.error(error)
        else:
            try:
                with st.spinner("Crunching the numbers…"):
                    st.session_state.result = predict(home, away)
                    st.session_state.result_pair = (home, away)
            except ValueError as exc:  # 400 from the API
                st.error(str(exc))
                st.session_state.pop("result", None)
            except requests.RequestException:
                st.error(_API_DOWN)
                st.session_state.pop("result", None)

    result = st.session_state.get("result")
    # Only show a result that matches the current selection.
    if not result or st.session_state.get("result_pair") != (home, away):
        _footer()
        return

    _render_result(result, home, away)
    _footer()


def _render_result(result: dict, home: str, away: str) -> None:
    probs = result["probabilities"]
    odds = result["implied_odds"]
    context = result["context"]
    explanation = result["explanation"]
    outcome = result["predicted_outcome"]

    # Result header: centred crests + names + predicted-outcome headline.
    if outcome == "home_win":
        headline = f"Predicted: {_team_label(home)} win"
    elif outcome == "away_win":
        headline = f"Predicted: {_team_label(away)} win"
    else:
        headline = f"Predicted: {_colored_span('Draw', DRAW_COLOR)}"

    with st.container(border=True):
        st.markdown(
            "<div style='text-align:center;line-height:1'>"
            f"{_logo_img(home, 72)}"
            "<span style='font-size:1.6rem;font-weight:700;margin:0 22px;"
            "vertical-align:middle'>VS</span>"
            f"{_logo_img(away, 72)}"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='text-align:center;margin-top:10px'>"
            f"{_team_label(home)} <span style='color:#6b7280'>(home)</span>"
            "&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"{_team_label(away)} <span style='color:#6b7280'>(away)</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h3 style='text-align:center;margin-top:12px'>{headline}</h3>",
            unsafe_allow_html=True,
        )

    # Probability cards — highlight the model's pick.
    cards = [
        (f"{display_name(home)} win", get_team_color(home), probs["p_home"], odds["home"]),
        ("Draw", DRAW_COLOR, probs["p_draw"], odds["draw"]),
        (f"{display_name(away)} win", get_team_color(away), probs["p_away"], odds["away"]),
    ]
    cols = st.columns(3)
    for col, (label, color, prob, odd) in zip(cols, cards):
        with col.container(border=True):
            st.markdown(_colored_span(label, color), unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:2.4rem;font-weight:700'>{prob * 100:.1f}%</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Implied odds: {odd:.2f}" if odd is not None else "Implied odds: —"
            )

    # Model confidence.
    with st.container(border=True):
        st.markdown("**Model confidence**")
        max_prob = max(probs["p_home"], probs["p_draw"], probs["p_away"])
        st.progress(
            int(round(max_prob * 100)),
            text=f"{max_prob * 100:.0f}% on the favoured outcome",
        )

    # Recent form.
    with st.container(border=True):
        st.markdown("**Recent form (last 5)**")
        fcol1, fcol2 = st.columns(2)
        for fcol, team, block in (
            (fcol1, home, context["home"]),
            (fcol2, away, context["away"]),
        ):
            fcol.markdown(_team_label(team), unsafe_allow_html=True)
            fcol.markdown(_form_chips(block["form"]), unsafe_allow_html=True)
            fcol.caption(
                f"PPG {block['ppg']:.1f} · Position #{block['league_position']} "
                f"· Clean sheets {block['clean_sheets']}"
            )

    with st.container(border=True):
        st.subheader("Elo rating history")
        st.altair_chart(
            elo_history_chart(context, home, away), use_container_width=True
        )

    with st.container(border=True):
        st.subheader("Goals scored vs conceded — rolling 5-match")
        st.altair_chart(
            rolling_goals_chart(context, home, away), use_container_width=True
        )
        st.caption("Solid line = goals scored  ·  Dashed line = goals conceded")

    scol, rcol = st.columns(2)
    with scol.container(border=True):
        st.subheader("Home / away splits")
        st.altair_chart(
            venue_splits_chart(context, home, away), use_container_width=True
        )
    with rcol.container(border=True):
        st.subheader("Team profile radar")
        st.pyplot(radar_figure(context, home, away, dark=_is_dark_theme()))

    with st.container(border=True):
        st.subheader("Head-to-head record")
        h2h = context["head_to_head"]
        h1, h2, h3 = st.columns(3)
        h1.metric(f"{display_name(home)} wins", h2h["home_wins"])
        h2.metric("Draws", h2h["draws"])
        h3.metric(f"{display_name(away)} wins", h2h["away_wins"])
        st.caption(f"{h2h['total']} previous meetings")

    with st.container(border=True):
        st.subheader("Why this prediction — top SHAP contributors")
        st.altair_chart(shap_chart(explanation), use_container_width=True)
        st.caption(
            f"Green pushes toward the predicted outcome "
            f"({outcome.replace('_', ' ')}), red away from it."
        )


def _footer() -> None:
    """Data-freshness caption pulled from the API."""
    try:
        meta = get_metadata()
        if meta.get("data_through_date"):
            st.caption(f"Predictions for the 2025/26 season")
    except requests.RequestException:
        pass
