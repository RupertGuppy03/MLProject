# Sprint 3 design decisions — season basis, UI, and the stadium map

This note records the reasoning behind three decisions made while building the Sprint 3 web app, so
the choices are traceable later (e.g. when the app rolls over to a new season).

## 1. Basing the app on the 2025/26 season

**Decision:** The app is fixed to the **2025/26 season** — the 20 clubs that played 25/26 are the only
selectable teams — and it stays that way until the **first data pull of the new (26/27) season**, at
which point everything rolls over.

**Why:**

- **League position is an important feature.** The model uses league position (and its home/away
  gap) as a signal, and a team's position is only meaningful *within a completed or in-progress
  season*. Switching the roster to 26/27 before any 26/27 matches exist would leave every promoted
  team with no position and every returning team with a stale/irrelevant one.
- **Don't discard good data prematurely.** The 25/26 season is our most recent complete signal —
  real Elo, real form, real positions for 20 real teams. Rolling over early would mean throwing that
  away and running the whole app on imputed placeholders (Elo 1450, league-average form,
  `is_new_team=1`) for the promoted sides, which is strictly worse than using the season we actually
  have data for.
- **A clean, dated rollover point.** Tying the switch to "the first 26/27 pull" gives one unambiguous
  trigger. Before that pull the app is honest about what it knows (25/26); after it, the new roster
  and fresh positions arrive together, and promoted teams get imputed only for the short window before
  they've played enough matches.

We *did* build a 26/27-roster version first (promoted teams in via imputation) and then reverted it
for the reasons above. The imputation path still exists for when the rollover actually happens.

## 2. UI revamp and club logos

The initial dashboard was functional but plain (dropdowns + numbers). We reworked it for a portfolio-
grade feel while keeping everything driven by real API data.

**Decisions:**

- **Multi-page layout** (`st.navigation`): a **Dashboard** page and an **About** page, so the app has
  a proper front door and a place to explain the model/data without cluttering the prediction view.
- **Everything in cards** (`st.container(border=True)`): each figure (probabilities, confidence, form,
  Elo history, rolling goals, splits, radar, H2H, SHAP) sits in its own bordered card, so the
  explanation panel reads as distinct, scannable sections rather than one long scroll.
- **Club logos.** Crests are shown beside the selected team names and enlarged in the result header
  (`[home crest] VS [away crest]`). Native `st.selectbox` can't render images *inside* options, so we
  put the logo next to the selected team instead. A short note in About records that crests are
  trademarks used for a non-commercial portfolio project.
- **Display names vs canonical names.** UI-only short labels (Brighton & Hove Albion FC → "Brighton",
  Wolverhampton Wanderers FC → "Wolves") via `team_display.display_name`. The **canonical** names are
  never changed — the API and model still use them — so display polish can't cause train/serve skew.
- **Theme-aware charts.** The matplotlib radar switches between light and dark palettes (transparent
  background + contrast colours) to match the active Streamlit theme automatically.
- **Chart correctness touches.** Elo shown as a "history" line with an integer match axis (no
  fractional ticks); rolling goals uses solid = scored / dashed = conceded with a caption instead of a
  confusing dual legend; SHAP features get human-readable labels.
- **UX niceties:** a home/away swap button, a spinner while predicting, brand-coloured team names, and
  a data-freshness footer.

## 3. Interactive 3D stadium map

**Decision:** After a prediction, show a 3D pydeck map of the league. Each of the 20 current-season
stadiums is a hexagonal bar whose **height = that team's current Elo rating**; the two selected teams
are in full club colour and the other 18 are greyed out; the view auto-centres on the home stadium and
each bar is labelled with its stadium name.

**Why these specific choices:**

- **Elo drives bar height, not league points.** Elo is already a served artifact (`current_elo.json`,
  exposed via the new `GET /elos` endpoint), so it needs no extra standings computation, and it's the
  model's own strength signal — a natural narrative tie-in to the prediction.
- **"Rest of the league greyed", not "other fixtures".** The original story idea was to grey out
  *other upcoming fixtures*, but our app has **no fixture calendar** — users pick any two teams, and
  the free-tier data source gives historical results, not a forward schedule. So the greyed bars are
  simply "the rest of the league," which delivers the same visual contrast without inventing data we
  don't have. Showing all 20 (vs just the two picked) also keeps the map visually rich instead of two
  lonely bars.
- **`ColumnLayer` with hexagonal columns, not `HexagonLayer`.** pydeck's `HexagonLayer` *aggregates*
  scattered points into binned hexagons and can't give one fixed-height bar per stadium. A
  `ColumnLayer` with `disk_resolution=6` gives the same hexagon look but one controllable bar per team
  (height = Elo), plus a `TextLayer` for the stadium labels.
- **Scoped to the 20 current-season clubs**, consistent with decision (1) and the rest of the app.
