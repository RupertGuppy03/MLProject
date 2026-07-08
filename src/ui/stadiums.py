"""Stadium locations for the current-season clubs, used by the 3D map on the dashboard.

Keyed by canonical team name (the same names the API/model use) so lookups match `get_team_color`,
`get_team_logo`, and the `/elos` payload without any extra mapping. Coordinates are the stadium's
latitude/longitude in decimal degrees.
"""

from __future__ import annotations

STADIUMS: dict[str, dict] = {
    "Arsenal FC": {"stadium": "Emirates Stadium", "lat": 51.5549, "lon": -0.1084},
    "Aston Villa FC": {"stadium": "Villa Park", "lat": 52.5091, "lon": -1.8848},
    "AFC Bournemouth": {"stadium": "Vitality Stadium", "lat": 50.7352, "lon": -1.8380},
    "Brentford FC": {"stadium": "Gtech Community Stadium", "lat": 51.4907, "lon": -0.2887},
    "Brighton & Hove Albion FC": {"stadium": "American Express Stadium", "lat": 50.8617, "lon": -0.0837},
    "Burnley FC": {"stadium": "Turf Moor", "lat": 53.7890, "lon": -2.2302},
    "Chelsea FC": {"stadium": "Stamford Bridge", "lat": 51.4817, "lon": -0.1910},
    "Crystal Palace FC": {"stadium": "Selhurst Park", "lat": 51.3983, "lon": -0.0855},
    "Everton FC": {"stadium": "Hill Dickinson Stadium", "lat": 53.4249, "lon": -3.0027},
    "Fulham FC": {"stadium": "Craven Cottage", "lat": 51.4749, "lon": -0.2216},
    "Leeds United FC": {"stadium": "Elland Road", "lat": 53.7778, "lon": -1.5722},
    "Liverpool FC": {"stadium": "Anfield", "lat": 53.4308, "lon": -2.9608},
    "Manchester City FC": {"stadium": "Etihad Stadium", "lat": 53.4831, "lon": -2.2004},
    "Manchester United FC": {"stadium": "Old Trafford", "lat": 53.4631, "lon": -2.2913},
    "Newcastle United FC": {"stadium": "St James' Park", "lat": 54.9756, "lon": -1.6217},
    "Nottingham Forest FC": {"stadium": "City Ground", "lat": 52.9400, "lon": -1.1327},
    "Sunderland AFC": {"stadium": "Stadium of Light", "lat": 54.9142, "lon": -1.3883},
    "Tottenham Hotspur FC": {"stadium": "Tottenham Hotspur Stadium", "lat": 51.6043, "lon": -0.0664},
    "West Ham United FC": {"stadium": "London Stadium", "lat": 51.5386, "lon": -0.0166},
    "Wolverhampton Wanderers FC": {"stadium": "Molineux Stadium", "lat": 52.5903, "lon": -2.1305},
}


def get_stadium(team: str) -> dict | None:
    """Return the stadium entry (name + lat/lon) for a team, or None if unmapped."""
    return STADIUMS.get(team)
