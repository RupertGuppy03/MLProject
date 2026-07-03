"""Team brand colours for the dashboard.

Maps each canonical dataset team name to its main club colour (hex), used to colour team names
in the UI and, later, the match-context charts. Values are converted exactly from the supplied
RGB brand colours. A neutral grey fallback keeps the UI safe if the dataset ever gains a team
that isn't mapped here.
"""

from __future__ import annotations

# Neutral fallback for any team without a mapped colour.
DEFAULT_COLOR = "#9CA3AF"

# Canonical team name -> hex brand colour.
TEAM_COLORS: dict[str, str] = {
    "AFC Bournemouth": "#B50E12",
    "Arsenal FC": "#EF0107",
    "Aston Villa FC": "#670E36",
    "Brentford FC": "#FFB400",
    "Brighton & Hove Albion FC": "#0057B8",
    "Burnley FC": "#FFF30D",
    "Chelsea FC": "#034694",
    "Crystal Palace FC": "#1B458F",
    "Everton FC": "#003399",
    "Fulham FC": "#CC0000",
    "Ipswich Town FC": "#3A64A3",
    "Leeds United FC": "#FFCD00",
    "Leicester City FC": "#FDBE11",
    "Liverpool FC": "#C8102E",
    "Luton Town FC": "#F78F1E",
    "Manchester City FC": "#1C2C5B",
    "Manchester United FC": "#DA020E",
    "Newcastle United FC": "#BBBCBC",
    "Nottingham Forest FC": "#DD0000",
    "Sheffield United FC": "#EE2737",
    "Southampton FC": "#22B24C",
    "Sunderland AFC": "#EB172B",
    "Tottenham Hotspur FC": "#132257",
    "West Ham United FC": "#7A263A",
    "Wolverhampton Wanderers FC": "#FDB913",
}


def get_team_color(team: str, default: str = DEFAULT_COLOR) -> str:
    """Return the hex brand colour for a team, or a neutral grey fallback if unmapped."""
    return TEAM_COLORS.get(team, default)
