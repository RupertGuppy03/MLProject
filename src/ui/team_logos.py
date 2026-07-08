"""Team crest lookup for the dashboard.

Maps each canonical team name to its logo file in ``logos/128x128/``. Returns None for any team
without a crest so the UI can fall back to the coloured name. Crests are club trademarks, used here
for a non-commercial portfolio project.
"""

from __future__ import annotations

from src.config import PROJECT_ROOT

LOGO_DIR = PROJECT_ROOT / "logos" / "128x128"

# Canonical team name -> filename slug (files are "<slug>.football-logos.cc.png").
_SLUGS: dict[str, str] = {
    "AFC Bournemouth": "bournemouth",
    "Arsenal FC": "arsenal",
    "Aston Villa FC": "aston-villa",
    "Brentford FC": "brentford",
    "Brighton & Hove Albion FC": "brighton",
    "Burnley FC": "burnley",
    "Chelsea FC": "chelsea",
    "Crystal Palace FC": "crystal-palace",
    "Everton FC": "everton",
    "Fulham FC": "fulham",
    "Leeds United FC": "leeds-united",
    "Liverpool FC": "liverpool",
    "Manchester City FC": "manchester-city",
    "Manchester United FC": "manchester-united",
    "Newcastle United FC": "newcastle",
    "Nottingham Forest FC": "nottingham-forest",
    "Sunderland AFC": "sunderland",
    "Tottenham Hotspur FC": "tottenham",
    "West Ham United FC": "west-ham",
    "Wolverhampton Wanderers FC": "wolves",
}


def get_team_logo(team: str) -> str | None:
    """Return the path to a team's crest, or None if there's no logo for it."""
    slug = _SLUGS.get(team)
    if slug is None:
        return None
    path = LOGO_DIR / f"{slug}.football-logos.cc.png"
    return str(path) if path.exists() else None
