"""Short display names for teams whose canonical name is long.

The canonical names (used by the API/model) stay unchanged; these are UI-only labels.
"""

from __future__ import annotations

DISPLAY_NAMES: dict[str, str] = {
    "Brighton & Hove Albion FC": "Brighton",
    "Wolverhampton Wanderers FC": "Wolves",
}


def display_name(team: str) -> str:
    """Short label for a team, or the canonical name if there's no override."""
    return DISPLAY_NAMES.get(team, team)
