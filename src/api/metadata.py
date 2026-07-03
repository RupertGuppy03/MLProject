"""Freshness metadata for the API.

`GET /metadata` needs to tell clients two dates:
  - ``data_through_date`` — the latest match date the served model was built on
  - ``last_updated_date`` — when the artifacts were last (re)generated

Both live in ``artifacts/last_updated.json``. This module writes and reads that file.
The writer defaults ``data_through_date`` to the ``as_of_date`` already stored in
``current_elo.json`` (both come from the same canonical-dataset walk), so the two artifacts
can't disagree. The Sprint 4 refresh script is expected to call `write_last_updated` too.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.config import ARTIFACTS_DIR

# Freshness artifact read by GET /metadata.
LAST_UPDATED_PATH = ARTIFACTS_DIR / "last_updated.json"
# Source of the default data_through_date (written by save_state / save_final_artifacts).
CURRENT_ELO_PATH = ARTIFACTS_DIR / "current_elo.json"


def _as_of_from_elo(path: Path = CURRENT_ELO_PATH) -> str:
    """Read the as_of_date recorded in the Elo state artifact."""
    payload = json.loads(Path(path).read_text())
    return payload["as_of_date"]


def write_last_updated(
    data_through_date: str | None = None,
    last_updated_date: str | None = None,
    path: Path = LAST_UPDATED_PATH,
    elo_path: Path = CURRENT_ELO_PATH,
) -> dict[str, str]:
    """Write the freshness artifact and return its payload.

    Args:
        data_through_date: latest match date the model was built on. Defaults to the
            ``as_of_date`` in ``current_elo.json`` so it always matches the served artifacts.
        last_updated_date: regeneration date. Defaults to today.
        path: where to write (defaults to ``artifacts/last_updated.json``).
        elo_path: Elo artifact used to derive the default data_through_date.
    """
    if data_through_date is None:
        data_through_date = _as_of_from_elo(elo_path)
    if last_updated_date is None:
        last_updated_date = date.today().isoformat()

    payload = {
        "last_updated_date": last_updated_date,
        "data_through_date": data_through_date,
    }

    # Atomic write (tmp + replace), matching _save_schema / save_state.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return payload


def read_last_updated(path: Path = LAST_UPDATED_PATH) -> dict[str, str] | None:
    """Return the freshness payload, or None if the artifact hasn't been written yet."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    """Generate artifacts/last_updated.json from the current Elo as_of_date."""
    payload = write_last_updated()
    print(f"Wrote {LAST_UPDATED_PATH.name}: {payload}")


if __name__ == "__main__":
    main()
