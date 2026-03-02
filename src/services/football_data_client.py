import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from src.config import FOOTBALL_DATA_API_TOKEN_ENV, FOOTBALL_DATA_BASE_URL


@dataclass
class FootballDataClient:
    base_url: str = FOOTBALL_DATA_BASE_URL
    token_env: str = FOOTBALL_DATA_API_TOKEN_ENV
    timeout_s: int = 30

    def _headers(self) -> Dict[str, str]:
        token = os.getenv(self.token_env)
        if not token:
            raise RuntimeError(
                f"Missing API token. Set environment variable: {self.token_env}"
            )
        return {"X-Auth-Token": token}

    def get_matches(self, competition_code: str, season: int) -> Dict[str, Any]:
        """
        Football-Data endpoint: /competitions/{code}/matches?season=YYYY
        Returns JSON with 'matches' list.
        """
        url = f"{self.base_url}/competitions/{competition_code}/matches"
        r = requests.get(
            url,
            headers=self._headers(),
            params={"season": season},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()
