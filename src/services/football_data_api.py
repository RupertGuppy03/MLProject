# src/services/football_data_api.py
"""
Small client for football-data.org API.

Why this exists:
- Keeps HTTP logic out of dataset/feature/model code (separation of concerns)
- Makes it easy to swap API providers later if needed
- Central place for retries / error handling
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests


class FootballDataAPI:
    """
    Minimal wrapper around requests.get for football-data.org.

    - base_url: API root (e.g., https://api.football-data.org/v4)
    - api_key: your X-Auth-Token
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")  # avoid double slashes
        self.api_key = api_key
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("Missing FOOTBALL_API_KEY. Add it to your .env file.")

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Makes a GET request and returns parsed JSON (dict).

        Adds:
        - auth header (X-Auth-Token)
        - retry with exponential backoff for temporary failures/rate limits
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"X-Auth-Token": self.api_key}

        # Retry a few times for transient errors
        for attempt in range(5):
            response = requests.get(
                url, headers=headers, params=params, timeout=self.timeout
            )

            # 429 = rate limited, 5xx = temporary server issues
            if response.status_code in (429, 502, 503, 504):
                time.sleep(2**attempt)  # 1s, 2s, 4s, 8s, 16s
                continue

            # For other non-200 errors, raise a clear exception
            response.raise_for_status()
            return response.json()

        raise RuntimeError(f"GET {url} failed after retries.")
