#!/usr/bin/env bash
# Smoke test for the FastAPI service. Hits every endpoint and prints the responses.
#
# Start the API first:  uvicorn src.api.main:app --reload
# Then run:             bash scripts/smoke_api.sh
# Override the target:  API_BASE_URL=http://192.168.1.20:8000 bash scripts/smoke_api.sh
set -euo pipefail

BASE_URL="${API_BASE_URL:-http://localhost:8000}"

echo "== GET /health =="
curl -sf "${BASE_URL}/health"
echo

echo "== GET /metadata =="
curl -sf "${BASE_URL}/metadata"
echo

echo "== GET /teams =="
curl -sf "${BASE_URL}/teams"
echo

echo "== POST /predict =="
curl -sf -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal FC", "away_team": "Chelsea FC"}'
echo

echo "Smoke test passed."
