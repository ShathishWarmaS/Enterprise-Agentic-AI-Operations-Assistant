#!/usr/bin/env bash
# Run every gate CI runs, locally. Assumes an active venv with `.[dev]` installed.
set -euo pipefail

echo "== ruff =="
ruff check .

echo "== black =="
black --check .

echo "== mypy =="
mypy app

echo "== import check =="
python -c "import app.main; import app.services.evaluation; import app.agents.orchestrator"

echo "== pytest =="
pytest -q

echo "== startup check =="
python - <<'PY'
from fastapi.testclient import TestClient
from app.main import app
with TestClient(app) as c:
    r = c.get("/health")
    r.raise_for_status()
    print("startup OK:", r.json())
PY

echo "== seed + eval =="
python scripts/seed.py
python scripts/run_eval.py --min-pass 0.7

echo
echo "all checks passed"
