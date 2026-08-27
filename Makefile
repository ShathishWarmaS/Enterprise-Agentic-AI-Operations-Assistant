.PHONY: install seed api frontend test lint format check eval docker-up docker-down

install:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

seed:
	uv run python scripts/seed.py

api:
	uv run uvicorn app.main:app --reload

frontend:
	uv run streamlit run frontend/streamlit_app.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run black --check .
	uv run mypy app

format:
	uv run ruff check --fix .
	uv run black .

check:
	bash scripts/check.sh

eval:
	uv run python scripts/run_eval.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
