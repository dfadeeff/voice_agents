.PHONY: setup setup-cloud run run-prod seed models test lint benchmark compare-latency clean

setup: models seed
	pip install -e "backend/.[dev]"
	pip install pre-commit
	pre-commit install

setup-cloud:
	pip install -e "backend/.[cloud,dev]"
	@echo "Set API keys in .env (see .env.cloud.example)"

models:
	cd backend && python3 scripts/download_models.py

seed:
	cd backend && python3 scripts/seed_calendar.py

calendar-full:
	cd backend && python3 scripts/fill_calendar.py

run:
	cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-prod:
	cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

test:
	cd backend && pytest -v --tb=short

lint:
	ruff check backend/ --config backend/pyproject.toml
	ruff format --check backend/ --config backend/pyproject.toml

benchmark:
	cd backend && python3 scripts/benchmark_models.py

# Compare per-component latency between two call logs (default: two most recent).
# Override with ARGS, e.g. make compare-latency ARGS="logs/<local>.json logs/<cloud>.json --labels local cloud"
compare-latency:
	cd backend && python3 scripts/compare_latency.py $(ARGS)

clean:
	rm -rf backend/data/ backend/models/ __pycache__
