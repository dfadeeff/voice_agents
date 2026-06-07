.PHONY: setup setup-cloud run run-prod seed models test lint benchmark clean

setup: models seed
	pip install -e "backend/.[dev]"
	pip install pre-commit
	pre-commit install

setup-cloud:
	pip install -e "backend/.[cloud,dev]"
	@echo "Set API keys in .env (see .env.cloud.example)"

models:
	python3 scripts/download_models.py

seed:
	mkdir -p data
	python3 scripts/seed_calendar.py

run:
	PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-prod:
	PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

test:
	cd backend && pytest -v --tb=short

lint:
	ruff check backend/ --config backend/pyproject.toml
	ruff format --check backend/ --config backend/pyproject.toml

benchmark:
	python3 scripts/benchmark_models.py

clean:
	rm -rf data/ models/ __pycache__
