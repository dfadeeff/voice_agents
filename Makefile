.PHONY: setup run seed models clean

setup: models seed
	pip install -e "backend/.[dev]"

models:
	python3 scripts/download_models.py

seed:
	mkdir -p data
	python3 scripts/seed_calendar.py

run:
	PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf data/ models/ __pycache__
