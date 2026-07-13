.PHONY: install test lint run compose-up compose-down

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

run:
	MULTIROUTER_CONFIG_DIR=configs uvicorn multirouter.api:app --reload --port 8000

compose-up:
	docker compose up --build

compose-down:
	docker compose down
