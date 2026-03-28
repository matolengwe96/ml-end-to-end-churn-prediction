.PHONY: install train test test-cov lint format app api docker-build docker-up pre-commit ci

install:
	pip install -r requirements.txt

train:
	python -m src.train

test:
	pytest

test-cov:
	pytest --cov=src --cov=api --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests app api

format:
	ruff format src tests app api

app:
	streamlit run app/app.py

api:
	uvicorn api.main:app --reload --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up -d

pre-commit:
	pre-commit run --all-files

ci: lint test
