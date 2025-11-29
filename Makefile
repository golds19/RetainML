.PHONY: install test lint format run clean help dev-setup

help:
	@echo "Available commands:"
	@echo "  make install     - Install project dependencies"
	@echo "  make dev-setup   - Set up development environment"
	@echo "  make test        - Run tests with coverage"
	@echo "  make lint        - Run linters (flake8)"
	@echo "  make format      - Format code with black and isort"
	@echo "  make run         - Start the FastAPI server"
	@echo "  make mlflow      - Start MLflow UI"
	@echo "  make clean       - Remove generated files and caches"
	@echo "  make data        - Generate synthetic data"

install:
	pip install -r requirements.txt

dev-setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ --max-line-length=100 --exclude=.venv,__pycache__

format:
	black src/ tests/ scripts/ --line-length=100
	isort src/ tests/ scripts/ --profile black

run:
	uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

data:
	python scripts/generate_data.py --size 50000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.log" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov/ .mypy_cache/ 2>/dev/null || true
	@echo "Cleaned up generated files and caches"
