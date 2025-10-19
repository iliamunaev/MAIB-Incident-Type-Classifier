# MAIB Incident Type Classifier - Makefile

.PHONY: help install install-dev train inference evaluate test clean docker-build docker-run

# Default target
help:
	@echo "MAIB Incident Type Classifier - Available Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install with development dependencies"
	@echo ""
	@echo "Training and Inference:"
	@echo "  train        Train the model"
	@echo "  inference    Run inference"
	@echo "  evaluate     Evaluate the model"
	@echo ""
	@echo "Development:"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        Clean build artifacts"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,jupyter]"

# Training
train:
	python scripts/train.py --data_path data/maib-incident-reports-dataset.jsonl

# Inference
inference:
	python scripts/inference.py --model_path outputs/best_model --interactive

# Evaluation
evaluate:
	python scripts/evaluate.py --model_path outputs/best_model --data_path data/maib-incident-reports-dataset.jsonl

# Testing
test:
	pytest tests/ -v

# Linting
lint:
	flake8 src/ scripts/
	mypy src/

# Formatting
format:
	black src/ scripts/

# Clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf outputs/
	rm -rf logs/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker
docker-build:
	docker build -t maib-classifier .

docker-run:
	docker run -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs maib-classifier

# Development setup
setup-dev: install-dev
	mkdir -p data outputs logs
	@echo "Development environment setup complete!"
	@echo "Place your data file in the data/ directory"
	@echo "Training outputs will be saved to outputs/"
	@echo "Logs will be saved to logs/"
