.DEFAULT_GOAL := help
UID := $(shell id -u)
GID := $(shell id -g)
export UID
export GID

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-18s %s\n", $$1, $$2}'

reproduce:  ## Run the full pipeline in Docker (~6 min)
	docker compose up --build --exit-code-from reproduce reproduce

validate:  ## Check the data contract only
	docker compose --profile validate run --rm validate

search:  ## Run the optimisation campaign
	docker compose --profile search run --rm search

test:  ## Run the test suite locally
	uv run pytest

lint:  ## Lint and format-check
	uv run ruff check src tests

dod:  ## Definition-of-Done gate for the current change
	uv run ruff check src tests
	uv run pytest -q
	@[ -z "$$(git status --porcelain)" ] || (echo "working tree dirty"; exit 1)
	@echo "Definition of Done satisfied."

verify:  ## Prove a container run reproduces the committed results
	docker compose up --build --exit-code-from reproduce reproduce
	git diff --exit-code reports/results.json
	@echo "Container output matches the committed golden results."

.PHONY: help reproduce validate search test lint dod verify
