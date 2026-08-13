# Sana corpus processing. Python (uv-managed); scores, labels, chunks and indexes the
# corpus the scraper collected. Requires uv on PATH (https://docs.astral.sh/uv/).

.PHONY: all help uv-check setup run lint typecheck test check clean

all: check

help:
	@echo "make setup        - create the venv + install dev tools (uv sync)"
	@echo "make run ARGS=…    - processing CLI (ARGS='sample --n 1000', ARGS='abstracts …')"
	@echo "make lint          - ruff check + format check"
	@echo "make typecheck     - mypy (strict)"
	@echo "make test          - pytest"
	@echo "make check         - lint + typecheck + test (the verification gate)"

uv-check:
	@command -v uv >/dev/null 2>&1 || \
	  { echo "Error: uv not found on PATH (https://docs.astral.sh/uv/)"; exit 1; }

setup: uv-check
	uv sync

run: uv-check
	uv run python -m corpus $(ARGS)

lint: uv-check
	uv run ruff check .
	uv run ruff format --check .

typecheck: uv-check
	uv run mypy

test: uv-check
	uv run pytest -q

check: lint typecheck test

clean:
	rm -rf .venv .mypy_cache .pytest_cache .ruff_cache
