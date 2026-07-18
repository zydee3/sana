# Sana scraper. Python (uv-managed); pulls open-access research PDFs into the corpus.
# Requires uv on PATH (https://docs.astral.sh/uv/); it owns the venv and tool installs.

.PHONY: all help uv-check setup run lint typecheck test check clean

# Deployable artifact (its own container) is not built yet; the CLI runs via `make run`.
all:
	@echo "scraper: CLI only so far (make run); no container artifact yet"

help:
	@echo "make setup        - create the venv + install dev tools (uv sync)"
	@echo 'make run QUERY=…   - pull one OA paper into ./corpus (QUERY="mindfulness anxiety")'
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
	uv run python -m scraper "$(QUERY)" $(ARGS)

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
