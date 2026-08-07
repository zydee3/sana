# Sana scraper. Python (uv-managed); crawls open-access research into the corpus.
# Requires uv on PATH (https://docs.astral.sh/uv/); it owns the venv and tool installs.

.PHONY: all help uv-check setup run build docker deploy lint typecheck test check clean

# Default: build the artifact (the sana-scraper image).
all: build

help:
	@echo "make setup        - create the venv + install dev tools (uv sync)"
	@echo "make run ARGS=…    - crawler CLI (ARGS='add-topic \"sleep\"', ARGS='run --once')"
	@echo "make build        - build the sana-scraper image"
	@echo "make deploy       - build, import into k3s, render topics.md -> ConfigMap, apply"
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
	uv run python -m scraper $(ARGS)

lint: uv-check
	uv run ruff check .
	uv run ruff format --check .

typecheck: uv-check
	uv run mypy

test: uv-check
	uv run pytest -q

check: lint typecheck test

build:
	docker build -t sana-scraper .

docker: build

# deploy is self-sufficient: first run creates /sana-data; later runs skip it.
# Credentials are a live hostPath mount of ~/.claude/.credentials.json (see
# deploy.yaml) — no secret to sync, and token rotation on the host propagates.
# Topics live in topics.md, re-rendered into the sana-topics ConfigMap each run.
deploy: build
	@command -v k3s >/dev/null || { echo "Error: k3s not found - install k3s first"; exit 1; }
	@[ -d /sana-data ] || { \
	  echo "creating /sana-data (one-time, needs sudo; symlink it to a disk first if preferred)"; \
	  sudo mkdir /sana-data && sudo chown 1000 /sana-data; }
	@[ -f $$HOME/.claude/.credentials.json ] || \
	  { echo "Error: ~/.claude/.credentials.json not found - install Claude Code and log in first"; exit 1; }
	docker save sana-scraper | sudo k3s ctr images import -
	kubectl create configmap sana-topics --from-file=topics=topics.md \
	  --dry-run=client -o yaml | kubectl apply -f -
	@# discord reporter token: copy from the discord-bot namespace (ns-scoped secrets);
	@# skipped quietly if that project's secret is absent (reporter then stays off)
	@TOKEN=$$(kubectl -n discord-bot get secret discord-bot-secrets \
	    -o jsonpath='{.data.DISCORD_BOT_TOKEN}' 2>/dev/null); \
	  [ -z "$$TOKEN" ] && echo "note: discord-bot-secrets not found; reporter disabled" || \
	  kubectl create secret generic discord-bot-token \
	    --from-literal=DISCORD_BOT_TOKEN="$$(echo $$TOKEN | base64 -d)" \
	    --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f deploy.yaml
	kubectl rollout restart deployment/sana-scraper

clean:
	rm -rf .venv .mypy_cache .pytest_cache .ruff_cache
