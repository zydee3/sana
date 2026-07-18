# Sana backend. Container runtime only so far — the pipeline/HTTP server is not built.

.PHONY: all help setup build deploy

# Default: build the artifact (the sana-server image).
all: build

help:
	@echo "make setup      - check deploy prerequisites (docker, k3s, claude login)"
	@echo "make build      - build the sana-server image"
	@echo "make deploy     - build, import the image into k3s, apply deploy.yaml"

# Nothing to install — server deps live in the image. Setup just verifies prereqs.
setup:
	@ok=1; \
	command -v docker >/dev/null && echo "ok: docker" || { echo "missing: docker"; ok=0; }; \
	command -v k3s >/dev/null && echo "ok: k3s" || { echo "missing: k3s"; ok=0; }; \
	[ -f $$HOME/.claude/.credentials.json ] && echo "ok: claude credentials" \
	  || { echo "missing: claude login (~/.claude/.credentials.json)"; ok=0; }; \
	[ $$ok = 1 ] || exit 1

build:
	docker build -t sana-server .

# deploy is self-sufficient: first run creates /sana-data and the credentials
# secret; later runs skip what already exists.
deploy: build
	@command -v k3s >/dev/null || { echo "Error: k3s not found - install k3s first"; exit 1; }
	@[ -d /sana-data ] || { \
	  echo "creating /sana-data (one-time, needs sudo; symlink it to a disk first if preferred)"; \
	  sudo mkdir /sana-data && sudo chown 1000 /sana-data; }
	@sudo k3s kubectl get secret claude-credentials >/dev/null 2>&1 || { \
	  [ -f $$HOME/.claude/.credentials.json ] || \
	    { echo "Error: ~/.claude/.credentials.json not found - install Claude Code and log in first"; exit 1; }; \
	  echo "creating claude-credentials secret (one-time)"; \
	  sudo k3s kubectl create secret generic claude-credentials \
	    --from-file=$$HOME/.claude/.credentials.json; }
	docker save sana-server | sudo k3s ctr images import -
	sudo k3s kubectl apply -f deploy.yaml
