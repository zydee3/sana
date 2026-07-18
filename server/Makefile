# Sana backend. Container runtime only so far — the pipeline/HTTP server is not built.

.PHONY: help build deploy

help:
	@echo "make build      - build the sana-server image"
	@echo "make deploy     - build, import the image into k3s, apply deploy.yaml"

build:
	docker build -t sana-server .

deploy: build
	docker save sana-server | sudo k3s ctr images import -
	sudo k3s kubectl apply -f deploy.yaml
