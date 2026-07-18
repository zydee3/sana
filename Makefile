# Monorepo task runner. Each project owns its build system (app/Makefile, ...);
# this file only delegates — projects will eventually ship as separate containers.

.PHONY: all help setup check docker deploy app-% server-% scraper-%

# Default: build every project's artifact.
all:
	@$(MAKE) -C app all
	@$(MAKE) -C server all
	@$(MAKE) -C scraper all

help:
	@echo "make            - build every project's artifact (app bundle, server image)"
	@echo "make setup      - first-time setup in every project (deps + prereq checks)"
	@echo "make check      - run every project's check gate (app, scraper)"
	@echo "make docker     - build every project's image (sana-app, sana-server)"
	@echo "make deploy     - deploy (currently: server -> k3s; app hosting not configured)"
	@echo "make app-<t>    - run target <t> in app/ (make app-web, make app-check, ...)"
	@echo "make server-<t> - run target <t> in server/"
	@echo "make scraper-<t> - run target <t> in scraper/"
	@echo "projects: app/ (Expo web+iOS) · server/ (claude runtime, pipeline not built) · scraper/ (Python corpus ingester)"

setup:
	@$(MAKE) -C app setup
	@$(MAKE) -C server setup
	@$(MAKE) -C scraper setup

check:
	@$(MAKE) -C app check
	@$(MAKE) -C scraper check

docker:
	@$(MAKE) -C app docker
	@$(MAKE) -C server build

# app deploy joins when a web host is chosen (app/Makefile deploy is a stub).
deploy:
	@$(MAKE) -C server deploy

app-%:
	@$(MAKE) -C app $*

server-%:
	@$(MAKE) -C server $*

scraper-%:
	@$(MAKE) -C scraper $*
