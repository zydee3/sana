# Monorepo task runner. Each project owns its build system (app/Makefile, ...);
# this file only delegates — projects will eventually ship as separate containers.

.PHONY: help check docker app-% server-%

help:
	@echo "make check      - run every project's check gate (currently: app)"
	@echo "make docker     - build every project's image (sana-app, sana-server)"
	@echo "make app-<t>    - run target <t> in app/ (make app-web, make app-check, ...)"
	@echo "make server-<t> - run target <t> in server/"
	@echo "projects: app/ (Expo web+iOS) · server/ (claude runtime, pipeline not built) · scraper/ (Python, not built)"

check:
	@$(MAKE) -C app check

docker:
	@$(MAKE) -C app docker
	@$(MAKE) -C server build

app-%:
	@$(MAKE) -C app $*
