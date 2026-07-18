# Monorepo task runner. Each project owns its build system (app/Makefile, ...);
# this file only delegates — projects will eventually ship as separate containers.

.PHONY: help check app-%

help:
	@echo "make check      - run every project's check gate (currently: app)"
	@echo "make app-<t>    - run target <t> in app/ (make app-web, make app-check, ...)"
	@echo "projects: app/ (Expo web+iOS) · scraper/ (Python, not built)"

check:
	@$(MAKE) -C app check

app-%:
	@$(MAKE) -C app $*
