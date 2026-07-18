# Sana app task runner. Requires Node >= 22 on PATH (see package.json engines);
# how it gets there (apt, nvm, brew, ...) is the machine's business, not the repo's.

.PHONY: help node-check install web ios typecheck lint test build check docker deploy

help:
	@echo "make install    - install dependencies"
	@echo "make web        - run the web dev server (http://localhost:8081)"
	@echo "make ios        - run the iOS dev server (needs macOS/simulator)"
	@echo "make typecheck  - tsc --noEmit"
	@echo "make lint       - eslint + prettier check"
	@echo "make test       - vitest run"
	@echo "make build      - export the web bundle to dist/"
	@echo "make check      - typecheck + lint + test + build"
	@echo "make docker     - build the sana-app image (web bundle served by nginx)"
	@echo "make deploy     - build, then deploy web (host not configured yet)"

node-check:
	@v=$$(node -v 2>/dev/null | cut -c2- | cut -d. -f1); \
	[ -n "$$v" ] && [ "$$v" -ge 22 ] || \
	  { echo "Error: Node >= 22 required (found: $$(node -v 2>/dev/null || echo none))"; exit 1; }

install: node-check
	npm install

web: node-check
	npm run web

ios: node-check
	npm run ios

typecheck: node-check
	node --stack-size=8000 ./node_modules/typescript/lib/tsc.js --noEmit

lint: node-check
	npm run lint

test: node-check
	npx vitest run

build: node-check
	npx expo export --platform web

check: typecheck lint test build

docker:
	docker build -t sana-app .

deploy: build
	@echo "Web bundle built to dist/."
	@echo "Deploy target not configured. Choose a host (EAS Hosting / Netlify / Vercel)"
	@echo "and it gets wired here. iOS distribution is a separate eas build/submit flow."
	@exit 1
