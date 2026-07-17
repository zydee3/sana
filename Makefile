# Sana task runner.
# Wraps npm/expo and pins Node 22 via nvm — the system Node is 18 and cannot run Expo.

NVM := export NVM_DIR="$$HOME/.nvm"; . "$$NVM_DIR/nvm.sh"; nvm use 22 >/dev/null

.PHONY: help install web ios typecheck lint build deploy

help:
	@echo "make install    - install dependencies"
	@echo "make web        - run the web dev server (http://localhost:8081)"
	@echo "make ios        - run the iOS dev server (needs macOS/simulator)"
	@echo "make typecheck  - tsc --noEmit"
	@echo "make lint       - eslint + prettier check"
	@echo "make build      - export the web bundle to dist/"
	@echo "make deploy     - build, then deploy web (host not configured yet)"

install:
	@$(NVM); npm install

web:
	@$(NVM); npm run web

ios:
	@$(NVM); npm run ios

typecheck:
	@$(NVM); node --stack-size=8000 ./node_modules/typescript/lib/tsc.js --noEmit

lint:
	@$(NVM); npm run lint

build:
	@$(NVM); npx expo export --platform web

deploy: build
	@echo "Web bundle built to dist/."
	@echo "Deploy target not configured. Choose a host (EAS Hosting / Netlify / Vercel)"
	@echo "and it gets wired here. iOS distribution is a separate eas build/submit flow."
	@exit 1
