# Sana

A wellness companion — a health chat app where you talk with an AI companion whose responses
are grounded in published scientific research.

Monorepo. Each project is self-contained with its own build system and will eventually run as
its own container:

- `app/` — the client: Expo / React Native + NativeWind, one codebase for web + iOS.
  See `app/README.md` for how it works and how to run it.
- `scraper/` — Python ingester that collects open-access research papers into the corpus the
  AI draws on. Walking skeleton works (`make run QUERY=...`); see `scraper/README.md`.
- `server/` — the backend (TypeScript/Node): decrypts chats in memory, retrieves from the
  corpus, calls Claude Code headless, returns replies. Container runtime exists (Claude Code
  image); the pipeline itself is not built. See `server/README.md`.

## Common tasks

```bash
make setup      # first-time: deps + prereq checks in every project
make            # build every project's artifact (app bundle, server image)
make deploy     # deploy (server -> k3s; deploys build first)
make check      # every project's check gate
make app-web    # any project's target: app-web, server-deploy, ...
```

Or work inside a project directly: `cd app && make web`.
