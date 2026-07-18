# Sana

A wellness companion — a health chat app where you talk with an AI companion whose responses
are grounded in published scientific research.

Monorepo. Each project is self-contained with its own build system and will eventually run as
its own container:

- `app/` — the client: Expo / React Native + NativeWind, one codebase for web + iOS.
  See `app/README.md` for how it works and how to run it.
- `scraper/` — Python ingester that collects research papers into the corpus the AI draws on.
  Not built yet.
- `server/` — the backend (TypeScript/Node): decrypts chats in memory, retrieves from the
  corpus, calls Claude Code headless, returns replies. Not built yet.

## Common tasks

```bash
make check      # every project's check gate
make app-web    # any app/ target: app-web, app-ios, app-check, ...
```

Or work inside a project directly: `cd app && make web`.
