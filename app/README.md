# Sana

A wellness companion — a health chat app where you talk with an AI companion. Cross-platform
(web + iOS), built with Expo / React Native + NativeWind.

## Run

Requires Node 22.

```bash
make web        # web dev server (http://localhost:8081)
make ios        # iOS dev server (needs macOS / simulator)
make build      # bundle web to dist/
make typecheck
make lint
make test       # vitest
make check      # typecheck + lint + test + build
```

`make` sets the Node version for you. Without it, activate Node 22 first (`nvm use 22`), then use
the `npm run web` / `npm run ios` scripts directly.

## How it works

- **Login is a key, not a password.** Your identity is a client-side 256-bit **encryption key**.
  Register generates it — save it by downloading it or storing it on this device. The key is never
  sent to a server. Log in by pasting the key, or by using the one saved on this device.
  Losing the key means losing access; there is no recovery.
- **Chats** are created and revisited from the home screen and stored on your device.

Assistant replies are currently a placeholder — the AI backend (Claude Code headless, `claude -p`)
is not connected yet.

## Layout

- `src/auth/` — key generation, device storage, session
- `src/chat/` — chat screens, state, local persistence
- `src/components/ui/` — react-native-reusables UI components
- `App.tsx` — auth gate → chat app
