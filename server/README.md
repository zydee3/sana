# server

Sana's backend: per message it decrypts the chat in memory, retrieves from the research corpus,
builds a prompt, calls Claude Code headless (`claude -p`), and returns the reply.

What exists so far: the **container runtime** — an image with Claude Code installed — and a k3s
deployment. The pipeline library and HTTP server are not built (the pod idles until they are).

```bash
make build      # build the sana-server image
make deploy     # build + import into k3s + kubectl apply deploy.yaml
```

## Prerequisites

docker, k3s, and Claude Code logged in on the host (`~/.claude/.credentials.json`). That's it —
`make deploy` handles everything else on first run:

- **`/sana-data`** — server state (the ciphertext chat DB) mounts from here; created if missing
  (writable by UID 1000, the container user). To place it on a specific disk, symlink it before
  the first deploy: `sudo ln -s /path/on/some/disk /sana-data`.
- **`claude-credentials` secret** — created from `~/.claude/.credentials.json` if missing.

Nothing in this repo is machine-specific. The hosted/GA deployment will authenticate with
Sana's own API key (env var) instead of personal credentials.

## Notes

- Logs go to stdout/stderr (`kubectl logs`); no log mounts. Prometheus `/metrics` comes with
  the server code. One `/data` volume is the only state.
- k3s uses containerd, not Docker — `make deploy` handles the `docker save | k3s ctr images
  import` step; `imagePullPolicy: Never` keeps k3s from trying a registry.
- The credentials secret is mounted via `subPath` (fine for secret volumes; the known subPath
  problem is with single-file hostPath mounts, which this avoids).
- Single-node assumption: `/sana-data` is a hostPath; with more than one node, pin the pod or
  switch to a local PV.
