# server

Sana's backend: per message it decrypts the chat in memory, retrieves from the research corpus,
builds a prompt, calls Claude Code headless (`claude -p`), and returns the reply.

What exists so far: the **container runtime** — an image with Claude Code installed — and a k3s
deployment. The pipeline library and HTTP server are not built (the pod idles until they are).

```bash
make build      # build the sana-server image
make deploy     # build + import into k3s + kubectl apply deploy.yaml
```

## One-time setup (per machine)

Nothing in this repo is machine-specific; each host provides two things under fixed names:

1. **`/sana-data`** — server state (the ciphertext chat DB) mounts from here. Create a
   directory, or symlink it to any disk. It must be writable by UID 1000 (the container user).

   ```bash
   sudo mkdir /sana-data && sudo chown 1000 /sana-data
   # or: sudo ln -s /path/on/some/disk /sana-data
   ```

2. **`claude-credentials` secret** — Claude Code credentials, created from your own login:

   ```bash
   sudo k3s kubectl create secret generic claude-credentials \
     --from-file=$HOME/.claude/.credentials.json
   ```

The hosted/GA deployment will authenticate with Sana's own API key (env var) instead of
personal credentials.

## Notes

- Logs go to stdout/stderr (`kubectl logs`); no log mounts. Prometheus `/metrics` comes with
  the server code. One `/data` volume is the only state.
- k3s uses containerd, not Docker — `make deploy` handles the `docker save | k3s ctr images
  import` step; `imagePullPolicy: Never` keeps k3s from trying a registry.
- The credentials secret is mounted via `subPath` (fine for secret volumes; the known subPath
  problem is with single-file hostPath mounts, which this avoids).
- Single-node assumption: `/sana-data` is a hostPath; with more than one node, pin the pod or
  switch to a local PV.
