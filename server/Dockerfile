# Backend runtime: Claude Code headless in a container.
# Credentials are NEVER baked into the image — mount them at runtime:
#   docker run -v ~/.claude/.credentials.json:/home/node/.claude/.credentials.json:ro sana-server
FROM node:22-slim
RUN npm install -g @anthropic-ai/claude-code@latest
# /data holds server state (ciphertext chat DB) — host-mounted at runtime.
RUN mkdir /data && chown node:node /data
# node is UID 1000, matching the host user — required: .credentials.json is mode 0600,
# and the host-mounted /data dir must be writable by the same UID.
# On a host where the user is not UID 1000: RUN useradd -m -u <UID> <user> instead.
USER node
WORKDIR /home/node
RUN mkdir -p /home/node/.claude
VOLUME /data
ENTRYPOINT ["claude"]
