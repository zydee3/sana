# Corpus crawler: Python (stdlib-only) + the claude CLI for triage.
# Credentials are NEVER baked into the image — mount them at runtime:
#   docker run -v ~/.claude:/host-claude:ro sana-scraper
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
  && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && npm install -g @anthropic-ai/claude-code@latest \
  && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
# UID 1000 matches the host user: .credentials.json is mode 0600, and the
# host-mounted /data dir must be writable by the same UID.
RUN useradd -m -u 1000 scraper && mkdir /data && chown scraper /data
COPY src/scraper /app/scraper
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 SANA_CORPUS=/data/corpus
USER scraper
WORKDIR /home/scraper
# symlink, not a mounted file: the host CLI rotates the token by atomic rename,
# so the path must be resolved on every open to see the current inode
RUN mkdir -p /home/scraper/.claude \
  && ln -s /host-claude/.credentials.json /home/scraper/.claude/.credentials.json
VOLUME /data
ENTRYPOINT ["python", "-m", "scraper"]
