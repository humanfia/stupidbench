# What a cell runs in: the three agent CLIs, and a Python that only has to run
# a pure-stdlib simulator. Nothing else — the smaller this is, the less of a
# hosted runner's six hours goes to pulling it.
FROM node:24-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        procps \
        python3 \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g --no-fund --no-audit \
        @anthropic-ai/claude-code@latest \
        @moonshot-ai/kimi-code@latest \
        @openai/codex@latest

# A cell runs as the runner's own uid, which no image knows, so everything a
# CLI wants to write goes to the workspace it is given.
ENV HOME=/workspace
WORKDIR /workspace
