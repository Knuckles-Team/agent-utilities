# syntax=docker/dockerfile:1
# A right-sized image for the graph-os serving plane. Slim multi-stage (install into a
# builder, ship only /usr/local — no dev-tools/source). Like every agents/* package
# (gitlab-api, servicenow-api, sql-mcp), the package is installed from the PUBLISHED
# PyPI wheel via uv — NOT built from source. The pipeline runs publish-docker AFTER
# publish-pypi (pipeline.yml `needs:`), so the version is on PyPI by the time this builds.
#
#   agent-utilities[serving] = the serving plane (graph-os MCP + host daemon:
#   mcp/engine/feeds/embeddings/neo4j/falkordb/auth/metrics/agent-headless, NO heavy
#   ML/native deps). Runs the graph-os MCP server, which is ALSO the MCP fleet gateway —
#   it loads other MCP servers on demand (the standalone mcp-multiplexer was folded in).
#   Push as: latest / main / <version>.
#
#   docker build --build-arg VERSION=0.52.0 -t agent-utilities:latest .
FROM python:3.11-slim AS builder-base
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=3600
# A C++11 toolchain for any [serving] dependency that ships only an sdist. Installed
# in the builder only — the final stage copies just /usr/local, so the compiler never
# ships in the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
# Relax pydantic-acp's over-strict `pydantic-ai-slim==1.106.0` transitive pin to the
# v2 floor for any acp-inclusive extra. No-op for [serving] (no acp); future-proofs
# acp-inclusive images. Mirrors pyproject [tool.uv] override-dependencies.
COPY overrides.txt /tmp/overrides.txt
ENV UV_OVERRIDE=/tmp/overrides.txt

# Full serving-plane dependency closure, from the published PyPI wheel.
FROM builder-base AS builder-agent
ARG VERSION=0.52.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --upgrade --break-system-packages --prerelease=allow "agent-utilities[serving]>=${VERSION}"

FROM python:3.11-slim AS runtime-base
ARG HOST=0.0.0.0
ARG PORT=8000
ARG TRANSPORT="stdio"
ARG AUTH_TYPE="none"
ENV HOST=${HOST} \
    PORT=${PORT} \
    TRANSPORT=${TRANSPORT} \
    AUTH_TYPE=${AUTH_TYPE} \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:/usr/local/bin:${PATH}" \
    UV_HTTP_TIMEOUT=3600 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# The graph-os MCP server / serving plane (also the MCP fleet gateway).
FROM runtime-base AS agent
COPY --from=builder-agent /usr/local /usr/local
# The graph-os MCP server (console script). Honors HOST/PORT/TRANSPORT.
CMD ["graph-os"]
