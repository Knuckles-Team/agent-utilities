# syntax=docker/dockerfile:1
# Two right-sized images from ONE Dockerfile, selected by build --target. Both are
# slim multi-stage (install into a builder, ship only /usr/local — no dev-tools/
# source). Like every agents/* package (gitlab-api, servicenow-api, sql-mcp), the
# package is installed from the PUBLISHED PyPI wheel via uv — NOT built from source.
# The pipeline runs publish-docker AFTER publish-pypi (pipeline.yml `needs:`), so the
# version is on PyPI by the time these build.
#
#   --target agent  (DEFAULT)  installs agent-utilities[serving] = the serving plane
#                              (graph-os MCP + host daemon: mcp/engine/feeds/
#                              embeddings/neo4j/falkordb/auth/metrics/agent-headless,
#                              NO heavy ML/native deps). Runs the graph-os MCP server.
#                              Push as the default tags: latest / main / <version>.
#
#   --target mcp               installs ONLY the mcp-multiplexer (CONCEPT:ECO-4.36):
#                              agent-utilities --no-deps + ~7 light runtime deps. The
#                              multiplexer imports none of the heavy base deps and
#                              reaches graph-os over MCP as a child. The slim variant.
#                              Push as the :mcp / :main-mcp tag.
#
#   docker build --target agent --build-arg VERSION=0.52.0 -t agent-utilities:latest .
#   docker build --target mcp   --build-arg VERSION=0.52.0 -t agent-utilities:mcp    .
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
# v2 floor for any acp-inclusive extra. No-op for [serving]/multiplexer (no acp);
# future-proofs acp-inclusive images. Mirrors pyproject [tool.uv] override-dependencies.
COPY overrides.txt /tmp/overrides.txt
ENV UV_OVERRIDE=/tmp/overrides.txt

# Full serving-plane dependency closure, from the published PyPI wheel.
FROM builder-base AS builder-agent
ARG VERSION=0.52.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --upgrade --break-system-packages --prerelease=allow "agent-utilities[serving]>=${VERSION}"

# Slim mcp-multiplexer closure: the published agent-utilities WITHOUT its heavy base
# deps (`--no-deps`) plus only the ~7 light runtime packages it actually uses.
FROM builder-base AS builder-mcp
ARG VERSION=0.52.0
COPY docker/requirements-multiplexer.txt /tmp/requirements-multiplexer.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
    -r /tmp/requirements-multiplexer.txt \
    && uv pip install --system --upgrade --break-system-packages --prerelease=allow --no-deps "agent-utilities>=${VERSION}"

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

# Slim image: the mcp-multiplexer (reaches graph-os and the fleet over MCP).
FROM runtime-base AS mcp
ENV TRANSPORT="streamable-http"
COPY --from=builder-mcp /usr/local /usr/local
CMD ["mcp-multiplexer", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]

# Full image (DEFAULT target): the graph-os MCP server / serving plane.
FROM runtime-base AS agent
COPY --from=builder-agent /usr/local /usr/local
# The graph-os MCP server (console script). Honors HOST/PORT/TRANSPORT.
CMD ["graph-os"]
