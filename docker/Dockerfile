# syntax=docker/dockerfile:1
# Slim multi-stage build (mirrors the agents/*-mcp pattern, e.g. gitlab-api):
# install into a builder, ship only /usr/local, and run the graph-os MCP server
# binary. Keeps the pushed image small so layer pushes finish inside the
# registry's blob-upload window.
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=3600
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --upgrade --break-system-packages --prerelease=allow "agent-utilities[all]>=0.47.1"

FROM python:3.11-slim

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

COPY --from=builder /usr/local /usr/local

# The graph-os MCP server binary (console script). Honors HOST/PORT/TRANSPORT.
CMD ["graph-os"]
