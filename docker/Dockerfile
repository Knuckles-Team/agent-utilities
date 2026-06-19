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
# hnswlib (HNSW vector index, an [all] dependency) ships no manylinux wheel and
# builds from sdist, which needs a C++11 toolchain. Install it in the builder
# only — the final stage copies just /usr/local, so the compiler never ships.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
# Build from the LOCAL checkout, not from PyPI. The pipeline runs publish-docker
# right after publish-pypi for the same version bump; pulling that version from
# PyPI races its propagation (and couples the image to a successful PyPI publish).
# Installing the source tree in the build context removes that race entirely — the
# image always matches the committed code. `.dockerignore` keeps the context lean.
COPY . /src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --upgrade --break-system-packages --prerelease=allow "/src[serving]"

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
