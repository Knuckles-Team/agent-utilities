# A right-sized image for the graph-os serving plane. Slim multi-stage (install into a
# builder, ship only /usr/local — no dev-tools/source). The default ``agent`` target
# consumes the exact published wheel. Release assembly uses ``agent-local`` so the OCI
# subject is built from the same digest-verified wheel that was certified and signed.
#
#   agent-utilities[serving] = the serving plane (graph-os MCP + host daemon:
#   mcp/engine/feeds/embeddings/neo4j/falkordb/auth/metrics/agent-headless, NO heavy
#   ML/native deps). Runs the graph-os MCP server, which is ALSO the MCP fleet gateway —
#   it loads other MCP servers on demand (the standalone mcp-multiplexer was folded in).
#   Push as: latest / main / <version>.
#
#   docker build --build-arg VERSION=0.52.0 -t agent-utilities:local .
#   docker build -f docker/Dockerfile --target agent-local \
#     --build-arg RELEASE_REQUIREMENTS=release-requirements.txt \
#     --build-arg RELEASE_REQUIREMENTS_SHA256=<requirements-sha256> \
#     --build-arg AGENT_UTILITIES_VERSION=<exact-version> \
#     --build-arg EPISTEMIC_GRAPH_VERSION=<exact-version> \
#     --build-arg LANGFUSE_AGENT_VERSION=<exact-version> .
FROM python:3.14-slim@sha256:cea0e6040540fb2b965b6e7fb5ffa00871e632eef63719f0ea54bca189ce14a6 AS builder-base
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=3600

# The published-package target may compile a dependency that has no wheel. This stage
# is deliberately not an ancestor of the exact offline release target.
FROM builder-base AS builder-published
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
FROM builder-published AS builder-agent
ARG VERSION=0.52.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages "agent-utilities[serving]==${VERSION}"

# Exact offline release input. The caller stages the complete approved dependency
# closure plus a hash-locked requirements file under ``release-wheelhouse/``. Roots are
# resolved by package name from that closed wheelhouse, so installers do not create PEP
# 610 local-source metadata. No source, credentials, certificate material, or
# workstation paths cross the build boundary.
FROM builder-base AS builder-agent-local
ARG RELEASE_REQUIREMENTS=release-requirements.txt
ARG RELEASE_REQUIREMENTS_SHA256
ARG AGENT_UTILITIES_VERSION
ARG EPISTEMIC_GRAPH_VERSION
ARG LANGFUSE_AGENT_VERSION
COPY release-wheelhouse/ /tmp/release-wheelhouse/
RUN test "$(basename -- "${RELEASE_REQUIREMENTS}")" = "${RELEASE_REQUIREMENTS}" \
    && case "${RELEASE_REQUIREMENTS}" in *.txt) ;; *) exit 64 ;; esac \
    && test -n "${AGENT_UTILITIES_VERSION}" \
    && test -n "${EPISTEMIC_GRAPH_VERSION}" \
    && test -n "${LANGFUSE_AGENT_VERSION}" \
    && printf '%s  %s\n' "${RELEASE_REQUIREMENTS_SHA256}" "/tmp/release-wheelhouse/${RELEASE_REQUIREMENTS}" \
        | sha256sum --check --strict - \
    && uv pip install --system --break-system-packages --offline --no-index \
        --find-links /tmp/release-wheelhouse --only-binary :all: --require-hashes \
        --requirement "/tmp/release-wheelhouse/${RELEASE_REQUIREMENTS}" \
    && python -c 'import importlib.metadata as m, pathlib, shutil, sys; au, eg, lf = sys.argv[1:]; assert sys.version_info[:2] == (3, 12); assert m.version("agent-utilities") == au; assert m.version("epistemic-graph") == eg; assert m.version("langfuse-agent") == lf; assert shutil.which("graph-os"); roots = {pathlib.Path(d.locate_file("")) for d in m.distributions()}; assert not any(path for root in roots for path in root.glob("*.dist-info/direct_url.json")); import epistemic_graph.numeric' \
        "${AGENT_UTILITIES_VERSION}" "${EPISTEMIC_GRAPH_VERSION}" "${LANGFUSE_AGENT_VERSION}"

FROM python:3.14-slim@sha256:cea0e6040540fb2b965b6e7fb5ffa00871e632eef63719f0ea54bca189ce14a6 AS runtime-base
ARG HOST=127.0.0.1
ARG PORT=8000
ARG TRANSPORT="stdio"
ARG AUTH_TYPE="none"
ENV HOST=${HOST} \
    PORT=${PORT} \
    TRANSPORT=${TRANSPORT} \
    AUTH_TYPE=${AUTH_TYPE} \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:${PATH}" \
    UV_HTTP_TIMEOUT=3600 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# The graph-os MCP server / serving plane (also the MCP fleet gateway).
# Fixed identity for least-privilege runtime and stable volume ownership.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --no-create-home \
        --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /app/data \
    && chown -R 10001:10001 /app
ENV HOME=/tmp \
    XDG_CONFIG_HOME=/tmp/.config \
    XDG_CACHE_HOME=/tmp/.cache

# Release target: byte-for-byte package input selected and verified above.
FROM runtime-base AS agent-local
USER 10001:10001
COPY --from=builder-agent-local /usr/local /usr/local
CMD ["graph-os"]

# The ordinary published-package target remains the default final stage.
FROM runtime-base AS agent
USER 10001:10001
COPY --from=builder-agent /usr/local /usr/local
# The graph-os MCP server (console script). Honors HOST/PORT/TRANSPORT.
CMD ["graph-os"]
