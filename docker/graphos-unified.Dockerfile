# W-C — the ONE unified graph-os container image (latest base, self-contained).
# See reports/unified-binary-program.md (workstream W-C) for full program context.
#
# Contains, in a SINGLE image, everything needed for a self-contained graph-os:
#   - the Rust epistemic-graph engine: epistemic-graph-server (+ lake-fixture-export /
#     restore / migrate-shards) and the epistemic_graph.numeric kernel, both bundled in
#     the staged, already-injected wheel (passes scripts/check_wheel_completeness.py).
#   - the au serving plane (graph-os / kg_server), installed EDITABLE from local source.
#   - messaging platforms (Telegram + Mattermost) + backends (neo4j/falkordb/postgres/redis).
#   - langfuse-agent (workspace source) for LLM observability — dropped from an earlier
#     build of this image (PyPI tops out below au's floor), restored here; see the step-2b
#     comment below for why it installs as its own package rather than via au's [langfuse]
#     extra.
#
# Base: ubuntu:26.04 (glibc 2.43). The current engine binary needs glibc >= 2.43
# (confirmed via `objdump -T | grep GLIBC_`), which Debian 13 "trixie"/glibc 2.41 — the
# current knucklessg1/agent-utilities:latest base — cannot satisfy. Hence latest Ubuntu,
# not a Debian bump.
#
# Python: 3.14 — ubuntu:26.04's own native/default python3 (apt-cache shows NO 3.12/3.13
# candidate on this release, only 3.14). Validated before writing this file: every
# dependency installed below resolves a prebuilt cp314-linux_x86_64 wheel via
# `pip download --only-binary=:all:` against this exact base image — no compiler needed,
# hence no build-essential / multi-stage compile split.
#
# Build context = this repo's worktree root. build-artifacts/eg-wheel/*.whl and
# build-artifacts/langfuse-agent-src/ are git-ignored, build-time-only paths — each
# populated by its own hostPath/NFS mount at build time (see the kaniko Job manifests
# alongside this file), never committed to git. Both live at the context ROOT, not under
# docker/ — this repo's .dockerignore excludes `docker/*` wholesale (the published-wheel
# Dockerfile builds from no local source), which would silently drop anything staged under
# docker/ from the kaniko build context too.
#
# This is the W-C VALIDATION image: it always installs au EDITABLE from local worktree
# source (not a published PyPI release), because the fixes this program depends on
# (session-currency, mattermost event-loop, [serving] messaging bundling) are source-only
# so far. The official, multi-target, published-wheel Dockerfile (docker/Dockerfile) is a
# later "official publish" step — out of scope here.
#
# Build (kaniko, no dockerd on the hosts — see docker/graphos-unified-kaniko-job.yaml):
#   kaniko executor --context=dir:///workspace \
#     --dockerfile=docker/graphos-unified.Dockerfile \
#     --destination=knucklessg1/graph-os-unified:latest
#
# The langfuse-agent restoration (step 2b below) needs a SECOND hostPath/NFS mount beyond
# what the original job used — see docker/graphos-unified-langfuse-kaniko-job.yaml for the
# variant that adds it and pushes the `:langfuse` validation tag.

FROM ubuntu:26.04@sha256:3131b4cc82a783df6c9df078f86e01819a13594b865c2cad47bd1bca2b7063bb

ARG HOST=127.0.0.1
ARG PORT=8000
ARG TRANSPORT="stdio"
ARG AUTH_TYPE="none"
ENV DEBIAN_FRONTEND=noninteractive \
    HOST=${HOST} \
    PORT=${PORT} \
    TRANSPORT=${TRANSPORT} \
    AUTH_TYPE=${AUTH_TYPE} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Ubuntu 26.04's own default Python (3.14) + certs for HTTPS (PyPI / registry pushes).
# `python3` (the dispatch metapackage) guarantees the unversioned /usr/bin/python3
# symlink; `python3.14` pins the exact interpreter this file was validated against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.14 \
        python3 \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv — fast resolver/installer (same pinned version as docker/Dockerfile, for consistency).
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /uvx /usr/local/bin/
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=3600

# 1) The engine. Installed by EXACT local wheel path (not a bare `epistemic-graph>=...`
#    requirement) so this staged, numeric-kernel-injected artifact is what lands — never a
#    same-numbered but incomplete build silently resolved from PyPI instead. Installing
#    this wheel drops epistemic_graph/numeric.abi3.so AND the epistemic-graph-server /
#    lake-fixture-export / restore / migrate-shards binaries (wheel `data/scripts`)
#    straight into /usr/local/bin — next to sys.executable, exactly where EngineResolver
#    (OS-5.63) looks for a co-located engine to autostart.
#    The wheel directory is deliberately NOT deleted here — step 2's resolver is pointed
#    back at it with `--find-links` so a re-resolution of `epistemic-graph[full]>=2.23.2`
#    lands on THIS staged artifact rather than walking off to PyPI (which tops out at
#    2.23.0, below the floor). It is removed after step 2 instead.
COPY build-artifacts/eg-wheel/epistemic_graph-2.23.2-py3-none-linux_x86_64.whl /tmp/wheels/
RUN uv pip install --system --break-system-packages \
        "/tmp/wheels/epistemic_graph-2.23.2-py3-none-linux_x86_64.whl[full]"

# 2) au itself — EDITABLE install from the local worktree source (the "editable-source
#    install model": a deployer can still bind-mount fresher source over
#    /opt/agent-utilities at runtime, exactly like today's /au NFS mount — but every
#    dependency is already resolved+installed at BUILD time, so pod start no longer
#    pip-installs anything).
#
#    `serving`'s OWN sub-extras are listed explicitly here (mcp, feeds, embeddings-openai,
#    neo4j, falkordb, auth, metrics, agent-headless [skills/pydantic-ai/pydantic-monty/
#    fasta2a], logfire, messaging-telegram, messaging-mattermost) rather than the `serving`
#    umbrella extra itself, MINUS `langfuse`: that sub-extra pulls
#    `langfuse-agent[mcp]>=1.0.3,<2`, which — like epistemic-graph above — has a
#    `[tool.uv.sources] langfuse-agent = { workspace = true }` entry, i.e. au's own
#    pyproject expects it from the ecosystem monorepo's sibling checkout, not PyPI (the
#    latest version actually published to PyPI is only 1.0.1, below the 1.0.3 floor). The
#    package itself is restored below (step 2b) from that same workspace checkout — but
#    NOT through this `langfuse` extra: the workspace source is now at version 2.0.0, past
#    au's OWN `<2` ceiling on it, so requesting this extra here would make pip refuse the
#    very artifact step 2b stages (confirmed). Bumping that ceiling is a separate,
#    coordinated au change (re-lock + test) out of scope for an image-only fix, so this
#    extra stays out of the list below; step 2b installs langfuse-agent directly instead.
#    `logfire` (the plain OTel SDK integration, a normal PyPI package, NOT
#    workspace-sourced) is kept. `postgresql` (psycopg[binary]/psycopg-pool) and
#    the first-party Pydantic AI Harness ACP extra are added on top.
# Targeted copy (not `COPY .`): only what `pip install -e` actually needs — the package
# itself + packaging metadata. Skips deploy/scripts/tests/docs (dev/CI-only; also already
# outside this repo's .dockerignore-admitted set) and avoids re-pulling in build-artifacts/.
COPY pyproject.toml build_backend.py README.md LICENSE /opt/agent-utilities/
COPY agent_utilities/ /opt/agent-utilities/agent_utilities/

# 2b) langfuse-agent — ALSO workspace-sourced (`[tool.uv.sources] langfuse-agent =
#     { workspace = true }` in au's own pyproject, same as epistemic-graph in step 1), and
#     for the same reason unresolvable from a bare `pip install`: PyPI tops out at 1.0.1.
#     It is a SIBLING repo (agent-packages/agents/langfuse-agent), not part of this
#     Dockerfile's own worktree, so — exactly like build-artifacts/eg-wheel above — it is
#     mounted into the build context at build time (see
#     docker/graphos-unified-langfuse-kaniko-job.yaml) rather than COPYable from a path
#     this repo owns. Targeted copy, same rationale as au's own source above: the package
#     + packaging metadata only, never `.git`/caches/tests (already outside
#     .dockerignore's admitted set, same as agent_utilities/ above).
COPY build-artifacts/langfuse-agent-src/pyproject.toml build-artifacts/langfuse-agent-src/build_backend.py build-artifacts/langfuse-agent-src/README.md build-artifacts/langfuse-agent-src/LICENSE /tmp/langfuse-agent-src/
COPY build-artifacts/langfuse-agent-src/langfuse_agent/ /tmp/langfuse-agent-src/langfuse_agent/
# D-OB-18 — uv here, NOT plain pip, and the reason is the whole point of this step.
#
# The image build resolves from THIS PACKAGE, in an isolated kaniko context with no
# sibling checkouts and no workspace root. The ecosystem's real dependency policy lives
# in `${WORKSPACE_ROOT}/pyproject.toml`'s `[tool.uv] override-dependencies`, which uv
# honours ONLY from the workspace root — and which plain pip cannot read at all (it is a
# uv-only table). So a plain-pip build of this package silently resolves a DIFFERENT
# dependency set than the workspace lock. That is not a theoretical hazard: it is exactly
# how the runtime came to ship fastmcp 3.4.5 / mcp 1.29.0 underneath fastmcp-4-targeted
# source, which silently killed `attach_fleet_loader` (`ImportError: cannot import name
# 'MCPError' from 'mcp.shared.exceptions'` — mcp 2.0.0 renamed `McpError`) and took every
# fleet meta-tool (`find_tools`/`load_tools`/`list_catalog`) down with it.
#
# `overrides.txt` (context root) is the build-side MIRROR of that root table, and
# `--override` applies it. The historical reason uv was dropped here — au's own
# `[tool.uv.sources] epistemic-graph = { workspace = true }` (+ langfuse-agent), which uv
# cannot resolve outside the full workspace — is handled by `--no-sources`, which makes uv
# ignore uv-only source tables and resolve the standard PEP 508 requirements exactly as
# pip did. `--find-links /tmp/wheels` then pins `epistemic-graph[full]>=2.23.2` to step
# 1's staged, kernel-injected wheel instead of PyPI's incomplete 2.23.0.
#
# No `--prerelease=allow`: the `>=4.0.0b1` override is itself an explicit prerelease
# requirement, which is all uv needs to admit fastmcp 4 for THAT package. A blanket
# prerelease mode bleeds elsewhere (verified: it pulled sqlalchemy 2.1.0b3).
#
# ONE invocation, not two: the exact Pydantic AI family pins below must be visible to the
# SAME resolution pass as au's own `>=2.14.1,<3.0.0` floor on pydantic-ai-slim, and
# `/tmp/langfuse-agent-src`'s own `agent-utilities[mcp]>=2.0.0,<3.0.0` requirement must
# see the au install this same command performs.
#
# apt's python3-pip pulls in `packaging` via dpkg rather than pip, which ships no pip
# RECORD file — an in-place upgrade then fails ("Cannot uninstall packaging 26.0 ... no
# RECORD file was found"). Upgrade JUST that one package first, with a narrowly scoped
# --ignore-installed (Debian/Ubuntu resolve dist-packages with /usr/local taking
# precedence over /usr/lib, so this shadows the apt-owned copy at import time without
# needing to uninstall it).
COPY overrides.txt /tmp/overrides.txt
RUN pip install --break-system-packages --no-cache-dir --ignore-installed "packaging>=26.2.0"
RUN uv pip install --system --break-system-packages --no-cache \
        --no-sources \
        --override /tmp/overrides.txt \
        --find-links /tmp/wheels \
        -e "/opt/agent-utilities[mcp,feeds,embeddings-openai,neo4j,falkordb,auth,metrics,agent-headless,owl,logfire,messaging-telegram,messaging-mattermost,postgresql,acp]" \
        "/tmp/langfuse-agent-src" \
        "redis>=5.0.0" \
        "neo4j>=6.2.0" \
        "falkordb>=1.6.2" \
        "fastmcp>=4.0.0b1" \
        "llama-index-embeddings-openai>=0.6.0" \
        "python-telegram-bot>=22.8" \
        "mattermostdriver>=7.3.2" \
        "pydantic-ai-slim[mcp,openai,ag-ui,ui,web,cli,google,groq]==2.21.0" \
        "pydantic-ai==2.21.0" \
        "pydantic-graph==2.21.0" \
        "pydantic-ai-harness[acp,dynamic-workflow]==0.14.0" \
        "pydantic-ai-skills==1.2.0" \
        "pydantic-monty==0.0.19" \
        "fasta2a[pydantic-ai]>=0.6.1" \
    && chmod -R a+rX /opt/agent-utilities \
    && rm -rf /tmp/langfuse-agent-src /tmp/wheels /tmp/overrides.txt
# ^ this pin list matches the exact stack the CURRENT split-image deploy pip-installs at
#   pod-start (`kubectl get deploy graph-os -n platform -o yaml`) — most of it
#   (neo4j/falkordb/telegram/mattermost/embeddings-openai) is already inside [serving];
#   repeated here explicitly for deploy-artifact parity/robustness against a future
#   narrowing of that extra. `redis` (no au extra covers it) and the broader
#   pydantic-ai-slim extras (ag-ui/ui/google/groq) + Harness ACP/DynamicWorkflow are the
#   genuinely new-vs-[serving] additions. `/tmp/langfuse-agent-src` (step 2b) is a THIRD
#   category — a workspace-sourced package restored whole, not a [serving] sub-extra —
#   resolved in this SAME pass so its own `agent-utilities[mcp]>=2.0.0,<3.0.0` requirement
#   sees the au install this same command is performing, not a second, separate pass; it
#   also pulls in `langfuse>=4,<5` (the SDK) from PyPI normally.

# Build-time self-check: fail the image build itself (not just a later validation pod) if
# the engine/kernel/serving-plane import chain is broken. `epistemic-graph-server --help`
# is a plain clap CLI (safe/non-blocking); `graph-os` itself is NOT invoked here — running
# it would start an MCP server (stdio) rather than return, so only its console-script
# presence is checked.
#
# D-OB-18 — the version assertion is no longer a hand-copied literal (`fastmcp == 3.4.5`),
# which is what let the image quietly drift a whole major version below the floor au's own
# `[mcp]` extra declares. `check_mcp_sdk_floor()` derives the fastmcp floor from
# agent-utilities' OWN installed metadata and mcp's transitively from fastmcp-slim's, so
# this assertion tracks the source of truth automatically and cannot go stale.
#
# `attach_fleet_loader` is imported EXPLICITLY here, not left to a runtime try/except: the
# tolerant handler at kg_server's attach site is by design (a dead fleet loader must not
# take graph-os down), but it also meant a version-mismatched image shipped green and only
# logged the loss of every fleet meta-tool. Failing the BUILD is where that belongs.
RUN python3 -c "import importlib.metadata as m; import agent_utilities.mcp.kg_server; import epistemic_graph.numeric; import langfuse_agent; import owlready2; import pyshacl; import rdflib; from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow; from pydantic_ai_harness.experimental.acp import PydanticAIACPAgent; from agent_utilities.mcp.multiplexer import attach_fleet_loader; from agent_utilities.mcp.protocol_compat import check_mcp_sdk_floor; r = check_mcp_sdk_floor(); assert r['ok'] is True, r['detail']; assert m.version('pydantic-ai-slim') == '2.21.0'; assert m.version('pydantic-ai-harness') == '0.14.0'; print('mcp_sdk_floor OK:', r['detail'])" \
    && epistemic-graph-server --help >/dev/null \
    && command -v graph-os >/dev/null

# Fixed identity for least-privilege runtime (matches docker/Dockerfile's convention).
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --no-create-home \
        --home-dir /tmp --shell /usr/sbin/nologin app
ENV HOME=/tmp \
    XDG_CONFIG_HOME=/tmp/.config \
    XDG_CACHE_HOME=/tmp/.cache

USER 10001:10001
# The graph-os MCP server (console script) — also the MCP fleet gateway. Honors
# HOST/PORT/TRANSPORT (W-D: stdio for packaged-local, streamable-http for served).
CMD ["graph-os"]
