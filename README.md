# Agent Utilities

A batteries-included Python harness for building, orchestrating, and running AI
agents against a shared knowledge graph — zero-infra by default.

[![PyPI - Version](https://img.shields.io/pypi/v/agent-utilities)](https://pypi.org/project/agent-utilities/)
[![Build](https://img.shields.io/github/actions/workflow/status/Knuckles-Team/agent-utilities/release.yml?branch=main)](https://github.com/Knuckles-Team/agent-utilities/actions/workflows/release.yml)
[![PyPI - License](https://img.shields.io/pypi/l/agent-utilities)](LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/agent-utilities)](https://pypi.org/project/agent-utilities/)
[![Docs](https://img.shields.io/badge/docs-published-blue)](https://knuckles-team.github.io/agent-utilities/)
[![Engine: epistemic-graph](https://img.shields.io/badge/engine-epistemic--graph-6f42c1)](https://github.com/Knuckles-Team/epistemic-graph)

*Version: 2.5.0*

> **New here?** Read **[docs/start-here.md](docs/start-here.md)** — one page,
> the real onboarding entry point. For AIs, **[llms.txt](llms.txt)** is the
> entry index.

## What it is

`agent-utilities` is a batteries-included harness for building Pydantic-AI
agents that ship with a knowledge graph, orchestration, memory, and tools
built in. Install it and use it three ways: as a **library** you import into
Python code (`from agent_utilities import create_agent`), as an **MCP server**
(`graph-os`) that hands an existing agent — Claude Code, Cursor, your own —
the knowledge graph and tool surface, or as an **HTTP/REST gateway** sharing
one KG backend across many clients. All three sit on one engine — a fast Rust
knowledge-graph engine that does compute, caching, semantics, and durable
storage — so whichever surface you pick, you're talking to the same brain.
Writes fan out asynchronously to optional durable mirrors (Postgres/pg-age,
Neo4j, FalkorDB) for interop, BI, or disaster recovery, but the engine stays
the one read/write authority; nothing downstream of it changes when you add
one. The default needs no databases or external services: the knowledge graph
runs in-process, so a fresh checkout can create, query, and persist a graph
before you've installed anything else — no Postgres, no Neo4j, no separate
graph server to stand up first. Full trade-offs:
**[Consumption Models](docs/guides/consumption-models.md)**.

## Quickstart

```bash
pip install agent-utilities          # zero external *service* deps to start
```

Point it at any model provider (`OPENAI_API_KEY`, or a local vLLM/Ollama
endpoint), then create an agent — skills, tools, and the in-process KG included:

```python
from agent_utilities import create_agent

agent, toolsets = create_agent(name="assistant", skill_types=["universal", "graphs"])
print(agent.run_sync("What can you do?").output)
```

Or stand up the whole platform and verify it — three commands, zero infra:

```bash
setup-config generate --profile tiny     # complete config.json (every option)
graph-os &                                # KG MCP server — no database needed
agent-utilities-doctor                    # one health sweep across every subsystem
```

Or work with the knowledge graph directly, no database required:

```python
from agent_utilities.mcp import kg_server   # epistemic-graph is the default — zero-infra

await kg_server._execute_tool("graph_write", action="add_node",
    node_id="svc:payments", node_type="Service",
    properties='{"team":"fintech","tier":"critical"}')

res = await kg_server._execute_tool("graph_query",
    cypher="MATCH (n:Service) WHERE n.tier='critical' RETURN n")
```

Scale up with `--profile single-node-prod`/`enterprise`, add Stardog + pg-age,
or let Claude set itself up — all in the **[Quick Start Guide](docs/guides/quick-start.md)**
and **[Self-Setup Guide](docs/guides/self-setup.md)**. The full capability
catalog (search, ingest, orchestrate, ontology, memory) is in
**[docs/capabilities.md](docs/capabilities.md)**; runnable code is in the
[reference agent](examples/reference_agent/).

> **Heads-up — this is two repos.** The heavy graph compute lives in a
> **separate** Rust engine, [`epistemic-graph`](https://github.com/Knuckles-Team/epistemic-graph)
> (reached out-of-process over MessagePack/UDS — no PyO3, no Rust toolchain
> needed here). AI agents pointed here to deploy this: follow **[Zero-to-deployed](AGENTS.md#-zero-to-deployed-genesis--deploying-this-for-an-operator)**
> in `AGENTS.md`.

## Key Features

Grouped by what they do — link out for the full catalog with every concept ID:
**[docs/guides/features.md](docs/guides/features.md)**.

- **Knowledge graph & memory** — one Rust engine is the authority for compute,
  cache, OWL semantics, and durable persistence; optional Postgres/Neo4j/
  FalkorDB mirrors fan out writes.
- **Ontology system** (Palantir-Foundry parity) — objects, links, interfaces,
  derived properties, action types, and object-set permissioning, graph-native.
- **Orchestration & self-evolution** — Spec-Driven Development, capability
  auto-activation, and a governed evolution loop that proposes changes for
  review and never auto-pushes them.
- **Enterprise integration (Company Brain)** — a document-source connector
  framework plus the ~58-server MCP fleet feed a 6-layer ingestion runtime
  with trust-decay conflict resolution, field-level survivorship, data ACLs
  and tenant scoping, and a human-correction→rule→eval feedback loop.
- **Scale-out planes, all opt-in** — externalized durable state, tenant-sharded
  engines with HRW routing, Kafka ingest scale-out, queue-driven agent
  dispatch, and a Prometheus-instrumented gateway; the zero-infra default is
  unchanged until you turn these on.
- **Inference acceleration** — KV-cache layering across vLLM/LMCache/the
  engine, plus a numpy-compatible numeric shim backed by the engine's own
  kernel.
- **Autonomy & governance** — a fail-closed `ActionPolicy` gate, server-minted
  identity, and a hardened MCP fleet gateway built into `graph-os`.

Benchmarked against a conventional stitched memory stack (separate vector DB +
BM25 + app-level fusion), the unified KG memory matches recall (1.000) while
retrieving ~3.6× faster and surviving a restart via a durable KV cold-tier
(100% survival) — full scorecard in the
[Phase-2 benchmark report](https://github.com/knuckles-team/epistemic-graph/blob/main/docs/benchmarks.md#phase-2-agent-memory--kv-cache-benchmark-measured).
A handful of capabilities are real and importable today but lightly
documented — causal reasoning, a `SKILL.md`-to-`GraphPlan` compiler, and
graph-native event sourcing among them; see
**[docs/guides/features.md](docs/guides/features.md)** for the full list.

## Architecture at a glance

<!-- BEGIN GENERATED: concepts -->

Synthesized from concept markers in the codebase into **1220 canonical concepts** across **9 pillars**.

> This count is generated from `docs/concepts.yaml` by `scripts/gen_docs.py` — do not edit by hand. The table below covers the 5 pillars agent-utilities itself owns; the other 4 (37 concepts) belong to the epistemic-graph engine's own pillar set. Live per-pillar status: [docs/status.md](docs/status.md).

| # | Pillar | Focus | Concepts | Docs |
|:-:|:-------|:------|:--------:|:-----|
| 1 | Graph Orchestration | Planning, SDD lifecycle, dynamic multi-layer execution | 220 | [docs/pillars/1_graph_orchestration.md](docs/pillars/1_graph_orchestration.md) |
| 2 | Epistemic Knowledge Graph | The one engine authority — ingestion, ontology, ETL, reasoning | 513 | [docs/pillars/2_epistemic_knowledge_graph.md](docs/pillars/2_epistemic_knowledge_graph.md) |
| 3 | Agentic Harness Engineering | Self-models, evaluation, governed self-evolution | 120 | [docs/pillars/3_agentic_harness_engineering.md](docs/pillars/3_agentic_harness_engineering.md) |
| 4 | Ecosystem & Peripherals | MCP fleet, messaging, connectors, UI surfaces | 141 | [docs/pillars/4_ecosystem_peripherals.md](docs/pillars/4_ecosystem_peripherals.md) |
| 5 | Agent OS Infrastructure | Auth, governance, deployment, scaling | 189 | [docs/pillars/5_agent_os_infrastructure.md](docs/pillars/5_agent_os_infrastructure.md) |

<!-- END GENERATED: concepts -->

All four consumption surfaces (library, `graph-os` MCP, REST gateway, and any
IDE/agent) talk to **one** `graph-os` MCP server — it serves the knowledge
graph natively and doubles as the fleet gateway for the other ~58 MCP servers,
loading them on demand via `find_tools`/`load_tools` so hundreds of fleet
tools stay out of context until asked for. Wiring it into Claude Code/Cursor/
etc. via `mcp_config.json` (generate with `setup-config mcp` — don't
hand-write it), self-contained vs. shared-engine configs, Keycloak-protected
fleets, and every env var: **[Consumption Models](docs/guides/consumption-models.md)**.
Full architecture, every pillar deep-dive, C4 diagrams, and the Company Brain
and Vendor-Neutral Enterprise Ontology write-ups: **[docs/index.md](docs/index.md)**.

## Installation & Deployment

```bash
pip install agent-utilities              # add "[all]" for MCP servers, UI, and external graph backends
```

Out of the box it runs zero-infrastructure — no database or graph server to
stand up; the bundled Rust `epistemic-graph` engine is the one authority for
compute, cache, and durable persistence. Add a durable Postgres mirror later
(`GRAPH_MIRROR_TARGETS`/`KG_CONNECTIONS`) if you need one for interop/BI/DR.
See the **[Installation Guide](docs/guides/installation.md)**.

Model providers, routing, and secrets are configured centrally via
`~/.config/agent-utilities/config.json` (every field has a matching
environment-variable override) — generate one with `setup-config generate`,
then see the **[Configuration Guide](docs/guides/configuration.md)** and the
**[Local Secret Storage Guide](docs/guides/secrets-auth.md)**.

To go beyond a laptop — `graph-os` over stdio/streamable-HTTP, the REST
gateway, Docker composes, and sharded/queue-driven production shapes — the
**[Deployment Configurations](docs/guides/deployment-configurations.md)**
guide walks every step from zero-infra to a governed, multi-tenant fleet. The
**[Enterprise Enablement Runbook](docs/guides/enterprise-enablement-runbook.md)**
is the ordered push → deploy → flag-enablement sequence for turning on the
opt-in scale-out and autonomy planes once you're already deployed. Once
installed, run **`agent-utilities install`** to drop the skill toolkit into
your agent tool (Claude Code, Cursor, etc.) — it unlocks the deployment,
evolution, and knowledge-graph skills the rest of this README assumes.

## Documentation

- **[docs/start-here.md](docs/start-here.md)** — the real onboarding entry
  point: what this is, the three ways to use it, and the zero-infra knowledge
  graph, in one page.
- **[AGENTS.md](AGENTS.md)** — contributor/agent working discipline:
  architecture reference, coding conventions, the branching/merge-queue
  workflow, and the zero-to-deployed genesis procedure.
- **[docs/status.md](docs/status.md)** — the live concept/capability status
  page: what's built vs. roadmap, by pillar.
- **[docs/journey.md](docs/journey.md)** — *optional deep-dive*: a narrative
  walkthrough of the platform in motion, for readers who prefer a story to
  config tables.
- **[CHANGELOG.md](CHANGELOG.md)** — release history and the roadmap direction
  beyond a single agent harness (distributed agentic evolution).

Everything else — architecture, pillar deep-dives, guides — is indexed from
**[docs/index.md](docs/index.md)**.

agent-utilities is also the entrypoint for the wider `agent-packages`
ecosystem — 65 connector packages, three frontends (geniusbot, agent-webui,
agent-terminal-ui), a skill library, and ontologies. See
**[The wider ecosystem](docs/ecosystem.md)**.

## Contributing

Fork the repo, write tests for new functionality (assertions required, not
just coverage), follow the established Pydantic models/structured
prompts/concept markers, and run `uv run pytest tests/ -q` before submitting
— a 60-second timeout applies to every test, so an unbounded `time.sleep`
fails automatically. Update `docs/` if your change affects a public API. See
**[CONTRIBUTING.md](CONTRIBUTING.md)** and **[AGENTS.md](AGENTS.md)** for the
full conventions and architecture rules.

## License

This project is licensed under the terms in the [LICENSE](LICENSE) file.

## Environment Variables

<!-- ENV-VARS-TABLE:START -->

#### Package environment variables

| Variable | Example | Description |
|----------|---------|-------------|
| `CHAT_MODELS` | `'[{"id":"chat-model","provider":"custom","base_url":"https://chat-model.example.test/v1","api_key_ref":"env://MODEL_API_KEY_1","headers_ref":"env://MODEL_HEADERS_1","intelligence_level":"normal"}]'` | ─────────────────────────────────────────────────────────────────────────────── 1. Primary Model Registry (JSON Overrides) ─────────────────────────────────────────────────────────────────────────────── [OPTIONAL] JSON string array of chat models. Overrides config.json registries. |
| `EMBEDDING_MODELS` | `'[{"id":"embedding-model","provider":"custom","base_url":"https://embedding-model.example.test/v1","api_key_ref":"env://MODEL_API_KEY_2","headers_ref":"env://MODEL_HEADERS_2","chunk_size":768}]'` | [OPTIONAL] JSON string array of embedding models. Overrides config.json registries. |
| `OPENAI_API_KEY` | secret-injected | ─────────────────────────────────────────────────────────────────────────────── 2. Process-only Provider Overrides ─────────────────────────────────────────────────────────────────────────────── These literal key variables are accepted only from the explicit process environment. Durable AgentConfig must use the provider's dedicated *_REF field. |
| `OPENAI_API_KEY_REF` | `vault://secret/data/agent-utilities/openai` | api_key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` |  |
| `ANTHROPIC_API_KEY` | secret-injected |  |
| `GEMINI_API_KEY` | secret-injected |  |
| `GROQ_API_KEY` | secret-injected |  |
| `MISTRAL_API_KEY` | secret-injected |  |
| `HUGGING_FACE_API_KEY` | secret-injected |  |
| `DEEPSEEK_API_KEY` | secret-injected |  |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` |  |
| `PROVIDER_CONFIGS` | `'{"example-provider":{"enabled":true,"endpoint_ref":"env://EXAMPLE_PROVIDER_ENDPOINT","credential_refs":{"EXAMPLE_PROVIDER_TOKEN":"secret://providers/example/token"},"selector_refs":{},"tls_profile_ref":"secret://tls/example-provider"}}'` | [OPTIONAL] Neutral external-provider profiles. Durable values are references; resolved endpoints, credentials, selectors, and TLS material remain runtime-only. |
| `DEFAULT_AGENT_NAME` | `Agent` | ─────────────────────────────────────────────────────────────────────────────── 3. Agent Identity & Workspaces ─────────────────────────────────────────────────────────────────────────────── [OPTIONAL] Agent display identity |
| `AGENT_DESCRIPTION` | `"AI Agent"` |  |
| `AGENT_SYSTEM_PROMPT` | — |  |
| `DEPLOYMENT_PROFILE` | `tiny` | tiny \| single-node-prod \| enterprise |
| `WORKSPACE_PATH` | `./runtime-workspace` | [OPTIONAL] System workspace path (where filesystem and repository tools execute) |
| `EVOLUTION_STAGING_ROOT` | `./runtime-evolution-staging` | Explicit private (0700) review-artifact root |
| `AGENT_UTILITIES_CONFIG_DIR` | — | Optional XDG root override |
| `HOST` | `127.0.0.1` | ─────────────────────────────────────────────────────────────────────────────── 4. Server Network Settings ─────────────────────────────────────────────────────────────────────────────── [OPTIONAL] Core protocol server network settings |
| `PORT` | `9000` |  |
| `DEBUG` | `false` |  |
| `GRAPH_SERVICE_ENDPOINTS` | `tls://... with a named or secret-ref-backed profile.` | [OPTIONAL] Native epistemic-graph TLS. External engines are selected only by |
| `ENGINE_TLS_PROFILE` | — |  |
| `ENGINE_TLS_PROFILE_REF` | — |  |
| `ENGINE_TLS_SERVER_NAME` | — |  |
| `MODEL_TLS_PROFILE` | — | [OPTIONAL] Model/embedder transport trust. Prefer named, secret-backed TLS profiles for custom CAs and mTLS. Private/loopback provider hosts are denied unless their exact hostname is listed; model traffic is DNS-pinned and does not inherit ambient proxies. |
| `MODEL_TLS_PROFILE_REF` | — |  |
| `EMBEDDING_TLS_PROFILE` | — |  |
| `EMBEDDING_TLS_PROFILE_REF` | — |  |
| `OAUTH2_TOKEN_TLS_PROFILE` | secret-injected |  |
| `OAUTH2_TOKEN_TLS_PROFILE_REF` | — |  |
| `MODEL_HTTP_ALLOWED_PRIVATE_HOSTS` | `[]` |  |
| `AIRGAP_MODE` | `false` | [OPTIONAL] Sovereign/air-gap deployment gate (CONCEPT:AU-OS.deployment.airgap-mode). When true, the canonical outbound HTTP factory (core/http_client.py) and the LLM client constructor (core/model_factory.py) refuse any request whose target host is not loopback/RFC1918-private/link-local — fail-closed with AirgapViolation instead of silently phoning out. Off by default. See docs/guides/sovereign-self-hosted.md §4. |
| `SOURCE_HTTP_ALLOWED_PRIVATE_HOSTS` | `[]` | [OPTIONAL] Shared outbound-source egress policy. Values are JSON arrays of exact hostnames; URLs, wildcards, private IP ranges, and implicit redirects are rejected. Private hosts require an explicit exact-host entry. TLS trust comes from SOURCE_HTTP_TLS_PROFILE or the global secret-backed TLS profile. Each DNS-approved hop is address-pinned while Host/SNI retain the logical hostname. Ambient proxies are excluded; use an explicit HTTP/SOCKS profile proxy (unsupported pin/SNI combinations fail closed). |
| `SOURCE_HTTP_ALLOWED_REDIRECT_HOSTS` | `[]` |  |
| `SOURCE_HTTP_MAX_RESPONSE_BYTES` | `10485760` |  |
| `SOURCE_HTTP_MAX_REDIRECTS` | `3` |  |
| `SOURCE_HTTP_ALLOW_BROWSER_FETCH` | `false` |  |
| `MODEL_CONTEXT_TOKEN_BUDGET` | secret-injected | [OPTIONAL] Mandatory governed model-context sizing/version boundaries. Compilation itself cannot be disabled; these values only tune the evidence budget and invalidate cache identities when ordering/redaction semantics change. |
| `MODEL_CONTEXT_ORDERING_VERSION` | `context-mmr-v1` |  |
| `MODEL_CONTEXT_REDACTION_VERSION` | `permissioning-v1` |  |
| `MODEL_CONTEXT_COMPILER_ENABLED` | `true` | Deployment opt-out of the RAG-by-default context compiler; default on |
| `ENABLE_WEB_UI` | `false` | [OPTIONAL] UI Toggles |
| `ENABLE_TERMINAL_UI` | `false` |  |
| `ENABLE_WEB_LOGS` | `false` |  |
| `ENABLE_ACP` | `false` | Deprecated gateway compatibility flag |
| `ACP_SESSION_ROOT` | `.acp-sessions` |  |
| `MAX_UPLOAD_SIZE` | `10485760` | Limit upload size in bytes (10MB default) |
| `AUTH_JWT_JWKS_URI` | `https://identity.provider/keys` | [OPTIONAL] OIDC / JWT Token Verification |
| `AUTH_JWT_ISSUER` | `https://identity.provider` |  |
| `AUTH_JWT_AUDIENCE` | `my-agent-client-id` |  |
| `MCP_JWT_ISSUER` | — | Legacy last-resort aliases the multiplexer's task-delegation auth checks after AUTH_JWT_ISSUER/AUTH_JWT_AUDIENCE and FASTMCP_SERVER_AUTH_JWT_ISSUER/AUDIENCE; prefer AUTH_JWT_ISSUER/AUTH_JWT_AUDIENCE above for new deployments. |
| `MCP_JWT_AUDIENCE` | — |  |
| `OIDC_CONFIG_URL` | `https://identity.provider/.well-known/openid-configuration` | [OPTIONAL] OIDC Client Delegation (RFC 8693 Token Exchange) |
| `OIDC_CLIENT_ID` | `client-id` |  |
| `OIDC_CLIENT_SECRET_REF` | — |  |
| `ENABLE_DELEGATION` | `false` |  |
| `ENABLE_DELEGATED_IDENTITY` | `off` | Per-agent on-behalf-of delegation rollout: off \| warn \| on; default off |
| `AUDIENCE` | `https://downstream.api/v1` | Target audience for exchange |
| `DELEGATED_SCOPES` | `api` | Space-separated scopes |
| `REMOTE_OAUTH_PROVIDERS_JSON` | `[]` | [OPTIONAL] Administrator-approved remote-MCP OAuth provider registry. The value is a JSON array of provider descriptor objects; malformed input disables the registry with a bounded diagnostic. Durable configuration may use the same typed field; no request may add providers at runtime. |
| `REMOTE_OAUTH_SUCCESS_REDIRECT_URL` | — | Fixed administrator-configured URL used after a successful remote OAuth flow. |
| `DATA_PREP_RUNTIME` | — | Startup-owned governed data-prep model/policy/shape declaration (JSON object). |
| `ALLOWED_ORIGINS` | — | Unset disables cross-origin requests |
| `CORS_ALLOW_CREDENTIALS` | `false` | Requires exact, non-wildcard origins |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1,[::1]` | Required for non-loopback listeners |
| `SECRETS_BACKEND` | `engine` | ─────────────────────────────────────────────────────────────────────────────── 6. Secrets Storage Backend ─────────────────────────────────────────────────────────────────────────────── [OPTIONAL] Secret backends (encrypted engine store or vault) |
| `SECRETS_VAULT_URL` | `http://127.0.0.1:8200` |  |
| `SECRETS_VAULT_MOUNT` | `secret` |  |
| `VAULT_AUTH_METHOD` | `auto` | auto \| oidc \| approle \| token \| kubernetes |
| `VAULT_AUTH_MOUNT` | `jwt` | Auth mount path |
| `VAULT_ROLE` | `default` | Login role name |
| `VAULT_PATH_PREFIX` | `agents/mcp/` | KV v2 path prefix |
| `EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF` | `vault://apps/graph-os/engine-data-key` | [REQUIRED OUTSIDE NON-PRODUCTION TINY MODE] External bootstrap reference for the packaged engine data-encryption key. secret:// is intentionally rejected. |
| `PERSISTENCE_IDENTITY_HMAC_KEY_REF` | `vault://apps/graph-os/identity-hmac-key` | [OPTIONAL] Privacy-safe durable identity references. Production templates use this neutral reference; the secret controller injects the referenced value. |
| `PERSISTENCE_PRIVACY_DENY_TERMS_REF` | `vault://apps/graph-os/privacy-deny-terms` |  |
| `USAGE_CONTENT_RETENTION` | `metadata` | [REQUIRED IN PRODUCTION] Metadata-only usage and Langfuse traces. |
| `LANGFUSE_CAPTURE_CONTENT` | `false` |  |
| `LANGFUSE_MCP_ENABLED` | — | auto-on when both key refs are present |
| `LANGFUSE_KG_AUTO_INGEST` | `false` | explicit parent-governed graph-write gate |
| `KG_FAILURE_EVOLUTION` | — | auto-on when both key refs are present |
| `KG_OPTIMIZATION_ENABLED` | `true` | propose-only native optimization sweep |
| `MEMENTO_RAW_RETENTION_ENABLED` | `false` | [OPTIONAL] Raw Memento recovery is OFF by default. It is enabled only when all three values are present; raw blocks are then stored as authenticated ciphertext. |
| `MEMENTO_RAW_RETENTION_POLICY` | `approved-encrypted-v1` |  |
| `MEMENTO_RAW_ENCRYPTION_KEY_REF` | `vault://apps/graph-os/memento-raw-key` |  |
| `GRAPH_MIRROR_TARGETS` | `age` | CSV/JSON of mirror connection names (e.g. age,neo4j,falkordb) |
| `GRAPH_DB_CONNECTION_PROFILE_REF` | `vault://graph/connections` | profile |
| `ASSET_MIRROR_TARGETS` | `servicenow,egeria` | CSV/JSON of enabled CMDB sinks |
| `SERVICENOW_ENABLE_WRITE` | `false` | gate live writes per sink (fail-closed) |
| `ERPNEXT_ENABLE_WRITE` | `false` |  |
| `EGERIA_ENABLE_WRITE` | `false` |  |
| `TWENTY_ENABLE_WRITE` | `false` |  |
| `ROUTING_STRATEGY` | `hybrid` | hybrid \| semantic \| rule-based |
| `GRAPH_PERSISTENCE_TYPE` | `file` | file \| memory |
| `GRAPH_PERSISTENCE_PATH` | `~/.local/share/agent-utilities/graph_state` |  |
| `ENABLE_LLM_VALIDATION` | `false` | Validate outputs via additional LLM passes |
| `GRAPH_ROUTER_TIMEOUT` | `300.0` | Router maximum timeout in seconds |
| `GRAPH_VERIFIER_TIMEOUT` | `300.0` | Verifier maximum timeout in seconds |
| `MIN_CONFIDENCE` | `0.4` | Routing decision confidence threshold |
| `VALIDATION_MODE` | `false` | Mock LLM generation for CI/testing |
| `APPROVAL_TIMEOUT` | `0.0` | Seconds to wait for manual human approval (0 = block) |
| `GRAPH_TIMEOUT` | `1200000` | Global network/query timeout in ms |
| `MAX_RECURSION_DEPTH` | `2` | Maximum graph expansion depth |
| `ROUTING_PERCENTILE` | `50.0` | Semantic routing percentile limit |
| `KG_EMBEDDING_DIM` | `768` | Knowledge Graph embedding dimension size |
| `ENABLE_KG_EMBEDDINGS` | `true` | [OPTIONAL] Knowledge Graph Sync and Concurrency |
| `KG_BACKUPS` | `3` | Keep N timestamped SQLite/DuckDB backups |
| `KG_INGESTION_WORKERS` | — | Max worker threads for codebase ingestion |
| `KG_LLM_CONCURRENCY` | `4` | Max concurrent LLM calls for KG analysis |
| `KG_ANALYSIS_MAX_DEPTH` | `2` | Ingestion graph analyzer recursion depth |
| `KNOWLEDGE_GRAPH_SYNC_BACKGROUND` | `true` | Sync knowledge graph in the background |
| `MODEL_REGISTRY_PATH` | — | Path to models.yaml registry |
| `ENABLE_OTEL` | `false` | ─────────────────────────────────────────────────────────────────────────────── 9. Observability & Telemetry ─────────────────────────────────────────────────────────────────────────────── [OPTIONAL] OpenTelemetry (OTLP) Exports |
| `TRACE_EXPORT_ENABLED` | `false` |  |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `https://telemetry.example.test/api/public/otel` |  |
| `OTEL_EXPORTER_OTLP_HEADERS_REF` | `env://OTEL_AUTH_HEADERS` |  |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY_REF` | `env://OTEL_PUBLIC_KEY` |  |
| `OTEL_EXPORTER_OTLP_SECRET_KEY_REF` | `env://OTEL_SECRET_KEY` |  |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` |  |
| `OTEL_TLS_PROFILE` | `runtime-trust` |  |
| `OTEL_TLS_PROFILE_REF` | `secret://runtime/otel-tls-profile` |  |
| `OTEL_TRACES_EXPORTER` | `otlp` | standard OTel var; "none" is a hard opt-out |
| `LANGFUSE_HOST` | `https://telemetry.example.test` | [OPTIONAL] Langfuse Observability Integration |
| `LANGFUSE_PUBLIC_KEY` | secret-injected | Process-only literal overrides; durable AgentConfig must use the *_REF fields above. |
| `LANGFUSE_SECRET_KEY` | secret-injected |  |
| `LANGFUSE_PUBLIC_KEY_REF` | `vault://observability/langfuse-public-key` |  |
| `LANGFUSE_SECRET_KEY_REF` | `vault://observability/langfuse-secret-key` |  |
| `LANGFUSE_PERSISTENCE_HMAC_KEY_REF` | `vault://observability/langfuse-persistence-hmac` |  |
| `LANGFUSE_TLS_PROFILE_REF` | `vault://observability/langfuse-tls-profile` |  |
| `LANGFUSE_CLIENT_CERT_REF` | `vault://observability/langfuse-client-certificate` |  |
| `LANGFUSE_CLIENT_KEY_REF` | `vault://observability/langfuse-client-private-key` |  |
| `LANGFUSE_CLIENT_KEY_PASSWORD_REF` | `vault://observability/langfuse-client-key-password` |  |
| `LANGFUSE_DATASET_CAPTURE_THRESHOLD` | `0.0` | Capture score threshold for dataset creation |
| `LANGFUSE_LATENCY_BASELINE_SECONDS` | `60.0` | Alert baseline latency |
| `LANGFUSE_TOKEN_BASELINE` | secret-injected | Expected token limit warning threshold |
| `LANGFUSE_VERIFIER_FALLBACK_LIMIT` | `1` | Retries before verification fallback |
| `GOOGLE_WORKSPACE_OAUTH_CLIENT_ID` | — | Optional Google Workspace OAuth bootstrap. No tenant/client defaults ship in the codebase; provide both values through runtime/XDG configuration. |
| `GOOGLE_WORKSPACE_OAUTH_BROKER_URL` | `https://oauth-broker.example.test` |  |
| `AGENT_UTILITIES_SELF_INGEST` | `false` | Master opt-in switch (default off) |
| `EPISTEMIC_GRAPH_OBS_ADDR` | `https://observability.example.test` | Engine obs endpoint base addr |
| `AGENT_UTILITIES_SELF_INGEST_MODE` | `otlp` | otlp → POST /v1/logs ; bulk → POST /_bulk |
| `AGENT_UTILITIES_SELF_INGEST_SERVICE` | `agent-utilities` | OTLP service.name |
| `AGENT_UTILITIES_SELF_INGEST_LEVEL` | `INFO` | Minimum log level to ship |
| `AGENT_UTILITIES_SELF_INGEST_BATCH` | `100` | Max records per batch |
| `AGENT_UTILITIES_SELF_INGEST_INTERVAL` | `2.0` | Background flush interval (seconds) |
| `AGENT_UTILITIES_SELF_INGEST_QUEUE_MAX` | `10000` | Bounded queue; overflow spills durably |
| `AGENT_UTILITIES_SELF_INGEST_TIMEOUT` | `3.0` | Per-request HTTP timeout (seconds) |
| `AGENT_UTILITIES_SELF_INGEST_MAX_RETRIES` | `3` | Resends before a batch spills durably |
| `AGENT_UTILITIES_SELF_INGEST_SPILL_PATH` | — | Durable overflow sqlite path (default: XDG data dir) |
| `AGENT_UTILITIES_SELF_INGEST_SPILL_MAX` | `50000` | Durable buffer cap; beyond this = true drop |
| `FLEET_EVENTS_TOKEN_REF` | — | Secret-provider reference |
| `KG_LOOP` | `true` | Hourly propose-only research/self-evolution cycle |
| `KG_LOOP_INTERVAL` | `3600` |  |
| `KG_LOOP_TOPICS` | `5` |  |
| `KG_FAILURE_EVOLUTION_INTERVAL` | `3600` |  |
| `KG_ANOMALY_CONSUMER` | `true` | Consume PerformanceAnomaly nodes → failure_gap topics (cheap, propose-only; default on) |
| `KG_GOLDEN_AUTO_MERGE` | `false` | Governed auto-merge: proposal→active promotion gated by the production PromotionGovernanceValidator (SHACL + recorded regression gate + MergePolicy + constitution rules). Keep FALSE until you trust the proposal stream. |
| `KG_GOLDEN_MERGE_THRESHOLD` | `0.85` |  |
| `ACTION_POLICY_PATH` | — | Empty = shipped conservative deploy/action-policy.default.yml |
| `FLEET_RECONCILER` | `true` | Leader-only desired-state reconciler tick (OS-5.25; default off) |
| `FLEET_RECONCILER_INTERVAL` | `120` |  |
| `FLEET_RECONCILER_MAX_ACTIONS` | `5` | Storm guard per tick |
| `FLEET_REGISTRY_PATH` | — | Empty = deploy/mcp-fleet.registry.yml |
| `FLEET_DESIRED_STATE_PATH` | — | Optional per-service replicas/desired/version override YAML |
| `FLEET_ACTUATOR` | `dryrun` | dryrun (records intent only) \| docker; Portainer wired via set_fleet_actuator() |
| `DEPLOY_WATCH_WINDOW` | `300` | Health watch after deploy/restart (OS-5.27) |
| `DEPLOY_WATCH_POLL` | `15` |  |
| `FLEET_AUTOSCALER` | `true` | Leader-only reactive replica autoscaler tick (OS-5.29; default off) |
| `FLEET_AUTOSCALER_INTERVAL` | `60` |  |
| `MCP_FLEET_REGISTRATION` | `true` | Each MCP server self-registers + heartbeats with the engine registry; false opts out |
| `MCP_FLEET_REGISTRATION_TTL_SECS` | `300` | Registration lease TTL, seconds |
| `SCALING_PROMETHEUS_URL` | — | Empty = zero-infra in-process gauges; set http://prometheus:9090 for PromQL signals |
| `ENGINE_SURFACE_PROMETHEUS_URL` | — | graph_promql's fallback Prometheus base URL, consulted only when the connected engine build has no native PromQL surface; empty = clean degrade, never an invented hostname |
| `A2A_BROKER` | `epistemic_graph` | ─────────────────────────────────────────────────────────────────────────────── 10. Agent-to-Agent (A2A) Discovery ─────────────────────────────────────────────────────────────────────────────── [OPTIONAL] Standardized A2A communication parameters. Persistence is always the native Epistemic Graph plane; no alternate broker/storage URL exists. |
| `A2A_STORAGE` | `epistemic_graph` |  |
| `A2A_BROKER_POLL_INTERVAL_MS` | `100` |  |
| `A2A_BROKER_LEASE_MS` | `300000` |  |
| `A2A_BROKER_PREFETCH` | `1` |  |
| `A2A_BROKER_MESSAGE_TTL_MS` | `86400000` |  |
| `A2A_BROKER_MAX_DELIVERY_COUNT` | `5` |  |
| `A2A_MAX_PAYLOAD_BYTES` | `262144` |  |
| `A2A_MAX_HISTORY` | `100` |  |
| `A2A_MAX_ARTIFACTS` | `50` |  |
| `A2A_MAX_CONTEXT_MESSAGES` | `100` |  |
| `A2A_STORAGE_UPDATE_RETRIES` | `4` |  |
| `A2A_DISPATCH_RECONCILE_INTERVAL_MS` | `1000` |  |
| `A2A_DISPATCH_RECONCILE_LIMIT` | `64` |  |
| `A2A_CANCELLATION_POLL_INTERVAL_MS` | `1000` |  |
| `A2A_CONFIG` | — | Path to external a2a_config.json |
| `A2A_REFRESH_INTERVAL` | `300` | Period (sec) for card refresh |
| `GRAPH_SERVICE_HEAVY_RPC_TIMEOUT` | `1200` | Seconds; matches the client's heavy-RPC budget |
| `GRAPH_SERVICE_CONNECT_TIMEOUT` | `10` | Seconds |
| `GRAPH_SERVICE_WRITE_TIMEOUT` | `30` | Seconds |
| `AGENT_UTILITIES_A2A_CALL_DEADLINE_MARGIN_SECONDS` | `60` | Extra headroom above heavy+connect+write |
| `CUSTOM_SKILLS_DIRECTORY` | — | Custom skill packages folder |
| `SKILL_TYPES` | — | Comma-separated list of active skill types |
| `X_TOOLS` | `0` | X/Grok social search via xAI (needs XAI_API_KEY); set 1 for production X use |
| `MEDIA_TOOLS` | `0` | Media generation/transcription services |
| `DB_TOOLS` | `0` | Native database traversal tools |
| `DATA_PREPTOOL` | `true` | Governed Arrow profile/clean/validate/commit surface |
| `MAX_TOKENS` | `16384` | ─────────────────────────────────────────────────────────────────────────────── 12. LLM Inference Parameters ─────────────────────────────────────────────────────────────────────────────── [OPTIONAL] Global inference controls and parameters |
| `TEMPERATURE` | `0.7` |  |
| `TOP_P` | `1.0` |  |
| `TIMEOUT` | `3600.0` |  |
| `TOOL_TIMEOUT` | `3600.0` |  |
| `PARALLEL_TOOL_CALLS` | `true` |  |
| `SEED` | — |  |
| `PRESENCE_PENALTY` | `0.0` |  |
| `FREQUENCY_PENALTY` | `0.0` |  |
| `LOGIT_BIAS` | — | e.g., '{"5028": -100}' |
| `STOP_SEQUENCES` | — | e.g., '["\n", "Human:"]' |
| `EXTRA_HEADERS` | — | Explicit process injection only |
| `EXTRA_BODY` | — | e.g., '{"presence_penalty": 0.2}' |
| `COGNITIVE_SCHEDULER_ENABLED` | `true` | ─────────────────────────────────────────────────────────────────────────────── 13. Cognitive Scheduler & OS Policies ─────────────────────────────────────────────────────────────────────────────── [OPTIONAL] Priority scheduling and memory policies |
| `MAX_CONCURRENT_AGENTS` | `5` |  |
| `AGENT_TOKEN_QUOTA` | secret-injected |  |
| `PREEMPTION_THRESHOLD_PCT` | `0.85` |  |
| `AGENT_POLICIES_PATH` | — | Path to agent_policies.json |
| `PERMISSIONS_SIGNING_KEY_REF` | `env://AGENT_PERMISSION_AUTHORITY` | Optional: unset self-provisions a durable shared key; set to pin an external/rotated key (wins) |
| `PERMISSIONS_IDENTITY_TTL_SECONDS` | `3600` | Bounded lifetime of a runtime-issued permission identity (auto-renewed on use; 0 disables expiry) |
| `PERMISSIONS_IDENTITY_REFRESH_SKEW_SECONDS` | `300` | Re-issue an identity within this window of expiry at the governed-execution boundary |
| `SPECIALIST_REGISTRY_PATH` | — | Local specialist agents catalog directory |
| `HOMEOSTATIC_DOWNGRADE_ENABLED` | `true` | Automatically downgrades to light model under load |
| `ADVERSARIAL_VERIFICATION` | `false` | Enable double verification checks |
| `MAINTENANCE_TOKEN_BUDGET` | secret-injected | Unlimited if 0 |
| `MAINTENANCE_PRIORITY` | `LOW` | LOW \| MEDIUM \| HIGH |
| `WATCHDOG_PATTERNS` | `'["pyproject.toml","mcp_config.json","requirements*.txt"]'` |  |
| `TOOL_GUARD_MODE` | `strict` | on \| strict (cannot be disabled) |
| `SENSITIVE_TOOL_PATTERNS` | `'[".*delete.*",".*remove.*",".*rm_.*",".*rmdir.*",".*drop.*",".*truncate.*",".*prune.*",".*kill.*",".*terminate.*",".*reboot.*",".*shutdown.*",".*install.*",".*uninstall.*",".*redeploy.*",".*bump.*",".*create.*",".*add.*",".*post.*",".*put.*",".*insert.*",".*upload.*",".*ingest.*",".*write.*",".*update.*",".*patch.*",".*set.*",".*reset.*",".*clear.*",".*revert.*",".*replace.*",".*rename.*",".*move.*",".*rotate.*",".*start.*",".*stop.*",".*restart.*",".*pause.*",".*unpause.*",".*execute.*",".*shell.*",".*run_shell.*",".*run_command.*",".*run_script.*",".*run_code.*",".*git_.*",".*clone.*",".*pull.*",".*maintain.*",".*setup.*",".*build.*",".*validate.*",".*sync.*",".*enable.*",".*disable.*",".*activate.*",".*approve.*",".*graphql.*",".*mutation.*",".*http.*",".*eval.*",".*exec.*",".*compile.*",".*socket.*",".*connect.*",".*os\\..*",".*subprocess\\..*",".*shutil\\..*"]'` |  |
| `MCP_URL` | — | Remote MCP SSE connection URL |
| `MCP_CONFIG` | `mcp_config.json` | Local multi-server definitions file |
| `KG_DAEMON_ROLE` | `auto` | ─────────────────────────────────────────────────────────────────────────────── 16. KG Background Daemon (single consolidated daemon — CONCEPT:EG-KG.storage.nonblocking-checkpoint / OS-5.9) ─────────────────────────────────────────────────────────────────────────────── The KG runs ONE background daemon (queue drain + graph writer + task workers + maintenance scheduler + file-watch). KG_DAEMON_ROLE selects who runs it: host   — run the full daemon (set on the API gateway / agent-webui, or run the standalone `graph-os-daemon` process). client — spawn NOTHING; enqueue work to the durable queue the host drains (set on the graph-os MCP server, CLI, and one-shot scripts). auto   — default: in-process consolidated daemon (standalone / dev). |
| `AGENTTOOL` | `true` | ─────────────────────────────────────────────────────────────────────────────── 18. MCP Condensed-Surface Domain Toggles ─────────────────────────────────────────────────────────────────────────────── |
| `ANALYSISTOOL` | `true` |  |
| `ANALYZE_SUITETOOL` | `true` |  |
| `ARGUMENTTOOL` | `true` |  |
| `BUSTOOL` | `true` |  |
| `CANDIDATE_CLAIMTOOL` | `true` |  |
| `CAPABILITYTOOL` | `true` |  |
| `CLAIMTOOL` | `true` |  |
| `CONFIGTOOL` | `true` |  |
| `ENGINETOOL` | `true` |  |
| `GRAPHOS_VERBOSETOOL` | `true` |  |
| `GRAPH_ENGINEERINGTOOL` | `true` |  |
| `INTENTTOOL` | `true` |  |
| `MCP_APPSTOOL` | `true` |  |
| `MEDIA_SIDECARTOOL` | `true` |  |
| `ONTOLOGYTOOL` | `true` |  |
| `OPS_CAUSALTOOL` | `true` |  |
| `QUANTTOOL` | `true` |  |
| `QUERYTOOL` | `true` |  |
| `REACHTOOL` | `true` |  |
| `SECRETTOOL` | `true` |  |
| `STATETOOL` | `true` |  |
| `SWETOOL` | `true` |  |
| `WRITE_INGESTTOOL` | `true` |  |
| `A2A_TOOLS` | `true` | ─────────────────────────────────────────────────────────────────────────────── 19. Tool Surface Toggles ─────────────────────────────────────────────────────────────────────────────── |
| `AGENT_KG_TOOLS` | `true` |  |
| `BROWSER_TOOLS` | `true` |  |
| `COMPUTER_USE_TOOLS` | `false` |  |
| `DEVELOPER_TOOL_MAX_OUTPUT_BYTES` | `65536` |  |
| `DEVELOPER_TOOL_MAX_TIMEOUT_SECONDS` | `600` |  |
| `DEVELOPER_TOOLS` | `true` |  |
| `GIT_TOOLS` | `true` |  |
| `SCHEDULER_TOOLS` | `true` |  |
| `SWE_TOOLS` | `false` |  |
| `WORKSPACE_TOOLS` | `true` |  |
| `BUS_HUB_ID` | — | ─────────────────────────────────────────────────────────────────────────────── 20. Messaging Platform Config ─────────────────────────────────────────────────────────────────────────────── |
| `GOOGLE_CHAT_SERVICE_ACCOUNT` | — |  |
| `GOOGLE_MEET_SERVICE_ACCOUNT` | — |  |
| `IRC_NICKNAME` | `agent_bot` |  |
| `IRC_PORT` | `6667` |  |
| `IRC_SERVER` | — |  |
| `LINE_CHANNEL_ACCESS_TOKEN` | secret-injected |  |
| `MATRIX_ACCESS_TOKEN` | secret-injected |  |
| `MATRIX_HOMESERVER` | — |  |
| `MATRIX_USER_ID` | — |  |
| `MATTERMOST_BOT_USER` | — |  |
| `MATTERMOST_URL` | — |  |
| `MESSAGING_AGENT` | — |  |
| `MESSAGING_ALERT_INTAKE_ALLOW_REMOTE` | `False` |  |
| `MESSAGING_ALERT_INTAKE_HOST` | `127.0.0.1` |  |
| `MESSAGING_ALERT_INTAKE_PORT` | — |  |
| `MESSAGING_ALERT_INTAKE_TOKEN_REF` | — |  |
| `MESSAGING_BURST_MAX_S` | `12` |  |
| `MESSAGING_BURST_WINDOW_S` | `2.5` |  |
| `MESSAGING_ADDRESSED_MODEL` | — |  |
| `MESSAGING_MODEL_TRIGGER` | — |  |
| `MESSAGING_DEFAULT_CHANNEL` | — |  |
| `MESSAGING_DEFAULT_PLATFORM` | `telegram` |  |
| `MESSAGING_INTAKE_ENABLED` | `false` |  |
| `MESSAGING_DISCORD_TOKEN` | secret-injected |  |
| `MESSAGING_ENABLED_BACKENDS` | — |  |
| `MESSAGING_ENRICH` | `1` |  |
| `MESSAGING_GOALS` | `1` |  |
| `MESSAGING_GOOGLECHAT_TOKEN` | secret-injected |  |
| `MESSAGING_GOOGLEMEET_TOKEN` | secret-injected |  |
| `MESSAGING_INBOX_RETRY_S` | `120` |  |
| `MESSAGING_IRC_CHANNELS` | — |  |
| `MESSAGING_IRC_NICKNAME` | `agent_bot` |  |
| `MESSAGING_IRC_PORT` | `6667` |  |
| `MESSAGING_IRC_SERVER` | — |  |
| `MESSAGING_KG_INGEST` | `true` |  |
| `MESSAGING_KG_MEMORY_TYPE` | `episodic` |  |
| `MESSAGING_LINE_TOKEN` | secret-injected |  |
| `MESSAGING_LISTEN_BACKOFF_BASE_S` | `1` |  |
| `MESSAGING_LISTEN_BACKOFF_MAX_S` | `60` |  |
| `MESSAGING_LISTEN_HEALTHY_RESET_S` | `60` |  |
| `MESSAGING_DEFAULT_MODEL` | — |  |
| `MESSAGING_LOG_LEVEL` | `INFO` |  |
| `MESSAGING_MATRIX_HOMESERVER` | — |  |
| `MESSAGING_MATRIX_TOKEN` | secret-injected |  |
| `MESSAGING_MATRIX_USER_ID` | — |  |
| `MESSAGING_MATTERMOST_TOKEN` | secret-injected |  |
| `MESSAGING_MATTERMOST_URL` | — |  |
| `MESSAGING_NEXTCLOUD_APP_ID` | — |  |
| `MESSAGING_NEXTCLOUD_TOKEN` | secret-injected |  |
| `MESSAGING_NEXTCLOUD_URL` | — |  |
| `REACTIONS` | `1` | Native model-agnostic reaction output |
| `MESSAGING_PROGRESS_STREAMING` | `False` |  |
| `MESSAGING_REPLY_TIMEOUT` | `45` |  |
| `MESSAGING_ROUTE_TO_PLANNER` | `true` |  |
| `MESSAGING_SIGNAL_TOKEN` | secret-injected |  |
| `MESSAGING_SLACK_APP_TOKEN` | secret-injected |  |
| `MESSAGING_SLACK_TOKEN` | secret-injected |  |
| `MESSAGING_SYNOLOGY_WEBHOOK_URL` | — |  |
| `MESSAGING_TEAMS_APP_ID` | — |  |
| `MESSAGING_TEAMS_APP_SECRET` | secret-injected |  |
| `MESSAGING_TELEGRAM_TOKEN` | secret-injected |  |
| `MESSAGING_TRANSPARENCY_FOOTER` | `true` |  |
| `MESSAGING_TWITCH_CHANNELS` | — |  |
| `MESSAGING_TWITCH_TOKEN` | secret-injected |  |
| `MESSAGING_VOICE` | `1` |  |
| `MESSAGING_VOICECALL_APP_ID` | — |  |
| `MESSAGING_VOICECALL_FROM_NUMBER` | — |  |
| `MESSAGING_VOICECALL_TOKEN` | secret-injected |  |
| `MESSAGING_VOICE_MODEL` | — |  |
| `MESSAGING_WEBHOOK_BASE_URL` | — |  |
| `MESSAGING_WEBHOOK_PORT` | `8443` |  |
| `MESSAGING_WEBHOOK_SECRET` | secret-injected |  |
| `MESSAGING_WHATSAPP_PHONE_NUMBER_ID` | — |  |
| `MESSAGING_WHATSAPP_TOKEN` | secret-injected |  |
| `MESSAGING_WHATSAPP_USE_BUSINESS_API` | `false` |  |
| `MSTEAMS_APP_PASSWORD` | secret-injected |  |
| `NEXTCLOUD_TOKEN` | secret-injected |  |
| `NEXTCLOUD_URL` | — |  |
| `NEXTCLOUD_USER` | — |  |
| `SIGNAL_PHONE_NUMBER` | — |  |
| `SLACK_APP_TOKEN` | secret-injected |  |
| `TWILIO_ACCOUNT_SID` | — |  |
| `TWILIO_AUTH_TOKEN` | secret-injected |  |
| `TWILIO_FROM_NUMBER` | — |  |
| `TWITCH_CHANNELS` | — |  |
| `TWITCH_OAUTH_TOKEN` | secret-injected |  |
| `ANSIBLE_ENABLE_WRITE` | `false` | ─────────────────────────────────────────────────────────────────────────────── 21. Writeback Sink Live-Write Gates ─────────────────────────────────────────────────────────────────────────────── |
| `ARCHIMATE_ENABLE_WRITE` | `false` |  |
| `CADDY_ENABLE_WRITE` | `false` |  |
| `CISO_ASSISTANT_ENABLE_WRITE` | `false` |  |
| `EMERALD_ENABLE_WRITE` | `false` |  |
| `GITHUB_ENABLE_WRITE` | `false` |  |
| `GITLAB_ENABLE_WRITE` | `false` |  |
| `HOMEASSISTANT_ENABLE_WRITE` | `false` |  |
| `JIRA_ENABLE_WRITE` | `false` |  |
| `KEYCLOAK_ENABLE_WRITE` | `false` |  |
| `LEANIX_ENABLE_WRITE` | `false` |  |
| `LEGAL_ENABLE_WRITE` | `false` |  |
| `LGTM_ENABLE_WRITE` | `false` |  |
| `MEALIE_ENABLE_WRITE` | `false` |  |
| `NEXTCLOUD_ENABLE_WRITE` | `false` |  |
| `OKF_ENABLE_WRITE` | `false` |  |
| `OKTA_ENABLE_WRITE` | `false` |  |
| `PLANE_ENABLE_WRITE` | `false` |  |
| `PORTAINER_ENABLE_WRITE` | `false` |  |
| `SALESFORCE_ENABLE_WRITE` | `false` |  |
| `TECHNITIUM_DNS_ENABLE_WRITE` | `false` |  |
| `UPTIME_KUMA_ENABLE_WRITE` | `false` |  |
| `WGER_ENABLE_WRITE` | `false` |  |
| `BACKSTAGE_FILE` | `catalog-info.yaml` | ─────────────────────────────────────────────────────────────────────────────── 22. External System Discovery (EA / ITSM / Observability endpoints) ─────────────────────────────────────────────────────────────────────────────── |
| `BAO_URL` | — |  |
| `BPMN_FILE` | `process.bpmn` |  |
| `BPM_PROVIDER` | `opensource` |  |
| `BPM_TOKEN` | secret-injected |  |
| `BPM_URL` | — |  |
| `CADDY_API_URL` | — |  |
| `CADDY_URL` | — |  |
| `CHECKLIST_FILE` | `task.md` |  |
| `EAR_TOKEN` | secret-injected |  |
| `EAR_URL` | — |  |
| `EMERALD_API_KEY` | secret-injected |  |
| `EMERALD_URL` | — |  |
| `ERPNEXT_TOKEN` | secret-injected |  |
| `ERPNEXT_URL` | — |  |
| `ESSENTIAL_EA_TOKEN` | secret-injected |  |
| `ESSENTIAL_EA_URL` | — |  |
| `GITHUB_API_KEY` | secret-injected |  |
| `GITHUB_TOKEN` | secret-injected |  |
| `GITHUB_URL` | `https://api.github.com` |  |
| `GITLAB_API_TOKEN` | secret-injected |  |
| `GLPI_TOKEN` | secret-injected |  |
| `GLPI_URL` | — |  |
| `GRAFANA_URL` | — |  |
| `JIRA_API_TOKEN` | secret-injected |  |
| `JIRA_TOKEN` | secret-injected |  |
| `JIRA_URL` | — |  |
| `KEYCLOAK_ADMIN_PASSWORD` | secret-injected |  |
| `LGTM_URL` | — |  |
| `LISTMONK_TOKEN` | secret-injected |  |
| `LISTMONK_URL` | — |  |
| `MATTERMOST_TOKEN` | secret-injected |  |
| `NEXTCLOUD_PASSWORD` | secret-injected |  |
| `OPENMAINT_TOKEN` | secret-injected |  |
| `OPENMAINT_URL` | — |  |
| `PLANE_API_TOKEN` | secret-injected |  |
| `PLANE_TOKEN` | secret-injected |  |
| `PLANE_URL` | — |  |
| `PORTAINER_PASSWORD` | secret-injected |  |
| `PORTAINER_TOKEN` | secret-injected |  |
| `PORTAINER_URL` | — |  |
| `POSTIZ_TOKEN` | secret-injected |  |
| `POSTIZ_URL` | — |  |
| `SCHOLARX_API_KEY` | secret-injected |  |
| `SCHOLARX_URL` | — |  |
| `SERVICENOW_PASSWORD` | secret-injected |  |
| `SERVICENOW_INSTANCE` | — |  |
| `SERVICENOW_USERNAME` | — |  |
| `SERVICENOW_URL` | — |  |
| `TECHNITIUM_TOKEN` | secret-injected |  |
| `TECHNITIUM_URL` | — |  |
| `TUNNEL_MANAGER_URL` | — |  |
| `TUNNEL_URL` | — |  |
| `TWENTY_API_TOKEN` | secret-injected |  |
| `TWENTY_TOKEN` | secret-injected |  |
| `TWENTY_URL` | — |  |
| `UPTIME_KUMA_URL` | — |  |
| `VAULT_URL` | — |  |
| `GRAPH_DB_POOL_MIN` | `4` | Minimum warm PostgreSQL graph-backend connections |
| `GRAPH_DB_POOL_MAX` | `32` | Maximum shared PostgreSQL graph-backend connections |
| `GRAPH_DB_POOL_TIMEOUT` | `5` |  |
| `GRAPH_FUSEKI_DATASET` | — |  |
| `GRAPH_FUSEKI_PASSWORD_REF` | `vault://graph/fuseki` | password |
| `GRAPH_FUSEKI_USER` | — |  |
| `GRAPH_PGGRAPH_SCHEMA` | `public` |  |
| `GRAPH_PG_AGE` | — |  |
| `LADYBUG_DB_READ_ONLY` | `0` |  |
| `LADYBUG_MAX_DB_SIZE` | — |  |
| `LADYBUG_TRANSIENT_CONNECTIONS` | — |  |
| `OWL_ALLOW_REMOTE_IMPORTS` | `false` |  |
| `OWL_BACKEND` | — |  |
| `OWL_DB_PATH` | — |  |
| `STARDOG_DATABASE` | — |  |
| `STARDOG_ENDPOINT` | — |  |
| `STARDOG_PASSWORD` | secret-injected |  |
| `STARDOG_USER` | — |  |
| `GRAPH_COMPUTE_BACKEND` | `rust` | ─────────────────────────────────────────────────────────────────────────────── 24. Graph Engine Service (sharding / connection) ─────────────────────────────────────────────────────────────────────────────── |
| `GRAPH_SERVICE_AUTH_SECRET` | secret-injected |  |
| `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` | — | Trusted signer ids -> HMAC keys, the SAME map the engine reads. A process can only sign an identity-admission operation as ITSELF (the engine requires signer == calling principal), so this must name this process's own principal. Unset for a packaged local engine, which mints its own bootstrap key. |
| `GRAPH_OS_WEBUI_PORT` | `8080` | Port the agent-webui dashboard binds when graph-os serves it in-process (ENABLE_WEB_UI). MUST differ from graph-os's own listener port. Default 8080. |
| `GRAPH_SERVICE_PERSIST_DIR` | — | Durable store for an auto-started engine. Leave unset ONLY where the implicit location under AGENT_UTILITIES_DATA_DIR is genuinely durable: in a container that path is routinely an emptyDir, so an unset value means the graph is discarded on every restart. Autostart logs a WARNING when it falls back. |
| `GRAPH_SERVICE_PERSIST_ON_SHUTDOWN` | `true` |  |
| `GRAPH_SERVICE_TCP_ADDR` | — | Extra listeners an auto-started engine arms, in ADDITION to the unix socket. Each is optional and omitted from the child's argv when unset, so leaving all four unset reproduces the socket-only behaviour exactly. The names match the engine's own clap `env =` declarations, so exporting the bare variable and passing the flag are equivalent. Set these where the engine must also serve consumers that dial it directly over the network -- e.g. a single-container topology fronted by a Service -- rather than only through this process. |
| `GRAPH_SERVICE_TLS_CERT` | — |  |
| `GRAPH_SERVICE_TLS_KEY` | secret-injected |  |
| `GRAPH_SERVICE_METRICS_ADDR` | — | Bind address for the engine's Prometheus metrics listener. Defaults to 127.0.0.1:9101 (loopback only) when unset, which is unreachable from outside the container -- set it explicitly if metrics must be scraped. |
| `EG_ANALYTICS_WORKER_CAPABILITIES` | `mining.association,pool:default` | Packaged-local overrides only; TCP must be loopback. |
| `EG_ANALYTICS_WORKER_LEASE_MS` | `60000` |  |
| `EG_ANALYTICS_WORKER_POLL_SECONDS` | `0.25` |  |
| `EG_ANALYTICS_WORKER_SLOTS` | `1` |  |
| `GRAPH_OS_ANALYTICS_PRINCIPAL` | — |  |
| `GRAPH_OS_ANALYTICS_TENANT` | — |  |
| `GRAPH_OS_BACKUP_PRINCIPAL` | — |  |
| `GRAPH_OS_BACKUP_TENANT` | — |  |
| `GRAPHOS_BACKUP_RETENTION_COUNT` | `2` |  |
| `EPISTEMIC_GRAPH_RESTORE_BIN` | `restore` |  |
| `EPISTEMIC_GRAPH_SERVER_BIN` | `epistemic-graph-server` |  |
| `RESTORE_VALIDATION_PORT` | `19100` |  |
| `PLACEMENT_CATALOG_ENABLED` | `true` | resolve graph placement via the engine's PlacementCatalog RPC |
| `COMPUTER_USE_DISPLAY` | `:1` |  |
| `COMPUTER_USE_USER` | `sandbox` |  |
| `COMPUTER_USE_HOME` | — |  |
| `KG_ADAPTIVE_CONCURRENCY` | `true` | ─────────────────────────────────────────────────────────────────────────────── 25. Knowledge Graph Engine Tuning ─────────────────────────────────────────────────────────────────────────────── |
| `KG_AGENT_AUTO_APPLY` | `false` |  |
| `KG_ASR_MODEL` | — |  |
| `KG_AUTO_INGEST_SKILLS` | `true` |  |
| `KG_BREADTH_LIBRARY_ROOTS` | — |  |
| `KG_BREADTH_REPO_ROOTS` | — |  |
| `KG_CARD_MODEL` | `lite` |  |
| `KG_CONCEPT_CODE_LINK` | `1` |  |
| `KG_CONNECTIONS` | — |  |
| `KG_CHUNKED_DRAIN` | `true` |  |
| `KG_DRAIN_MAX_PAGES` | `5000` |  |
| `KG_DRAIN_PAGE_SIZE` | `100` |  |
| `KG_DAEMON_LOG_LEVEL` | `INFO` |  |
| `KG_DEFAULT_GRAPH` | `__commons__` |  |
| `KG_DEV_MODE` | `false` |  |
| `KG_OPTIMIZATION_INTERVAL` | `10800.0` |  |
| `KG_EA_WRITEBACK` | `false` |  |
| `KG_EMBED_TIMEOUT` | `30.0` |  |
| `KG_ENABLE_HARD_NEGATIVE_MINING` | `false` |  |
| `KG_ENGINE_DETACHED` | — |  |
| `KG_ENGINE_POOL_DROP_ON_EVICT` | `false` |  |
| `KG_ENGINE_POOL_SIZE` | `8` |  |
| `KG_ENGINE_TOOL_POOL_SIZE` | `16` |  |
| `KG_ENRICH_CHUNK_THRESHOLD` | `12000` |  |
| `KG_ENRICH_MAX_CHUNKS` | `64` |  |
| `KG_EVAL_CAPTURE` | `false` |  |
| `KG_EXTRACT_MAX_RETRIES` | `4` |  |
| `KG_FAILURE_EVOLUTION_WINDOW` | `86400.0` |  |
| `KG_FAILURE_REGRESSION_DATASET` | `false` |  |
| `KG_FUSEKI_ENDPOINT` | `https://fuseki.example.test` |  |
| `KG_FUSEKI_PUBLISH` | `false` |  |
| `KG_FUSEKI_PUBLISH_INTERVAL` | `3600.0` |  |
| `KG_GRAPH_NAME` | `__commons__` |  |
| `KG_GRAPH_OWNERSHIP_ENFORCED` | `false` | Enforce per-graph ownership disposition; default off |
| `KG_INGEST_AUTO_EMBED` | `true` |  |
| `KG_INGEST_INFLIGHT` | — |  |
| `KG_INGEST_PROFILE` | — |  |
| `KG_INGEST_SHARD_FANOUT` | `false` |  |
| `KG_LLM_PRIORITY_RESERVE` | — |  |
| `KG_LLM_PRIORITY_RESERVE_FRACTION` | `0.34` |  |
| `KG_MEDIA_TENANT_ISOLATED_BLOBS` | `false` |  |
| `KG_MIN_KEYWORD_DISCOVER_RELEVANCE_THRESHOLD` | `0.1` | Relevance floor for engine-native discover() keyword-overlap hits (separate from the lexical-fallback and vector-score floors) |
| `KG_MIN_LEXICAL_RELEVANCE_THRESHOLD` | `0.15` | Relevance floor for lexical-fallback hits (separate from the vector-score floor below) |
| `KG_MIN_RELEVANCE_THRESHOLD` | — |  |
| `KG_PARSE_BATCH` | `512` |  |
| `KG_POLICY_VERSION` | — |  |
| `KG_POOL_MEMORY_GEN_CAP` | — |  |
| `KG_POOL_ACQUISITION_FLOOR` | — | Reserved acquisition workers; default max(1, worker_count // 4) |
| `KG_PROCESS_WRITEBACK` | `false` |  |
| `KG_PROVIDER_ADAPTER_BACKEND` | `static` |  |
| `KG_RERANK_BASE_URL` | — |  |
| `KG_RERANK_LOCAL_NEURAL` | `false` |  |
| `KG_RERANK_MODEL` | — |  |
| `KG_RESEARCH_EXTERNAL` | `0` |  |
| `KG_RESEARCH_FEED` | `true` |  |
| `KG_RESEARCH_FEED_INTERVAL` | `1800.0` |  |
| `KG_RETRIEVAL_QUALITY_GATE` | `true` |  |
| `KG_RSS_FEEDS` | — |  |
| `KG_SCHED_CODEBASE_CAP` | — |  |
| `KG_SCHED_PER_LANE_MIN` | `1` | Minimum worker coverage per active scheduling lane |
| `KG_SCHED_RESERVED` | `1` | Hot-spare workers reserved from ordinary lane allocation |
| `KG_STAGED_PIPELINE` | `1` |  |
| `KG_STRICT_SOURCE_PARTITION` | `false` |  |
| `KG_TASKS_PARTITIONS` | `6` |  |
| `KG_TENANT_GC_INTERVAL` | `300.0` |  |
| `KG_TRUST_HIERARCHY` | — |  |
| `KG_WATCH_DIRS` | — |  |
| `KG_WORKFLOW_SHAPE_GATE` | `true` |  |
| `KG_WRITE_DELTA` | `1` |  |
| `LAKEHOUSE_S3_ENDPOINT` | — | SeaweedFS S3-compatible gateway (services/lakehouse-seaweedfs) |
| `LAKEKEEPER_CATALOG_URI` | — | Lakekeeper Iceberg REST catalog base URI (services/lakekeeper); must end in /catalog |
| `LAKEKEEPER_WAREHOUSE` | `lakehouse` | Iceberg warehouse name registered with the Lakekeeper catalog |
| `LAKEKEEPER_OAUTH2_SCOPE` | `lakekeeper` | Keycloak OAuth2 scope for the Lakekeeper catalog (client default "catalog" is NOT granted) |
| `LAKEKEEPER_DB_URI_REF` | — | Runtime secret reference for the Lakekeeper Postgres DSN (services/lakekeeper-db) |
| `TRINO_ENDPOINT` | — | Trino coordinator HTTP endpoint (services/trino) |
| `SPARK_RUNNER_ENDPOINT` | — | Spark driver pod's Spark UI endpoint (services/spark, Deployment spark-runner) |
| `ARD_REGISTRIES` | — | ─────────────────────────────────────────────────────────────────────────────── 26. Knowledge Graph Core Tuning ─────────────────────────────────────────────────────────────────────────────── |
| `CLAUDE_MEMORY_DIR` | — |  |
| `DOCKERHUB_NAMESPACE` | — |  |
| `DOCKERHUB_NAMESPACES` | — |  |
| `EPISTEMIC_GRAPH_REDB_SHARDS` | — |  |
| `EPISTEMIC_GRAPH_SOCKET` | `/tmp/epistemic-graph.sock` |  |
| `EVENT_BACKEND` | — |  |
| `FRESHRSS_MAX_BATCHES` | `3` |  |
| `FRESHRSS_USE_NOVELTY` | `false` |  |
| `GITLAB_TOKEN` | secret-injected |  |
| `GITLAB_URL` | `https://gitlab.com` |  |
| `REDPANDA_BROKERS` | — |  |
| `REDPANDA_CONSUMER_GROUP` | — |  |
| `REDPANDA_SECURITY_PROTOCOL` | — |  |
| `SOURCE_SYNC_ALLOW_EMPTY_TOMBSTONE` | — |  |
| `CONFLUENCE_SPACE_IDS` | — | Comma-separated fallback Confluence space IDs for the default instance |
| `JIRA_PROJECT_KEYS` | — | Comma-separated fallback Jira project keys for the default instance |
| `PLANE_PROJECT_IDS` | — | Comma-separated fallback Plane project IDs for the default instance |
| `ASSIMILATION_ENGINE_PAGERANK` | — | ─────────────────────────────────────────────────────────────────────────────── 27. Knowledge Graph (misc) ─────────────────────────────────────────────────────────────────────────────── |
| `ASSIMILATION_SYNTH_TIMEOUT_S` | `30` |  |
| `KG_AMBIENT_EPISTEMIC` | `true` | background epistemic enrichment during ingestion |
| `KG_AMBIENT_EPISTEMIC_DISABLED_SOURCES` | — | comma-separated source names to exclude |
| `AUDIT_REVIEW_TIMEOUT_S` | `45` | wall-clock budget for the KG_LOOP_AUDIT review round |
| `KG_INSIGHT_AUTONOMY` | `false` |  |
| `KG_LOOP_AUDIT` | `false` | opt-in loop entry: harness.audit_gap_detector |
| `KG_LOOP_AUTO_DEVELOP` | `false` |  |
| `KG_LOOP_BELIEF_REVISION` | `true` |  |
| `KG_LOOP_BREADTH` | `true` |  |
| `KG_LOOP_DISCOVER` | `false` |  |
| `KG_LOOP_DISTILL` | `false` |  |
| `KG_LOOP_GOAL_EVAL_ENABLED` | `true` | score each loop cycle against its stated goal |
| `KG_LOOP_GOAL_EVAL_THRESHOLD` | `0.7` |  |
| `KG_LOOP_INSIGHT_VALIDATION` | `true` |  |
| `KG_LOOP_MAX_CONSECUTIVE_FAILURES` | `3` |  |
| `KG_LOOP_MAX_DURATION_S` | `0.0` | 0 = no wall-clock cap |
| `KG_LOOP_MINE_DISCOVERY` | `true` |  |
| `KG_LOOP_NO_PROGRESS_WINDOW` | `3` |  |
| `KG_LOOP_SKILL_EVOLUTION` | `true` |  |
| `KG_LOOP_STANDARDIZE` | `false` |  |
| `KG_LOOP_TRACE_MINING` | `true` |  |
| `KG_SAI_FACTORY` | `true` |  |
| `KG_SAI_FACTORY_INTERVAL` | `3600.0` |  |
| `KG_SKILL_EVOLUTION_LANGFUSE_HOLDOUT_DATASET` | — | Langfuse dataset name held out from training |
| `KG_SKILL_EVOLUTION_LANGFUSE_SCORE_NAME` | — | Langfuse score name read as the reward signal |
| `KG_SKILL_EVOLUTION_LANGFUSE_TRAIN_DATASET` | — | Langfuse dataset name used for training |
| `KG_SKILL_EVOLUTION_LANGFUSE_WEIGHT` | `0.5` | blend weight vs. the native reward signal |
| `AGENT_USER_MAP` | `{}` | ─────────────────────────────────────────────────────────────────────────────── 30. Enrichment Pipeline ─────────────────────────────────────────────────────────────────────────────── |
| `FRESHRSS_URL` | — | ─────────────────────────────────────────────────────────────────────────────── 31. Research & Assimilation ─────────────────────────────────────────────────────────────────────────────── |
| `KG_ARXIV_CATEGORIES` | — | arXiv ingestion is OPT-IN: an unscoped arXiv query is not a useful source, so the connector stays inert until KG_ARXIV_CATEGORIES names at least one category. |
| `KG_ARXIV_MAX_RESULTS` | `50` |  |
| `PLACEMENT_CONTROL_LOOP_ENABLED` | `false` |  |
| `SKILL_GRAPH_CRAWLER` | — | ─────────────────────────────────────────────────────────────────────────────── 32. Skill-Graph Distillation ─────────────────────────────────────────────────────────────────────────────── |
| `SKILL_GRAPH_CRAWLER_PYTHON` | — |  |
| `SKILL_GRAPH_CRAWL_TIMEOUT` | — |  |
| `SKILL_GRAPH_MAX_PAGES` | — |  |
| `AGENTS_ROOT` | — | ─────────────────────────────────────────────────────────────────────────────── 33. Ontology / Connector Governance ─────────────────────────────────────────────────────────────────────────────── |
| `CA26_EXTERNAL_POLICY_SYNC_ENABLED` | `false` |  |
| `CONFLUENCE_INSTANCES` | — | ─────────────────────────────────────────────────────────────────────────────── 35. Multi-Instance Source Connectors ─────────────────────────────────────────────────────────────────────────────── |
| `GITLAB_INSTANCES` | — |  |
| `JIRA_INSTANCES` | — |  |
| `PLANE_INSTANCES` | — |  |
| `ARD_REQUIRE_SIGNATURE` | `false` | ─────────────────────────────────────────────────────────────────────────────── 36. Source Connectors ─────────────────────────────────────────────────────────────────────────────── |
| `JINA_API_KEY` | secret-injected |  |
| `ARCHIVEBOX_URL` | — | ─────────────────────────────────────────────────────────────────────────────── 37. External Service Endpoints ─────────────────────────────────────────────────────────────────────────────── |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka.example.test:9092` |  |
| `KAFKA_CDC_DEBEZIUM_ENABLED` | `false` |  |
| `KAFKA_ENABLED` | `false` |  |
| `KAFKA_ENABLE_WRITE` | `false` |  |
| `KAFKA_OPENLINEAGE_CONSUMER_ENABLED` | `false` |  |
| `KAFKA_TOPIC` | — |  |
| `NATS_URL` | — |  |
| `OPENSEARCH_CA_CERTS` | — |  |
| `OPENSEARCH_PASSWORD` | secret-injected |  |
| `OPENSEARCH_TIMEOUT_S` | `30` |  |
| `OPENSEARCH_URL` | `http://localhost:9200` |  |
| `OPENSEARCH_USER` | — |  |
| `OPENSEARCH_VERIFY_CERTS` | `true` |  |
| `SPARQL_ENDPOINTS` | `["https://query.wikidata.org/sparql"]` |  |
| `VLLM_BASE_URL` | — |  |
| `MCP_CHILD_BREAKER_COOLDOWN` | `15.0` | ─────────────────────────────────────────────────────────────────────────────── 38. MCP Multiplexer & Child Process Management ─────────────────────────────────────────────────────────────────────────────── |
| `MCP_CHILD_BREAKER_THRESHOLD` | `5` |  |
| `MCP_CHILD_MAX_CONCURRENCY` | `8` |  |
| `MCP_CHILD_MAX_RESTARTS` | `5` |  |
| `MCP_CHILD_POOL_SIZE` | `1` |  |
| `MCP_CHILD_QUEUE_TIMEOUT` | `30.0` |  |
| `MCP_CHILD_RESTART_WINDOW` | `300.0` |  |
| `MCP_DYNAMIC_TOP_K` | `8` |  |
| `MCP_DYNAMIC_DISCOVERY_TIMEOUT` | `5.0` |  |
| `MCP_CATALOG_PROBE_TTL` | `300` | Seconds a probed child-server catalog entry stays fresh (1-86400) |
| `AGENT_UTILITIES_MCP_TASK_CHANNEL_SECRET` | secret-injected | Local stdio child multiplexing binds a per-connection-generation channel secret that authenticates delegated-task requests from that specific child process; the multiplexer always generates this itself (secrets.token_urlsafe) and injects it into the child's environment — never set it yourself, and a child-supplied value in its own configured env block is rejected as parent-controlled. |
| `MCP_ALWAYS_LOAD` | `["tunnel-manager-mcp","systems-manager-mcp","repository-manager-mcp","container-manager-mcp"]` | Fleet servers mounted EAGERLY on a session's first contact with graph-os, before any find_tools round trip. Fail-soft: a broken server degrades to lazy discovery and never blocks startup. Set to [] for fully-lazy. |
| `MCP_ALWAYS_LOAD_TOOLS` | `["github-mcp:github_issues","github-mcp:github_pulls","gitlab-mcp:gitlab_issues","gitlab-mcp:gitlab_merge_requests"]` | INDIVIDUAL tools mounted eagerly, for servers too large to mount whole. Entries are <server>:<tool> (preferred) or an already-prefixed name. |
| `FRONTEND_CONTRIBUTION_TRUSTED_SIGNERS` | `[]` | Bounded signer-key-id allowlist for installed frontend contributions. Empty/unset denies every contribution. |
| `MCP_TOOL_MODE` | `intent` | intent \| condensed \| verbose \| both |
| `AGENT_EXECUTIONTOOL` | `true` |  |
| `DOMAIN_OPSTOOL` | `true` |  |
| `DURABLETOOL` | `true` |  |
| `EVOLUTIONTOOL` | `true` |  |
| `GOVERNANCETOOL` | `true` |  |
| `JOBTOOL` | `true` |  |
| `RLMTOOL` | `true` |  |
| `WORKFLOWTOOL` | `true` |  |
| `AGENT_ID` | — | ─────────────────────────────────────────────────────────────────────────────── 39. MCP Server / OIDC Auth ─────────────────────────────────────────────────────────────────────────────── |
| `GRAPH_FANOUT_TIMEOUT` | — |  |
| `KEYCLOAK_CLIENT_ID` | — |  |
| `KEYCLOAK_REALM` | `master` |  |
| `KEYCLOAK_URL` | — |  |
| `MCP_PUBLIC_BASE_URL` | — |  |
| `OIDC_AUDIENCE` | `agent-services` |  |
| `OIDC_BASE_URL` | — |  |
| `OIDC_ISSUER` | — |  |
| `OIDC_SCOPE` | — |  |
| `OIDC_TOKEN_URL` | secret-injected |  |
| `OPENAPI_CLIENT_ID` | — |  |
| `OPENAPI_CLIENT_SECRET_REF` | — |  |
| `OPENAPI_PASSWORD_REF` | — |  |
| `OPENAPI_USERNAME` | — |  |
| `MCP_FLEET_SECRET_REFS` | secret-injected |  |
| `SESSION_ID` | — |  |
| `AGENT_UTILITIES_RUNTIME_DIR` | — | ─────────────────────────────────────────────────────────────────────────────── 40. Security & Secrets ─────────────────────────────────────────────────────────────────────────────── |
| `ARD_SIGNING_PRIVATE_KEY` | secret-injected |  |
| `MAX_TOOL_CALLS_PER_SESSION` | `50` |  |
| `MAX_TOOL_REPEATS` | `3` |  |
| `OIDC_TLS_PROFILE` | — |  |
| `OIDC_TLS_PROFILE_REF` | — |  |
| `OIDC_HTTP_ALLOWED_PRIVATE_HOSTS` | `[]` |  |
| `MCP_BASIC_AUTH_PASSWORD_REF` | — |  |
| `MCP_BEARER_TOKEN_FILE` | secret-injected |  |
| `SECURITY_PROMPT_THRESHOLD` | `0.8` |  |
| `SOURCE_CREDENTIALS` | `{}` |  |
| `VAULT_ROLE_ID` | — |  |
| `VAULT_SECRET_ID` | secret-injected |  |
| `VAULT_TOKEN` | secret-injected |  |
| `ESCALATION_BLAST_FANOUT` | `20` | ─────────────────────────────────────────────────────────────────────────────── 41. Observability & Escalation ─────────────────────────────────────────────────────────────────────────────── |
| `ESCALATION_CI_RETRY_CAP` | `3` |  |
| `ESCALATION_DIFF_FILES` | `10` |  |
| `ESCALATION_REWARD_FLOOR` | `0.3` |  |
| `HITL_ESCALATION_TIMEOUT` | `300` |  |
| `INCIDENT_NOTIFY_URL` | — |  |
| `INCIDENT_PLANE_PROJECT` | — | Target project/board key for the selected INCIDENT_TICKET_BACKEND writeback adapter (agent_utilities/observability/incident_router.py). Each adapter declares its own key, so only the one matching the active backend is read. |
| `INCIDENT_TICKET_BACKEND` | `none` |  |
| `INCIDENT_TICKET_ENABLE` | `false` |  |
| `INFRA_INVENTORY_PATH` | — |  |
| `OTEL_SERVICE_NAME` | — |  |
| `TRM_WRITEBACK_BACKEND` | `none` |  |
| `ACTION_IRREVERSIBILITY_AVERSION` | `false` | ─────────────────────────────────────────────────────────────────────────────── 42. Orchestration & Dispatch ─────────────────────────────────────────────────────────────────────────────── |
| `AU_ACTIVATION_DIAGNOSTIC_MODE` | `false` | ops escape hatch for a non-production worker; NEVER set in production (BUG-001) |
| `FLEET_MCP_URL_TEMPLATE` | `https://{server}.example.test/mcp` |  |
| `JIRA_TICKET_WORKFLOW` | — | Optional graph-workflow ID dispatched after Jira ticket ingestion |
| `PLANE_TICKET_WORKFLOW` | — | Optional graph-workflow ID dispatched after Plane ticket ingestion |
| `CIRCUIT_BREAKER_THRESHOLD` | `3` | ─────────────────────────────────────────────────────────────────────────────── 43. Engine Lifecycle & Circuit Breaker ─────────────────────────────────────────────────────────────────────────────── |
| `ENGINE_BREAKER_COOLDOWN` | `15.0` |  |
| `ENGINE_BREAKER_THRESHOLD` | `5` |  |
| `ENGINE_IDLE_SHUTDOWN_SECS` | `60` |  |
| `ENGINE_LIFECYCLE` | `refcounted` |  |
| `ENGINE_SURFACETOOL` | `true` |  |
| `EPISTEMIC_GRAPH_STARTUP_TIMEOUT_SECS` | `300` |  |
| `EPISTEMIC_GRAPH_MAX_RESIDENT_GRAPHS` | `256` |  |
| `EPISTEMIC_GRAPH_LAZY_OPEN_PAGE_SIZE` | `4096` |  |
| `EPISTEMIC_GRAPH_MAX_NODES_PER_GRAPH` | `250000` |  |
| `EPISTEMIC_GRAPH_MAX_REQUEST_BYTES` | `67108864` |  |
| `EPISTEMIC_GRAPH_MAX_RESPONSE_BYTES` | `67108864` |  |
| `EPISTEMIC_GRAPH_MAX_MSGPACK_ITEMS` | `1000000` |  |
| `EPISTEMIC_GRAPH_CONNECTION_IO_TIMEOUT_SECS` | `120` |  |
| `EPISTEMIC_GRAPH_TLS_HANDSHAKE_TIMEOUT_SECS` | `10` |  |
| `EPISTEMIC_GRAPH_AST_MAX_FILES` | `4096` |  |
| `EPISTEMIC_GRAPH_AST_MAX_SOURCE_BYTES` | `4194304` |  |
| `EPISTEMIC_GRAPH_AST_MAX_TOTAL_BYTES` | `33554432` |  |
| `EPISTEMIC_GRAPH_MODALITY_MAX_BUNDLE_BYTES` | `4194304` |  |
| `EPISTEMIC_GRAPH_MODALITY_MAX_SOURCE_BYTES` | `16777216` |  |
| `EPISTEMIC_GRAPH_SQLITE_TRANSFER_ROOT_REF` | — |  |
| `EPISTEMIC_GRAPH_SQLITE_MAX_BYTES` | `268435456` |  |
| `EPISTEMIC_GRAPH_SQLITE_MAX_ROWS` | `1000000` |  |
| `EPISTEMIC_GRAPH_BACKUP_ROOT_REF` | — |  |
| `AGENT_BUS_LOG_BACKEND` | — | ─────────────────────────────────────────────────────────────────────────────── 44. Agent Bus / Dispatch ─────────────────────────────────────────────────────────────────────────────── |
| `AGENT_BUS_PARTITIONS` | `6` |  |
| `AGENT_CLAIM_BACKEND` | `workitem` | work-item claim backend for engine_claim.py |
| `AGENT_DISPATCH_CLAIM_TTL_S` | `120.0` |  |
| `AGENT_DISPATCH_MAX_DEPTH` | `100000` |  |
| `AGENT_DISPATCH_RENEW_INTERVAL_S` | `30.0` |  |
| `AGENT_EXECUTION_TIMEOUT` | `120.0` |  |
| `AGENT_TURNS_PARTITIONS` | `6` |  |
| `AGENT_USER_TOKEN` | secret-injected |  |
| `AGENT_THINKING_EFFORT` | — | ─────────────────────────────────────────────────────────────────────────────── 45. Agent Runtime ─────────────────────────────────────────────────────────────────────────────── |
| `AGENT_UTILITIES_SKIP_LIVE_MOUNT_CHECK` | — | skip the D-EGK-1 live-mount drift check (agent_utilities/core/live_mount_guard.py) |
| `AGENT_UTILITIES_TESTING` | — |  |
| `KUBERNETES_SERVICE_HOST` | — | injected by Kubernetes itself, never set by hand; read as the in-pod signal for the D-EGK-1 live-mount drift check |
| `DURABLE_EXECUTION_DB` | — | durable-execution store override (orchestration/durable_execution.py) |
| `STATE_DB_POOL_SIZE` | `8` |  |
| `STATE_DB_URI` | — |  |
| `TASK_QUEUE_BACKEND` | — |  |
| `USAGE_DB_BACKEND` | `sqlite` |  |
| `USAGE_DB_PATH` | — |  |
| `USAGE_DB_URI` | — |  |
| `USAGE_DUCKDB_PATH` | — |  |
| `USAGE_GATEWAY_URL` | — |  |
| `USAGE_TENANT_ID` | — |  |
| `USAGE_TRACKING_ENABLED` | `true` |  |
| `GATEWAY_METRICS` | `false` | ─────────────────────────────────────────────────────────────────────────────── 47. Gateway Rate Limiting ─────────────────────────────────────────────────────────────────────────────── |
| `GATEWAY_RATE_BURST` | `0.0` |  |
| `GATEWAY_RATE_LIMIT` | `0.0` | Explicit rate; remote REST falls back to 50 req/s |
| `GATEWAY_WORKERS` | `1` |  |
| `ENABLE_PROGRESSIVE_SYNTHESIS` | `true` | ─────────────────────────────────────────────────────────────────────────────── 48. Parallelism & Synthesis Tuning ─────────────────────────────────────────────────────────────────────────────── |
| `MAX_PARALLEL_AGENTS` | `60` |  |
| `PARALLEL_BATCH_SIZE` | `25` |  |
| `PLACEMENT_CATALOG_TTL_S` | `5.0` |  |
| `SYNTHESIS_RATIO` | `10` |  |
| `SYNTHESIS_STRATEGY` | `auto` |  |
| `WORKER_POOL_SIZE` | `8` |  |
| `AGENT_REQUEST_LIMIT` | — | ─────────────────────────────────────────────────────────────────────────────── 49. Graph Routing & Planning ─────────────────────────────────────────────────────────────────────────────── |
| `AGENT_UTILITIES_GWT_STRICT` | — |  |
| `ARPO_BRANCH_ENTROPY` | `0.6` |  |
| `ARPO_MAX_BRANCHES` | `3` |  |
| `GRAPH_DIRECT_DISPATCH` | `true` |  |
| `PLANNER_REQUEST_LIMIT` | `6` |  |
| `VERIFIER_REQUEST_LIMIT` | `4` |  |
| `EPISTEMIC_GRAPH_KVCACHE_ADDR` | — | ─────────────────────────────────────────────────────────────────────────────── 50. KV-Cache Layering ─────────────────────────────────────────────────────────────────────────────── |
| `EPISTEMIC_GRAPH_KVCACHE_MAX_CONNECTIONS` | `32` |  |
| `EPISTEMIC_GRAPH_KVCACHE_TIMEOUT_S` | `2.0` |  |
| `EPISTEMIC_GRAPH_KVCACHE_TLS_PROFILE` | — |  |
| `EPISTEMIC_GRAPH_KVCACHE_TLS_PROFILE_REF` | — |  |
| `EPISTEMIC_GRAPH_KVCACHE_TOKEN` | secret-injected |  |
| `EPISTEMIC_GRAPH_KVCACHE_URL` | — |  |
| `KV_CACHE_CHARS_PER_TOKEN` | secret-injected |  |
| `KV_CACHE_LAYERING` | `true` |  |
| `KV_CACHE_MIN_CONTEXT_TOKENS` | `2048` |  |
| `KV_CACHE_MIN_HISTORY_TURNS` | `1` |  |
| `KV_CACHE_MIN_PREFIX_TOKENS` | `1024` |  |
| `AU_PROMPT_CACHE` | `true` | Provider-native prompt cache (CONCEPT:AU-ORCH.optimization.provider-prompt-cache) and the opt-in semantic response cache (CONCEPT:AU-KG.memory.semantic-response-cache). |
| `AU_SEMANTIC_CACHE` | `false` |  |
| `MODEL_CONTEXT_COMPILER_CACHE_ENABLED` | `false` | Compiled model-context bundle cache (CONCEPT:AU-KG.retrieval.context-compiler-kv-seam, agent_utilities/core/contextual_model.py). The cache itself is fully wired and tenant/ACL-scoped, but it is DELIBERATELY OFF BY DEFAULT and must stay that way until D-DPF-2 closes: its bundle key is scoped by evidence-ID SET, not node content, so a caller that mutates a node in place while reusing the same id observes a stale hit. Flipping the default to true reproduced exactly that as a real cross-test collision in tests/retrieval/test_context_compiler_delegated_run_default.py. Enabling it is a reviewed, per-deployment decision that asserts no caller in that deployment recycles evidence ids across differing content — not a safe general default. MAXSIZE/TTL_S apply only when ENABLED is true; an explicit set_context_compiler_cache() override always wins over all three. |
| `MODEL_CONTEXT_COMPILER_CACHE_MAXSIZE` | `512` |  |
| `MODEL_CONTEXT_COMPILER_CACHE_TTL_S` | `300.0` |  |
| `ENABLE_RLM` | `false` | ─────────────────────────────────────────────────────────────────────────────── 51. RLM Sandbox ─────────────────────────────────────────────────────────────────────────────── |
| `FORKD_SNAPSHOT_TAG` | `pyagent` |  |
| `FORKD_TOKEN` | secret-injected |  |
| `FORKD_URL` | `http://127.0.0.1:8889` |  |
| `RLM_SANDBOX` | `auto` |  |
| `RLM_CONTAINER_IMAGE_REF` | — |  |
| `RLM_CONTAINER_MEMORY` | `512m` |  |
| `RLM_CONTAINER_CPUS` | `1.0` |  |
| `RLM_CONTAINER_PIDS_LIMIT` | `256` |  |
| `RLM_CONTAINER_TIMEOUT_SECONDS` | `120` |  |
| `RLM_WASM_PYTHON` | — |  |
| `COMFYUI_URL` | `https://image-workflow.example.test` | ─────────────────────────────────────────────────────────────────────────────── 52. Media Generation / Transcription Gateways ─────────────────────────────────────────────────────────────────────────────── |
| `FASTER_WHISPER_URL` | `https://transcription.example.test` |  |
| `FLUX_URL` | — |  |
| `HUNYUAN_IMAGE_URL` | — |  |
| `HUNYUAN_URL` | — |  |
| `LTX_URL` | — |  |
| `OPENAI_TTS_URL` | `https://speech.example.test` |  |
| `QWEN_IMAGE_URL` | — |  |
| `SD35_URL` | — |  |
| `WHISPER_URL` | `https://transcription.example.test` |  |
| `XTTS_URL` | `https://speech.example.test` |  |
| `ARCHI_MODEL_PATH` | — | ─────────────────────────────────────────────────────────────────────────────── 53. Ecosystem Integrations ─────────────────────────────────────────────────────────────────────────────── |
| `ARD_FEDERATION_MODE` | `none` |  |
| `ARD_PUBLISHER_DOMAIN` | — |  |
| `ARD_SPEC_VERSION` | `draft-0` |  |
| `HERMES_HOME` | — |  |
| `LEANIX_API_TOKEN` | secret-injected |  |
| `LEANIX_TOKEN` | secret-injected |  |
| `LEANIX_URL` | — |  |
| `BINANCE_API_KEY` | secret-injected | ─────────────────────────────────────────────────────────────────────────────── 54. Finance / Crypto Connectors ─────────────────────────────────────────────────────────────────────────────── |
| `BINANCE_SECRET` | secret-injected |  |
| `BINANCE_SECRET_KEY` | secret-injected |  |
| `DERIVATIVES_API_KEY` | secret-injected |  |
| `ETHERSCAN_API_KEY` | secret-injected |  |
| `ENABLE_KG_EXTERNAL_GRAPHS` | `true` | ─────────────────────────────────────────────────────────────────────────────── 55. Schema Pack ─────────────────────────────────────────────────────────────────────────────── |
| `ENABLE_KG_KB` | `true` |  |
| `ENABLE_KG_ONTOLOGY_BOOTSTRAP` | `false` |  |
| `ENABLE_KG_OWL` | `true` |  |
| `ENABLE_KG_SHACL_GATE` | `true` |  |
| `ENABLE_KG_WORKSPACE_SYNC` | `true` |  |
| `GRAPH_SCHEMA_AUDIT_DIR` | — |  |
| `GRAPH_SCHEMA_AUDIT_VERBOSE` | — |  |
| `GRAPH_SCHEMA_PACK` | `core` |  |
| `DOMAIN_PACKS_ROOT` | — |  |
| `INGESTION_CONFIDENCE_THRESHOLDS` | — |  |
| `AGENT_UTILITIES_CACHE_DIR` | — | ─────────────────────────────────────────────────────────────────────────────── 56. Core Runtime Tuning ─────────────────────────────────────────────────────────────────────────────── |
| `AGENT_UTILITIES_DATA_DIR` | — |  |
| `AGENT_UTILITIES_HOST_INVENTORY` | — | JSON host identities -> abstract admission roles; unset/malformed fails closed |
| `AGENT_UTILITIES_LOG_DIR` | — |  |
| `AGENT_UTILITIES_MEMORY_DIR` | — |  |
| `AGENT_UTILITIES_PROMPTS_DIR` | — |  |
| `AGENT_UTILITIES_SKILLS_DIR` | — |  |
| `CONTINUOUS_STARDOG_MIRROR` | `false` |  |
| `ENABLE_KG_REGISTRY_FETCH` | `true` |  |
| `ENABLE_SDD_WATCHER` | `true` |  |
| `EVOLUTION_WORKTREE_ROOT` | — |  |
| `GPU_CONCURRENCY_BUDGETS` | `{}` |  |
| `GPU_RESERVED_ROLES` | — |  |
| `MODEL_AUTOSCALE_UPDATE_INTERVAL_S` | — |  |
| `MODEL_AUTOSCALE_UPDATE_SAMPLES` | — |  |
| `MODEL_AUTOSCALE_VLLM_METRICS` | `true` |  |
| `MODEL_AUTOSCALE_WINDOW` | — |  |
| `MODEL_BREAKER_BACKOFF_FACTOR` | — |  |
| `MODEL_BREAKER_BASE_COOLDOWN_S` | — |  |
| `MODEL_BREAKER_FAIL_THRESHOLD` | — |  |
| `MODEL_BREAKER_MAX_COOLDOWN_S` | — |  |
| `MODEL_CIRCUIT_BREAKER` | `true` |  |
| `MODEL_LATENCY_GRADIENT_TARGET` | — |  |
| `MODEL_MAX_CONCURRENCY` | — |  |
| `MODEL_MAX_CONCURRENT_REQUESTS` | — |  |
| `MODEL_ROLE_ROUTING` | — |  |
| `POSTGRES_DSN` | — |  |
| `PRICING_CATALOG_PATH` | — | Versioned operator-owned JSON pricing catalog |
| `PRICING_LITELLM_URL` | — |  |
| `REDIS_CONNECTION_PROFILE_REF` | — |  |
| `REDIS_TLS_PROFILE` | — |  |
| `REDIS_TLS_PROFILE_REF` | — |  |
| `RESOURCE_WEIGHT_COST` | `0.4` |  |
| `RESOURCE_WEIGHT_LATENCY` | `0.3` |  |
| `RESOURCE_WEIGHT_QUALITY` | `0.3` |  |
| `SESSION_COST_BUDGET_USD` | `5.0` |  |
| `SESSION_LATENCY_BUDGET_MS` | `30000` |  |
| `SESSION_TOKEN_BUDGET` | secret-injected |  |
| `TELEGRAM_BOT_TOKEN` | secret-injected |  |
| `AGENT_INVENTORY_YAML` | — | ─────────────────────────────────────────────────────────────────────────────── 57. SDD Watcher ─────────────────────────────────────────────────────────────────────────────── |
| `AGENT_MCP_CONFIG_JSON` | — |  |
| `AGENT_RESEARCH_DIR` | — |  |
| `SCHOLARX_PAPERS_DIR` | — |  |
| `GRAPHOS_BASE_URL` | `http://127.0.0.1:8000` | ─────────────────────────────────────────────────────────────────────────────── 58. Harness / MemoryData ─────────────────────────────────────────────────────────────────────────────── |
| `GRAPHOS_TOKEN` | secret-injected |  |
| `AGENT_UTILITIES_RELEASE_CHANNEL` | `stable` | ─────────────────────────────────────────────────────────────────────────────── 59. Deployment / Doctor ─────────────────────────────────────────────────────────────────────────────── Highest-priority source for the active release channel (agent_utilities/core/release_channel.py): env -> `release_channel` in the loaded config -> `stable`. One of stable\|beta\|canary\|edge. |
| `APP_PROFILE` | — |  |
| `EXTERNAL_GRAPH_CONNECTORS` | `[]` | Reference-only external source declarations. For GraphQL/`graphql_document`, mapping_policy_ref is required; endpoint, auth headers, queries, mappings, variables, and TLS material stay in their respective runtime secret documents. |
| `TLS_PROFILE` | — | ─────────────────────────────────────────────────────────────────────────────── 60. Shared TLS and Authorization Profiles ─────────────────────────────────────────────────────────────────────────────── All certificate/key/proxy material is supplied by secret reference. Named profiles are resolved from AgentConfig/XDG configuration; no endpoint-specific trust bypasses are supported. |
| `TLS_PROFILE_REF` | — |  |
| `TLS_PROFILES_REF` | — |  |
| `TLS_CA_BUNDLE_REF` | — |  |
| `TLS_CLIENT_CERT_REF` | — |  |
| `TLS_CLIENT_KEY_REF` | — |  |
| `TLS_CLIENT_KEY_PASSWORD_REF` | — |  |
| `TLS_PROXY_URL_REF` | — |  |
| `TLS_SYSTEM_TRUST` | `true` |  |
| `TLS_TRUST_ENV` | `true` |  |
| `AUTH_JWT_ALGORITHMS` | `["RS256","ES256","EdDSA"]` |  |
| `IDENTITY_GROUP_CAPABILITY_MAP` | `{}` |  |
| `OAUTH_UPSTREAM_CLIENT_SECRET_REF` | — |  |
| `KEYCLOAK_CLIENT_SECRET_REF` | — |  |
| `LANGFUSE_TLS_PROFILE` | — | [OPTIONAL] Langfuse-specific profile layered over shared TLS policy. |
| `LANGFUSE_CA_BUNDLE_REF` | — |  |
| `LANGFUSE_PROXY_URL_REF` | — |  |
| `EUNOMIA_API_KEY_REF` | — | [OPTIONAL] Remote policy authorization transport. |
| `EUNOMIA_TLS_PROFILE` | — |  |
| `EUNOMIA_TLS_PROFILE_REF` | — |  |
| `EUNOMIA_ALLOWED_PRIVATE_HOSTS` | `[]` |  |
| `EUNOMIA_TIMEOUT_SECONDS` | `10.0` |  |
| `EUNOMIA_MAX_RESPONSE_BYTES` | `1048576` |  |
| `EUNOMIA_BULK_CHECK_MAX` | `100` |  |
| `BPM_TLS_PROFILE` | — | [OPTIONAL] Per-connector TLS profile overrides (CONCEPT:AU-OS.config.dynamic-env-family). Each resolves through resolve_configured_tls_profile(service=...); when unset, resolution falls back to the shared TLS_PROFILE/TLS_PROFILE_REF above. |
| `BPM_TLS_PROFILE_REF` | — |  |
| `CERTIFICATION_PROMETHEUS_TLS_PROFILE` | — |  |
| `CERTIFICATION_PROMETHEUS_TLS_PROFILE_REF` | — |  |
| `EXTERNAL_GRAPH_TLS_PROFILE` | — |  |
| `EXTERNAL_GRAPH_TLS_PROFILE_REF` | — |  |
| `FUSEKI_TLS_PROFILE` | — |  |
| `FUSEKI_TLS_PROFILE_REF` | — |  |
| `GITLAB_TLS_PROFILE` | — |  |
| `GITLAB_TLS_PROFILE_REF` | — |  |
| `GLOBAL_TLS_PROFILE` | — | Doctor's whole-deployment TLS check; distinct from the bare TLS_PROFILE default above |
| `GLOBAL_TLS_PROFILE_REF` | — |  |
| `GRAPHQL_DOCUMENT_TLS_PROFILE` | — |  |
| `GRAPHQL_DOCUMENT_TLS_PROFILE_REF` | — |  |
| `GRAPH_OS_TLS_PROFILE` | — |  |
| `GRAPH_OS_TLS_PROFILE_REF` | — |  |
| `LEANIX_TLS_PROFILE` | — |  |
| `LEANIX_TLS_PROFILE_REF` | — |  |
| `MCP_TLS_PROFILE` | — |  |
| `MCP_TLS_PROFILE_REF` | — |  |
| `MCP_CHILD_TLS_PROFILE` | — |  |
| `MCP_CHILD_TLS_PROFILE_REF` | — |  |
| `MESSAGING_MEDIA_TLS_PROFILE` | — |  |
| `MESSAGING_MEDIA_TLS_PROFILE_REF` | — |  |
| `NEO4J_TLS_PROFILE` | — |  |
| `NEO4J_TLS_PROFILE_REF` | — |  |
| `NEXTCLOUD_TLS_PROFILE` | — |  |
| `NEXTCLOUD_TLS_PROFILE_REF` | — |  |
| `OBSERVABILITY_SELF_INGEST_TLS_PROFILE` | — |  |
| `OBSERVABILITY_SELF_INGEST_TLS_PROFILE_REF` | — |  |
| `ONTOLOGY_TLS_PROFILE` | — |  |
| `ONTOLOGY_TLS_PROFILE_REF` | — |  |
| `SYNOLOGY_CHAT_TLS_PROFILE` | — |  |
| `SYNOLOGY_CHAT_TLS_PROFILE_REF` | — |  |
| `WHATSAPP_BUSINESS_TLS_PROFILE` | — |  |
| `WHATSAPP_BUSINESS_TLS_PROFILE_REF` | — |  |
| `MEASUREMENT_LOAD_THRESHOLD` | — | [OPTIONAL] Load-average threshold above which the measurement harness (agent_utilities/measurement/load_gate.py) refuses to emit a pass/fail verdict at all, returning TOO_LOADED_TO_MEASURE instead. Default is 1.5x cpu_count(); override with an absolute load-average number when a host's normal operating load differs. |
| `MCP_ALLOWED_HOSTS` | — | ─────────────────────────────────────────────────────────────────────────────── 61. MCP and Runtime Server Boundaries ─────────────────────────────────────────────────────────────────────────────── Exact hosts/origins and trusted proxy networks are required for exposed listeners. Certificate paths are runtime-mounted inputs, never tracked values. |
| `MCP_ALLOWED_ORIGINS` | — |  |
| `MCP_HTTP_ALLOWED_PRIVATE_HOSTS` | `[]` |  |
| `MCP_TLS_CERTFILE` | — |  |
| `MCP_TLS_KEYFILE` | — |  |
| `MCP_TLS_TERMINATED` | `false` |  |
| `MCP_STDIO_PROHIBITED` | `false` | Prohibit spawning stdio-transport MCP children in THIS process (a stdio child is a subprocess spawned inside the calling process itself). Default permissive; set true for a fan-out deployment that must never spawn a child process in-pod (e.g. agent-webui). See enforce_mcp_stdio_permitted() in agent_utilities/core/config.py. |
| `MCP_TRUSTED_PROXY_CIDRS` | — |  |
| `MCP_MAX_REQUEST_BYTES` | `4194304` |  |
| `MCP_MAX_CONNECTIONS` | `128` |  |
| `MCP_LISTEN_BACKLOG` | `256` |  |
| `MCP_METRICS_TOKEN_REF` | — |  |
| `SERVER_TLS_CERTFILE` | — |  |
| `SERVER_TLS_KEYFILE` | — |  |
| `SERVER_TLS_TERMINATED` | `false` |  |
| `SERVER_TRUSTED_PROXY_CIDRS` | `[]` |  |
| `SERVER_MAX_CONNECTIONS` | `256` |  |
| `DATABASE_TYPE` | — | ─────────────────────────────────────────────────────────────────────────────── 62. Database and External-System Connection Inputs ─────────────────────────────────────────────────────────────────────────────── Credentials and complete connection profiles are secret references. Hosts and database names remain unset until supplied by AgentConfig at deployment time. |
| `DB_HOST` | — |  |
| `DB_PORT` | — |  |
| `DBNAME` | — |  |
| `DB_USERNAME_REF` | — |  |
| `DB_PASSWORD_REF` | — |  |
| `DOCUMENT_DIRECTORY` | — |  |
| `SVD_URL` | — |  |
| `POSTGRES_TLS_PROFILE` | — |  |
| `POSTGRES_TLS_PROFILE_REF` | — |  |
| `POSTGRES_REQUEST_TIMEOUT` | `30` |  |
| `POSTGRES_MAX_POOL_SIZE` | `20` |  |
| `QDRANT_API_KEY_REF` | — |  |
| `QDRANT_TLS_PROFILE` | — |  |
| `QDRANT_TLS_PROFILE_REF` | — |  |
| `QDRANT_HTTP_ALLOWED_PRIVATE_HOSTS` | `[]` |  |
| `QDRANT_REQUEST_TIMEOUT` | `30` |  |
| `MONGODB_URI_REF` | — |  |
| `MONGODB_TLS_PROFILE` | — |  |
| `MONGODB_TLS_PROFILE_REF` | — |  |
| `MONGODB_REQUEST_TIMEOUT_MS` | `30000` |  |
| `MONGODB_MAX_POOL_SIZE` | `20` |  |
| `STARDOG_PASSWORD_REF` | — |  |
| `SYNOLOGY_CHAT_WEBHOOK_URL_REF` | — |  |
| `KG_AUTH_TOKEN_REF` | — | ─────────────────────────────────────────────────────────────────────────────── 63. Identity, Signing, and Governed Runtime Inputs ─────────────────────────────────────────────────────────────────────────────── |
| `KG_IDENTITY_OAUTH2` | `{}` |  |
| `KG_ADMIN_BROKER_OAUTH2` | `{}` |  |
| `BUS_IDENTITY_HMAC_KEY_REF` | — |  |
| `ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF` | — |  |
| `ONTOLOGY_RELEASE_TRUSTED_PUBLIC_KEYS` | `[]` |  |
| `VAULT_K8S_SA_TOKEN_PATH` | secret-injected |  |
| `INCIDENT_ACTUATION_ENABLED` | `false` |  |
| `KG_EPISTEMIC_LIGHT_DEFAULT` | `true` |  |
| `KG_LOOP_ALLOW_HOST_VALIDATION` | `false` |  |
| `KG_LOOP_HOST_VALIDATION_EXECUTABLES` | `pytest,ruff,mypy,pyright,nox,tox,cargo,go` |  |
| `GRAPH_RAFT_GROUP_ENDPOINTS` | `{}` | ─────────────────────────────────────────────────────────────────────────────── 64. Distributed Runtime and Capacity Controls ─────────────────────────────────────────────────────────────────────────────── |
| `AGENT_BUS_MAX_CONSUMERS` | `32` |  |
| `AGENT_BUS_MAX_DEPTH` | `100000` |  |
| `AGENT_BUS_MAX_TOPIC_SUBSCRIBERS` | `1024` |  |
| `AGENT_BUS_DELIVERY_LEASE_SECONDS` | `300` |  |
| `FLEET_ACTUATOR_K8S_NAMESPACE` | `platform` | namespace the fleet actuator operates in |
| `FLEET_REPLICA_COST_USD_PER_HOUR` | `0.05` |  |
| `FLEET_SCALE_BUDGET_USD_PER_HOUR` | — |  |
| `RUNTIME_WORKSPACE_IMAGES` | `[]` |  |
| `RUNTIME_WORKSPACE_NETWORK` | `none` |  |
| `RUNTIME_MAX_SESSIONS` | `16` |  |
| `RUNTIME_SESSION_TTL_SECONDS` | `3600` |  |
| `RUNTIME_MAX_EVENTS` | `1000` |  |
| `RLM_AUTO_TRIGGER` | `false` |  |
| `AUDITTOOL` | `true` | ─────────────────────────────────────────────────────────────────────────────── 65. Condensed Tool-Surface Toggles ─────────────────────────────────────────────────────────────────────────────── |
| `COMPLIANCETOOL` | `true` |  |
| `EPISTEMICTOOL` | `true` |  |
| `INCIDENTTOOL` | `true` |  |
| `AGENT_PACKAGES_ROOT` | — | ─────────────────────────────────────────────────────────────────────────────── 66. Maintainer / CI Scripts (repo-internal tooling) ─────────────────────────────────────────────────────────────────────────────── |
| `AGENT_UTILITIES_TOKEN_SECRET` | secret-injected |  |
| `SKILL_HARNESS_CALL_TIMEOUT` | `10` | fleet_harness functional-checks per-call timeout (s) |
| `SKILL_HARNESS_CONNECT_TIMEOUT` | `8` | fleet_harness functional-checks connect timeout (s) |
| `SKILL_HARNESS_GRAPH_OS_URL` | — | How fleet_harness functional checks reach graph-os. Deployment-varying with no universal default: an explicit SSE/HTTP URL wins; otherwise a stdio command is spawned (default `graph-os` from PATH). Unset both and the functional layer reports SKIPPED-unreachable rather than a false PASS. |
| `SKILL_HARNESS_GRAPH_OS_COMMAND` | `graph-os` |  |
| `CERTIFICATION_MODE` | `disabled` | disabled \| production |
| `CERT_RELEASE_MANIFEST` | — |  |
| `CERT_ARTIFACTS_DIR` | — |  |
| `CERT_HARDWARE_CLASS` | — |  |
| `CERT_LOAD_COMMAND` | `[]` | bounded argv (JSON array), absolute executable |
| `CERT_METRICS_COMMAND` | `[]` |  |
| `CERT_HOOK_COMMANDS` | `{}` | JSON {hook_name: argv} |
| `CERT_FAULT_ACTION_COMMANDS` | `{}` | JSON {fault_name: argv} |
| `CERT_FAULT_PROBE_COMMANDS` | `{}` | JSON {fault_name: argv} |
| `CERT_EVIDENCE_SIGNER_COMMAND` | `[]` |  |
| `CERT_EVIDENCE_VERIFIER_COMMAND` | `[]` |  |
| `CERT_PROMETHEUS_URL` | — |  |
| `CERT_PROMETHEUS_BEARER_TOKEN_REF` | — |  |
| `CERT_PROMETHEUS_TLS_PROFILE` | — |  |
| `CERT_PROMETHEUS_TLS_PROFILE_REF` | — |  |
| `SKILL_CERT_RUNTIME_CONFIGURATION` | — | Exact skill certification deployment references (all absolute paths). |
| `SKILL_CERT_RUNTIME_PROFILE` | — |  |
| `SKILL_CERT_RELEASE_SPEC` | — |  |
| `SKILL_CERT_PROMOTION_EVIDENCE` | — |  |
| `SKILL_CERT_GRAPHOS_ENDPOINT` | — | loopback http(s) only, e.g. http://127.0.0.1:8100 |
| `SKILL_CERT_GRAPHOS_COMMAND` | `[]` |  |
| `SKILL_VALIDATION_EVIDENCE_SIGNER_COMMAND` | `[]` |  |
| `SKILL_VALIDATION_EVIDENCE_VERIFIER_COMMAND` | `[]` |  |
| `SKILL_CERT_IDENTITY_AUTHORITY_MODE` | `ephemeral-https-loopback` |  |
| `SKILL_CERT_IDENTITY_TOKEN_TTL_SECONDS` | secret-injected | 180-3600 |
| `DATA_RESIDENCY_REGION` | — | e.g. eu-west, us-east |
| `KG_DAEMON_METRICS` | `true` | serve the daemon's /metrics endpoint |
| `KG_DAEMON_METRICS_HOST` | `0.0.0.0` |  |
| `KG_DAEMON_METRICS_PORT` | `9110` |  |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | — | ── OpenTelemetry / Prometheus export (agent_utilities/observability) ──────── Per-signal OTLP endpoint overrides; unset falls back to the generic OTEL_EXPORTER_OTLP_ENDPOINT above. Empty = no override. |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | — |  |
| `PROMETHEUS_MULTIPROC_DIR` | — | Multiprocess Prometheus collector directory (gunicorn/uvicorn workers). Empty = single-process registry. |
| `DISTILLATION_PROMOTION_THRESHOLD` | `3` | runs a distilled prompt must win before promotion |
| `DISTILLATION_QUALITY_SCORE_MINIMUM` | `0.6` | minimum quality score to be promotable |
| `AGENT_UTILITIES_MEMORY_DISTILL` | `false` | opt in to recurring-memory procedural distillation |
| `AGENT_UTILITIES_MEMORY_DISTILL_WORKING_SET` | `500` |  |
| `AGENT_UTILITIES_MEMORY_DISTILL_MIN_RECURRENCE` | `3` |  |
| `AGENT_UTILITIES_MEMORY_DISTILL_MAX_CLUSTER` | `32` |  |
| `AGENT_UTILITIES_MEMORY_DISTILL_MAX_CLUSTERS` | `4` |  |
| `AGENT_UTILITIES_MEMORY_DISTILL_FLYWHEEL` | `true` |  |
| `AGENT_UTILITIES_MEMORY_DISTILL_SEED_REWARD` | `0.5` |  |
| `AGENT_UTILITIES_DISTILL_BASE_MODEL` | — | optional default base-model ID for memory-to-weights runs |
| `AGENT_UTILITIES_DISTILL_MAX_EXAMPLES` | `512` | maximum exported training examples |
| `AGENT_UTILITIES_MEMORY_LIFECYCLE` | `false` | opt in to summary, consolidation, decay, and eviction maintenance |
| `AGENT_UTILITIES_MEMORY_LIFECYCLE_WORKING_SET` | `500` |  |
| `AGENT_UTILITIES_MEMORY_LIFECYCLE_MIN_CLUSTER` | `3` |  |
| `AGENT_UTILITIES_MEMORY_LIFECYCLE_MIN_AGE_HOURS` | `6.0` |  |
| `AGENT_UTILITIES_MEMORY_LIFECYCLE_MAX_CLUSTER` | `32` |  |
| `AGENT_UTILITIES_MEMORY_LIFECYCLE_HALF_LIFE_SECS` | `604800.0` |  |
| `AGENT_UTILITIES_MEMORY_LIFECYCLE_DECAY_FLOOR` | `0.05` |  |
| `AGENT_UTILITIES_MEMORY_LIFECYCLE_EVICT_MAX` | `0` | 0 delegates the eviction cap to the engine |
| `MCP_SDK_FLOOR_ENFORCE` | `error` | error \| warn |
| `GRAPH_CLUSTER_ID` | — | no default; unset means "accept any cluster" |
| `GRAPH_CLUSTER_DISCOVERY_MAX_AGE_S` | `30.0` | reject a topology snapshot older than this |
| `GRAPH_CLUSTER_DISCOVERY_CLOCK_SKEW_S` | `5.0` | tolerated peer clock skew before rejection |
| `GRAPH_DRAIN_TIMEOUT_S` | `15.0` | cooperative drain budget before a shrink is abandoned |
| `EPISTEMIC_GRAPH_KVCACHE_TENANT` | — | verified tenant binding |
| `EPISTEMIC_GRAPH_KVCACHE_PRINCIPAL` | — | verified worker principal |
| `EPISTEMIC_GRAPH_KVCACHE_REQUIRE_AUTH` | — | true ⇒ refuse unauthenticated/unscoped use |

#### Inherited agent-utilities variables (apply to every connector)

| Variable | Example | Description |
|----------|---------|-------------|
| `TRANSPORT` | `stdio` | MCP transport: `stdio` \| `streamable-http` \| `sse` |
| `MCP_ENABLED_TOOLS` | — | Comma-separated tool allow-list |
| `MCP_DISABLED_TOOLS` | — | Comma-separated tool deny-list |
| `MCP_ENABLED_TAGS` | — | Comma-separated tag allow-list |
| `MCP_DISABLED_TAGS` | — | Comma-separated tag deny-list |
| `EUNOMIA_TYPE` | `none` | Authorization mode: `none` \| `embedded` \| `remote` |
| `EUNOMIA_POLICY_FILE` | `mcp_policies.json` | Embedded Eunomia policy file |
| `EUNOMIA_REMOTE_URL` | — | Remote Eunomia authorization server URL |
| `MCP_CLIENT_AUTH` | — | Outbound MCP child auth: `oidc-client-credentials` \| `basic` \| `none` |
| `MCP_BASIC_AUTH_USERNAME` | — | HTTP Basic username (`MCP_CLIENT_AUTH=basic`) |
| `PYTHONUNBUFFERED` | `1` | Unbuffered stdout (recommended in containers) |
| `PROVIDER` | — | Operator-configured LLM provider for the agent |
| `MODEL_ID` | — | Operator-configured model id for the agent |

_1080 package + 13 inherited variable(s). Auto-generated from `.env.example` + the shared agent-utilities set — do not edit._
<!-- ENV-VARS-TABLE:END -->
