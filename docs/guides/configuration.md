# Configuration Reference

This document provides a unified reference for all environment variables, configuration files, and CLI flags used across the `agent-utilities` ecosystem.

## Environment Variables

### LLM Configuration
All LLM configuration (models, API keys, endpoints) is now managed centrally via the XDG configuration file at `~/.config/agent-utilities/config.json`.

Environment variables for `LLM_BASE_URL`, `LLM_MODEL_ID`, etc., are **deprecated** and will be ignored. API keys can optionally be provided in the `.env` file or directly inside the `config.json` model entries.

### Graph Database

The **epistemic-graph engine is the one database — the authority**. It serves all
reads and is where every write commits first. `GRAPH_BACKEND=fanout` additionally
fans committed writes out asynchronously and losslessly to the **mirrors** named
in `GRAPH_MIRROR_TARGETS` (Postgres/pg-age, Neo4j, FalkorDB, Ladybug) for
interop/BI/DR — mirrors are never the authority and never on the read path.

| Variable | Default | Description |
|---|---|---|
| `GRAPH_BACKEND` | `epistemic_graph` | `epistemic_graph` (the engine only — self-contained, zero-infra), `fanout` (engine + mirrors), or `memory` (ephemeral, tests/CI) |
| `GRAPH_MIRROR_TARGETS` | *None* | Comma-separated mirrors to fan out to under `fanout`: `postgresql`, `neo4j`, `falkordb`, `ladybug` |
| `GRAPH_DB_PATH` | `knowledge_graph.db` | File path for the engine store (or a LadybugDB mirror) |
| `GRAPH_DB_HOST` | `localhost` | Host for a Neo4j/FalkorDB mirror |
| `GRAPH_DB_PORT` | `7687` | Port for a Neo4j/FalkorDB mirror |
| `GRAPH_DB_URI` | *None* | Direct connection URI for a PostgreSQL/Neo4j mirror (overrides Host/Port) |
| `GRAPH_DB_USER` | `neo4j` | Username for a remote mirror (Neo4j/PostgreSQL) |
| `GRAPH_DB_PASSWORD` | *None* | Password for a remote mirror (Neo4j/PostgreSQL) |
| `GRAPH_DB_NAME` | `agent_graph` | Database/graph name for a FalkorDB/PostgreSQL mirror |
| `GRAPH_POOL_MIN` | `2` | Minimum PostgreSQL mirror connection pool size |
| `GRAPH_POOL_MAX` | `10` | Maximum PostgreSQL mirror connection pool size |
| `GRAPH_PGGRAPH_SCHEMA` | `public` | Schema for pg-age mirror table registration |

### OWL Reasoning
| Variable | Default | Description |
|---|---|---|
| `OWL_BACKEND` | `stardog` | Reasoning backend (`stardog`, `hermit`) |
| `OWL_DB_PATH` | `ontology.owl` | Path to local OWL ontology file |
| `STARDOG_ENDPOINT` | `http://localhost:5820` | Stardog server URL |
| `STARDOG_USERNAME` | `admin` | Stardog authentication user |
| `STARDOG_PASSWORD` | `admin` | Stardog authentication password |

### Document Storage
| Variable | Default | Description |
|---|---|---|
| `DOC_BACKEND` | `sqlite` | Backend for document pipeline (`sqlite`, `mongodb`, `postgresql`) |
| `DOC_DB_URI` | *None* | Connection string for MongoDB/Postgres document stores |

### Secrets & Auth (CONCEPT:OS-5.1)
| Variable | Default | Description |
|---|---|---|
| `SECRETS_BACKEND` | `inmemory` | Storage for secrets (`inmemory`, `sqlite`, `vault`). See [secrets-auth.md](../5_agent_os_infrastructure/secrets-auth.md) |
| `SECRETS_SQLITE_PATH` | `~/.agent-utilities/secrets.db` | Path for SQLite secrets DB |
| `SECRETS_VAULT_URL` | *None* | URL for HashiCorp Vault & OpenBao |
| `SECRETS_VAULT_MOUNT` | `secret` | Vault/OpenBao KV v2 mount point |
| `ENABLE_API_AUTH` | `False` | Enable JWT validation on server endpoints |
| `AUTH_JWT_JWKS_URI` | *None* | URI to fetch JSON Web Key Sets |
| `AUTH_JWT_ISSUER` | *None* | Expected JWT issuer |
| `AUTH_JWT_AUDIENCE` | *None* | Expected JWT audience |
| `AGENT_API_KEY` | *None* | Static API key for basic auth |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOWED_HOSTS` | `*` | Comma-separated trusted hosts |

### Graph Execution
| Variable | Default | Description |
|---|---|---|
| `GRAPH_DIRECT_EXECUTION`| `True` | Direct graph dispatch in AG-UI/ACP (bypasses LLM tool-call hop) |
| `VALIDATION_MODE` | `False` | Disables real LLM calls for unit testing and CI |
| `WORKSPACE_TOOLS` | `True` | Enable workspace filesystem and grep tools |
| `GIT_TOOLS` | `True` | Enable Git tools |
| `BROWSER_TOOLS` | `True` | Enable browser and web search tools |
| `A2A_TOOLS` | `True` | Enable Agent-to-Agent discovery and messaging |

### RLM & AHE Observability
| Variable | Default | Description |
|---|---|---|
| `ENABLE_RLM` | `True` | Enable Recursive Language Model execution |
| `RLM_MAX_DEPTH` | `3` | Maximum recursion depth for RLM sub-shells |
| `RLM_USE_CONTAINER` | `True` | Run RLM in an isolated container if available |
| `AHE_TRACE_THRESHOLD` | `0.7` | Quality threshold triggering distillation traces |

### Swarm & First Principles
| Variable | Default | Description |
|---|---|---|
| `SWARM_MODE` | `False` | Enable swarm orchestration in dispatcher |
| `SWARM_MAX_DEPTH` | `3` | Maximum recursion depth for sub-swarms |
| `SWARM_MAX_AGENTS` | `10` | Maximum agents per swarm |

### Observability
| Variable | Default | Description |
|---|---|---|
| `OTEL_ENABLE_OTEL` | `False` | Enable OpenTelemetry exports |
| `LANGFUSE_PUBLIC_KEY` | *None* | Langfuse integration key |
| `LANGFUSE_SECRET_KEY` | *None* | Langfuse integration secret |
| `LOGFIRE_TOKEN` | *None* | Pydantic Logfire token |
| `AGENT_UTILITIES_SELF_INGEST` | `False` | **Opt-in** — ship our own logs + RunTrace/ToolCall into the epistemic-graph engine obs store (KG-2.304, dogfooding). Requires `EPISTEMIC_GRAPH_OBS_ADDR`. |
| `EPISTEMIC_GRAPH_OBS_ADDR` | *None* | Engine OTLP/HTTP log endpoint base address (engine EG-160). Empty ⇒ self-ingest disabled. |
| `AGENT_UTILITIES_SELF_INGEST_MODE` | `otlp` | `otlp` → `POST /v1/logs`; `bulk` → `POST /_bulk` |
| `AGENT_UTILITIES_SELF_INGEST_SERVICE` | `agent-utilities` | OTLP `service.name` stamped on records |
| `AGENT_UTILITIES_SELF_INGEST_LEVEL` | `INFO` | Minimum log level shipped by the handler |
| `AGENT_UTILITIES_SELF_INGEST_BATCH` | `100` | Max records per batch |
| `AGENT_UTILITIES_SELF_INGEST_INTERVAL` | `2.0` | Background flush interval (seconds) |
| `AGENT_UTILITIES_SELF_INGEST_QUEUE_MAX` | `10000` | Bounded queue size; overflow is dropped (never blocks) |
| `AGENT_UTILITIES_SELF_INGEST_TIMEOUT` | `3.0` | Per-request HTTP timeout (seconds) |

### MCP Tooling
| Variable | Default | Description |
|---|---|---|
| `MCP_CONFIG` | `mcp_config.json` | Path to the MCP server configuration map |
| `MCP_SEMAPHORE_LIMIT` | `30` | Max parallel subprocesses during tool discovery |
| `TOOL_GUARD_MODE` | `on` | Strictness of the tool execution guard (`on`, `off`, `custom`) |
| `DISABLE_TOOL_GUARD` | `False` | Completely bypass tool elicitation and safety checks |

### A2A Agent Discovery (CONCEPT:ECO-4.0)
| Variable | Default | Description |
|---|---|---|
| `A2A_CONFIG` | *None* | Path to `a2a_config.json` for external A2A agent discovery |
| `A2A_REFRESH_INTERVAL` | `300` | Seconds between periodic `.well-known/agent-card.json` re-fetch |

### CLI Execution
The preferred method for running `agent-utilities` servers is via the standardized `uv` scripts:

| Script | Command | Description |
|---|---|---|
| **KG Server** | `uv run graph-os` | Launches the Knowledge Graph (graph-os) MCP server |
| **Main Server** | `python -m agent_utilities` | Launches the unified protocol server (ACP/A2A/AG-UI) |

## CLI Flags

When running `agent-utilities` commands (or `python -m agent_utilities`), the following standard flags are available:

| Flag | Equivalent Env Var | Description |
|---|---|---|
| `--base-url` | Base URL (Overrides `config.json`) |
| `--api-key` | API Key (Overrides `config.json`) |
| `--port` | *None* | Server listen port (default: 8000) |
| `--host` | *None* | Server bind host (default: 0.0.0.0) |
| `--web` | *None* | Enables the bundled web UI proxy if present |
| `--mcp-config` | `MCP_CONFIG` | Path to MCP config file |
| `--debug` | *None* | Sets log level to DEBUG |
| `--skill-types`| *None* | Comma-separated list of skills to load (`universal`, `graphs`) |


## Configuration & Environment Variables

All LLM and embedding configuration now routes **exclusively** through the `chat_models` and `embedding_models` registries in `config.json`.

### Unified Agent Configuration (`config.json`)

The centralized `config.json` at `~/.config/agent-utilities/config.json` (XDG-compliant) is the **single source of truth** for all configuration.

#### Configuration Precedence Chain

```
config.json registry → AgentConfig defaults
```

Environment variables are no longer part of the LLM configuration chain. API keys can be specified per-model in the registry.

#### Full `config.json` Schema

```json
{
  // ── Agent Identity ──────────────────────────────────────────────
  "default_agent_name": "Agent",
  "agent_description": "AI Agent",
  "agent_system_prompt": null,

  // ── Server ──────────────────────────────────────────────────────
  "host": "0.0.0.0",
  "port": 9000,
  "debug": false,
  "enable_web_ui": false,
  "enable_terminal_ui": false,
  "enable_web_logs": true,
  "enable_acp": false,
  "acp_port": 8001,
  "acp_session_root": ".acp-sessions",
  "mcp_config": null,
  "max_upload_size": 10485760,

  // ── Authentication & Security ───────────────────────────────────
  "agent_api_key": null,
  "enable_api_auth": false,
  "auth_jwt_jwks_uri": null,
  "auth_jwt_issuer": null,
  "auth_jwt_audience": null,
  "allowed_origins": null,
  "allowed_hosts": null,
  "tool_guard_mode": "strict",
  "sensitive_tool_patterns": [".*delete.*", ".*remove.*", "..."],

  // ── Secrets Backend ─────────────────────────────────────────────
  "secrets_backend": "inmemory",
  "secrets_sqlite_path": null,
  "secrets_vault_url": null,
  "secrets_vault_mount": "secret",

  // ── Graph Execution ─────────────────────────────────────────────
  "routing_strategy": "hybrid",
  "graph_persistence_type": "file",
  "graph_persistence_path": "~/.local/share/agent-utilities/graph_state",
  "enable_llm_validation": false,
  "graph_router_timeout": 300.0,
  "graph_verifier_timeout": 300.0,
  "graph_direct_execution": true,
  "min_confidence": 0.4,
  "validation_mode": false,
  "approval_timeout": 0.0,

  // ── Knowledge Graph ─────────────────────────────────────────────
  "enable_kg_embeddings": true,
  "kg_backups": 3,
  "knowledge_graph_sync_background": true,

  // ── Observability (OTEL / Langfuse) ─────────────────────────────
  "enable_otel": true,
  "otel_exporter_otlp_endpoint": "http://langfuse.example.com/api/public/otel",
  "otel_exporter_otlp_headers": null,
  "otel_exporter_otlp_public_key": "lf_pk_...",
  "otel_exporter_otlp_secret_key": "lf_sk_...",
  "otel_exporter_otlp_protocol": "http/protobuf",
  "langfuse_host": "http://langfuse.example.com",
  "langfuse_public_key": "lf_pk_...",
  "langfuse_secret_key": "lf_sk_...",
  "langfuse_dataset_capture_threshold": 0.0,

  // ── A2A Agent Discovery ─────────────────────────────────────────
  "a2a_broker": "in-memory",
  "a2a_broker_url": null,
  "a2a_storage": "in-memory",
  "a2a_storage_url": null,
  "a2a_config": null,
  "a2a_refresh_interval": 300,

  // ── LLM Inference Parameters ────────────────────────────────────
  "max_tokens": 16384,
  "temperature": 0.7,
  "top_p": 1.0,
  "timeout": 32400.0,
  "tool_timeout": 32400.0,
  "parallel_tool_calls": true,
  "seed": null,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "logit_bias": null,
  "stop_sequences": null,
  "extra_headers": null,
  "extra_body": null,

  // ── Cognitive Scheduler & Agent Policies ─────────────────────────
  "cognitive_scheduler_enabled": true,
  "max_concurrent_agents": 5,
  "agent_token_quota": 100000,
  "preemption_threshold_pct": 0.85,
  "agent_policies_path": null,
  "permissions_signing_key": null,
  "specialist_registry_path": null,
  "homeostatic_downgrade_enabled": true,
  "adversarial_verification": false,
  "maintenance_token_budget": 0,
  "maintenance_priority": "LOW",
  "watchdog_patterns": ["pyproject.toml", "mcp_config.json", "requirements*.txt"],

  // ── Skills ──────────────────────────────────────────────────────
  "custom_skills_directory": null,
  "skill_types": null,

  // ── Model Registries (PRIMARY CONFIG) ───────────────────────────
  "chat_models": [
    {
      "id": "qwen/qwen3.6-27b",
      "provider": "openai",
      "base_url": "http://vllm.arpa/v1",
      "supports_json": false,
      "vision": true,
      "reasoning": true,
      "tools_enabled": true,
      "parallel_instances": 3,
      "context_window": 256000,
      "intelligence_level": "normal",
      "can_route": true,
      "can_kg": true
    }
  ],
  "embedding_models": [
    {
      "id": "text-embedding-nomic-embed-text-v2-moe",
      "provider": "openai",
      "base_url": "http://vllm-embed.arpa/v1",
      "parallel_instances": 4,
      "chunk_size": 768
    }
  ],

  // ── Workspace & Paths ───────────────────────────────────────────
  "workspace_path": "/home/apps/workspace",
  "agent_utilities_config_dir": "~/.config/agent-utilities"
}
```

> **Note:** JSON does not support comments. The `//` annotations above are for documentation purposes only. Your actual `config.json` must not include comments.

#### Chat Model Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Model identifier (e.g., `gpt-4o-mini`, `qwen/qwen3.6-27b`) |
| `provider` | string | ✅ | Provider name (`openai`, `anthropic`, `google`, etc.) |
| `base_url` | string | ❌ | Override API endpoint (e.g., for LM Studio, Ollama) |
| `api_key` | string | ❌ | Per-model API key override |
| `intelligence_level` | string | ❌ | Routing hint: `light`, `normal`, `high` |
| `supports_json` | bool | ❌ | Whether the model supports structured JSON output |
| `vision` | bool | ❌ | Whether the model supports image inputs |
| `reasoning` | bool | ❌ | Whether the model supports extended reasoning/thinking |
| `tools_enabled` | bool | ❌ | Whether the model supports tool/function calling |
| `parallel_instances` | int | ❌ | Max concurrent requests to this model |
| `context_window` | int | ❌ | Maximum context window in tokens |
| `can_route` | bool | ❌ | Whether the model can serve as a router in graph orchestration |
| `can_kg` | bool | ❌ | Whether the model can serve KG analysis tasks |

#### Embedding Model Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Model identifier |
| `provider` | string | ✅ | Provider name |
| `base_url` | string | ❌ | Override API endpoint |
| `api_key` | string | ❌ | Per-model API key override |
| `parallel_instances` | int | ❌ | Max concurrent embedding requests |
| `chunk_size` | int | ❌ | Embedding dimension size (default: 768) |

#### Per-Model Provider Routing

The registry supports per-model `base_url` and `api_key` overrides, enabling configurations like:
- **LM Studio local**: `base_url: "http://vllm.arpa/v1"` (your GPU server)
- **Official OpenAI**: `api_key: "sk-..."` (no `base_url` needed, hits api.openai.com)
- **Ollama**: `base_url: "http://localhost:11434/v1"`, `api_key: "ollama"`
- **Azure OpenAI**: `base_url: "https://my-resource.openai.azure.com"`, `api_key: "..."`

This allows configuring multiple models from the same provider hitting different endpoints.

#### Migration from `.env` to `config.json`

1. Move all `LLM_*`, `LITE_LLM_*`, `SUPER_LLM_*`, and `EMBEDDING_*` variables from `.env` into `chat_models`/`embedding_models` registry entries
2. API keys go directly in per-model entries via the `api_key` field
3. Non-LLM environment variables (e.g., `GRAPH_BACKEND`, `OTEL_ENABLE_OTEL`) are now also configurable via `config.json`

> **Full Documentation:** See [docs/models.md](docs/pillars/2_epistemic_knowledge_graph/models.md) for advanced schema options, local model fallbacks, and routing logic.
