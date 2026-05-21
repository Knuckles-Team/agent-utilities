# Configuration Reference

This document provides a unified reference for all environment variables, configuration files, and CLI flags used across the `agent-utilities` ecosystem.

## Environment Variables

### LLM Configuration
All LLM configuration (models, API keys, endpoints) is now managed centrally via the XDG configuration file at `~/.config/agent-utilities/config.json`.

Environment variables for `LLM_BASE_URL`, `LLM_MODEL_ID`, etc., are **deprecated** and will be ignored. API keys can optionally be provided in the `.env` file or directly inside the `config.json` model entries.

### Graph Database
| Variable | Default | Description |
|---|---|---|
| `GRAPH_BACKEND` | `ladybug` | Backend to use (`ladybug`, `falkordb`, `neo4j`) |
| `GRAPH_DB_PATH` | `knowledge_graph.db` | File path for LadybugDB (DuckDB) |
| `GRAPH_DB_HOST` | `localhost` | Host for Neo4j/FalkorDB |
| `GRAPH_DB_PORT` | `7687` | Port for Neo4j/FalkorDB |
| `GRAPH_DB_URI` | *None* | Direct connection URI (overrides Host/Port) |
| `GRAPH_DB_USER` | `neo4j` | Username for remote DBs |
| `GRAPH_DB_PASSWORD` | *None* | Password for remote DBs |

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
| `SECRETS_VAULT_URL` | *None* | URL for HashiCorp Vault |
| `SECRETS_VAULT_MOUNT` | `secret` | Vault KV v2 mount point |
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
| **KG Server** | `uv run agent-utilities-kg` | Launches the Knowledge Graph MCP server |
| **Main Server** | `uv run agent-utilities-server` | Launches the unified protocol server (ACP/A2A/AG-UI) |

## CLI Flags

When running `agent-utilities` commands (or `python -m agent_utilities.server`), the following standard flags are available:

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

### Standardized LLM Environment Variables

The ecosystem relies on a standardized set of API keys in `.env` for security, while routing and capabilities are managed via the unified JSON configuration:

- **API Keys**: While models are defined in `config.json`, sensitive API keys like `LLM_API_KEY` can optionally remain in the `.env` file to prevent committing secrets to version control.

> **Full Documentation:** See [docs/configuration.md](docs/pillars/5_agent_os_infrastructure/configuration.md) for a complete list of environment variables.

### Unified Agent Configuration (`config.json`)

Define a registry of models mapped to routing tiers (`light`, `normal`, `super`) and capabilities directly in the XDG-compliant `~/.config/agent-utilities/config.json`. The graph orchestrator autonomously selects the right model for each task based on required complexity.

**Configuration Example:**
```json
{
  "chat_models": [
    {
      "id": "gpt-4o-mini",
      "provider": "openai",
      "intelligence_level": "normal",
      "supports_json": true,
      "vision": true
    }
  ]
}
```

*The graph orchestrator automatically accesses this file globally via `AgentConfig`.*
> **Full Documentation:** See [docs/models.md](docs/pillars/2_epistemic_knowledge_graph/models.md) for advanced schema options, local model fallbacks, and routing logic.
