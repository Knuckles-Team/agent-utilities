# Configuration Reference

This document provides a unified reference for all environment variables, configuration files, and CLI flags used across the `agent-utilities` ecosystem.

## Environment Variables

### LLM Configuration
| Variable | Default | Description |
|---|---|---|
| `PROVIDER` | `openai` | Primary LLM provider (e.g., `openai`, `anthropic`, `groq`, `ollama`) |
| `MODEL_ID` | `llama-3.2-3b-instruct` | Specific model identifier to use |
| `LLM_BASE_URL` | `http://host.docker.internal:1234/v1` | Base URL for the LLM API |
| `LLM_API_KEY` | `llama` | API key for the provider |
| `MODELS_CONFIG` | *None* | Path to a multi-model registry JSON configuration |
| `MODEL_OVERRIDE` | *None* | Override model specified per-request via middleware |
| `LLM_CUSTOM_HEADERS` | *None* | Comma-separated list of custom headers to pass to the LLM |

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

### Secrets & Auth (AU-011)
| Variable | Default | Description |
|---|---|---|
| `SECRETS_BACKEND` | `inmemory` | Storage for secrets (`inmemory`, `sqlite`, `vault`) |
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

## CLI Flags

When running `python -m agent_utilities.server` or custom agent scripts, the following standard flags are available:

| Flag | Equivalent Env Var | Description |
|---|---|---|
| `--provider` | `PROVIDER` | LLM Provider |
| `--model-id` | `MODEL_ID` | LLM Model |
| `--base-url` | `LLM_BASE_URL` | Base URL |
| `--api-key` | `LLM_API_KEY` | API Key |
| `--port` | *None* | Server listen port (default: 8000) |
| `--host` | *None* | Server bind host (default: 0.0.0.0) |
| `--web` | *None* | Enables the bundled web UI proxy if present |
| `--mcp-config` | `MCP_CONFIG` | Path to MCP config file |
| `--debug` | *None* | Sets log level to DEBUG |
| `--skill-types`| *None* | Comma-separated list of skills to load (`universal`, `graphs`) |
