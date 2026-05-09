# Development Guide

## Commands

```bash
# Run tests (unit + integration, excludes live)
uv run pytest -x -v

# Run with coverage
uv run pytest --cov=agent_utilities --cov-report=term-missing

# Lint
uv run ruff check agent_utilities/ tests/
uv run ruff format --check agent_utilities/ tests/

# Type check
uv run mypy agent_utilities/

# Run the server (development)
uv run python -m agent_utilities.server --debug --provider openai --model-id llama-3.2-3b-instruct

# Install with all optional dependencies
uv pip install -e ".[all]"
```

## Test Status & Markers

| Marker | Scope | When to Use |
|---|---|---|
| `integration` | In-process, no external services | `TestClient`, subprocess, fixture-based |
| `live` | Requires real LLM endpoints / network | API round-trips, end-to-end flows |

Default: `pytest -m "not live"` runs unit + integration.

## Runtime Prerequisites

**Required Environment Variables:**

| Variable | Purpose | Default |
|---|---|---|
| `PROVIDER` | LLM provider name (e.g., `openai`, `anthropic`, `groq`) | `openai` |
| `MODEL_ID` | LLM model identifier | `llama-3.2-3b-instruct` |
| `LLM_BASE_URL` | LLM API base URL | `http://host.docker.internal:1234/v1` |
| `LLM_API_KEY` | LLM API key | `llama` |
| `GRAPH_DB_PATH` | Path to knowledge graph database | `knowledge_graph.db` |

**Optional Variables:**

| Variable | Purpose | Default |
|---|---|---|
| `DEFAULT_AGENT_NAME` | Override agent display name | Loaded from `main_agent.json` |
| `AGENT_SYSTEM_PROMPT` | Override system prompt | Built from workspace |
| `TOOL_GUARD_MODE` | `on`, `off`, `custom` | `on` |
| `DISABLE_TOOL_GUARD` | Boolean to disable tool guard | `False` |
| `MODELS_CONFIG` | Path to multi-model registry config | None |
| `ENABLE_DELEGATION` | Enable OIDC token delegation | `False` |
| `GRAPH_BACKEND` | Backend type: `ladybug`, `falkordb`, `neo4j` | `ladybug` |
| `GRAPH_DIRECT_EXECUTION` | Direct graph dispatch in AG-UI/ACP (bypasses LLM tool-call hop) | `True` |
| `SECRETS_BACKEND` | Secrets storage backend: `inmemory`, `sqlite`, `vault` | `inmemory` |
| `SECRETS_SQLITE_PATH` | SQLite secrets database path | `~/.agent-utilities/secrets.db` |
| `SECRETS_VAULT_URL` | HashiCorp Vault URL | None |
| `SECRETS_VAULT_MOUNT` | Vault KV v2 mount point | `secret` |
| `AGENT_SECRETS_MASTER_KEY` | Fernet encryption key (base64) | *(auto-generated)* |
| `AUTH_JWT_JWKS_URI` | JWKS URI for JWT Bearer token verification | None |
| `AUTH_JWT_ISSUER` | Expected JWT issuer claim | None |
| `AUTH_JWT_AUDIENCE` | Expected JWT audience claim | None |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins | `*` |
| `ALLOWED_HOSTS` | Comma-separated trusted hosts | `*` |
| `AGENT_USER_TOKEN` | Session token forwarded to MCP subprocesses | None |

## Validation & Diagnostics

### Pre-Flight Checks
Before modifying any file, verify:
1. `uv run pytest -x` passes (baseline green)
2. `uv run ruff check agent_utilities/` is clean

### Post-Change Verification
After every change:
1. Run `uv run pytest -x -v` — all tests must pass
2. Run `uv run ruff check --fix agent_utilities/ tests/`
3. Run `uv run ruff format agent_utilities/ tests/`
4. Run `uv run mypy agent_utilities/` — no new errors

### Diagnostics
If tests fail unexpectedly:
- Check `AGENT_UTILITIES_TESTING=true` is set (auto-set via `pytest.ini`)
- Check for singleton pollution: `IntelligenceGraphEngine._ACTIVE_ENGINE` and `knowledge_graph/backends.__init__._ACTIVE_BACKEND` leak state across tests. Tests needing `None` must `monkeypatch.setattr(...)`.

## Project Structure Quick Reference

```
agent_utilities/
├── __init__.py              # Public API re-exports
├── agent_utilities.py       # Legacy exports and helper wrappers
├── base_utilities.py        # Shared utility functions (env expansion, type coercion)
├── mcp_utilities.py         # MCP server creation and context helpers
│
├── core/                    # Foundational Primitives
│   ├── workspace.py         # Workspace discovery and initialization
│   ├── config.py            # Centralized configuration constants
│   ├── exceptions.py        # Core domain exceptions
│   └── decorators.py        # Cross-cutting decorators
│
├── agent/                   # Agent Lifecycle & Setup
│   ├── factory.py           # CLI agent creation helpers
│   ├── discovery.py         # Specialist discovery
│   └── registry_builder.py  # Prompt → KG registry synchronization
│
├── protocols/               # External Interfaces
│   ├── acp_adapter.py       # ACP protocol adapter
│   ├── a2a.py               # A2A protocol adapter
│   └── agui_emitter.py      # AG-UI wire format translator
│
├── server/                  # FastAPI Application
│   ├── app.py               # App factory and middleware
│   ├── dependencies.py      # Route dependencies
│   └── routers/             # Endpoint definitions
│
├── security/                # Auth & Permissions
│   ├── auth.py              # JWT validation
│   └── cors.py              # Cross-origin policies
│
├── graph/                   # Orchestration Engine
│   ├── builder.py           # Graph initialization
│   ├── unified.py           # Execution layer
│   ├── runner.py            # Lifecycle management
│   ├── steps.py             # Orchestration nodes (router, verifier, etc.)
│   └── state.py             # GraphState definitions
│
├── knowledge_graph/         # MAGMA Memory System
│   ├── engine.py            # IntelligenceGraphEngine
│   ├── maintainer.py        # Pruning, decay, maintenance
│   ├── hybrid_retriever.py  # Vector + topological search
│   └── owl_bridge.py        # LPG ↔ OWL transitive reasoning
│
├── harness/                 # Agentic Harness Engineering (AHE)
│   ├── verifier.py          # Decision observability
│   ├── trace_distiller.py   # Execution trace distillation
│   └── evolve_agent.py      # Prompt evolution loop
│
├── mcp/                     # MCP Orchestration
│   ├── manager.py           # Subprocess management
│   └── utilities.py         # MCP tool extraction
│
├── tools/                   # Agent Tools
│   ├── agent_tools.py       # Core agent tools
│   ├── developer_tools.py   # File system tools
│   └── ...                  # 16 other tool categories
│
├── models/                  # Pydantic Schemas
│   ├── knowledge_graph.py   # RegistryNode, Edge schemas
│   └── sdd.py               # Spec, Plan, Tasks
│
├── prompts/                 # JSON Prompt Blueprints
│   └── *.json
│
├── rlm/                     # Recursive Language Models
│   └── repl.py              # Sub-shell execution
│
├── sdd/                     # Spec-Driven Development
│   └── orchestrator.py      # Pipeline engine
│
└── agent_data/              # Runtime data directory (git-ignored)
```

## Code Style & Conventions

- **Python**: `ruff` for linting/formatting, `mypy` for type checking
- **Line length**: 88 characters (Black-compatible)
- **Target version**: Python 3.11
- **Imports**: `ruff` manages import ordering (isort-compatible)
- **Type hints**: Required for all public functions; `from __future__ import annotations` where needed
- **Docstrings**: Google-style docstrings for public API
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Error handling**: Use `raise ... from None` (B904) inside `except` blocks when re-raising different exceptions
- **Mutable defaults**: Never use mutable defaults (`list`, `dict`); use `None` with runtime initialization
- **Lazy imports**: Use lazy imports for heavy dependencies (e.g., `pydantic_ai`, `fastmcp`)

## Safety & Boundaries

> **⚠️ Pydantic AI VercelAIAdapter Note**: The `VercelAIAdapter` class in Pydantic AI is **internal and unstable**. Do NOT subclass or directly modify it. Any streaming or event-format changes should be made through composition and middleware, not by patching VercelAIAdapter.

- **Do NOT commit real API keys, tokens, or credentials** -- use env vars and `.env.example`
- **Do NOT add provider-specific auth code** to `agent-utilities` -- it is auth-agnostic
- **Do NOT reference internal/proprietary project names, hostnames, or vendor codenames**
- **Do NOT add `print()` for debugging** -- use `logger.debug()`

## Troubleshooting

### Startup Timeouts
If agents timeout during "Ingesting MCP tools", ensure:
1. All MCP servers are reachable and start within 10-15s individually.
2. Parallel ingestion is not disabled (default is 5 concurrent connections).

### Database Lock Contention
When running multiple agents on the same host, LadybugDB (DuckDB) may encounter file locks if multiple processes try to sync to the same `knowledge_graph.db`.
- **Recommendation**: Set a unique `GRAPH_DB_PATH` per agent (e.g., `GRAPH_DB_PATH=./agent_data/graph.db`).
- **Resilience**: The backend includes a 5-attempt retry mechanism with exponential backoff and jitter to handle transient lock contention.

## Adding New Modules

1. Follow existing code style and conventions
2. Add type hints and comprehensive docstrings
3. Add unit tests in `tests/`
4. Export in `__init__.py` if part of public API
5. Use lazy imports for heavy dependencies
6. Update this documentation to describe the new module

## When Stuck

Key entry points for understanding the codebase:
- `agent_utilities.py` → `create_agent` implementation
- `agent_factory.py` → CLI agent creation
- `mcp_utilities.py` → MCP tool registration
- `graph/builder.py` → Graph initialization and workspace discovery
- `knowledge_graph/engine.py` → Intelligence Graph API
