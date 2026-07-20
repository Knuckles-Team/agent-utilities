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
uv run python -m agent_utilities --debug --provider openai --model-id llama-3.2-3b-instruct

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

**Core runtime settings:**

| Variable | Purpose | Default |
|---|---|---|
| `PROVIDER` | LLM provider name (e.g., `openai`, `anthropic`, `groq`) | `openai` |

**Optional Variables:**

| Variable | Purpose | Default |
|---|---|---|
| `DEFAULT_AGENT_NAME` | Override agent display name | Loaded from `main_agent.json` |
| `AGENT_SYSTEM_PROMPT` | Override system prompt | Built from workspace |
| `TOOL_GUARD_MODE` | `on` or `strict` | `strict` |
| `PERMISSIONS_SIGNING_KEY_REF` | Runtime secret reference for the stable identity-signing authority | None |
| `ENABLE_DELEGATION` | Enable OIDC token delegation | `False` |
| `GRAPH_DB_CONNECTION_PROFILE_REF` | Runtime secret reference for an optional external graph connection; the engine authority needs no external database profile | None |
| `GRAPH_MIRROR_TARGETS` | Neutral `KG_CONNECTIONS` aliases that receive the governed mirror stream; declaring mirrors automatically wraps the engine authority in fan-out | None |
| `SECRETS_BACKEND` | Secrets storage backend: encrypted `engine` or `vault` | `engine` |
| `SECRETS_VAULT_URL` | HashiCorp Vault URL | None |
| `SECRETS_VAULT_MOUNT` | Vault KV v2 mount point | `secret` |
| `AUTH_JWT_JWKS_URI` | JWKS URI for JWT Bearer token verification | None |
| `AUTH_JWT_ISSUER` | Expected JWT issuer claim | None |
| `AUTH_JWT_AUDIENCE` | Expected JWT audience claim | None |
| `KG_POLICY_VERSION` | Required GraphSession policy revision | None |
| `ALLOWED_ORIGINS` | Exact CORS origins; unset disables CORS | None |
| `CORS_ALLOW_CREDENTIALS` | Allow credentials for exact configured origins | `false` |
| `ALLOWED_HOSTS` | Comma-separated trusted hosts | loopback authorities |
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
├── base_utilities.py        # Shared utility functions (env expansion, type coercion)
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
├── mcp/                     # Model Context Protocol runtime
│   ├── server_factory.py    # Server construction and authentication middleware
│   ├── context_helpers.py   # Progress, elicitation, logging, state, and sampling
│   └── kg_server.py         # GraphOS tool registration and governed dispatch
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
│   ├── executor.py          # Execution layer (step descriptions, specialist dispatch)
│   ├── lifecycle.py         # Lifecycle management
│   ├── steps.py             # Orchestration nodes (router, verifier, etc.)
│   └── state.py             # GraphState definitions
│
├── knowledge_graph/         # Epistemic-graph authority facade and governance
│   ├── core/engine.py       # IntelligenceGraphEngine
│   ├── core/maintainer.py   # Pruning, decay, maintenance
│   ├── retrieval/hybrid_retriever.py  # Vector + topological search
│   └── core/owl_bridge.py   # LPG ↔ OWL transitive reasoning
│
├── harness/                 # Agentic Harness Engineering (AHE)
│   ├── verifier.py          # Decision observability
│   └── evolve_agent.py      # Prompt evolution loop
│   # (trace distillation lives in knowledge_graph/adaptation/trace_distiller.py)
│
├── mcp/                     # MCP Orchestration
│   ├── kg_server.py         # graph-os KG MCP server entry point
│   └── server_factory.py    # MCP server creation helpers
│
├── tools/                   # Agent Tools
│   ├── agent_tools.py       # Core agent tools
│   ├── developer_tools.py   # Read-only code and KG discovery
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

- **Do NOT commit real API keys, tokens, or credentials** -- use AgentConfig
  reference fields plus the private XDG runtime-secret source or an external
  secret backend.
- **Do NOT add provider-specific auth code** to `agent-utilities` -- it is auth-agnostic
- **Do NOT reference internal/proprietary project names, hostnames, or vendor codenames**
- **Do NOT add `print()` for debugging** -- use `logger.debug()`

## Troubleshooting

### Startup Timeouts
If agents timeout during "Ingesting MCP tools", ensure:
1. All MCP servers are reachable and start within 10-15s individually.
2. Parallel ingestion is not disabled (default is 5 concurrent connections).

### Mirror Lock Contention

The epistemic-graph engine is the only primary authority in development and
production. A file-backed LadybugDB/Kuzu connection is optional and may be used
only as a governed mirror or read source. Its single-writer lock is owned by one
mirror drainer; do not let agent processes open the mirror directly. Check
`graph_configure action=mirror_status` and reconcile the named mirror after
repairing its runtime connection profile.

## Adding New Modules

1. Follow existing code style and conventions
2. Add type hints and comprehensive docstrings
3. Add unit tests in `tests/`
4. Export in `__init__.py` if part of public API
5. Use lazy imports for heavy dependencies
6. Update this documentation to describe the new module

## When Stuck

Key entry points for understanding the codebase:
- `agent/factory.py` → `create_agent` implementation and CLI agent creation
- `mcp/server_factory.py` → MCP server construction and authentication middleware
- `mcp/context_helpers.py` → standardized per-tool context operations
- `mcp/kg_server.py` → GraphOS tool registration and governed dispatch
- `graph/builder.py` → Graph initialization and workspace discovery
- `knowledge_graph/core/engine.py` → `IntelligenceGraphEngine` (Intelligence Graph API)


## API Documentation

Every agent server automatically hosts an interactive Swagger UI for its APIs.

- **URL**: `http://localhost:8000/docs`
- **Spec**: `http://localhost:8000/openapi.json`

This interface allows you to test the `/health`, `/acp`, and `/mcp` endpoints directly from your browser.
