# Changelog

All notable changes to agent-utilities will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **A2A Config File (AU-028)** — File-based external A2A agent discovery via `a2a_config.json`. Mirrors `mcp_config.json` symmetry. Features include:
  - `secret://`, `env://`, `vault://` URI-based auth token resolution via `SecretsClient.resolve_ref()`.
  - Soft-fail startup: unreachable `.well-known/agent-card.json` endpoints log a warning and skip, never blocking server startup.
  - Periodic background re-fetch (`A2A_REFRESH_INTERVAL`, default 300s) to detect capability changes from remote agents.
  - Full KG ingestion: agent cards are registered as `CallableResource` nodes with embeddings, making them eligible for affinity-based swarm selection.
  - Cache invalidation: bulk ingestion triggers `invalidate_registry_cache()` (AU-024) to keep the hot cache synchronized.
- **Unified Specialist Model (AU-029)** — Collapses the artificial `prompt` / `mcp` agent type distinction into a single `specialist` type. Any specialist can now host any combination of MCP tools and/or agent skills. A2A agents remain their own type (`a2a`) due to the fundamentally different remote execution protocol. Legacy `agent_type` values are normalized at read time for full backward compatibility.
- New `A2A_CONFIG` and `A2A_REFRESH_INTERVAL` environment variables.
- New module: `agent_utilities/protocols/a2a_config.py` (config loader, auto-discovery, periodic refresh).
- Updated Concept Galaxy to 29 concepts (from 27).

### Changed
- `MCPAgent.agent_type` default changed from `"prompt"` to `"specialist"`.
- `DiscoveredSpecialist.source` values unified to `"specialist"` or `"a2a"`.
- `discover_agents()` collapsed from 3 type branches to 2 (specialist + a2a).
- Executor `_execute_agent_package_logic()` simplified from 3 execution paths to 2 (remote A2A vs. unified specialist).
- `build_agent_app()` and `create_agent_server()` accept new `a2a_config` parameter.
- `initialize_graph_from_workspace()` accepts `a2a_config` parameter and syncs A2A agents alongside MCP agents during startup.

## [0.3.0] - 2026-05-02

### Added
- **First Principles Architecture** — Four new foundational concepts (AU-024 through AU-027) that rewire the routing, dispatch, and feedback layers from first principles.
- **Registry Hot Cache (AU-024)** — Session-scoped `_RegistryCache` singleton providing O(1) specialist lookups. Replaces full registry scans on every routing call, reducing prompt bloat by filtering to only the top-7 relevant specialists per query.
- **Event-Driven Cache Invalidation** — Cache auto-invalidates on MCP reload (`/mcp/reload`), pipeline completion, Self-Model session updates, and TeamConfig promotions. No stale-cache risk.
- **TeamConfig Promotion (AU-025)** — `promote_coalition_to_template()` persists successful specialist coalitions as reusable `TeamConfigNode` templates in the Knowledge Graph. `find_matching_team_config()` enables 3-stage hybrid routing: TeamConfig → Self-Model bias → LLM planning.
- **TeamConfig Reward Tracking** — `record_team_outcome()` records success/failure outcomes against team templates, enabling reward-weighted team selection over time.
- **RLM + TeamConfig Synergy** — When a `TeamConfig` is selected and input exceeds size thresholds, RLM capability is auto-attached to specialists via `capability_overrides`.
- **AgentCapability Type System (AU-026)** — `AgentCapabilityNode` formalized as a first-class KG node with `auto_activate`, `trigger_conditions`, and `handler_module` fields. Capabilities are auto-activated in the executor based on input constraints (e.g., RLM for large payloads, critic for code).
- **PlannerGraphSkill (AU-027)** — A2A-native routing entry point via `PlannerGraphSkill` registered in `server/app.py`. When a `graph_bundle` is available, A2A requests bypass LLM orchestration and route directly through the graph planner.
- **Self-Model Feedback Loop** — Post-execution verification in `synthesizer_step` now feeds outcomes back to `SelfModel.update_after_session()` and `record_team_outcome()`, enabling recursive learning.
- **WorkspaceAttention Scoring** — `WorkspaceAttention` (GWT) scores are computed and logged per-specialist during execution for data-driven specialist prioritization.
- **Process Lifecycle Management** — `atexit` and `SIGTERM`/`SIGINT` handlers in `server/__init__.py` ensure all child processes (MCP servers, TUI, background threads) are gracefully killed on server exit. Uses child-only `pgrep` pattern instead of `killpg` to avoid self-termination.
- **USES_PROMPT Edges** — `link_prompt_to_agent()` creates `USES_PROMPT` edges between specialist nodes and their JSON prompt templates for full traceability.
- 33 new unit tests across `test_config_helpers.py`, `test_team_config.py`, and `test_capability_nodes.py`.
- 3 new documentation files: `first-principles.md`, `registry-cache.md`, `process-lifecycle.md`.

### Changed
- Router now performs 3-stage hybrid routing: (1) TeamConfig match → (2) Self-Model proficiency bias → (3) LLM planning fallback.
- Specialist filtering reduced from O(N) full registry scan to O(7) via `get_relevant_specialists()` using query-keyed caching.
- Verification synthesizer now feeds execution outcomes back to both Self-Model and TeamConfig for continuous improvement.
- `RegistryNodeType` enum extended with `TEAM_CONFIG` and `AGENT_CAPABILITY` types.
- `RegistryEdgeType` enum extended with `HAS_CAPABILITY`, `REUSED_TEAM`, and `USES_PROMPT` types.
- Server process cleanup uses targeted `pgrep -P` child enumeration instead of `os.killpg()` to avoid killing the process group (which would terminate test runners).

### Fixed
- mypy `call-arg` error: `SelfModel.update_after_session()` now correctly receives `GraphState` instead of kwargs.
- mypy `assignment` error: `log_file_path` now guards against `None` before assigning to env dict.
- ruff `F401`: Removed unused `contextlib.suppress` import from `server/__init__.py`.
- Bandit `B110` suppressions added for defensive `try/except pass` patterns in cleanup handlers.

## [0.2.41] - 2026-04-29

### Added
- Direct graph execution fast-path in AG-UI endpoint — bypasses LLM tool-call overhead
- `AGUIGraphEmitter` module for translating graph events to AG-UI wire format (0:/2:/8:/9: prefixes)
- `run_graph_iter()` and `execute_graph_iter()` — step-by-step graph execution using `graph.iter()` beta API
- Per-step state snapshots and elicitation hooks in the iter-based execution path
- `GRAPH_DIRECT_EXECUTION` env var (default: `true`) to toggle direct dispatch
- `SecretsClient` with pluggable backends: InMemory (Fernet-encrypted), SQLite (persistent + encrypted), HashiCorp Vault (enterprise)
- URI-style secret references: `vault://`, `env://`, `sqlite://` schemes
- `SECRETS_BACKEND`, `SECRETS_SQLITE_PATH`, `SECRETS_VAULT_URL` configuration
- `secrets_client` field on `GraphDeps` for graph execution credential resolution
- `docs/secrets-auth.md` comprehensive documentation (CONCEPT:AU-011)
- Concept marker backfill: AU-004, AU-007, AU-008, AU-009, AU-010, AU-011 across tests and source
- `auth.py` JWT Bearer token validation using `authlib` + JWKS caching (CONCEPT:AU-011)
- Combined auth dependency: accepts API key OR JWT Bearer token (gradual migration)
- `AUTH_JWT_JWKS_URI`, `AUTH_JWT_ISSUER`, `AUTH_JWT_AUDIENCE` configuration
- `ALLOWED_ORIGINS` and `ALLOWED_HOSTS` for configurable CORS/host restriction
- MCP subprocess token forwarding via `AGENT_USER_TOKEN` env injection
- `auth = ["authlib>=1.4.0"]` optional extra in `pyproject.toml`
- 69 new unit tests (30 secrets, 20 auth, 19 emitter/iter)

### Changed
- ACP adapter refactored to use pydantic-acp's `agent_factory` callback for per-session agent creation
- Removed `REQUESTED_MODEL_ID_CTX` workaround from ACP's `run_graph_flow` closure
- Unified execution layer (`graph/unified.py`) now exports `execute_graph_iter` as a first-class entry point
- `cryptography>=44.0.0` added as core dependency for Fernet encryption
- `vault` optional extra added to `pyproject.toml` for `hvac>=2.3.0`
- CORS middleware now reads `ALLOWED_ORIGINS` instead of hardcoded `["*"]`
- TrustedHostMiddleware now reads `ALLOWED_HOSTS` instead of hardcoded `["*"]`

## [0.2.40] - 2026-04-28

### Added
- LLM Council integration with 7 role-based advisor prompts
- Hybrid OWL Reasoning sidecar with HermiT/Stardog inference
- Standard ontology schemas (BFO, Schema.org, PROV-O, Dublin Core, SKOS)
- Concept traceability markers (`@pytest.mark.concept`) for doc-test alignment
- `.env.example` template for developer onboarding
- Project Structure section in AGENTS.md

### Changed
- Restructured test suite into domain-driven subdirectories (core, graph, integration)
- Deprecated `MEMORY.md` in favor of Knowledge Graph native storage
- Bumped pre-commit hooks: ruff 0.15.12, mypy 1.20.2, bandit 1.9.4

### Fixed
- Git merge conflict artifacts in atlassian-agent, langfuse-agent, repository-manager
- Duplicate import errors in protocol adapters
- TOML configuration errors in .bumpversion.cfg
- Broken file references in AGENTS.md

## [0.2.39] - 2026-04-26

### Added
- AG-UI protocol adapter for web and terminal frontends
- Human-in-the-loop tool approval with `ApprovalManager`
- MCP elicitation callback support via `global_elicitation_callback()`

### Changed
- Unified protocol layer: ACP, A2A, MCP, AG-UI all served from single FastAPI server
- Migrated flat-file state (MEMORY.md, USER.md, HEARTBEAT.md) to Knowledge Graph

## [0.2.38] - 2026-04-24

### Added
- 14-phase Unified Intelligence Pipeline
- LadybugDB as default embedded graph backend
- Knowledge Base layer with LLM-maintained wiki
- MAGMA-inspired orthogonal reasoning views (Semantic, Temporal, Causal, Entity)

### Changed
- Replaced Neo4j-only backend with pluggable graph abstraction (LadybugDB, FalkorDB, Neo4j)
