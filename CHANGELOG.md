# Changelog

All notable changes to agent-utilities will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
