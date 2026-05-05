# Changelog

All notable changes to agent-utilities will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **AU-064: Distributed Agentic Evolution** - Transitioned the harness into an open-source hive mind where autonomous agents evolve globally. Features include:
  - **Evolutionary Vector (`genius-agent`)**: A background daemon (`--evolve` flag) that runs `SelfImprovementCycle` indefinitely, creating synthetic tasks and writing new skills to close gaps.
  - **Autonomous PR Generator**: A central `autonomous-contribution` skill in `universal-skills` that formats local KG breakthroughs (TeamConfigs, Skills) into Git branches and upstream PRs.
  - **Community Telemetry & Identity**: `TeamConfigNode` and `CallableResourceNode` metadata now include `origin` (`local` | `community` | `upstream`), deterministic hash identifiers, precise timestamps, and author fields to prevent duplication and establish primacy.
  - **Human-in-the-loop Guardrails**: All autonomously generated skills are explicitly flagged with `Author: Autonomous` in the SKILL.md frontmatter, requiring maintainer approval before centralized CI merge and global distribution.
- **AU-059: HTN & LATS** - Integrated Hierarchical Task Networks (`Task.subtasks`) and `LATSPlanner` (Monte Carlo Tree Search fallback) into the orchestration planner for recursive goal decomposition.
- **AU-060: Tiered Virtual Context Blocks** - Introduced `VirtualContextBlockNode` for tiered working/episodic memory scaling.
- **AU-061: Multi-Agent BFT Consensus** - Integrated `execute_bft_consensus` into `A2AClient` for Byzantine Fault Tolerance across agent peers.
- **AU-062: Execution Budgets & Cost Governors** - Implemented `ExecutionBudget` and integrated enforcement inside `dispatcher_step` to prevent infinite LLM loops and cap USD costs.
- **AU-063: Quiet-STaR Rationale Persistence** - Created `QuietStarRationaleNode` to persist internal chain-of-thought traces with reward gradients for self-improvement.
- **Context-Aware Entity Representations (AU-058)** — Injects topological graph structure directly into node vector embeddings. Features include:
  - Expands Deep GraphRAG principles by appending multi-hop contexts (up to 2 levels of parents/children) directly into the stringified node description before embedding generation.
  - Automatically fetches and appends OWL-inferred relationships (e.g., transitive subclasses) into the node's context space.
  - Enables "topology-aware" vector semantic searches for free, drastically improving multi-hop accuracy.
  - `ContextualRepresentationBuilder` dynamically controls the depth and breadth of injected structural logic.
  - `re_embed_node` pipeline immediately re-embeds nodes when the OWL bridge downfeeds new inferred facts.
- **Wide-Search Orchestration (AU-056)** — Pydantic-native Graph node architecture for orchestrating large-scale extractions. Features include:
  - Automates batch decomposition within the SDD pipeline by instructing orchestrators (planners/routers) to partition large extractions into parallel `ExecutionStep`s (`is_parallel=True`).
  - Implements a hybrid validation strategy inside `join_step` using `WideSearchWorkboard`.
  - Fast-path: Native Pydantic schema validation for expected row counts and schema conformity.
  - Slow-path: `wide_search_joiner` LLM repair node to standardization data, fix schema mismatches, or signal re-plans on fast-path failures.
- **Trace Distillation Error Categorization (AU-057)** — Categorizes orchestrator (`ORCHESTRATOR_SKILL`) vs worker (`WORKER_SKILL`) failure modes through AHE skill distillation to enable self-evolving updates. RLM prompt updated to support proposing targeted ComponentEdits for orchestration logic vs worker tools.
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
- **Confidence-Gated Model Router (AU-039)** — Adaptive model tier routing using runtime confidence signals from specialist consensus. When WorkspaceAttention scores indicate high agreement across specialist outputs, the model tier is automatically downgraded to reduce cost. Low confidence triggers escalation to heavier tiers. Based on the Squeeze Evolve multi-model orchestration framework (Maheswaran et al., 2026). Features include:
  - `ModelRegistry.pick_for_task_adaptive()` with `confidence_signal` and `routing_percentile` parameters.
  - Integration with `pick_specialist_model()` in the graph executor for automatic confidence-gated routing.
  - Composition with AU-033 (Homeostatic Downgrade): budget pressure routes first, then confidence adjusts within the budget-allowed tier range.
  - `routing_confidence_log` on `GraphState` for per-specialist routing decision observability.
  - Soft SelfModel (AU-016) integration: blends 70% runtime confidence + 30% historical proficiency when available; degrades gracefully when absent.
- **Evolutionary Aggregation Engine (AU-040)** — Group-level diversity scoring and three-tier aggregation for specialist outputs. Extends WorkspaceAttention (AU-017) with group fitness computation (confidence and diversity signals) and routes aggregation to the most cost-effective strategy:
  - `MAJORITY_VOTE`: Free — no LLM call when all specialists agree.
  - `LIGHT_MODEL`: Cheap model synthesis for moderate-confidence groups.
  - `HEAVY_MODEL`: Deep aggregation with reasoning-tier models for low-confidence, high-diversity groups.
  - `ConvergenceMonitor` in the CognitiveScheduler (AU-030) detects diversity collapse and triggers early loop termination.
  - Configurable `population_size` (N=4) and `group_size` (K=2) for fully adjustable evolutionary loops.
- New module: `agent_utilities/graph/evolutionary_aggregation.py`.
- New documentation: `docs/squeeze-evolve-routing.md`.
- `routing_percentile` field on `GraphDeps` (env var: `ROUTING_PERCENTILE`, default 50.0).
- `ROUTING_DECISION` node type and `ROUTED_BY`/`AGGREGATED_FROM` edge types in Knowledge Graph.
- Updated Concept Galaxy to 40 concepts (from 38).
- 46 new unit tests across `test_confidence_routing.py` and `test_evolutionary_aggregation.py`.
- **Schema Packs (AU-041)** — Domain-configurable KG profiles that scope the active node types, edge types, retrieval boosts, and OWL extensions to a specific domain. Inspired by gbrain#587 schema-pack proposal. Features include:
  - `SchemaPack` base model with dual operating modes: `ADDITIVE` (layer on top of core) and `EXCLUSIVE` (only pack + protected core types).
  - Protected core types (memory, episode, person, concept, etc.) that are always active regardless of mode.
  - Per-pack retrieval boost multipliers for domain-specific edge weighting.
  - Schema pack registry with `get_schema_pack()` factory and `register_schema_pack()` for runtime extensions.
  - Four pre-built packs: `core` (default), `research-state`, `biomedical`, `finance`.
  - `SchemaPackNode` for KG persistence of active pack configuration.
  - OWL Bridge integration: `PROMOTABLE_NODE_TYPES` and `PROMOTABLE_EDGE_TYPES` filtered through active pack.
- **Backlink-Density Retrieval Boost (AU-042)** — Logarithmic in-degree-based scoring modifier in `HybridRetriever` that boosts hub entities with many inbound edges. Pack-configurable via `backlink_boost_strategy` (`global`, `context_only`, `disabled`) and `backlink_boost_factor` (default 0.1). Based on gbrain's observed +31% P@5 improvement.
- **KG Eval Capture (AU-043)** — Lightweight regression testing harness for Knowledge Graph retrieval. Records query-result pairs to a separate SQLite database (`eval_log.db`) to prevent KG contamination. Features include:
  - `KGEvalCapture.capture()` — append-only recording of queries, results, scores, and latency.
  - `KGEvalCapture.replay()` — re-runs captured queries and reports Jaccard@k, top-1 stability, and latency delta.
  - `export()` / `purge()` for maintenance. Controlled by `KG_EVAL_CAPTURE` env var (disabled by default).
- New module: `agent_utilities/models/schema_pack.py`.
- New package: `agent_utilities/models/schema_packs/` with 4 pre-built domain profiles.
- New module: `agent_utilities/knowledge_graph/eval_capture.py`.
- `SCHEMA_PACK` node type and `USES_SCHEMA_PACK` edge type in Knowledge Graph.
- Updated Concept Galaxy to 43 concepts (from 40).
- 47 new unit tests across `test_schema_packs.py`, `test_backlink_boost.py`, and `test_eval_capture.py`.
- **Conductor Workflow Specification (AU-044)** — Refined natural-language subtask instructions per specialist step. The router/planner now generates a focused `refined_subtask` field on each `ExecutionStep`, crafting targeted sub-goals instead of forwarding the raw user query. Inspired by the RL Conductor's per-step subtask specification (Nielsen et al., ICLR 2026).
- **Execution Visibility Graph (AU-045)** — Per-step `access_list` controlling which prior step results are visible to each specialist. `_resolve_access_context()` helper filters `results_registry` before injection. Supports `["all"]`, specific node_ids, or empty for no context sharing.
- **Model Synergy Tracker (AU-046)** — Per-model-combination EMA success tracking in SelfModel (AU-016). `model_synergies` dict on `SelfModelNode` tracks sorted pipe-delimited model combination keys. `SelfModel.get_best_synergies()` filters by available models for intelligent recombination.
- **Recursive Graph Orchestration (AU-047)** — Nested `run_graph()` calls for self-referential test-time scaling. `recursive_orchestrator` specialist spawns inner graph executions with parent context. `RecursiveContext` dataclass and `MAX_RECURSION_DEPTH` env var (default 2) for depth control.
- New module: `agent_utilities/graph/recursive_executor.py`.
- New documentation: `docs/conductor-orchestration.md`.
- Updated Concept Galaxy to 47 concepts (from 43).
- ~60 new unit tests across `test_conductor_workflow.py`, `test_visibility_graph.py`, `test_model_synergy.py`, and `test_recursive_orchestration.py`.
- **Structural Fingerprint Engine (AU-048)** — AST-based signature extraction and three-level change classification (NONE/COSMETIC/STRUCTURAL) for incremental KG updates. Generic capability that avoids costly full re-ingestion when only cosmetic changes occur. Includes `FingerprintManager` for workspace-level scanning and `detect_stale_files()` for git-based staleness detection.
- **Graph Integrity Validator (AU-053)** — Non-blocking 4-tier graph validation inspired by Understand-Anything's `graph-reviewer`. Features include:
  - Tier 1 (Auto-fix): LLM type alias normalization (30+ node aliases, 30+ edge aliases), score clamping, missing name defaults.
  - Tier 2 (Integrity): Dangling edges, missing node types, untyped edges, duplicate IDs.
  - Tier 3 (Quality): Orphan nodes, self-referencing edges, generic descriptions, underscored hub detection.
  - Tier 4 (Fatal): Zero-node graphs, graph fragmentation below 50% threshold.
  - Integrated as 15th pipeline phase (`validate`) with `KGEvalCapture` (AU-043) trend storage.
- **Entity-Claim Extraction / MAGMA Completion (AU-054)** — Two-phase entity-claim extraction that fills the MAGMA epistemic view with real data. Features include:
  - Deterministic Phase 1: Regex-based extraction of citations, wikilinks, and assertion patterns.
  - `ClaimNode` model with confidence scoring, claim types, and epistemic metadata.
  - New edge types: `BUILDS_ON`, `EXEMPLIFIES`, `AUTHORED_BY` (joining existing `CONTRADICTS` and `CITES`).
  - `retrieve_epistemic_view()` fully implemented with real Cypher queries (replacing stub) and NetworkX fallback.
- New module: `agent_utilities/knowledge_graph/fingerprint.py`.
- New module: `agent_utilities/knowledge_graph/graph_validator.py`.
- New module: `agent_utilities/knowledge_graph/kb/entity_claim_extractor.py`.
- New pipeline phase: `validate` (15th phase, runs after `knowledge_base`).
- `CLAIM` node type and `BUILDS_ON`/`EXEMPLIFIES`/`AUTHORED_BY` edge types in Knowledge Graph.
- Updated Concept Galaxy to 54 concepts (from 47).
- 43 new unit tests across `test_graph_validator.py`, `test_entity_claim_extractor.py`, and `test_fingerprint.py`.

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
- `docs/secrets-auth.md` comprehensive documentation (CONCEPT:OS-5.1)
- Concept marker backfill: AU-004, AU-007, AU-008, AU-009, AU-010, AU-011 across tests and source
- `auth.py` JWT Bearer token validation using `authlib` + JWKS caching (CONCEPT:OS-5.1)
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
