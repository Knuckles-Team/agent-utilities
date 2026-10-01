# AU capability coverage and disposition

This map routes each AU capability to the specification that owns it and, in the second half, routes each AU package directory to the requirement that removes, relocates or keeps it. A mention here is a routing decision, not acceptance. The sibling specs contain the complete AU behavior, design and test contract. A capability can have a separate owner implementation in another public repository; AU completion still requires its own caller and deletion proof.

| AU capability or disposition | Native spec | Requirement definitions |
|---|---|---|
| Agent execution, routing, swarm, task claims | [`agent-control-plane`](../agent-control-plane/spec.md) | [requirements](../agent-control-plane/requirements.md) |
| Harness capture, training, work market and governed change | [`harness-evolution`](../harness-evolution/spec.md) | [requirements](../harness-evolution/requirements.md) |
| Security, identity, guardrails and cache invalidation | [`agent-security-policy`](../agent-security-policy/spec.md) | [requirements](../agent-security-policy/requirements.md) |
| Portable tests, liveness and type coverage | [`au-boundary-quality`](../au-boundary-quality/spec.md) | [requirements](../au-boundary-quality/requirements.md) |
| Semantic schema and generated EG client | [`au-semantic-client`](../au-semantic-client/spec.md) | [requirements](../au-semantic-client/requirements.md) |
| Served, connector, graph and state authority cuts | [`au-boundary-deconstruction`](spec.md) | [requirements](requirements.md) |
| Cross-cutting audit of duplicate graph engine code | [`au-engine-duplicate-retirement`](../au-engine-duplicate-retirement/spec.md) | [requirements](../au-engine-duplicate-retirement/requirements.md) |
| Retrieval/context and finance agent roles | [`agent-context-and-finance`](../agent-context-and-finance/spec.md) | [requirements](../agent-context-and-finance/requirements.md) |
| Public contributor environment and skill lifecycle | [`au-developer-environment`](../au-developer-environment/spec.md) | [requirements](../au-developer-environment/requirements.md) |
| Final composition, ontology/SHACL/OWL clean cut, public application control plane and public documentation | [`au-integration-reliability`](../au-integration-reliability/spec.md) | [requirements](../au-integration-reliability/requirements.md) |

Where a capability has more than one owner, the native AU spec describes its own boundary and explicit input/output contract; it does not make another repository's implementation state an AU acceptance claim. Each contributor must confirm the current source tree and evidence at the exact head before moving a task state.

## Deletion and relocation inventory

One row per directory in the top two levels under `agent_utilities/`. It states what this spec and [`au-engine-duplicate-retirement`](../au-engine-duplicate-retirement/spec.md) say happens to that directory, and which requirement in [requirements.md](requirements.md) or the [retirement requirements](../au-engine-duplicate-retirement/requirements.md) covers it. The cut is complete when every row has a disposition other than "undecided" and every covering requirement is delivered.

How to read it:

- **Approx. lines** counts every tracked file under the directory, including data files and any child directory that has its own row. The package totals 1,973 tracked files and about 802,000 lines.
- **Disposition** is taken from the requirement text. "Split" means the requirements send different files to different owners. "Undecided" means neither spec names the directory or its files; the note then states only what the directory contains. A directory marked "undecided" is not thereby kept: no requirement says so either.
- A requirement that deletes an AU copy because another repository already owns the behavior is shown as a relocation to that repository.
- "AU-RETIRE-R001 (inventory obligation only)" marks a `knowledge_graph/` directory that AU-RETIRE-R001 requires to receive a disposition but for which no requirement states one.
- Short paths in requirement text such as `core/…`, `enrichment/…`, `ontology/…` and `kg/…` are read as relative to `agent_utilities/knowledge_graph/` where no directory of that name with that file exists directly under `agent_utilities/`.

| Directory | Approx. lines | Disposition | Covering requirement ID(s) | Notes |
|---|---:|---|---|---|
| `agent_utilities/` | 802,061 | split: relocate to agent-connector-sdk (`base_utilities.py`); relocate to graph-os (`release_catalogs.py`); remainder undecided | AU-BOUNDARY-R010, AU-BOUNDARY-R014 | Count is the whole package; 12 files sit directly here (2,060 lines). The 10 unnamed top-level files (681 lines) are package metadata and small helpers: `__init__.py`, `__main__.py`, `api_utilities.py`, `content.py`, `file_safety.py`, `nested_context.py`, `_version.py`, two JSON catalogs and `py.typed`. |
| `agent_utilities/agent/` | 2,203 | undecided | — | Agent factory, registry builder, sampling profiles and capability resolver; 2 of 8 modules import `knowledge_graph`. |
| `agent_utilities/agent_chat/` | 54 | undecided | — | One chat-input parser module and a README. |
| `agent_utilities/analysis/` | 287 | undecided | — | A single `analyzer.py` module. |
| `agent_utilities/api/` | 1,737 | keep in AU | AU-BOUNDARY-R013, AU-BOUNDARY-R019 | The public integration boundary. R013 widens it for graph-os; R019 keeps the single typed engine composition here. R017 expects a messaging module here that does not exist yet. |
| `agent_utilities/automation/` | 2,644 | split: relocate to agent-connector-sdk (3 modules); remainder undecided | AU-BOUNDARY-R025 | `worldmodel_pipeline`, `feed_sources`, `file_watcher` are named. `research_pipeline.py` (1,517 lines) and `__init__.py` are not named by any requirement. |
| `agent_utilities/caching/` | 898 | undecided | — | Semantic cache and prompt cache modules; 2 of 3 import `knowledge_graph`. |
| `agent_utilities/capabilities/` | 5,799 | undecided | — | Agent capability implementations (governed dynamic workflow, output repair, guardrails, checkpointing, model fallback). Includes `kg_audit_sink.py` and `eg_history_source.py`, which read or write graph state. |
| `agent_utilities/capabilities/compaction/` | 100 | undecided | — | Context-compaction helpers for agent runs. |
| `agent_utilities/claude_harness/` | 930 | undecided | — | Coding-harness guard modules: a fence, a pre-tool-use check and an unattended runner. |
| `agent_utilities/cli/` | 1,190 | undecided | — | The unified `agent-utilities` command line entry point; it imports `governance/`, which R037 relocates. |
| `agent_utilities/control_plane/` | 74 | delete as duplicate of engine | AU-BOUNDARY-R012 | The shim named by R012; three re-export modules. |
| `agent_utilities/core/` | 36,947 | split: relocate to agent-connector-sdk (6 modules); relocate to graph-os (9 modules); relocate to epistemic-graph (3 modules); `config.py` split three ways; remainder undecided | AU-BOUNDARY-R010, AU-BOUNDARY-R014, AU-BOUNDARY-R016, AU-BOUNDARY-R020 | Count includes the three child rows. 54 files sit directly here; 19 are named. The 35 unnamed modules (15,441 lines) are model factory and providers, model routing and concurrency, schedulers, sessions, embedding utilities and resource budgeting. |
| `agent_utilities/core/checkpoint/` | 769 | undecided | — | One checkpoint manager module; uses a SQL store and imports `knowledge_graph`. |
| `agent_utilities/core/execution/` | 984 | undecided | — | Unified execution contract package: protocol, adapters, provider proxy, stream handlers. |
| `agent_utilities/core/registry/` | 3,118 | split: delete as duplicate of engine (`kg_adapter.py`); remainder undecided | AU-BOUNDARY-R019 | `kg_adapter.py` (1,802 lines) is replaced by the engine composition in `api/`. The service, package and plugin adapters (5 files, 1,316 lines) are not named. |
| `agent_utilities/data/` | 16 | undecided | — | One 16-line MCP configuration JSON file. |
| `agent_utilities/data_prep/` | 4,481 | relocate to agent-connector-sdk | AU-BOUNDARY-R024 | Named wholesale as `data_prep/**`. |
| `agent_utilities/data_prep/certification/` | 643 | relocate to agent-connector-sdk | AU-BOUNDARY-R024 | Covered by the parent glob. |
| `agent_utilities/deployment/` | 19,006 | relocate to graph-os | AU-BOUNDARY-R002 | Named wholesale as `deployment/**`; the console scripts that target it go with it. |
| `agent_utilities/domains/` | 13,947 | undecided | — | Count includes the five child rows; only a 53-line `__init__.py` sits directly here. |
| `agent_utilities/domains/finance/` | 12,991 | split: delete (12 modules that produced fabricated data); remainder undecided in these two specs | AU-BOUNDARY-R041 | R041 names 12 of the 45 modules. The other 33 (about 9,950 lines) are outside these two specs; [`agent-context-and-finance`](../agent-context-and-finance/spec.md) owns the remaining finance cut. |
| `agent_utilities/domains/government/` | 92 | undecided | — | Typed domain models only. |
| `agent_utilities/domains/hr/` | 570 | undecided | — | Typed domain models and a workforce manager module. |
| `agent_utilities/domains/law/` | 134 | undecided | — | Typed domain models only. |
| `agent_utilities/domains/medical/` | 107 | undecided | — | Typed domain models only. |
| `agent_utilities/ecosystem/` | 4,619 | split: relocate to graph-os (2 modules); relocate to agent-connector-sdk (`ea_clients.py`, `media/`); relocate to epistemic-graph (`ard_*`); remainder undecided | AU-BOUNDARY-R014, AU-BOUNDARY-R024, AU-BOUNDARY-R025, AU-BOUNDARY-R028 | 9 direct modules (2,237 lines) are not named: governance workflow and agent, configuration staleness auditor, agent-instructions reflector, lint hook, permission policy, plugin bundle, bridge. |
| `agent_utilities/ecosystem/media/` | 777 | relocate to agent-connector-sdk | AU-BOUNDARY-R025 | Named wholesale as `ecosystem/media/**`. |
| `agent_utilities/gateway/` | 13,940 | relocate to graph-os | AU-BOUNDARY-R001 | Named wholesale; the AU copy is deleted after AU-only behavior is ported. |
| `agent_utilities/gateway/schemas/` | 2,604 | relocate to graph-os | AU-BOUNDARY-R001 | `graph_analyze` is AU-only; `graph_core` is a drifted duplicate to reconcile. |
| `agent_utilities/gateway/widgets/` | 4,540 | relocate to graph-os | AU-BOUNDARY-R001 | 73 widget modules; the `gateway-widgets` extra is removed with them. |
| `agent_utilities/gateway_client/` | 320 | relocate to graph-os | AU-BOUNDARY-R014 | Named wholesale as `gateway_client/**`. |
| `agent_utilities/governance/` | 11,993 | relocate to repository-manager; relocate to epistemic-graph (`relational_authority.py`) | AU-BOUNDARY-R037, AU-BOUNDARY-R028 | The destination is repository-manager, which is none of the three target repositories. The `relational_authority.json` data file next to the module is not named. |
| `agent_utilities/graph/` | 26,425 | undecided | — | The agent execution state machine (executor, router, planner, parallel engine, verification); not a graph database. 20 of 40 direct modules import `knowledge_graph`. Count includes the four child rows. |
| `agent_utilities/graph/planning/` | 486 | undecided | — | Planning package for the agent state machine. |
| `agent_utilities/graph/reactive/` | 844 | undecided | — | Reactive dispatch, budgets and an engine subscription module. |
| `agent_utilities/graph/reasoning/` | 1,898 | undecided | — | Prompt-level reasoning strategies (chain, tree and graph of thought, ReAct). |
| `agent_utilities/graph/routing/` | 1,871 | undecided | — | Routing strategies and capability enrichers for the agent router. |
| `agent_utilities/harness/` | 28,964 | undecided | — | Evaluation, optimization and self-improvement harness (78 direct files, 27,128 lines). Includes trace, state and memory modules that hold durable records. Not named by either of these two specs; the sibling [`harness-evolution`](../harness-evolution/spec.md) spec covers harness behavior but names no directory. |
| `agent_utilities/harness/memorydata/` | 1,836 | undecided | — | Benchmark adapter for the served memory stack, with seven YAML configurations. |
| `agent_utilities/httpsupport/` | 2,089 | relocate to agent-connector-sdk | AU-BOUNDARY-R010 | Named wholesale as `httpsupport/**`; deleted in AU once connectors use the SDK. |
| `agent_utilities/images/` | 208 | undecided | — | One PNG asset. |
| `agent_utilities/ingestion/` | 1,479 | relocate to agent-connector-sdk | AU-BOUNDARY-R025 | Named as the top-level `ingestion/**`. |
| `agent_utilities/ingestion/agent_sources/` | 1,174 | relocate to agent-connector-sdk | AU-BOUNDARY-R025 | Covered by the parent glob. |
| `agent_utilities/integrations/` | 257 | undecided | — | One git issue and pull-request resolver module. |
| `agent_utilities/knowledge_graph/` | 347,980 | split by child row; direct files: delete as duplicate of engine (`facade.py`); relocate to epistemic-graph (2 worker modules); relocate to graph-os (`readiness.py`); relocate to agent-connector-sdk (`governance_import.py`); remainder undecided | AU-BOUNDARY-R014, AU-BOUNDARY-R018, AU-BOUNDARY-R020, AU-BOUNDARY-R024, AU-RETIRE-R001 | Count is the whole subtree. 17 files sit directly here; 12 (3,716 lines) are not named: workflow store and compilers, process-plan compiler, migrations, `__main__.py`, `_engine_protocol.py`, `durable_execution_kg.py`, `ontology.lock`. |
| `agent_utilities/knowledge_graph/actions/` | 1,837 | delete as duplicate of engine | AU-BOUNDARY-R022 | Deleted once native engine coverage is proven. |
| `agent_utilities/knowledge_graph/adaptation/` | 5,281 | undecided | AU-RETIRE-R001 (inventory obligation only) | Failure analysis, feedback, belief revision, contradiction detection, skill evolution and remediation playbooks. |
| `agent_utilities/knowledge_graph/argumentation/` | 871 | delete as duplicate of engine | AU-BOUNDARY-R022 | Deleted once native engine coverage is proven. |
| `agent_utilities/knowledge_graph/assimilation/` | 4,807 | relocate to epistemic-graph | AU-BOUNDARY-R027 | Named wholesale. |
| `agent_utilities/knowledge_graph/backends/` | 12,810 | split: delete as duplicate of engine (`sparql/`); relocate to epistemic-graph (10 federation backends); keep in AU (`epistemic_graph_backend.py`, per design.md); remainder undecided | AU-BOUNDARY-R032, AU-BOUNDARY-R033, AU-RETIRE-R001 | `base.py` and the 1,065-line `__init__.py` dispatch are not named. `epistemic_graph_backend.py` is kept only by design.md, with no requirement ID. AU-RETIRE-R001 also names `backends/owl/`, which does not exist. |
| `agent_utilities/knowledge_graph/core/` | 69,181 | split: delete as duplicate of engine (33 modules); relocate to epistemic-graph (25 modules); relocate to agent-connector-sdk (6 modules); relocate to graph-os (2 modules); remainder undecided | AU-BOUNDARY-R014, AU-BOUNDARY-R018, AU-BOUNDARY-R019, AU-BOUNDARY-R020, AU-BOUNDARY-R021, AU-BOUNDARY-R022, AU-BOUNDARY-R024, AU-BOUNDARY-R025, AU-BOUNDARY-R030, AU-RETIRE-R001, AU-RETIRE-R005, AU-RETIRE-R008 | 66 of 93 modules are named. The 27 unnamed modules (10,665 lines) include `fleet_catalog_tables.py` (2,790), `source_catalog.py` (1,004), `optimal_execution.py`, `nl_planner.py`, `markov_regime.py`, `ecosystem_topology.py`, `materialization.py`, `table_ingest.py` and `archimate_layer.py`. |
| `agent_utilities/knowledge_graph/distillation/` | 7,491 | undecided | AU-RETIRE-R001 (inventory obligation only) | Skill-graph distillation pipeline, skill synthesizer, deduplicator and an LSH index. |
| `agent_utilities/knowledge_graph/domain_packs/` | 1,650 | relocate; destination not stated | AU-BOUNDARY-R031 | R031 says the directory is moved after the pack loader uses engine-generated types, without naming the receiving repository. |
| `agent_utilities/knowledge_graph/enrichment/` | 18,839 | split: relocate to agent-connector-sdk (extractors, write-back, git history, source adapters); relocate to epistemic-graph (6 deterministic modules); remainder undecided | AU-BOUNDARY-R026, AU-BOUNDARY-R027, AU-RETIRE-R002, AU-RETIRE-R004, AU-RETIRE-R005, AU-RETIRE-R008 | 73 of 93 files have a disposition. `pipeline.py` (1,354) and `models.py` (349) are changed by the AU-RETIRE requirements but given no destination; 18 further modules (4,339 lines) are not named, among them `cards.py`, `ops_causal_graph.py`, `topic_classifier.py`, `synthesize.py` and `orchestration.py`. |
| `agent_utilities/knowledge_graph/etl/` | 1,600 | relocate to epistemic-graph | AU-BOUNDARY-R027 | Named wholesale. |
| `agent_utilities/knowledge_graph/extraction/` | 5,724 | split: relocate to agent-connector-sdk (readers, PDF); relocate to epistemic-graph (`schema_discovery.py`); remainder undecided | AU-BOUNDARY-R025, AU-BOUNDARY-R030 | 10 modules (3,879 lines) are not named: fact extractor, structure router, job manager, candidate claims, ontology grounding, extraction schema and a note-sync module. R023 says AU retains model-backed candidate claims without naming a path. |
| `agent_utilities/knowledge_graph/id_management/` | 369 | delete as duplicate of engine | AU-BOUNDARY-R022 | Deleted once native engine coverage is proven. |
| `agent_utilities/knowledge_graph/infra/` | 711 | relocate to epistemic-graph | AU-BOUNDARY-R028 | Named wholesale. |
| `agent_utilities/knowledge_graph/ingestion/` | 29,521 | split: relocate to epistemic-graph (commit and derivation half); relocate to agent-connector-sdk (10 source modules); `engine.py` dissolved; remainder undecided | AU-BOUNDARY-R023, AU-BOUNDARY-R024, AU-BOUNDARY-R025, AU-RETIRE-R006, AU-RETIRE-R007, AU-RETIRE-R008 | R023 names the directory and lists 14 modules; R024 and R025 name 10 more. 13 modules (2,380 lines) fall under the directory glob only, with no stated owner, among them `skill_classification.py`, `staged_pipeline.py`, `manifest.py`, `batch_orchestrator.py` and the fleet harvest modules. The AU-RETIRE requirements change `engine.py` and `embedding_admission.py` before they move. |
| `agent_utilities/knowledge_graph/integrations/` | 2,419 | split: relocate to agent-connector-sdk (connector certification, attestation and its CLI); delete as duplicate of engine (legacy SPARQL sync and ingestor) | AU-BOUNDARY-R011, AU-BOUNDARY-R032 | Fully covered. |
| `agent_utilities/knowledge_graph/kb/` | 3,702 | split: relocate to agent-connector-sdk (non-model half); model half undecided | AU-BOUNDARY-R025 | R025 moves "the non-model half" without listing which of the 11 files that is. |
| `agent_utilities/knowledge_graph/live_artifacts/` | 414 | undecided | AU-RETIRE-R001 (inventory obligation only) | Live refreshable artifact models, store and refresh logic. |
| `agent_utilities/knowledge_graph/maintenance/` | 1,801 | delete as duplicate of engine | AU-BOUNDARY-R022 | Deleted once native engine coverage is proven. |
| `agent_utilities/knowledge_graph/memory/` | 13,347 | relocate to epistemic-graph | AU-BOUNDARY-R040, AU-BOUNDARY-R027 | Everything moves; `timeseries/` under R027, the rest under R040. AU keeps only a thin engine client. |
| `agent_utilities/knowledge_graph/neural/` | 911 | relocate to epistemic-graph | AU-BOUNDARY-R034 | Named wholesale. |
| `agent_utilities/knowledge_graph/ontology/` | 69,600 | split: relocate to epistemic-graph (object model, emitters behind pack compilation); relocate to agent-connector-sdk (connector manifest files, `sync_conflict`); files kept for AU typing not listed | AU-BOUNDARY-R029, AU-BOUNDARY-R030, AU-BOUNDARY-R011 | Largest directory after `core`. R029 excepts "the files kept for AU's own semantic typing" without listing them. `leanix_metamodel` is sent to the SDK by R029 and behind engine pack compilation by R030. |
| `agent_utilities/knowledge_graph/orchestration/` | 6,187 | split: delete as duplicate of engine (8 `engine_*` modules); remainder undecided | AU-BOUNDARY-R018 | 5 files (1,300 lines) are not named: research orchestrator, research subagent, data analyst and a budget controller. |
| `agent_utilities/knowledge_graph/pipeline/` | 3,925 | undecided | AU-RETIRE-R001 (inventory obligation only) | Phased pipeline: scan, parse, embedding, SHACL check, sync, communities, centrality; plus document ingest, update and deletion. |
| `agent_utilities/knowledge_graph/quantum/` | 276 | relocate to epistemic-graph | AU-BOUNDARY-R028 | Named wholesale. |
| `agent_utilities/knowledge_graph/research/` | 19,712 | split: relocate to epistemic-graph (`placement_mining.py`); remainder undecided | AU-BOUNDARY-R028, AU-RETIRE-R001 (inventory obligation only) | 36 of 37 files (17,567 lines) are not named: research loop controller (4,431 lines), change publisher, evolution state, evidence, promotion governance and an artifact compiler subpackage. |
| `agent_utilities/knowledge_graph/retrieval/` | 49,120 | split: relocate to epistemic-graph (18 retrieval engines); keep in AU (context compilation, capability index); remainder undecided | AU-BOUNDARY-R034 | 32 files (39,549 lines) are not named; 29,070 of those lines are one generated JSON catalog. R034 keeps "context compilation and the capability index" without listing files; the remaining modules (context planes, query analysis, evaluation corpus, score checks) have no stated owner. |
| `agent_utilities/knowledge_graph/search/` | 2,473 | relocate to epistemic-graph | AU-BOUNDARY-R034 | Named wholesale. |
| `agent_utilities/knowledge_graph/search_synthesis/` | 786 | undecided | AU-RETIRE-R001 (inventory obligation only) | Search-task synthesis over the graph: question formulation, evidence subgraph, shortcut-risk checks. |
| `agent_utilities/knowledge_graph/security/` | 2,462 | relocate to epistemic-graph | AU-BOUNDARY-R028 | Named wholesale. |
| `agent_utilities/knowledge_graph/setup/` | 685 | delete as duplicate of engine | AU-BOUNDARY-R032 | Legacy external-store setup; the `setup-databases` script goes with it. |
| `agent_utilities/knowledge_graph/shapes/` | 670 | undecided | AU-RETIRE-R001 (inventory obligation only) | Eight SHACL shape files in Turtle; no Python. |
| `agent_utilities/knowledge_graph/standardization/` | 1,322 | split: relocate to epistemic-graph; `drift.py` to agent-connector-sdk sync and the engine | AU-BOUNDARY-R028, AU-BOUNDARY-R035 | R028 moves everything except drift handling; R035 places drift handling. |
| `agent_utilities/knowledge_graph/streams/` | 504 | relocate to epistemic-graph | AU-BOUNDARY-R027 | Named wholesale. |
| `agent_utilities/kvcache/` | 6,097 | undecided | — | Remote key-value cache connector for the engine: eligibility, tiering, remote backend, checkpointing. |
| `agent_utilities/mcp/` | 70,511 | split: relocate to graph-os (legacy host, action manifest, multiplexer family, host-only modules); relocate to agent-connector-sdk (9 toolkit modules); keep in AU (`toolset_factory`, `tool_specs`, `agent_manager`); remainder undecided | AU-BOUNDARY-R003, AU-BOUNDARY-R004, AU-BOUNDARY-R010 | Count includes `mcp/tools`. 38 files sit directly here; 12 (2,800 lines) are not named: an environment-variable drift checker (1,625 lines), README generators, and small type modules. |
| `agent_utilities/mcp/tools/` | 34,738 | relocate to graph-os | AU-BOUNDARY-R003 | Named wholesale as `mcp/tools/**`; every action is mapped to a graph-os operation or dropped with a reason. |
| `agent_utilities/measurement/` | 1,611 | undecided | — | Measurement harness for benchmark and check runs: provenance, load checks, copy integrity. |
| `agent_utilities/media/` | 1,695 | undecided | — | Media sidecar delegation for audio, image, PDF and video; all 7 modules import `knowledge_graph`. |
| `agent_utilities/messaging/` | 13,819 | split: relocate to graph-os (service, daemon, listener, routing); relocate to epistemic-graph (durable bus and log); keep in AU (`bus_log.py` adapter) | AU-BOUNDARY-R017 | Only `bus_log.py` is named by path. The other 24 direct modules (9,200 lines) are assigned by role, not by file. The two `orchestration/` modules R017 says AU keeps do not exist. |
| `agent_utilities/messaging/backends/` | 3,423 | relocate to graph-os | AU-BOUNDARY-R017 | The 17 channel adapters; assigned by role ("channel adapters"), not by path. |
| `agent_utilities/models/` | 18,833 | split: relocate to epistemic-graph (graph-schema DTOs replaced by generated types); remainder undecided | AU-BOUNDARY-R031 | 10 direct modules plus `schema_packs/` are named. 17 direct files (4,694 lines) are not: model registry, company and goal models, execution manifest, model profile, SDD models. |
| `agent_utilities/models/domains/` | 438 | undecided | — | Domain-specific graph models (finance, infrastructure, enterprise, RLM). |
| `agent_utilities/models/schema_packs/` | 505 | relocate to epistemic-graph | AU-BOUNDARY-R031 | Named wholesale. |
| `agent_utilities/numeric/` | 789 | undecided | — | One 789-line thin adapter over the engine's numeric kernel. |
| `agent_utilities/observability/` | 17,558 | split: relocate to graph-os (health and metrics modules); relocate to epistemic-graph (3 modules); keep in AU (telemetry emission, per design.md); files not listed | AU-BOUNDARY-R014, AU-BOUNDARY-R028 | Only `trace_ontology`, `self_ingest` and `audit_logger` are named. The other 29 files (15,876 lines) are covered by description only: R014 says "health and metrics modules" and design.md says AU retains telemetry emission. |
| `agent_utilities/ontology/` | 189 | undecided | — | No direct files; see child row. |
| `agent_utilities/ontology/shapes/` | 189 | undecided | — | One SHACL shape file in Turtle. |
| `agent_utilities/orchestration/` | 41,801 | undecided | — | Largest directory with no requirement: 55 modules. Agent runner (6,348 lines), dispatch and activation workers, durable execution, action policy, plus fleet reconciler, autoscaler, actuation, scaling authorities and deploy watch. 23 modules import `knowledge_graph`. |
| `agent_utilities/patterns/` | 1,680 | undecided | — | Agentic design-pattern implementations (prompt chain, exploration, prioritization, test-driven loops). |
| `agent_utilities/policies/` | 562 | undecided | — | A 15-line `__init__.py`; see child row. |
| `agent_utilities/policies/engineering_rules/` | 547 | undecided | — | 13 Markdown rule summaries; no Python. |
| `agent_utilities/pricing/` | 505 | undecided | — | Model pricing catalog and normalization. |
| `agent_utilities/prompting/` | 2,667 | undecided | — | Structured prompt builder, provider adapter and a JSON schema. |
| `agent_utilities/prompts/` | 4,576 | undecided | — | About 97 agent role prompts as JSON, plus one loader module. |
| `agent_utilities/protocols/` | 24,023 | split by child row; direct files: relocate to graph-os (A2A, ACP, AG-UI); delete (`a2a_epistemic.py`, `universal_connector.py`) | AU-BOUNDARY-R011, AU-BOUNDARY-R015 | All 7 non-empty direct modules are named. Count includes the four child rows. |
| `agent_utilities/protocols/enterprise/` | 436 | relocate to epistemic-graph | AU-BOUNDARY-R028 | Named wholesale. |
| `agent_utilities/protocols/epistemic_operations/` | 6,277 | delete as duplicate of engine | AU-BOUNDARY-R012 | The second engine projection. |
| `agent_utilities/protocols/source_connectors/` | 11,532 | relocate to agent-connector-sdk | AU-BOUNDARY-R011 | Deleted in AU once the SDK owns cursors, conflict handling, mapping and certification. |
| `agent_utilities/protocols/voice_supply_chain/` | 708 | undecided | — | Voice-model acquisition, manifest and license registry. |
| `agent_utilities/rlm/` | 7,848 | undecided | — | Recursive language model runtime: REPL, prompt optimizer, evaluation-set optimizer. Count includes the two child rows. |
| `agent_utilities/rlm/benchmarks/` | 966 | undecided | — | Long-context benchmark tasks and scoreboard. |
| `agent_utilities/rlm/sandboxes/` | 3,377 | undecided | — | Code sandbox backends (container, microVM, WASM, local) behind one contract. |
| `agent_utilities/runtime/` | 3,981 | undecided | — | Developer-workspace runtime for the coding agent: workspaces, events, bridge, computer-use tier. |
| `agent_utilities/runtime/run_vcs/` | 1,288 | undecided | — | Version control of agent runs: kernel, commits, replay. |
| `agent_utilities/sdd/` | 2,573 | split: relocate to agent-connector-sdk (`watcher.py`); remainder undecided | AU-BOUNDARY-R025, AU-BOUNDARY-R003 | R003 first removes one function from `watcher.py`. The 1,294-line `__init__.py` and `orchestrator.py` are not named. |
| `agent_utilities/security/` | 15,365 | split: relocate to epistemic-graph (admission, permissions, entitlements); relocate to graph-os (request identity, auth, error surface, middleware); relocate to agent-connector-sdk (duplicated modules, not listed); remainder undecided | AU-BOUNDARY-R010, AU-BOUNDARY-R014, AU-BOUNDARY-R021 | 14 of 42 direct modules are named or identifiable. R010 says "the duplicated `security/**` modules" without listing them, so the other 28 (7,847 lines) cannot be assigned: guardrails, tool guard, secrets client, credential providers, sandboxed executor, threat defense. |
| `agent_utilities/security/conformance/` | 565 | relocate to graph-os | AU-BOUNDARY-R014 | Named wholesale. |
| `agent_utilities/server/` | 7,223 | relocate to graph-os | AU-BOUNDARY-R015 | Named wholesale as `server/**`. |
| `agent_utilities/server/routers/` | 3,626 | relocate to graph-os | AU-BOUNDARY-R015 | Covered by the parent glob. |
| `agent_utilities/skills/` | 15,911 | split: relocate to graph-os (validation and runtime-harness code, by description); skill content undecided | AU-BOUNDARY-R002 | R002 says "the moved `skills/` subpackages" without listing them. Four files sit directly here (5,756 lines): two validation modules and a profile. |
| `agent_utilities/skills/agent-utilities-deployment/` | 242 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/agent-utilities-development/` | 853 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/agent-utilities-evolution/` | 123 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/agent-utilities-self-evolution/` | 1,401 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/agent-utilities-source-integration/` | 190 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/autonomous-contribution/` | 51 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/fleet_harness/` | 1,202 | relocate to graph-os | AU-BOUNDARY-R002 | The only Python subpackage under `skills/`; matched to R002 by description, not by path. |
| `agent_utilities/skills/graph-engine-and-modalities/` | 168 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/graph-ingestion-and-integration/` | 557 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/graph-modeling-and-mutation/` | 165 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/graph-orchestration-and-automation/` | 153 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/graph-query-and-explanation/` | 631 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/graph-research-and-analysis/` | 178 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/graph-runtime-and-governance/` | 201 | undecided | — | Skill content (Markdown instructions, agent YAML, assets); no Python package. |
| `agent_utilities/skills/skill_graphs/` | 1,327 | undecided | — | Skill content; includes one deployment script. |
| `agent_utilities/skills/workflows/` | 2,713 | undecided | — | 26 files of workflow skill content, including Helm chart assets. |
| `agent_utilities/tools/` | 6,344 | undecided | — | Agent tool implementations (knowledge, code intelligence, git, workspace, scheduler, memory). 8 of 32 modules import `knowledge_graph`. |
| `agent_utilities/tools/browser/` | 878 | undecided | — | Browser automation tools. |
| `agent_utilities/usage/` | 2,113 | relocate to epistemic-graph | AU-BOUNDARY-R036 | The local store moves; AU keeps only event emission. |
| `agent_utilities/usage/backends/` | 977 | relocate to epistemic-graph | AU-BOUNDARY-R036 | The SQLite and SQL analytics stores. |
| `agent_utilities/workflows/` | 3,240 | undecided | — | Workflow catalog and runner, skill compiler and an engine sync module. |

### Requirements that own the open rows

The rows marked undecided or covered only in part are not left to chance. Each is owned by a
requirement that must record the decision and prove it:

- `AU-BOUNDARY-R042` — the inventory itself: every directory has one disposition and the tree check enforces it.
- `AU-BOUNDARY-R043` — agent execution, evaluation, runtime tooling, prompts, pricing and policy packages.
- `AU-BOUNDARY-R044` — fleet, scaling and deployment modules inside `orchestration/`.
- `AU-BOUNDARY-R045` — `knowledge_graph/` packages with no disposition or without file-level owners.
- `AU-BOUNDARY-R046` — the other partly covered packages, file by file.
- `AU-BOUNDARY-R047` — `kvcache/`, `numeric/`, `caching/` and `media/`.
- `AU-BOUNDARY-R048` — domain models, shapes, assets and skill content.

The path corrections listed below for messaging, the finance task module and the duplicate-engine
modules have been applied to the requirement text.

### Directories with no covering requirement

No requirement in either spec names these directories or any file in them. Line counts here exclude child directories that have their own entry, so the figures add up: 63 directories, about 163,600 lines.

| Group | Directories (lines) |
|---|---|
| Agent execution and orchestration | `orchestration/` (41,801), `graph/` (21,326), `graph/planning/` (486), `graph/reactive/` (844), `graph/reasoning/` (1,898), `graph/routing/` (1,871), `agent/` (2,203), `capabilities/` (5,699), `capabilities/compaction/` (100), `core/checkpoint/` (769), `core/execution/` (984), `patterns/` (1,680), `workflows/` (3,240), `tools/` (5,466), `tools/browser/` (878) |
| Evaluation and self-improvement | `harness/` (27,128), `harness/memorydata/` (1,836), `measurement/` (1,611), `rlm/` (3,505), `rlm/benchmarks/` (966), `rlm/sandboxes/` (3,377) |
| Runtime and developer tooling | `runtime/` (2,693), `runtime/run_vcs/` (1,288), `cli/` (1,190), `claude_harness/` (930), `integrations/` (257), `analysis/` (287), `agent_chat/` (54) |
| Engine-facing adapters and caches | `kvcache/` (6,097), `numeric/` (789), `caching/` (898), `media/` (1,695) |
| Prompts, pricing and policy data | `prompts/` (4,576), `prompting/` (2,667), `pricing/` (505), `policies/` (15), `policies/engineering_rules/` (547) |
| Domain models | `domains/` (53), `domains/government/` (92), `domains/hr/` (570), `domains/law/` (134), `domains/medical/` (107), `models/domains/` (438) |
| Shapes, assets and data | `ontology/` (0), `ontology/shapes/` (189), `images/` (208), `data/` (16), `protocols/voice_supply_chain/` (708) |
| Skill content | `skills/agent-utilities-deployment/` (242), `skills/agent-utilities-development/` (853), `skills/agent-utilities-evolution/` (123), `skills/agent-utilities-self-evolution/` (1,401), `skills/agent-utilities-source-integration/` (190), `skills/autonomous-contribution/` (51), `skills/graph-engine-and-modalities/` (168), `skills/graph-ingestion-and-integration/` (557), `skills/graph-modeling-and-mutation/` (165), `skills/graph-orchestration-and-automation/` (153), `skills/graph-query-and-explanation/` (631), `skills/graph-research-and-analysis/` (178), `skills/graph-runtime-and-governance/` (201), `skills/skill_graphs/` (1,327), `skills/workflows/` (2,713) |

Six `knowledge_graph/` directories are covered only by the AU-RETIRE-R001 obligation to assign a disposition, and no requirement assigns one (18,567 lines): `knowledge_graph/distillation/` (7,491), `knowledge_graph/adaptation/` (5,281), `knowledge_graph/pipeline/` (3,925), `knowledge_graph/search_synthesis/` (786), `knowledge_graph/shapes/` (670), `knowledge_graph/live_artifacts/` (414).

### Directories covered only in part

A requirement names some files in these directories; the remainder has no stated owner. Counts are files directly in the directory.

| Directory | Files without a stated owner | Lines | What is missing |
|---|---:|---:|---|
| `knowledge_graph/retrieval/` | 32 | 39,549 | 29,070 lines are one generated JSON catalog. AU-BOUNDARY-R034 keeps "context compilation and the capability index" without listing files. |
| `knowledge_graph/research/` | 36 | 17,567 | Only `placement_mining.py` is named. |
| `observability/` | 29 | 15,876 | Assigned by description only (AU-BOUNDARY-R014, design.md); three files are named by AU-BOUNDARY-R028. |
| `core/` | 35 | 15,441 | Model, provider, scheduler, session and embedding modules. |
| `knowledge_graph/core/` | 27 | 10,665 | The modules listed in the table note. |
| `domains/finance/` | 33 | 9,948 | Outside these two specs; the finance cut is owned by [`agent-context-and-finance`](../agent-context-and-finance/spec.md). |
| `messaging/` | 24 | 9,200 | AU-BOUNDARY-R017 assigns by role; only `bus_log.py` is named. |
| `security/` | 28 | 7,847 | AU-BOUNDARY-R010 says "the duplicated `security/**` modules" without listing them. |
| `skills/` and `skills/fleet_harness/` | 10 | 6,958 | AU-BOUNDARY-R002 says "the moved `skills/` subpackages" without listing them. |
| `knowledge_graph/enrichment/` | 20 | 6,042 | Includes `pipeline.py` and `models.py`, which the AU-RETIRE requirements change but do not place. |
| `models/` | 17 | 4,694 | Non-graph models. |
| `knowledge_graph/extraction/` | 10 | 3,879 | Model-backed extraction; AU-BOUNDARY-R023 implies retention without a path. |
| `knowledge_graph/` (direct files) | 12 | 3,716 | Workflow store, compilers, migrations. |
| `knowledge_graph/kb/` | 11 | 3,702 | AU-BOUNDARY-R025 moves "the non-model half" without a file list. |
| `messaging/backends/` | 18 | 3,423 | Assigned to graph-os by role ("channel adapters"), not by path. |
| `mcp/` | 12 | 2,800 | Drift checker, README generators, type modules. |
| `knowledge_graph/ingestion/` | 13 | 2,380 | Covered by the directory glob but in neither the commit/derivation list nor the source list. |
| `ecosystem/` | 9 | 2,237 | Governance workflow, auditors, hooks. |
| `knowledge_graph/backends/` | 4 | 2,218 | `epistemic_graph_backend.py` is kept by design.md only; `base.py` and `__init__.py` are not named. |
| `knowledge_graph/domain_packs/` | 6 | 1,650 | AU-BOUNDARY-R031 moves it without naming the destination. |
| `automation/` | 2 | 1,524 | `research_pipeline.py`. |
| `sdd/` | 3 | 1,428 | `__init__.py`, `orchestrator.py`, README. |
| `core/registry/` | 5 | 1,316 | Service, package and plugin adapters. |
| `knowledge_graph/orchestration/` | 5 | 1,300 | Research orchestrator, subagent, data analyst. |
| `agent_utilities/` (top-level files) | 10 | 681 | Package metadata and small helpers. |
| `knowledge_graph/ontology/` | not listed | — | AU-BOUNDARY-R029 excepts "the files kept for AU's own semantic typing" without listing them (directory total 69,600 lines). |

### Requirements that name paths not in the tree

| Requirement | Path as written | Finding |
|---|---|---|
| AU-RETIRE-R001 | `core/owl_bridge.py` | No such file; no tracked file under `agent_utilities/` has `owl` in its name. |
| AU-RETIRE-R001 | `backends/owl/` | No such directory; `knowledge_graph/backends/` contains only `sparql/` and `contrib/` as subdirectories. |
| AU-BOUNDARY-R017 | `orchestration/agent_bus.py` | No such file. The `AgentBus` class is in `messaging/bus.py`. |
| AU-BOUNDARY-R017 | `orchestration/messaging_handler.py` | No such file. `ActionPolicy` is defined in `orchestration/action_policy.py`. |
| AU-BOUNDARY-R017 | `agent_utilities.api.messaging` | Does not exist yet; `api/` has no messaging module. This is a module the requirement creates. |
| AU-BOUNDARY-R035 | `candidate.py`, `contract_store`, `activation` | No schema-drift package with these modules is in the tree. The only drift module is `knowledge_graph/standardization/drift.py`. The requirement is written for a package that has not been merged. |
| AU-BOUNDARY-R002 | "the moved `skills/` subpackages" | `deployment/` has no `skills/` subdirectory. The only match is the top-level `agent_utilities/skills/`, whose single Python subpackage is `fleet_harness/`. |
| AU-BOUNDARY-R041 | `quant_tasks` | Not in `domains/finance/`; the only file of that name is `knowledge_graph/core/quant_tasks.py`. |
| AU-BOUNDARY-R041 | candle rule sets, Merton-KMV specifics, sentiment lexicon | These are not modules. The content sits inside `domains/finance/` modules the requirement does not list (`credit_quality.py`, `pattern_classifier.py`, `visual_ta.py`, `kronos_forecaster.py`, `sentiment_fusion.py`). |

Two further inconsistencies, where the paths exist:

- `leanix_metamodel` is relocated to agent-connector-sdk by AU-BOUNDARY-R029 and placed behind engine pack compilation by AU-BOUNDARY-R030.
- AU-BOUNDARY-R030 writes `core/ontology_publisher.py` and `extraction/schema_discovery.py`, AU-BOUNDARY-R026 writes `enrichment/git_history` and `enrichment/source_adapters`, and the AU-RETIRE requirements write `core/…`, `enrichment/…` and `ingestion/…`; all of these exist only under `agent_utilities/knowledge_graph/`.

Every console script and packaging extra named by a requirement (`graph-os`, `graph-os-daemon`, `agent-utilities-acp`, `agent-utilities-messaging`, `kg-ingest-worker`, `graph-os-analytics-worker`, `setup-databases`, `graph-os-certify-connector`, `gateway-widgets`) is declared in `pyproject.toml`. All other paths named by the two specs resolve to tracked files.
