# graph-os Capability Power Descriptors (generated)

> **GENERATED — do not edit by hand.** Regenerate with `python3 scripts/gen_capability_power.py --write`; `scripts/check_cpd.py` gates drift in CI/pre-commit (CONCEPT:AU-KG.retrieval.capability-power-descriptor). Seam 8 Phase 1 — `plans/program-design-2026-07-11-epistemic-tool-routing.md` section 2b.
>
> 95 capabilities · generated 2026-07-11T22:23:17Z. Every field is derived from a live source (the MCP tool registry, the generated graph-os action manifest, the EG-P0-1 capability ledger, transcribed measured benchmarks) — an empty field means the source had no answer, never a fabricated one.

## Index

| Capability | Intent verbs | One-line power | Actions | REST |
|---|---|---|---:|---|
| [`ask_data`](#askdata) | ask | answer a DATA question over the Knowledge Graph with a DB-GPT-style, multi-step data-analysis agent  | 1 | `/graph/ask-data` |
| [`concept_registry`](#conceptregistry) | act | Atomically claim/list/release concept ids across parallel sessions & worktrees (CONCEPT:AU-OS.govern | 1 | `/concept/registry` |
| [`document_process`](#documentprocess) | ask | Document → ontology processing (CONCEPT:AU-KG.ingest.chunk-overlap-stage): extract → chunk(overlap)  | 1 | `/document/process` |
| [`engine_admin`](#engineadmin) | ask | Low-level epistemic-graph engine surface for the 'admin' domain (ops/maintenance: online backup + re | 2 | `/engine/admin` |
| [`engine_analytics`](#engineanalytics) | ask | Low-level epistemic-graph engine surface for the 'analytics' domain (centrality + (personalized) Pag | 5 | `/engine/analytics` |
| [`engine_blob`](#engineblob) | ask | Low-level epistemic-graph engine surface for the 'blob' domain (streamed content-addressed media blo | 11 | `/engine/blob` |
| [`engine_broker`](#enginebroker) | ask | Low-level epistemic-graph engine surface for the 'broker' domain (native message broker: exchange/qu | 21 | `/engine/broker` |
| [`engine_channels`](#enginechannels) | ask | Low-level epistemic-graph engine surface for the 'channels' domain (dynamic agent communication chan | 8 | `/engine/channels` |
| [`engine_consensus`](#engineconsensus) | ask | Low-level epistemic-graph engine surface for the 'consensus' domain (zero-trust identity + multisig  | 2 | `/engine/consensus` |
| [`engine_datascience`](#enginedatascience) | ask | Low-level epistemic-graph engine surface for the 'datascience' domain (estimators + primitives + tra | 15 | `/engine/datascience` |
| [`engine_edges`](#engineedges) | ask | Low-level epistemic-graph engine surface for the 'edges' domain (edge CRUD, temporal invalidate/supe | 9 | `/engine/edges` |
| [`engine_finance`](#enginefinance) | ask | Low-level epistemic-graph engine surface for the 'finance' domain (quantitative finance (optimize/ri | 67 | `/engine/finance` |
| [`engine_graph`](#enginegraph) | ask | Low-level epistemic-graph engine surface for the 'graph' domain (graph algorithms, AST parse/index,  | 29 | `/engine/graph` |
| [`engine_graphlearn`](#enginegraphlearn) | ask | Low-level epistemic-graph engine surface for the 'graphlearn' domain (KAN graph-learning: fit/predic | 2 | `/engine/graphlearn` |
| [`engine_ledger`](#engineledger) | ask | Low-level epistemic-graph engine surface for the 'ledger' domain (audit ledger get/clear/apply). | 3 | `/engine/ledger` |
| [`engine_lifecycle`](#enginelifecycle) | manage | Low-level epistemic-graph engine surface for the 'lifecycle' domain (prune/decay/evict, batch_update | 10 | `/engine/lifecycle` |
| [`engine_mining`](#enginemining) | ask | Low-level epistemic-graph engine surface for the 'mining' domain (association-rule mining (Apriori/F | 18 | `/engine/mining` |
| [`engine_nodes`](#enginenodes) | ask | Low-level epistemic-graph engine surface for the 'nodes' domain (node CRUD, batch/union reads, degre | 20 | `/engine/nodes` |
| [`engine_query`](#enginequery) | ask | Low-level epistemic-graph engine surface for the 'query' domain (SQL / Cypher / GraphQL / UQL / unif | 21 | `/engine/query` |
| [`engine_rbac`](#enginerbac) | manage | Low-level epistemic-graph engine surface for the 'rbac' domain (RBAC policy administration: roles +  | 5 | `/engine/rbac` |
| [`engine_rdf`](#enginerdf) | ask | Low-level epistemic-graph engine surface for the 'rdf' domain (RDF triples + SPARQL + OWL reasoning) | 9 | `/engine/rdf` |
| [`engine_reasoning`](#enginereasoning) | ask | Low-level epistemic-graph engine surface for the 'reasoning' domain (forward-chaining OWL/RDFS Datal | 1 | `/engine/reasoning` |
| [`engine_resharding`](#engineresharding) | manage | Low-level epistemic-graph engine surface for the 'resharding' domain (M3 catalog/reshard/rebalance a | 7 | `/engine/resharding` |
| [`engine_streaming`](#enginestreaming) | ask | Low-level epistemic-graph engine surface for the 'streaming' domain (CDC / continuous queries / watc | 9 | `/engine/streaming` |
| [`engine_tenants`](#enginetenants) | manage | Low-level epistemic-graph engine surface for the 'tenants' domain (multi-tenant graph create/delete/ | 3 | `/engine/tenants` |
| [`engine_timeseries`](#enginetimeseries) | ask | Low-level epistemic-graph engine surface for the 'timeseries' domain (native TSDB append/range/windo | 6 | `/engine/timeseries` |
| [`engine_txn`](#enginetxn) | ask | Low-level epistemic-graph engine surface for the 'txn' domain (server-side OCC ACID transactions). | 17 | `/engine/txn` |
| [`graph_analyze`](#graphanalyze) | ask | Ops / structural analysis over the KG. | 68 | `/graph/analyze` |
| [`graph_ask`](#graphask) | ask | ask the Knowledge Graph in plain English. | 1 | `/graph/ask` |
| [`graph_broker`](#graphbroker) | ask | the epistemic-graph engine message broker (AMQP-style exchanges + queues + streams), distinct from t | 1 | `/graph/broker` |
| [`graph_bus`](#graphbus) | ask | the agent-to-agent communication bus: let this session talk to other Claude/LLM sessions (any provid | 1 | `/graph/bus` |
| [`graph_code`](#graphcode) | ask | Understand a CODEBASE via the ingested code graph — query this before grep. | 1 | `/graph/code` |
| [`graph_code_nav`](#graphcodenav) | ask | Navigate the resolved code graph (CONCEPT:AU-KG.backend.declared-columns-so-schema). action: 'find_d | 1 | `/graph/code-nav` |
| [`graph_configure`](#graphconfigure) | manage | Manage backend configurations, system credentials, and tool registration within the unified agent ec | 31 | `/graph/configure` |
| [`graph_context`](#graphcontext) | act, ask | store/fetch curated context for invoker→spawned-agent handoff, persisted in the epistemic-graph so a | 4 | `/graph/context` |
| [`graph_document_tree`](#graphdocumenttree) | ask, find | Reasoning-tree (vectorless) document retrieval over a per-document section tree (CONCEPT:AU-KG.retri | 4 | `/graph/document-tree` |
| [`graph_etl`](#graphetl) | write | Unified ETL pipeline between systems over the canonical KG hub (CONCEPT:AU-KG.ontology.one-source). | 2 | `/graph/etl` |
| [`graph_evaluate`](#graphevaluate) | why | Evaluate agents/harnesses and reason over learned world models. | 1 | `/graph/evaluate` |
| [`graph_explain`](#graphexplain) | why | The UNIVERSAL context plane (CONCEPT:AU-KG.retrieval.route-question-its-domain): route a question to | 1 | `/graph/explain` |
| [`graph_federated_search`](#graphfederatedsearch) | ask | federated search fanned across registered external graph references. | 1 | `/graph/federated-search` |
| [`graph_feedback`](#graphfeedback) | write | Record a human correction so the brain learns: correction_type 'outcome' adjusts an entity's reward, | 1 | `/graph/feedback` |
| [`graph_feeds`](#graphfeeds) | manage | Manage the unified RSS/Atom feed registry (CONCEPT:AU-KG.ingest.rss-feed-connector/2.122). | 4 | `/graph/feeds` |
| [`graph_fork`](#graphfork) | ask | warm-fork fan-out over the ORCH-1.86..93 warm-fork primitive (LMCache KV / copy-on-write sandboxes): | 1 | `/graph/fork` |
| [`graph_gis`](#graphgis) | ask | the engine's GIS surface. | 1 | `/graph/gis` |
| [`graph_goals`](#graphgoals) | act, manage | Orchestrate background/autonomous loops (action in 'create', 'list', 'iterations', 'cancel'). | 4 | `/graph/goals` |
| [`graph_hydrate`](#graphhydrate) | manage | Hydrate the Knowledge Graph from configured external sources. | 1 | `/graph/hydrate` |
| [`graph_ingest`](#graphingest) | write | Smart ingestion for codebases, documents, directories, and conversation logs. | 38 | `/graph/ingest` |
| [`graph_kvcache`](#graphkvcache) | ask | the engine's shared, content-addressed KV-cache over the EG-187 HTTP surface, driven through the KG- | 5 | `/graph/kvcache` |
| [`graph_learn`](#graphlearn) | ask | a pure-Rust KAN (Kolmogorov-Arnold) link-predictor over the resident graph, whose learned per-featur | 2 | `/graphlearn/fit` |
| [`graph_loops`](#graphloops) | act, manage | The single entrypoint for long-running objectives (CONCEPT:AU-KG.research.these-properties-carry). | 9 | `/graph/loops` |
| [`graph_memory`](#graphmemory) | ask | the engine's EG-318 memory surface: episodic→semantic memory, the spatial scene graph, and RL trajec | 3 | `/graph/memory` |
| [`graph_message`](#graphmessage) | act, ask | bidirectional, cross-process, ordered message channel between an invoking agent and a spawned agent, | 5 | `/graph/message` |
| [`graph_mine`](#graphmine) | ask | the unified data-mining surface over the engine, compute-near-data (mining runs where the graph live | 10 | `/mining/associate` |
| [`graph_mine_deep`](#graphminedeep) | ask | the deep-learning / heavy-Python family the engine core deliberately does NOT implement (no torch/GP | 5 | `/mining/deep/deep_forecast` |
| [`graph_observe`](#graphobserve) | why | Reason over the KG-native observability subgraph — traces, online-scores, assertion verdicts, genera | 3 | `/graph/observe` |
| [`graph_ontology`](#graphontology) | manage | Hosted-ontology lifecycle CRUD (CONCEPT:AU-KG.ontology.manage-arbitrary) — manage arbitrary OWL/RDF  | 11 | `/graph/ontology` |
| [`graph_ops_causal`](#graphopscausal) | ask | Enterprise operations causal graph (Codex X-2): joins Langfuse traces -> agent/tool/model -> service | 5 | `/ops/causal` |
| [`graph_orchestrate`](#graphorchestrate) | act | Orchestrate multi-agent workflows, dispatch subagents, and manage execution loops. | 34 | `/graph/orchestrate` |
| [`graph_promql`](#graphpromql) | ask | query the engine's observability metrics with PromQL. action='instant' (a single evaluation at 'time | 2 | `/graph/promql` |
| [`graph_query`](#graphquery) | ask | Execute a read-only Cypher query against the Knowledge Graph. | 1 | `/graph/query` |
| [`graph_reach`](#graphreach) | ask | reach the user over a messaging backend (Telegram, Slack, Discord, ...). | 5 | `/graph/reach` |
| [`graph_research`](#graphresearch) | ask | Run the research/assimilation pipeline. | 1 | `/graph/research` |
| [`graph_runvcs`](#graphrunvcs) | act, manage | Agent-native run version-control (CONCEPT:AU-ORCH.runvcs.run-commit): fork, revert and review a LIVE | 7 | `/graph/runvcs` |
| [`graph_sandbox`](#graphsandbox) | act, manage | Inspect and control the native warm-fork sandbox runtime (CONCEPT:AU-ORCH.sandbox.graph-sandbox-surf | 3 | `/graph/sandbox` |
| [`graph_schedules`](#graphschedules) | act, manage | Inspect and control the unified scheduler (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent). | 6 | `/graph/schedules` |
| [`graph_search`](#graphsearch) | ask | Search the Knowledge Graph using multiple strategies (hybrid, concept, analogy, memory, discover, dc | 1 | `/graph/search` |
| [`graph_search_synthesis`](#graphsearchsynthesis) | ask | Synthesize a shortcut-resistant deep-search task from the evidence graph, or diagnose realized searc | 2 | `/graph/search-synthesis` |
| [`graph_secret`](#graphsecret) | manage | Manage secrets (CONCEPT:AU-OS.identity.encrypted-secret-store) in the durable, engine-encrypted __se | 4 | `/graph/secret` |
| [`graph_sessions`](#graphsessions) | manage | Manage durable sessions (action in 'list', 'get', 'delete', 'reply', 'cancel'). | 5 | `/graph/sessions` |
| [`graph_share`](#graphshare) | manage | Share a private node (CONCEPT:AU-KG.compute.data-is-private-its). | 4 | `/graph/share` |
| [`graph_table`](#graphtable) | ask, write | mirror data into native engine SQL tables (DataFusion + pg-wire) and manage them. | 6 | `/graph/table` |
| [`graph_traces`](#graphtraces) | ask | search or fetch distributed traces from the engine's observability surface. action='search' (filter  | 2 | `/graph/traces` |
| [`graph_write`](#graphwrite) | write | Write nodes, relationships, or register external graphs to the Knowledge Graph. | 14 | `/graph/write` |
| [`graph_writeback`](#graphwriteback) | write | Backfeed KG-derived knowledge into an external system-of-record (CONCEPT:EG-KG.storage.nonblocking-c | 1 | `/graph/writeback` |
| [`ingest_sessions`](#ingestsessions) | write | Ingest AI agent chat/session history into the usage store + KG (CONCEPT:AU-ECO.mcp.client-side-chat- | 3 | `/usage/ingest-sessions` |
| [`nl_query`](#nlquery) | ask | ask the Knowledge Graph in plain English, planned by agent-utilities' OWN configured fleet LLM (the  | 1 | `/graph/nl-query` |
| [`object_edits`](#objectedits) | ask | Durable object-edit ledger (CONCEPT:AU-KG.ontology.edit-ledger-writeback): record a structured edit  | 4 | `/object/edits` |
| [`object_index`](#objectindex) | ask | Object Index Lifecycle / Object Data Funnel (CONCEPT:AU-KG.ontology.batch-incremental-sync-live): ba | 3 | `/object/index` |
| [`object_permissioning`](#objectpermissioning) | ask | Fine-grained object permissioning (CONCEPT:AU-KG.ontology.redact-object-materialize-restricted): red | 3 | `/object/permissioning` |
| [`object_set`](#objectset) | ask | Object Set Service (CONCEPT:AU-KG.ontology.link-type-pivot/2.38): search/filter/search_around/pivot/ | 9 | `/object/set` |
| [`ontology_derive`](#ontologyderive) | ask | Compute derived (function/cypher/sparql/embedding-backed) properties live at read time (CONCEPT:AU-K | 4 | `/ontology/derive` |
| [`ontology_function`](#ontologyfunction) | ask | Typed, versioned ontology functions: list or invoke through the governed runtime (CONCEPT:AU-KG.onto | 2 | `/ontology/function` |
| [`ontology_interface`](#ontologyinterface) | ask | Ontology interfaces: resolve implementers (targeting), check conformance, or emit OWL (CONCEPT:AU-KG | 4 | `/ontology/interface` |
| [`ontology_leanix_sync`](#ontologyleanixsync) | ask | Discover the live LeanIX metamodel and mirror it natively as OWL/RDF: regenerates ontology_leanix.tt | 1 | `/ontology/leanix-sync` |
| [`ontology_link_materialize`](#ontologylinkmaterialize) | ask | Reify a many-to-many ontology link as a (junction_node, edge_a, edge_b) triple and write it (CONCEPT | 1 | `/ontology/link-materialize` |
| [`ontology_property_types`](#ontologypropertytypes) | ask | List the ontology property-type registry and resolve/validate a Palantir-style type ref (CONCEPT:AU- | 4 | `/ontology/property-types` |
| [`ontology_sampling_profile`](#ontologysamplingprofile) | ask | Task-aware LLM sampling profiles (CONCEPT:AU-ORCH.routing.sampling-profile-selection/KG-2.94): list/ | 6 | `/ontology/sampling-profiles` |
| [`ontology_value_types`](#ontologyvaluetypes) | ask | List/describe constrained ontology value types and validate or coerce a value (CONCEPT:AU-KG.ontolog | 4 | `/ontology/value-types` |
| [`quant`](#quant) | ask | The Ultimate Quant System Tool. | 14 | `/quant` |
| [`research_artifact`](#researchartifact) | manage | Agent-Native Research Artifacts over the one ontology-driven KG (CONCEPT:AU-KG.research.best-effort- | 1 | `/research/artifact` |
| [`source_connector`](#sourceconnector) | ask | Document-source connectors (CONCEPT:AU-ECO.connector.document-source-framework–4.29, KG-2.59): list  | 2 | `/connector/source` |
| [`source_drain`](#sourcedrain) | write | Watch a chunked async drain started by source_sync(mode='full') on a LARGE corpus (CONCEPT:AU-KG.ont | 1 | `/source/drain` |
| [`source_sync`](#sourcesync) | write | THE canonical connector→KG ingestion tool (CONCEPT:AU-KG.ingest.enterprise-source-extractor) — one e | 1 | `/source/sync` |
| [`spec_ticket`](#specticket) | write | Link a KG SDD spec/feature to a Plane/Jira work item and make agents assignable (CONCEPT:AU-KG.inges | 1 | `/spec/ticket` |
| [`usage_query`](#usagequery) | write | Query usage/cost/observability analytics (CONCEPT:AU-ECO.mcp.usage-cost-observability-surface): toke | 11 | `/usage/query` |

## Capabilities

### `ask_data`

**ask data**

answer a DATA question over the Knowledge Graph with a DB-GPT-style, multi-step data-analysis agent (distinct from the single-shot nl_query).

- **Intent verbs:** ask
- **REST route:** `/graph/ask-data`
- **MCP tags:** data-analysis, graph-os, nl, query
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `ask_data` → (no EG ledger match)

**Typed input:**

- `question` (string, required): The natural-language DATA question to answer.
- `dialect` (string): 'auto' (planner chooses, prefers uql) or 'uql'|'cypher'|'sql'|'sparql'.
- `max_corrections` (integer): Bounded self-correction retries after a failed query (0 disables).
- `limit` (integer): Max result rows to return.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `concept_registry`

**concept registry**

Atomically claim/list/release concept ids across parallel sessions & worktrees (CONCEPT:AU-OS.governance.atomic-concept-id-reservation). action='reserve' mints the next free id in a namespace (a pillar like 'EG-KG.compute.backend'/'OS-5' or a package prefix like 'KEY') and appends it to the committed, merge=union ledger so two sessions never collide; 'list' shows reservations; 'release' frees one; 'reconcile' marks landed/expired.

- **Intent verbs:** act
- **REST route:** `/concept/registry`
- **MCP tags:** concept, governance, graph-os, ontology
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `concept_registry` → (no EG ledger match)

**Typed input:**

- `action` (string): 'reserve', 'list', 'release', or 'reconcile'.
- `namespace` (string): For 'reserve': pillar ('EG-KG.compute.backend','OS-5') or package prefix ('KEY','GL').
- `session_id` (string): Claiming session id (defaults to host:pid).
- `design_doc` (string): Optional design-doc path recorded with the reservation.
- `concept_id` (string): For 'release': the id to free.
- `status` (string): For 'list': filter by status (reserved/landed/expired).
- `ttl_seconds` (integer): Reservation TTL before it is reclaimable.
- `repo` (string): Repo root whose ledger to use (defaults to agent-utilities).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `document_process`

**document process**

Document → ontology processing (CONCEPT:AU-KG.ingest.chunk-overlap-stage): extract → chunk(overlap) → embed → materialize a Document + linked Chunk objects through the live graph write path.

- **Intent verbs:** ask
- **REST route:** `/document/process`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `document_process` → (no EG ledger match)

**Typed input:**

- `document` (string, required): A file path or raw text content to process.
- `text` (string): Optional pre-extracted text (OCR/external).
- `source` (string): Provenance label (path/URL).
- `chunk_size` (integer): Target chunk size in characters.
- `overlap` (integer): Overlap characters between chunks.
- `contextual` (boolean): Enable contextual-retrieval enrichment (CONCEPT:AU-KG.enrichment.contextual-retrieval-enrichment): situate each chunk within the document and embed context+chunk for better recall.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_admin`

**engine admin**

Low-level epistemic-graph engine surface for the 'admin' domain (ops/maintenance: online backup + restore (ADMIN)).

- **Intent verbs:** ask
- **REST route:** `/engine/admin`
- **MCP tags:** admin, engine, graph-os
- **Side effects:** 2/2 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Saga', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `backup` → EG `Backup` (confidence 1.0)
- `restore` → EG `Restore` (confidence 1.0)

**Typed input:**

- `action` (string): engine_admin method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_analytics`

**engine analytics**

Low-level epistemic-graph engine surface for the 'analytics' domain (centrality + (personalized) PageRank).

- **Intent verbs:** ask
- **REST route:** `/engine/analytics`
- **MCP tags:** analytics, engine, graph-os
- **Side effects:** 3/5 actions matched an EG ledger Method; any_mutates=False; durability=['None']; txn=['Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `betweenness_centrality` → EG `BetweennessCentrality` (confidence 1.0)
- `degree_centrality` → EG `DegreeCentrality` (confidence 1.0)
- `degree_centrality_all` → EG `DegreeCentrality` (confidence 1.0)
- `pagerank` → (no EG ledger match)
- `personalized_pagerank` → (no EG ledger match)

**Typed input:**

- `action` (string): engine_analytics method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_blob`

**engine blob**

Low-level epistemic-graph engine surface for the 'blob' domain (streamed content-addressed media blobs).

- **Intent verbs:** ask
- **REST route:** `/engine/blob`
- **MCP tags:** blob, engine, graph-os
- **Side effects:** 9/11 actions matched an EG ledger Method; any_mutates=True; durability=['BlobRedb', 'None']; txn=['Atomic', 'Saga', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `begin` → EG `BlobBegin` (confidence 1.0)
- `chunk_get` → EG `BlobChunkGet` (confidence 1.0)
- `chunk_put` → EG `BlobChunkPut` (confidence 1.0)
- `commit` → EG `Commit` (confidence 1.0)
- `fetch` → EG `BlobFetchBegin` (confidence 0.667)
- `fetch_begin` → EG `BlobBegin` (confidence 1.0)
- `fetch_end` → EG `BlobFetchEnd` (confidence 1.0)
- `gc` → EG `BlobGc` (confidence 1.0)
- `incref` → (no EG ledger match)
- `store` → (no EG ledger match)
- `unref` → EG `BlobUnref` (confidence 1.0)

**Typed input:**

- `action` (string): engine_blob method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_broker`

**engine broker**

Low-level epistemic-graph engine surface for the 'broker' domain (native message broker: exchange/queue/stream admin + routed publish/consume).

- **Intent verbs:** ask
- **REST route:** `/engine/broker`
- **MCP tags:** broker, engine, graph-os
- **Side effects:** 21/21 actions matched an EG ledger Method; any_mutates=True; durability=['None', 'Outbox']; txn=['Atomic', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `ack` → EG `BrokerAck` (confidence 1.0)
- `ack_tag` → EG `BrokerAck` (confidence 1.0)
- `bind_queue` → EG `BindQueue` (confidence 1.0)
- `consume` → EG `BrokerConsume` (confidence 1.0)
- `declare_exchange` → EG `DeclareExchange` (confidence 1.0)
- `declare_queue` → EG `DeclareQueue` (confidence 1.0)
- `delete_exchange` → EG `DeleteExchange` (confidence 1.0)
- `nack_tag` → EG `BrokerNackTag` (confidence 1.0)
- `publish` → EG `Publish` (confidence 1.0)
- `publish_confirmed` → EG `Publish` (confidence 1.0)
- `publish_ex` → EG `Publish` (confidence 1.0)
- `publish_idempotent` → EG `Publish` (confidence 1.0)
- `reject` → EG `BrokerReject` (confidence 1.0)
- `stream_commit_offset` → EG `StreamCommitOffset` (confidence 1.0)
- `stream_committed_offset` → EG `StreamCommittedOffset` (confidence 1.0)
- `stream_declare` → EG `StreamDeclare` (confidence 1.0)
- `stream_publish` → EG `Publish` (confidence 1.0)
- `stream_read` → EG `StreamRead` (confidence 1.0)
- `stream_trim` → EG `StreamTrim` (confidence 1.0)
- `sweep_expired` → EG `SweepExpired` (confidence 1.0)
- `unbind_queue` → EG `UnbindQueue` (confidence 1.0)

**Typed input:**

- `action` (string): engine_broker method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_channels`

**engine channels**

Low-level epistemic-graph engine surface for the 'channels' domain (dynamic agent communication channels).

- **Intent verbs:** ask
- **REST route:** `/engine/channels`
- **MCP tags:** channels, engine, graph-os
- **Side effects:** 8/8 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Atomic', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `close` → EG `CloseChannel` (confidence 1.0)
- `create` → EG `CreateChannel` (confidence 1.0)
- `get_members` → EG `GetChannelMembers` (confidence 1.0)
- `get_messages` → EG `GetChannelMessages` (confidence 1.0)
- `join` → EG `JoinChannel` (confidence 1.0)
- `leave` → EG `LeaveChannel` (confidence 1.0)
- `list` → EG `ListChannels` (confidence 1.0)
- `send_message` → EG `SendMessage` (confidence 1.0)

**Typed input:**

- `action` (string): engine_channels method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_consensus`

**engine consensus**

Low-level epistemic-graph engine surface for the 'consensus' domain (zero-trust identity + multisig mutation).

- **Intent verbs:** ask
- **REST route:** `/engine/consensus`
- **MCP tags:** admin, consensus, engine, graph-os
- **Side effects:** 2/2 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Atomic']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `apply_multisig_mutation` → EG `ApplyMutation` (confidence 1.0)
- `register_identity` → EG `RegisterIdentity` (confidence 1.0)

**Typed input:**

- `action` (string): engine_consensus method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_datascience`

**engine datascience**

Low-level epistemic-graph engine surface for the 'datascience' domain (estimators + primitives + training kernels).

- **Intent verbs:** ask
- **REST route:** `/engine/datascience`
- **MCP tags:** datascience, engine, graph-os
- **Side effects:** 0/15 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `adam_step` → (no EG ledger match)
- `compute_stats` → (no EG ledger match)
- `cross_entropy` → (no EG ledger match)
- `dpo_loss` → (no EG ledger match)
- `fit_estimator` → (no EG ledger match)
- `grpo_surrogate` → (no EG ledger match)
- `kl_divergence` → (no EG ledger match)
- `kmeans` → (no EG ledger match)
- `linear_regression` → (no EG ledger match)
- `log_softmax` → (no EG ledger match)
- `pca` → (no EG ledger match)
- `predict_estimator` → (no EG ledger match)
- `sgd_step` → (no EG ledger match)
- `softmax` → (no EG ledger match)
- `train_test_split` → (no EG ledger match)

**Typed input:**

- `action` (string): engine_datascience method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_edges`

**engine edges**

Low-level epistemic-graph engine surface for the 'edges' domain (edge CRUD, temporal invalidate/supersede, batch reads).

- **Intent verbs:** ask
- **REST route:** `/engine/edges`
- **MCP tags:** edges, engine, graph-os
- **Side effects:** 8/9 actions matched an EG ledger Method; any_mutates=True; durability=['GraphRedb', 'None']; txn=['Atomic', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `add` → EG `AddEdge` (confidence 1.0)
- `count` → EG `EdgeCount` (confidence 1.0)
- `has` → EG `HasEdge` (confidence 1.0)
- `invalidate` → EG `InvalidateEdge` (confidence 1.0)
- `list` → (no EG ledger match)
- `properties` → EG `GetEdgeProperties` (confidence 0.667)
- `properties_batch` → EG `GetEdgePropertiesBatch` (confidence 0.75)
- `remove` → EG `RemoveEdge` (confidence 1.0)
- `supersede` → EG `SupersedeEdge` (confidence 1.0)

**Typed input:**

- `action` (string): engine_edges method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_finance`

**engine finance**

Low-level epistemic-graph engine surface for the 'finance' domain (quantitative finance (optimize/risk/regime/signals/HFT/derivatives)).

- **Intent verbs:** ask
- **REST route:** `/engine/finance`
- **MCP tags:** engine, finance, graph-os
- **Side effects:** 67/67 actions matched an EG ledger Method; any_mutates=False; durability=['None']; txn=['None']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `adf_test` → EG `FinanceAdfTest` (confidence 1.0)
- `alpha_combination_engine` → EG `FinanceAlphaCombinationEngine` (confidence 1.0)
- `avellaneda_stoikov` → EG `FinanceAvellanedaStoikov` (confidence 1.0)
- `bayesian_kelly` → EG `FinanceBayesianKelly` (confidence 1.0)
- `black_litterman` → EG `FinanceBlackLitterman` (confidence 1.0)
- `breakeven_alpha` → EG `FinanceBreakevenAlpha` (confidence 1.0)
- `brier_score` → EG `FinanceBrierScore` (confidence 1.0)
- `combine_alphas` → EG `FinanceCombineAlphas` (confidence 1.0)
- `convergence_gate` → EG `FinanceConvergenceGate` (confidence 1.0)
- `cross_sectional_rank` → EG `FinanceCrossSectionalRank` (confidence 1.0)
- `cvar` → EG `FinanceCvar` (confidence 1.0)
- `deflated_sharpe` → EG `FinanceDeflatedSharpe` (confidence 1.0)
- `detect_regimes` → EG `FinanceDetectRegimes` (confidence 1.0)
- `diebold_mariano` → EG `FinanceDieboldMariano` (confidence 1.0)
- `downside_deviation` → EG `FinanceDownsideDeviation` (confidence 1.0)
- `drawdown_series` → EG `FinanceDrawdownSeries` (confidence 1.0)
- `effective_independent_n` → EG `FinanceEffectiveIndependentN` (confidence 1.0)
- `efficient_frontier` → EG `FinanceEfficientFrontier` (confidence 1.0)
- `empirical_kelly` → EG `FinanceEmpiricalKelly` (confidence 1.0)
- `ewma` → EG `FinanceEwma` (confidence 1.0)
- `expected_pnl_rate` → EG `FinanceExpectedPnlRate` (confidence 1.0)
- `forensic_report` → EG `FinanceForensicReport` (confidence 1.0)
- `glosten_milgrom_spread` → EG `FinanceGlostenMilgromSpread` (confidence 1.0)
- `glt_quotes` → EG `FinanceGltQuotes` (confidence 1.0)
- `hardiman_bouchaud` → EG `FinanceHardimanBouchaud` (confidence 1.0)
- `hawkes_mle` → EG `FinanceHawkesMle` (confidence 1.0)
- `information_coefficient` → EG `FinanceInformationCoefficient` (confidence 1.0)
- `information_ratio` → EG `FinanceInformationRatio` (confidence 1.0)
- `kalman_beta` → EG `FinanceKalmanBeta` (confidence 1.0)
- `kalman_filter_1d` → EG `FinanceKalmanFilter1d` (confidence 0.667)
- `kalman_volatility` → EG `FinanceKalmanVolatility` (confidence 1.0)
- `kelly_fraction` → EG `FinanceKellyFraction` (confidence 1.0)
- `kyle_lambda` → EG `FinanceKyleLambda` (confidence 1.0)
- `logit_quotes` → EG `FinanceLogitQuotes` (confidence 1.0)
- `market_impact` → EG `FinanceMarketImpact` (confidence 1.0)
- `markov_transition_matrix` → EG `FinanceMarkovTransitionMatrix` (confidence 1.0)
- `match_orders` → EG `FinanceMatchOrders` (confidence 1.0)
- `max_drawdown` → EG `FinanceMaxDrawdown` (confidence 1.0)
- `mean_reversion` → EG `FinanceMeanReversion` (confidence 1.0)
- `microprice_series` → EG `FinanceMicropriceSeries` (confidence 1.0)
- `momentum` → EG `FinanceMomentum` (confidence 1.0)
- `monte_carlo_var` → EG `FinanceVar` (confidence 1.0)
- `ofi_series` → EG `FinanceOfiSeries` (confidence 1.0)
- `optimize_portfolio` → EG `FinanceOptimizePortfolio` (confidence 1.0)
- `order_book_imbalance` → EG `FinanceOrderBookImbalance` (confidence 1.0)
- `ou_calibrate` → EG `FinanceOuCalibrate` (confidence 1.0)
- `ou_optimal_thresholds` → EG `FinanceOuOptimalThresholds` (confidence 1.0)
- `pairs_trading` → EG `FinancePairsTrading` (confidence 1.0)
- `posterior_credible_interval` → EG `FinancePosteriorCredibleInterval` (confidence 1.0)
- `probability_backtest_overfit` → EG `FinanceProbabilityBacktestOverfit` (confidence 1.0)
- `purged_cpcv` → EG `FinancePurgedCpcv` (confidence 1.0)
- `queue_imbalance` → EG `FinanceQueueImbalance` (confidence 1.0)
- `realized_vol_tick` → EG `FinanceRealizedVolTick` (confidence 1.0)
- `risk_metrics` → EG `Metrics` (confidence 1.0)
- `risk_parity` → EG `FinanceRiskParity` (confidence 1.0)
- `rolling_zscore` → EG `FinanceRollingZscore` (confidence 1.0)
- `sabr_calibrate` → EG `FinanceSabrCalibrate` (confidence 1.0)
- `sabr_implied_vol` → EG `FinanceSabrImpliedVol` (confidence 1.0)
- `sabr_smile` → EG `FinanceSabrSmile` (confidence 1.0)
- `signal_decay` → EG `FinanceSignalDecay` (confidence 1.0)
- ... and 7 more actions

**Typed input:**

- `action` (string): engine_finance method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_graph`

**engine graph**

Low-level epistemic-graph engine surface for the 'graph' domain (graph algorithms, AST parse/index, semantic+embedding compute).

- **Intent verbs:** ask
- **REST route:** `/engine/graph`
- **MCP tags:** engine, graph, graph-os
- **Side effects:** 27/29 actions matched an EG ledger Method; any_mutates=True; durability=['GraphRedb', 'None']; txn=['Atomic', 'None', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `add_embedding` → EG `AddEmbedding` (confidence 1.0)
- `batch_cosine_similarity` → EG `BatchCosineSimilarity` (confidence 1.0)
- `batch_l2_normalize` → EG `BatchL2Normalize` (confidence 1.0)
- `blast_radius` → (no EG ledger match)
- `clear` → EG `ClearGraph` (confidence 1.0)
- `community_detect_ephemeral` → EG `CommunityDetectEphemeral` (confidence 1.0)
- `community_detection` → EG `CommunityDetection` (confidence 1.0)
- `compute_similarity_edges` → EG `ComputeSimilarityEdges` (confidence 1.0)
- `connected_components` → EG `ConnectedComponents` (confidence 1.0)
- `discover` → EG `Discover` (confidence 1.0)
- `find_cycle` → EG `FindCycle` (confidence 1.0)
- `find_similar_pairs` → EG `FindSimilarPairs` (confidence 1.0)
- `get_subgraph` → EG `GetSubgraph` (confidence 1.0)
- `graph_coloring` → EG `GraphColoring` (confidence 1.0)
- `hypergraph_encode_interaction` → EG `HypergraphEncodeInteraction` (confidence 1.0)
- `index_repository` → EG `IndexRepository` (confidence 1.0)
- `match_ontology_terms` → EG `MatchOntologyTerms` (confidence 1.0)
- `minimum_spanning_tree` → EG `MinimumSpanningTree` (confidence 1.0)
- `observe_screen` → EG `ObserveScreen` (confidence 1.0)
- `parse_file` → EG `ParseFile` (confidence 1.0)
- `parse_files` → EG `ParseFiles` (confidence 1.0)
- `parse_repository` → EG `ParseRepository` (confidence 1.0)
- `resolve_candidates` → EG `ResolveCandidates` (confidence 1.0)
- `semantic_search` → EG `SemanticSearch` (confidence 1.0)
- `shortest_path` → (no EG ledger match)
- `spectral_cluster` → EG `SpectralCluster` (confidence 1.0)
- `strongly_connected_components` → EG `ConnectedComponents` (confidence 1.0)
- `topological_sort` → EG `TopologicalSort` (confidence 1.0)
- `vf2_subgraph_match` → EG `Vf2SubgraphMatch` (confidence 1.0)

**Typed input:**

- `action` (string): engine_graph method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_graphlearn`

**engine graphlearn**

Low-level epistemic-graph engine surface for the 'graphlearn' domain (KAN graph-learning: fit/predict a learned per-feature edge-function link predictor).

- **Intent verbs:** ask
- **REST route:** `/engine/graphlearn`
- **MCP tags:** engine, graph-os, graphlearn
- **Side effects:** 2/2 actions matched an EG ledger Method; any_mutates=True; durability=['GraphRedb']; txn=['Atomic']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `fit` → EG `GraphLearnFit` (confidence 1.0)
- `predict` → EG `GraphLearnPredict` (confidence 1.0)

**Typed input:**

- `action` (string): engine_graphlearn method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_ledger`

**engine ledger**

Low-level epistemic-graph engine surface for the 'ledger' domain (audit ledger get/clear/apply).

- **Intent verbs:** ask
- **REST route:** `/engine/ledger`
- **MCP tags:** engine, graph-os, ledger
- **Side effects:** 3/3 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Atomic', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `apply` → EG `ApplyLedger` (confidence 1.0)
- `clear` → EG `ClearLedger` (confidence 1.0)
- `get` → EG `GetLedger` (confidence 1.0)

**Typed input:**

- `action` (string): engine_ledger method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_lifecycle`

**engine lifecycle**

Low-level epistemic-graph engine surface for the 'lifecycle' domain (prune/decay/evict, batch_update, context view, (de)serialize).

- **Intent verbs:** manage
- **REST route:** `/engine/lifecycle`
- **MCP tags:** engine, graph-os, lifecycle
- **Side effects:** 10/10 actions matched an EG ledger Method; any_mutates=True; durability=['GraphRedb', 'None']; txn=['Atomic', 'None', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `batch_update` → EG `BatchUpdate` (confidence 1.0)
- `decay_sweep` → EG `DecayNode` (confidence 1.0)
- `evict_lru` → EG `EvictLRU` (confidence 1.0)
- `from_msgpack` → EG `FromMsgpack` (confidence 1.0)
- `get_context_view` → EG `GetContextView` (confidence 1.0)
- `metrics` → EG `Metrics` (confidence 1.0)
- `multi_graph_batch_update` → EG `BatchUpdate` (confidence 1.0)
- `prune` → EG `PruneByLifecycle` (confidence 0.667)
- `to_msgpack` → EG `ToMsgpack` (confidence 1.0)
- `touch_nodes` → EG `TouchNodes` (confidence 1.0)

**Typed input:**

- `action` (string): engine_lifecycle method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_mining`

**engine mining**

Low-level epistemic-graph engine surface for the 'mining' domain (association-rule mining (Apriori/FP-Growth/Eclat: support/confidence/lift)).

- **Intent verbs:** ask
- **REST route:** `/engine/mining`
- **MCP tags:** engine, graph-os, mining
- **Side effects:** 18/18 actions matched an EG ledger Method; any_mutates=True; durability=['GraphRedb', 'None']; txn=['Atomic', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `anomaly` → EG `MineAnomaly` (confidence 1.0)
- `associate` → EG `MineAssociate` (confidence 1.0)
- `causal_impact` → EG `MineCausalImpact` (confidence 1.0)
- `classify_fit` → EG `MineClassifyFit` (confidence 1.0)
- `classify_predict` → EG `MineClassifyPredict` (confidence 1.0)
- `cluster` → EG `MineCluster` (confidence 1.0)
- `community` → EG `MineCommunity` (confidence 1.0)
- `entity_resolve` → EG `MineEntityResolve` (confidence 1.0)
- `forecast` → EG `MineForecast` (confidence 1.0)
- `ontology_gap` → EG `MineOntologyGap` (confidence 1.0)
- `process` → EG `MineProcess` (confidence 1.0)
- `reduce` → EG `MineReduce` (confidence 1.0)
- `retrieval_quality` → EG `MineRetrievalQuality` (confidence 1.0)
- `risk_propagation` → EG `MineRiskPropagation` (confidence 1.0)
- `root_cause` → EG `MineRootCause` (confidence 1.0)
- `sequence` → EG `MineSequence` (confidence 1.0)
- `subgraph` → EG `MineSubgraph` (confidence 1.0)
- `text` → EG `MineText` (confidence 1.0)

**Typed input:**

- `action` (string): engine_mining method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_nodes`

**engine nodes**

Low-level epistemic-graph engine surface for the 'nodes' domain (node CRUD, batch/union reads, degree/neighbour queries).

- **Intent verbs:** ask
- **REST route:** `/engine/nodes`
- **MCP tags:** engine, graph-os, nodes
- **Side effects:** 16/20 actions matched an EG ledger Method; any_mutates=True; durability=['GraphRedb', 'None']; txn=['Atomic', 'Snapshot']
- **Cost:** (unmeasured)
- **Latency:** {'add': {'eg_method': 'AddNode', 'p50_ms': 0.187, 'p99_ms': 0.223, 'source': 'epistemic-graph/docs/benchmarks.md#results (2026-06-01, UDS, in-memory graph)', 'kind': 'measured'}, 'claim_next': {'eg_method': 'ClaimNext', 'source': "epistemic-graph/docs/benchmarks-soak.md (queue/claim primitive is measured in the same soak run; approximates AgentBus queue latency per that doc's own caveat — see the doc for the current number, not duplicated here to avoid a second copy drifting from the source)", 'kind': 'measured'}, 'compare_and_set': {'eg_method': 'CompareAndSetNodeFields', 'p50_ms': 14.17, 'p95_ms': 43.72, 'p99_ms': 57.24, 'source': 'epistemic-graph/docs/benchmarks-soak.md#phase-a (2026-07-11 soak run, shared/contended box — upper bound per source doc)', 'kind': 'measured'}, 'properties': {'eg_method': 'GetNodeProperties', 'p50_ms': 0.179, 'p99_ms': 0.21, 'source': 'epistemic-graph/docs/benchmarks.md#results (2026-06-01, UDS, in-memory graph)', 'kind': 'measured'}}
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `add` → EG `AddNode` (confidence 1.0)
- `claim_next` → EG `ClaimNext` (confidence 1.0)
- `compare_and_set` → EG `CompareAndSetNodeFields` (confidence 0.8)
- `count` → EG `NodeCount` (confidence 1.0)
- `has` → EG `HasNode` (confidence 1.0)
- `has_batch` → EG `HasNode` (confidence 1.0)
- `ids` → EG `NodeIds` (confidence 1.0)
- `in_degree` → EG `InDegree` (confidence 1.0)
- `list` → (no EG ledger match)
- `list_by_label` → EG `GetNodesByLabel` (confidence 0.75)
- `list_by_label_union` → EG `UnionGetNodesByLabel` (confidence 0.8)
- `neighbors` → (no EG ledger match)
- `neighbors_union` → EG `UnionGetNodeProperties` (confidence 0.5)
- `out_degree` → EG `OutDegree` (confidence 1.0)
- `predecessors` → (no EG ledger match)
- `properties` → EG `GetNodeProperties` (confidence 0.667)
- `properties_batch` → EG `GetNodePropertiesBatch` (confidence 0.75)
- `properties_union` → EG `UnionGetNodeProperties` (confidence 0.75)
- `remove` → EG `RemoveNode` (confidence 1.0)
- `successors` → (no EG ledger match)

**Typed input:**

- `action` (string): engine_nodes method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_query`

**engine query**

Low-level epistemic-graph engine surface for the 'query' domain (SQL / Cypher / GraphQL / UQL / unified cross-modal query / federation).

- **Intent verbs:** ask
- **REST route:** `/engine/query`
- **MCP tags:** engine, graph-os, query
- **Side effects:** 19/21 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Atomic', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `causal_estimate` → EG `CausalEstimate` (confidence 1.0)
- `cypher` → EG `CypherQuery` (confidence 1.0)
- `epistemic_status` → EG `EpistemicStatus` (confidence 1.0)
- `explain_belief` → EG `ExplainBelief` (confidence 1.0)
- `explain_evidence` → EG `ExplainEvidence` (confidence 1.0)
- `explain_plan` → EG `ExplainPlan` (confidence 1.0)
- `explain_policy` → EG `ExplainPolicy` (confidence 1.0)
- `explain_provenance` → EG `ExplainProvenance` (confidence 1.0)
- `explain_provenance_by_ids` → EG `ExplainProvenance` (confidence 1.0)
- `export_sqlite_file` → EG `ExportSqliteFile` (confidence 1.0)
- `graphql` → (no EG ledger match)
- `import_sqlite_file` → EG `ImportSqliteFile` (confidence 1.0)
- `materialization_status` → EG `MaterializationStatus` (confidence 1.0)
- `nl_query` → EG `NlQuery` (confidence 1.0)
- `rank_by_provenance` → EG `RankByProvenance` (confidence 1.0)
- `register_foreign_source` → EG `RegisterForeignSource` (confidence 1.0)
- `register_materialization` → EG `RegisterMaterialization` (confidence 1.0)
- `sql` → EG `Sql` (confidence 1.0)
- `unified` → EG `UnifiedQuery` (confidence 1.0)
- `uql` → (no EG ledger match)
- `what_changed` → EG `WhatChanged` (confidence 1.0)

**Typed input:**

- `action` (string): engine_query method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_rbac`

**engine rbac**

Low-level epistemic-graph engine surface for the 'rbac' domain (RBAC policy administration: roles + resource/action grants (ADMIN)).

- **Intent verbs:** manage
- **REST route:** `/engine/rbac`
- **MCP tags:** admin, engine, graph-os, rbac
- **Side effects:** 0/5 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `add_grant` → (no EG ledger match)
- `add_role` → (no EG ledger match)
- `list` → (no EG ledger match)
- `remove_grant` → (no EG ledger match)
- `remove_role` → (no EG ledger match)

**Typed input:**

- `action` (string): engine_rbac method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_rdf`

**engine rdf**

Low-level epistemic-graph engine surface for the 'rdf' domain (RDF triples + SPARQL + OWL reasoning).

- **Intent verbs:** ask
- **REST route:** `/engine/rdf`
- **MCP tags:** engine, graph-os, rdf
- **Side effects:** 8/9 actions matched an EG ledger Method; any_mutates=True; durability=['GraphRedb', 'None']; txn=['Atomic', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `add_triples` → EG `AddTriples` (confidence 1.0)
- `drop_named_graph` → EG `DropNamedGraph` (confidence 1.0)
- `explain` → (no EG ledger match)
- `get_triples` → EG `GetTriples` (confidence 1.0)
- `owl_reason` → EG `OwlReason` (confidence 1.0)
- `owl_reason_distributed` → EG `OwlReason` (confidence 1.0)
- `remove_triples` → EG `RemoveTriples` (confidence 1.0)
- `sparql` → EG `Sparql` (confidence 1.0)
- `sparql_virtual` → EG `Sparql` (confidence 1.0)

**Typed input:**

- `action` (string): engine_rdf method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_reasoning`

**engine reasoning**

Low-level epistemic-graph engine surface for the 'reasoning' domain (forward-chaining OWL/RDFS Datalog closure).

- **Intent verbs:** ask
- **REST route:** `/engine/reasoning`
- **MCP tags:** engine, graph-os, reasoning
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `reason` → (no EG ledger match)

**Typed input:**

- `action` (string): engine_reasoning method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_resharding`

**engine resharding**

Low-level epistemic-graph engine surface for the 'resharding' domain (M3 catalog/reshard/rebalance admin (redb)).

- **Intent verbs:** manage
- **REST route:** `/engine/resharding`
- **MCP tags:** admin, engine, graph-os, resharding
- **Side effects:** 7/7 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Saga', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `catalog_assign` → EG `CatalogAssign` (confidence 1.0)
- `catalog_list` → EG `CatalogList` (confidence 1.0)
- `catalog_reassign` → EG `CatalogReassign` (confidence 1.0)
- `catalog_remove` → EG `CatalogRemove` (confidence 1.0)
- `rebalance_execute` → EG `RebalanceExecute` (confidence 1.0)
- `rebalance_plan` → EG `RebalancePlan` (confidence 1.0)
- `reshard` → EG `Reshard` (confidence 1.0)

**Typed input:**

- `action` (string): engine_resharding method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_streaming`

**engine streaming**

Low-level epistemic-graph engine surface for the 'streaming' domain (CDC / continuous queries / watch / triggers).

- **Intent verbs:** ask
- **REST route:** `/engine/streaming`
- **MCP tags:** engine, graph-os, streaming
- **Side effects:** 9/9 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Atomic', 'None', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `cdc_read` → EG `StreamRead` (confidence 1.0)
- `drop_continuous_query` → EG `DropContinuousQuery` (confidence 1.0)
- `drop_trigger` → EG `DropTrigger` (confidence 1.0)
- `fired_triggers` → EG `FiredTriggers` (confidence 1.0)
- `list_triggers` → EG `ListTriggers` (confidence 1.0)
- `read_continuous_query` → EG `StreamRead` (confidence 1.0)
- `register_continuous_query` → EG `RegisterContinuousQuery` (confidence 1.0)
- `register_trigger` → EG `RegisterTrigger` (confidence 1.0)
- `watch` → EG `Watch` (confidence 1.0)

**Typed input:**

- `action` (string): engine_streaming method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_tenants`

**engine tenants**

Low-level epistemic-graph engine surface for the 'tenants' domain (multi-tenant graph create/delete/list).

- **Intent verbs:** manage
- **REST route:** `/engine/tenants`
- **MCP tags:** admin, engine, graph-os, tenants
- **Side effects:** 2/3 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Atomic']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `create` → EG `CreateGraph` (confidence 1.0)
- `delete` → EG `DeleteGraph` (confidence 1.0)
- `list` → (no EG ledger match)

**Typed input:**

- `action` (string): engine_tenants method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_timeseries`

**engine timeseries**

Low-level epistemic-graph engine surface for the 'timeseries' domain (native TSDB append/range/window/asof/gapfill).

- **Intent verbs:** ask
- **REST route:** `/engine/timeseries`
- **MCP tags:** engine, graph-os, timeseries
- **Side effects:** 0/6 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `append` → (no EG ledger match)
- `asof_join` → (no EG ledger match)
- `gap_fill` → (no EG ledger match)
- `range` → (no EG ledger match)
- `register_series` → (no EG ledger match)
- `window` → (no EG ledger match)

**Typed input:**

- `action` (string): engine_timeseries method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `engine_txn`

**engine txn**

Low-level epistemic-graph engine surface for the 'txn' domain (server-side OCC ACID transactions).

- **Intent verbs:** ask
- **REST route:** `/engine/txn`
- **MCP tags:** engine, graph-os, txn
- **Side effects:** 17/17 actions matched an EG ledger Method; any_mutates=True; durability=['GraphRedb', 'None']; txn=['Atomic', 'Saga', 'Snapshot']
- **Cost:** (unmeasured)
- **Latency:** {'add_node': {'eg_method': 'AddNode', 'p50_ms': 0.187, 'p99_ms': 0.223, 'source': 'epistemic-graph/docs/benchmarks.md#results (2026-06-01, UDS, in-memory graph)', 'kind': 'measured'}}
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `add_edge` → EG `AddEdge` (confidence 1.0)
- `add_embedding` → EG `AddEmbedding` (confidence 1.0)
- `add_measurement` → EG `TxnAddMeasurement` (confidence 1.0)
- `add_node` → EG `AddNode` (confidence 1.0)
- `axiom` → EG `TxnAxiom` (confidence 1.0)
- `begin` → EG `BeginTxn` (confidence 1.0)
- `blob_ref` → EG `TxnBlobRef` (confidence 1.0)
- `cas` → EG `TxnCas` (confidence 1.0)
- `commit` → EG `Commit` (confidence 1.0)
- `construct` → EG `TxnConstruct` (confidence 1.0)
- `materialize_belief` → EG `TxnMaterializeBelief` (confidence 1.0)
- `plan_writeback` → EG `TxnPlanWriteback` (confidence 1.0)
- `remove_edge` → EG `RemoveEdge` (confidence 1.0)
- `remove_node` → EG `RemoveNode` (confidence 1.0)
- `rollback` → EG `Rollback` (confidence 1.0)
- `unified_query` → EG `UnifiedQuery` (confidence 1.0)
- `unified_query_plan` → EG `UnifiedQuery` (confidence 1.0)

**Typed input:**

- `action` (string): engine_txn method to call (empty ⇒ list actions).
- `params_json` (string): JSON object of keyword arguments for the method.
- `graph` (string): Target graph name (empty ⇒ the deployment default graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_analyze`

**graph analyze**

Ops / structural analysis over the KG.

- **Intent verbs:** ask
- **REST route:** `/graph/analyze`
- **MCP tags:** analysis, analyze, graph-os
- **Side effects:** 1/68 actions matched an EG ledger Method; any_mutates=False; durability=['None']; txn=['None']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `adr` → (no EG ledger match)
- `arch_report` → (no EG ledger match)
- `assimilation_benchmark` → (no EG ledger match)
- `background_research` → (no EG ledger match)
- `blast_radius` → (no EG ledger match)
- `call_graph` → (no EG ledger match)
- `causal` → (no EG ledger match)
- `change_coupling` → (no EG ledger match)
- `check_constraints` → (no EG ledger match)
- `cleanup_documents` → (no EG ledger match)
- `close` → (no EG ledger match)
- `code_context` → (no EG ledger match)
- `code_evolution` → (no EG ledger match)
- `code_metrics` → EG `Metrics` (confidence 1.0)
- `context` → (no EG ledger match)
- `contradictions` → (no EG ledger match)
- `cross_repo_usages` → (no EG ledger match)
- `deep_extract` → (no EG ledger match)
- `distill_memory` → (no EG ledger match)
- `distill_search` → (no EG ledger match)
- `enrichment_coverage` → (no EG ledger match)
- `epistemic_sync` → (no EG ledger match)
- `evaluate` → (no EG ledger match)
- `evaluate_alpha` → (no EG ledger match)
- `evaluate_harness` → (no EG ledger match)
- `evolve_agent` → (no EG ledger match)
- `evolve_code` → (no EG ledger match)
- `evolve_model` → (no EG ledger match)
- `evolve_variants` → (no EG ledger match)
- `explain` → (no EG ledger match)
- `extract_claims` → (no EG ledger match)
- `forecast` → (no EG ledger match)
- `guard_corpus` → (no EG ledger match)
- `harness_benchmark` → (no EG ledger match)
- `harness_certify` → (no EG ledger match)
- `harness_evolve` → (no EG ledger match)
- `harness_gate` → (no EG ledger match)
- `infer_links` → (no EG ledger match)
- `infra_sweep` → (no EG ledger match)
- `inspect` → (no EG ledger match)
- `invariant` → (no EG ledger match)
- `latent_efficiency_benchmark` → (no EG ledger match)
- `night_shift` → (no EG ledger match)
- `pick_skill` → (no EG ledger match)
- `placement_plan` → (no EG ledger match)
- `process_writeback` → (no EG ledger match)
- `quant_arb` → (no EG ledger match)
- `quant_banking` → (no EG ledger match)
- `quant_crypto` → (no EG ledger match)
- `quant_exchange` → (no EG ledger match)
- `quant_insider` → (no EG ledger match)
- `quant_microstructure` → (no EG ledger match)
- `quant_regime` → (no EG ledger match)
- `quant_strategy` → (no EG ledger match)
- `recommend` → (no EG ledger match)
- `recursive_distill` → (no EG ledger match)
- `relevance_sweep` → (no EG ledger match)
- `research_ingest` → (no EG ledger match)
- `routes` → (no EG ledger match)
- `security_scan` → (no EG ledger match)
- ... and 8 more actions

**Typed input:**

- `action` (string): Ops/structural action: inspect | enrichment_coverage | process_writeback | placement_plan | infra_sweep | security_scan. (Codebase→graph_code, research→graph_research, eval→graph_evaluate, Q&A→graph_explain, traces→graph_observe.)
- `query` (string): Query or path for the analysis.
- `top_k` (integer): Number of results or complexity budget.
- `node_id` (string): Specific node ID to analyze (e.g., for blast_radius).
- `depth` (integer): Depth of traversal (e.g., for blast_radius).
- `target` (string): Target for the analysis or inspection.
- `envelope` (string): 'raw' (default; byte-identical legacy shape) or 'bundle' (additionally wrap the result as an EvidenceBundle — code_context, executable_rag). Additive/opt-in; every other action ignores it.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_ask`

**graph ask**

ask the Knowledge Graph in plain English.

- **Intent verbs:** ask
- **REST route:** `/graph/ask`
- **MCP tags:** graph-os, nl, query
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_ask` → (no EG ledger match)

**Typed input:**

- `question` (string, required): The natural-language question to answer.
- `dialect` (string): 'auto' (model chooses) or 'cypher'|'sql'|'sparql' to force one.
- `execute` (boolean): When false, return only the generated query (preview/dry-run).
- `limit` (integer): Max result rows to return.
- `envelope` (string): 'raw' (default; byte-identical legacy shape) or 'bundle' (additionally wrap the result as an EvidenceBundle under `evidence_bundle`). Additive/opt-in.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_broker`

**graph broker**

the epistemic-graph engine message broker (AMQP-style exchanges + queues + streams), distinct from the agent-to-agent 'graph_bus'.

- **Intent verbs:** ask
- **REST route:** `/graph/broker`
- **MCP tags:** broker, engine, engine_surface, graph-os, messaging
- **Side effects:** 1/1 actions matched an EG ledger Method; any_mutates=True; durability=['Outbox']; txn=['Atomic']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_broker` → EG `BrokerConsume` (confidence 0.5)

**Typed input:**

- `action` (string): Broker method: declare_exchange | declare_queue | bind | publish | consume | stats | list_queues | list_exchanges | ...
- `exchange` (string): Exchange name.
- `queue` (string): Queue name.
- `routing_key` (string): Routing/binding key.
- `payload` (string): Message body (publish).
- `exchange_type` (string): Exchange type: direct | fanout | topic (declare).
- `params_json` (string): JSON object of extra kwargs, e.g. {"max_messages":10,"ack":true,"durable":true}. Merged over the typed fields.
- `graph` (string): Target graph (empty ⇒ deployment default).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': 'BrokerConsume', 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_bus`

**graph bus**

the agent-to-agent communication bus: let this session talk to other Claude/LLM sessions (any provider, any host) through the shared graph-os hub.

- **Intent verbs:** ask
- **REST route:** `/graph/bus`
- **MCP tags:** a2a, bus, graph-os, messaging
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_bus` → (no EG ledger match)

**Typed input:**

- `action` (string): register | heartbeat | roster | send | receive | subscribe | unsubscribe | ack | dispatch | leave | status
- `agent_id` (string): This participant's id (most actions).
- `sender` (string): Sender agent id (send/dispatch).
- `to` (string): Recipient agent id (send, direct).
- `topic` (string): Topic name (send/subscribe/unsubscribe).
- `payload` (string): Message body (send).
- `objective` (string): Work objective (dispatch).
- `kind` (string): Loop kind for dispatch: develop|research|skill.
- `priority` (string): Bucket 0-3 or critical|high|normal|background (dispatch).
- `provider` (string): Provider label, e.g. anthropic|openai|google (register/roster).
- `host` (string): Host this session runs on (register).
- `capabilities` (string): Comma-separated capability tags (register); single tag filter (roster).
- `session_id` (string): Originating session id (register).
- `message_id` (string): Message id to ack.
- `since` (integer): Cursor: messages already consumed (receive).
- `online_only` (boolean): Roster: only online peers.
- `reason` (string): Audit reason (send/dispatch).
- `url` (string): Peer hub base URL (register_hub).
- `group` (string): Message group to forward (federate).
- `origin` (string): Origin hub id (federate_in).
- `scope` (string): Marking scope for federation: commons|org|private (federate).
- `replay_recent` (boolean): Subscribe: backfill a bounded recent topic window for a late joiner.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_code`

**graph code**

Understand a CODEBASE via the ingested code graph — query this before grep.

- **Intent verbs:** ask
- **REST route:** `/graph/code`
- **MCP tags:** analyze_suite, code, graph-os
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_code` → (no EG ledger match)

**Typed input:**

- `action` (string): code_context | cross_repo_usages | call_graph | similar_code | routes | change_coupling | code_evolution | blast_radius | code_metrics | arch_report | adr
- `query` (string): Question / area / symbol name / repo path.
- `top_k` (integer): Result count.
- `node_id` (string): Symbol/:Code node id (call_graph, similar_code, blast_radius).
- `depth` (integer): Traversal depth (blast_radius).
- `target` (string): how|usage|impact (code_context) or callees|callers|inherits (call_graph).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_code_nav`

**graph code nav**

Navigate the resolved code graph (CONCEPT:AU-KG.backend.declared-columns-so-schema). action: 'find_definition' (locate a symbol's :Code node), 'find_references' (callers of a symbol), 'trace_call_graph' (transitive callees), 'impact_of_change' (transitive callers = blast radius), 'connects' (shortest path between TWO symbols — set symbol/node_id AND target_symbol/target_node_id — rendered hop-by-hop with each edge's relation + confidence, CONCEPT:EG-KG.compute.handled-outside-single-anchor).

- **Intent verbs:** ask
- **REST route:** `/graph/code-nav`
- **MCP tags:** code, graph-os, query
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `connects` → (no EG ledger match)

**Typed input:**

- `action` (string, required): find_definition | find_references | trace_call_graph | impact_of_change | connects
- `symbol` (string): Symbol name to start from (function/class/method).
- `node_id` (string): Exact :Code node id (overrides 'symbol' when set).
- `target_symbol` (string): For action='connects': the destination symbol name.
- `target_node_id` (string): For action='connects': the destination :Code node id.
- `source_system` (string): Optional source_system filter, e.g. 'gitlab:gitlab.com'.
- `depth` (integer): Max hops for trace_call_graph / impact_of_change (1-10).
- `limit` (integer): Max rows to return.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_configure`

**graph configure**

Manage backend configurations, system credentials, and tool registration within the unified agent ecosystem.

- **Intent verbs:** manage
- **REST route:** `/graph/configure`
- **MCP tags:** analysis, configure, graph-os
- **Side effects:** 2/31 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Saga', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `add_connection` → (no EG ledger match)
- `config_doctor` → (no EG ledger match)
- `config_reference` → (no EG ledger match)
- `doctor` → (no EG ledger match)
- `generate_config` → (no EG ledger match)
- `get_config` → (no EG ledger match)
- `harness_fence` → (no EG ledger match)
- `imprint_connection` → (no EG ledger match)
- `install_hooks` → (no EG ledger match)
- `list_config` → (no EG ledger match)
- `list_connections` → (no EG ledger match)
- `mirror_status` → (no EG ledger match)
- `preflight` → (no EG ledger match)
- `profile_connection` → (no EG ledger match)
- `pull_from_stardog` → (no EG ledger match)
- `push_to_stardog` → (no EG ledger match)
- `reconcile` → EG `Reconcile` (confidence 1.0)
- `register_mcp` → (no EG ledger match)
- `remove_connection` → (no EG ledger match)
- `schema_candidates` → (no EG ledger match)
- `schema_pack` → (no EG ledger match)
- `set_config` → (no EG ledger match)
- `set_default_connection` → (no EG ledger match)
- `set_role_routing` → (no EG ledger match)
- `set_secret` → (no EG ledger match)
- `setup_databases` → (no EG ledger match)
- `stardog_sparql` → EG `Sparql` (confidence 1.0)
- `system_doctor` → (no EG ledger match)
- `uninstall_hooks` → (no EG ledger match)
- `vault_sync` → (no EG ledger match)
- `verify_databases` → (no EG ledger match)

**Typed input:**

- `action` (string): Operation ('set_secret', 'vault_sync', 'register_mcp', 'install_hooks', 'uninstall_hooks', 'harness_fence', 'doctor', 'set_role_routing', 'schema_pack', 'schema_candidates', 'add_connection', 'remove_connection', 'list_connections', 'set_default_connection'). CONCEPT:AU-OS.deployment.vault-seed-service — 'vault_sync' reconciles a service's secrets with the store (read-existing to skip re-prompting + seed new): config_key=service, config_value=JSON {"env_keys":[...],"values":{KEY:VAL},"overwrite":false}; returns {refs:{KEY:"vault://<service>/<KEY>"},present,written,missing} so resolvable vault:// refs drop straight into config.json. CONCEPT:AU-OS.deployment.governance-derived-claude-code — 'harness_fence' writes a governance-derived Claude Code permission fence (settings.json allow/ask/deny + defaultMode=acceptEdits, plus .claudeignore) so the CLI can run unattended safely; config_key=target Claude config dir (default ~/.claude), config_value optional {"policy":<ActionPolicy yaml>,"dry_run":true}; the deny list is regenerated from the live ActionPolicy each run. 'schema_pack' with config_key=<name> sets the active domain Schema Pack, or with empty config_key returns the active pack plus available packs; 'schema_candidates' reviews out-of-pack types seen on write (CONCEPT:AU-KG.ontology.schema-pack-lifecycle-audit). CONCEPT:AU-KG.backend.multi-connection-registry — 'add_connection' registers a named graph backend (config_key=name, config_value=JSON spec e.g. {"backend":"neo4j","uri":"bolt://...","user":"...","password":"..."}; use backend 'age' for Postgres native openCypher; CONCEPT:AU-KG.backend.connection-registry — spec may set role 'read'(default, query-only data source)|'read_write'|'mirror', and password/user/uri may be a vault://path or env://VAR ref; the connection is persisted to config.json so it survives restart); 'remove_connection' (config_key=name); 'list_connections' returns per-connection health + role; 'set_default_connection' (config_key=name) repoints the default target. 'profile_connection' (config_key=name) read-only-introspects a registered external graph's schema (labels, relationship types, property keys, per-label counts + sample property shapes); 'imprint_connection' profiles it, maps each external label onto our ontology (interfaces + our node types; unmatched flagged 'novel'), and writes a self-describing ExternalGraphReference catalog node (no credentials) into the authority KG so the foreign graph becomes discoverable+usable. CONCEPT:AU-KG.backend.mirror-health-repair — 'mirror_status' returns per-mirror replication health (lag/failures/stalled) for a GRAPH_BACKEND=fanout deployment; 'reconcile' (optional config_key=<mirror name>, empty=all) runs a full authority→mirror drift-repair pass. 'setup_databases' provisions the Stardog + pg-age environment end-to-end (config_key=profile 'dev'|'prod', config_value=JSON options e.g. {"postgres_mode":"managed_image","dsn":"postgresql://...","sparql_target":"builtin"}); 'verify_databases' probes a Postgres for the age/vector/pg_search extensions (config_key or config_value.dsn = DSN). CONCEPT:AU-KG.query.stardog-instance-data — Stardog instance-data sync (push/pull/query of real KG data, distinct from the ontology/TBox): 'push_to_stardog' writes KG nodes+edges into Stardog, partitioned into urn:source:<system> named graphs (config_value optional {"sources":["leanix","servicenow"],"connection":<registered name>} — omit sources to push everything; resolves a Stardog backend from config_key/connection name or inline {"endpoint","database","username","password"} or STARDOG_* env); 'pull_from_stardog' re-ingests Stardog data back into the KG (config_value optional {"source":"leanix"} or {"graph_uri":"urn:source:..."} to scope to one named graph, {"limit":N}); 'stardog_sparql' runs a SPARQL SELECT/ASK/CONSTRUCT/UPDATE against Stardog (config_value={"query":"SELECT ..."} or a bare query string). For continuous live replication instead, register Stardog as a role='mirror' connection (add_connection {"backend":"stardog",...}) under GRAPH_BACKEND=tiered/fanout and use 'reconcile' to backfill. 'generate_config' writes a COMPLETE profile-seeded config.json covering every option (config_key=profile 'tiny'|'single-node-prod'|'enterprise', config_value optional {"out":path,"redact_secrets":true}); 'config_doctor' validates a deployment's config completeness/health (config_key=profile, config_value optional {"config":path}); 'config_reference' returns every option grouped by subsystem. CONCEPT:AU-KG.backend.connection-registry — 'get_config' (config_key=env name) returns a live value; 'set_config' (config_key=env name, config_value=scalar or JSON) validates against config_reference, persists to config.json + applies live, and flags 'restart_required' for engine-rebuild settings; 'list_config' returns every current value (secrets redacted). 'system_doctor' runs a holistic deployment health sweep (brew/flutter-doctor style) across config/engine/backend/secrets/auth/mcp-fleet/hooks/observability, each with a remediation + skill (config_value optional {"only":[...],"fix":true,"live":true}). 'preflight' checks whether THIS HOST has the runtimes/tools to deploy a profile BEFORE installing (Python 3.11-<3.15, uv/pip, the epistemic-graph engine binary — Rust only as a fallback, Docker when not the tiny profile, and per-component deps): config_key=profile 'tiny'|'single-node-prod'|'enterprise', config_value optional {"components":["agent-webui","geniusbot","agent-terminal-ui"]}.
- `config_key` (string): The key or ID of the configuration/secret (for 'schema_pack', the pack name e.g. 'research-state'; for connection actions, the connection name).
- `config_value` (string): JSON string containing the payload or secret value.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_context`

**graph context**

store/fetch curated context for invoker→spawned-agent handoff, persisted in the epistemic-graph so a SEPARATELY-spawned agent can read it by id.

- **Intent verbs:** act, ask
- **REST route:** `/graph/context`
- **MCP tags:** context, graph-os, orchestrate, query
- **Side effects:** 1/4 actions matched an EG ledger Method; any_mutates=False; durability=['None']; txn=['Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `get` → EG `GetContextView` (confidence 0.667)
- `list` → (no EG ledger match)
- `prune` → (no EG ledger match)
- `put` → (no EG ledger match)

**Typed input:**

- `action` (string): put | get | list
- `content` (string): Context text to store (action=put).
- `context_id` (string): ContextBlob id (action=get).
- `session_id` (string): Session scope key.
- `key` (string): Optional sub-key within the session.
- `ttl_s` (integer): Optional time-to-live in seconds (0 = persistent).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_document_tree`

**graph document tree**

Reasoning-tree (vectorless) document retrieval over a per-document section tree (CONCEPT:AU-KG.retrieval.section-tree/tree-navigation; distills PageIndex). action: 'build' (build + optionally persist the section tree from text or a stored document), 'structure' (return the text-free table-of-contents map = get_document_structure), 'content' (fetch section bodies for cited char ranges like '96..208,300..420' = get_page_content), 'retrieve' (walk the tree by relevance and return sections with cited start..end ranges).

- **Intent verbs:** ask, find
- **REST route:** `/graph/document-tree`
- **MCP tags:** document, graph-os, query, retrieval, tree
- **Side effects:** 0/4 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `build` → (no EG ledger match)
- `content` → (no EG ledger match)
- `retrieve` → (no EG ledger match)
- `structure` → (no EG ledger match)

**Typed input:**

- `action` (string, required): build | structure | content | retrieve
- `document_id` (string): Target Document id (structure/content, and build/retrieve when loading from the graph).
- `text` (string): action=build/retrieve — inline document text (markdown) to build the tree from, instead of loading a stored document.
- `query` (string): action=retrieve — the natural-language query to navigate the tree with.
- `ranges` (string): action=content — char ranges to fetch, e.g. '96..208,300..420'.
- `top_k` (integer): action=retrieve — max sections to return.
- `persist` (boolean): action=build — write the Section nodes/edges to the graph (requires document_id).
- `thin` (boolean): action=build — collapse tiny sections into their parent (token-budget thinning).
- `summarize` (boolean): action=build — compute per-node summaries for a text-free structure map.
- `use_llm` (boolean): action=retrieve — try LLM tree navigation before the lexical walk.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_etl`

**graph etl**

Unified ETL pipeline between systems over the canonical KG hub (CONCEPT:AU-KG.ontology.one-source).

- **Intent verbs:** write
- **REST route:** `/graph/etl`
- **MCP tags:** etl, graph-os, ingestion, ontology
- **Side effects:** 0/2 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `lineage` → (no EG ledger match)
- `list` → (no EG ledger match)

**Typed input:**

- `action` (string): 'run' | 'list' | 'lineage'.
- `source` (string): Ingestion source to pull into the KG (inbound).
- `sink` (string): Write-back domain or graph-store/connection name (outbound).
- `mode` (string): Inbound sync mode: delta|full|reconcile.
- `sources_json` (string): JSON list of source systems to filter a graph-store push.
- `ids_json` (string): JSON list of record ids to narrow the inbound sync.
- `ops_json` (string): JSON write-back payload (inferences/enrichments/creations/…).
- `dry_run` (boolean): Write-back dry-run (fail-closed default).
- `limit` (integer): Max rows for action='lineage'.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_evaluate`

**graph evaluate**

Evaluate agents/harnesses and reason over learned world models.

- **Intent verbs:** why
- **REST route:** `/graph/evaluate`
- **MCP tags:** analyze_suite, evaluate, graph-os
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_evaluate` → (no EG ledger match)

**Typed input:**

- `action` (string): evaluate | evaluate_alpha | evaluate_harness | guard_corpus | harness_gate | check_constraints | specialize | world_model_rollout | latent_efficiency_benchmark | evolve_model | forecast | causal | invariant
- `query` (string): Subject of the evaluation (JSON / id / start state).
- `top_k` (integer): Steps / result count.
- `node_id` (string): Optional node id.
- `depth` (integer): Optional depth.
- `target` (string): Optional target.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_explain`

**graph explain**

The UNIVERSAL context plane (CONCEPT:AU-KG.retrieval.route-question-its-domain): route a question to its domain provider and return ONE grounded, cited answer. action='explain' with target='domain:intent' (e.g. 'ops:why', 'code:usage', 'deploy:status', 'entity:health') — or a bare intent with the domain inferred, or target='domains' to list providers. action='context' returns a synthesized context bundle.

- **Intent verbs:** why
- **REST route:** `/graph/explain`
- **MCP tags:** analyze_suite, explain, graph-os
- **Side effects:** 1/1 actions matched an EG ledger Method; any_mutates=False; durability=['None']; txn=['Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_explain` → EG `ExplainPlan` (confidence 0.5)

**Typed input:**

- `action` (string): explain | context
- `query` (string): The question.
- `top_k` (integer): Result count.
- `node_id` (string): Optional anchor node id.
- `depth` (integer): Optional depth.
- `target` (string): 'domain:intent' | bare intent | 'domains'.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': 'ExplainPlan', 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_federated_search`

**graph federated search**

federated search fanned across registered external graph references.

- **Intent verbs:** ask
- **REST route:** `/graph/federated-search`
- **MCP tags:** engine, engine_surface, federated, graph-os, search
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_federated_search` → (no EG ledger match)

**Typed input:**

- `query` (string, required): Search query (natural language or keywords).
- `references` (string): Comma-separated external graph reference ids (empty ⇒ all).
- `top_k` (integer): Max results to return.
- `params_json` (string): JSON object of extra engine kwargs.
- `graph` (string): Target graph (empty ⇒ deployment default).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_feedback`

**graph feedback**

Record a human correction so the brain learns: correction_type 'outcome' adjusts an entity's reward, 'rule' persists a durable governance/voice/source rule consulted at retrieval time, 'eval' adds a regression case, 'reads_avoided' closes the code_context reads-avoided loop (target_id=capability_id, corrected_value=JSON {reads_avoided,files_read,correct,query}) (CONCEPT:AU-AHE.evaluation.reads-avoided-feedback), 'action_outcome' closes the loop on ANY autonomous action — a context answer, a deploy, a ticket close, a routing choice (target_id=action/capability id, corrected_value=JSON {success,reward?,expected?,observed?,query?}) so routing/playbooks prefer actions that achieve their goal (CONCEPT:AU-AHE.evaluation.action-outcome-feedback), 'gotcha' pins a hard-won trap to a file/module (target_id=path, corrected_value=the note) so code_context surfaces it when an agent next touches that area (CONCEPT:AU-KG.ingest.gotcha-feedback-capture), 'selective_erasure' forgets the learned reward for superseded designations (target_id + optional corrected_value list of ids) so the router re-learns them instead of carrying stale utility across a source/model regime change (CONCEPT:AU-KG.memory.generation-scoped-selective-reward).

- **Intent verbs:** write
- **REST route:** `/graph/feedback`
- **MCP tags:** feedback, graph-os, learning, write_ingest
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_feedback` → (no EG ledger match)

**Typed input:**

- `correction_type` (string, required): outcome | rule | eval | reads_avoided | action_outcome | gotcha | selective_erasure.
- `target_id` (string, required): Entity/episode/query the correction is about.
- `corrected_value` (string): The corrected value (reward, expected output, etc.).
- `reason` (string): Why — the human's explanation.
- `rule_scope` (string): For rule corrections: governance | voice | source | preference.
- `rule_kind` (string): For rule corrections: forbid | prefer | demote.
- `actor_id` (string): Who issued the correction.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_feeds`

**graph feeds**

Manage the unified RSS/Atom feed registry (CONCEPT:AU-KG.ingest.rss-feed-connector/2.122).

- **Intent verbs:** manage
- **REST route:** `/graph/feeds`
- **MCP tags:** feeds, graph-os, state
- **Side effects:** 0/4 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `add` → (no EG ledger match)
- `list` → (no EG ledger match)
- `remove` → (no EG ledger match)
- `sync` → (no EG ledger match)

**Typed input:**

- `action` (string): list|add|remove|sync
- `url` (string): Feed URL (single add/remove).
- `urls` (string): BULK add/remove: many feed URLs in ONE call — a JSON array ('["https://a/feed","https://b/rss"]') or a comma/newline-separated string. Combined with `url` and deduped.
- `mode` (string): delta|full (sync).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_fork`

**graph fork**

warm-fork fan-out over the ORCH-1.86..93 warm-fork primitive (LMCache KV / copy-on-write sandboxes): pay warm-up ONCE for a parent context, then fork N copy-on-write branches to run per-branch computations concurrently and return each branch's result.

- **Intent verbs:** ask
- **REST route:** `/graph/fork`
- **MCP tags:** engine, engine_surface, fanout, fork, graph-os, warm-fork
- **Side effects:** 1/1 actions matched an EG ledger Method; any_mutates=False; durability=['None']; txn=['Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_fork` → EG `Fork` (confidence 1.0)

**Typed input:**

- `code` (string): A single code snippet run on each of 'n' branches (ignored when 'branches_json' is provided).
- `n` (integer): Fan-out count when using 'code' (branches to fork).
- `branches_json` (string): JSON list of per-branch code snippets; overrides code/n.
- `vars_json` (string): JSON object seeding the namespace forked into every branch.
- `sandbox` (string): Preferred warm-fork rung name (empty ⇒ cheapest available).
- `context_query` (string): Optional: retrieve an engine cross-modal candidate set (vector+graph+text) for this query ONCE and fork it into every branch (reused, no recompute).
- `candidate_var` (string): Namespace name the cross-modal candidate set is bound to in each branch (only used when context_query is set).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': 'Fork', 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_gis`

**graph gis**

the engine's GIS surface.

- **Intent verbs:** ask
- **REST route:** `/graph/gis`
- **MCP tags:** engine, engine_surface, geospatial, gis, graph-os
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_gis` → (no EG ledger match)

**Typed input:**

- `action` (string): GIS method: route | tile | nearest | geo_task | ...
- `params_json` (string): JSON object of kwargs for the GIS method (coordinates, profile, tile z/x/y, task name, ...).
- `graph` (string): Target graph (empty ⇒ deployment default).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_goals`

**graph goals**

Orchestrate background/autonomous loops (action in 'create', 'list', 'iterations', 'cancel').

- **Intent verbs:** act, manage
- **REST route:** `/graph/goals`
- **MCP tags:** goals, graph-os, state
- **Side effects:** 0/4 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `cancel` → (no EG ledger match)
- `create` → (no EG ledger match)
- `iterations` → (no EG ledger match)
- `list` → (no EG ledger match)

**Typed input:**

- `action` (string, required): Action: 'create', 'list', 'iterations', 'cancel'
- `goal_id` (string): Target goal ID
- `goal` (string): Goal description/instruction for 'create' action
- `max_iterations` (integer): Max iterations for the autonomous loop

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_hydrate`

**graph hydrate**

Hydrate the Knowledge Graph from configured external sources.

- **Intent verbs:** manage
- **REST route:** `/graph/hydrate`
- **MCP tags:** graph-os, hydration, state
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_hydrate` → (no EG ledger match)

**Typed input:**

- `source` (string): The source connector to hydrate (any registered source), or 'all' to sweep every configured source.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:16Z'}*

---

### `graph_ingest`

**graph ingest**

Smart ingestion for codebases, documents, directories, and conversation logs.

- **Intent verbs:** write
- **REST route:** `/graph/ingest`
- **MCP tags:** graph-os, ingest, write_ingest
- **Side effects:** 0/38 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `agent_toolkit` → (no EG ledger match)
- `archivebox_sync` → (no EG ledger match)
- `build_skill_graph` → (no EG ledger match)
- `cancel` → (no EG ledger match)
- `clear` → (no EG ledger match)
- `cohort_create` → (no EG ledger match)
- `cohort_status` → (no EG ledger match)
- `corpus` → (no EG ledger match)
- `curate_wiki` → (no EG ledger match)
- `distill` → (no EG ledger match)
- `extract_jobs` → (no EG ledger match)
- `extract_jsonl` → (no EG ledger match)
- `extract_pause` → (no EG ledger match)
- `extract_resume` → (no EG ledger match)
- `extract_status` → (no EG ledger match)
- `extract_submit` → (no EG ledger match)
- `fact_extract` → (no EG ledger match)
- `fleet_relevance` → (no EG ledger match)
- `gitlab_sync` → (no EG ledger match)
- `gitlab_webhook` → (no EG ledger match)
- `import_pack` → (no EG ledger match)
- `ingest` → (no EG ledger match)
- `ingest_knowledge_pack` → (no EG ledger match)
- `ingest_url` → (no EG ledger match)
- `job_status` → (no EG ledger match)
- `jobs` → (no EG ledger match)
- `materialize` → (no EG ledger match)
- `materialize_source` → (no EG ledger match)
- `observe` → (no EG ledger match)
- `prioritize` → (no EG ledger match)
- `profile` → (no EG ledger match)
- `rebuild_indexes` → (no EG ledger match)
- `rebuild_skill_graph` → (no EG ledger match)
- `reflect` → (no EG ledger match)
- `skill_graph_status` → (no EG ledger match)
- `skill_workflows` → (no EG ledger match)
- `status` → (no EG ledger match)
- `sync` → (no EG ledger match)

**Typed input:**

- `target_path` (string): Path or JSON list of paths to ingest.
- `max_depth` (integer): Maximum directory depth for codebase ingestion.
- `agent_id` (string): ID of the agent performing the ingestion.
- `action` (string): Action to perform (ingest, ingest_url, archivebox_sync, skill_workflows, fact_extract, distill, import_pack, ingest_knowledge_pack, agent_toolkit, corpus, jobs, job_status, status, cancel, clear, prioritize, rebuild_indexes, observe, materialize, materialize_source, sync, reflect). 'ingest_url' content-aware single-URL ingest (CONCEPT:AU-KG.research.skill-graph-distillation): target_path=URL → fetch via the unified resolver (ArchiveBox→crawl4ai→requests) into a Document, and for a research roundup (auto-detected, or forced with description='extract_papers' / disabled with 'no_papers') download the cited papers via scholarx and ingest them too, linking page→paper; runs inline. 'archivebox_sync' pulls preserved ArchiveBox snapshots into the KG (corpus_name='full' = pull ALL, else delta; base_path=JSON list of snapshot ids to select). 'skill_workflows' ingests the universal-skills workflow corpus (workflows/<domain>/<name>/SKILL.md) into the KG as dispatchable WorkflowDefinition DAGs (+WorkflowStep depends_on edges +USES_SKILL links) in the exact WorkflowStore shape execute_workflow reads, so kg-delegate / graph_orchestrate execute_workflow can discover and fire them; target_path optionally overrides the corpus root, default=installed universal_skills package; idempotent (content-addressed re-ingest is a no-op); runs as a BACKGROUND job (returns a job_id immediately — the full corpus takes ~150s, over the call ceiling — poll with action=job_status job_id=<id>). 'materialize_source' runs an enterprise source extractor (corpus_name=category, e.g. 'camunda'/'aris'/'egeria'; description=optional JSON extractor config), persists its BusinessProcess/BusinessTask/FLOWS_TO batch into the graph via an in-process vendor client, then runs one OWL reasoning cycle so the new process structure folds into the cross-vendor crosswalk. 'fact_extract' turns a document (description=raw text, or target_path=file) into atomic (subject)-[predicate]->(object) fact edges with confidence/evidence/tags, dedups them, persists to the graph, and returns the facts + JSONL. 'extract_submit'/'extract_jobs'/'extract_status'/'extract_pause'/'extract_resume'/'extract_jsonl' run extraction as a GPU-slot-scheduled job (preempt/backfill/resume on the single GPU) addressed by job_id; max_depth sets rounds. 'distill' exports a KG subgraph to a portable skill-graph (target_path=out dir; corpus_name=seed node id OR description=query; max_depth=hop depth). 'import_pack' re-ingests a distilled skill-graph dir back into the KG (target_path=dir; corpus_name='dedup' to merge duplicates). 'build_skill_graph' runs the UNIFIED skill-graph pipeline (CONCEPT:AU-KG.research.skill-graph-distillation): acquire from ANY source kind into one standardized skill-graph (corpus_name=name; target_path=output parent dir; base_path=JSON list of sources [{kind,uri,options}] OR 'kind=uri,kind=uri' shorthand over web/pdf/office/dir/url_reader/rest/database/mcp_tool/generated/kg_query; description=optional human description) — always writes the offline corpus + a sources.json provenance/freshness manifest, and ALSO ingests into the KG when the daemon is reachable (degrades cleanly otherwise). 'skill_graph_status' reports freshness of an existing skill-graph (target_path=dir; corpus_name='quick' to skip network sources). 'rebuild_skill_graph' re-acquires from the recorded sources and bumps the version (target_path=dir). Queue control: 'cancel' (job_id), 'clear' (target_path=status filter pending|running|completed|failed|cancelled|zombie|all, default completed), 'prioritize' (job_id, target_path=high|normal). Research evolution (CONCEPT:AU-KG.ingest.batch-research-cohort): 'cohort_create' (base_path=JSON list of paper URLs, target_path=JSON list of repo paths, description=goal) batch-ingests a cohort of papers+repos whose self-polling barrier synthesizes the comparative feature/innovation matrix (KG-2.173) when every member drains; 'cohort_status' (job_id=cohort_id) returns per-member progress + the matrix counts; 'profile' (corpus_name=lane|type|tkind, CONCEPT:AU-OS.observability.per-lane-latency-metrics) returns per-lane/stage latency percentiles + token/cost + the parallelism factor.
- `job_id` (string): ID of the job to check status for.
- `corpus_name` (string): Name of the corpus to add/update.
- `base_path` (string): Base path for the corpus.
- `description` (string): Description of the corpus.
- `content_type` (string): Internal override only — leave empty. The content type (codebase, document, config, prompt, skill, mcp_server, kb, conversation, policy) is auto-detected from the path, and heavy types (codebase/document) always run on the async job queue. Only set this to force a specific category for an ambiguous path.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_kvcache`

**graph kvcache**

the engine's shared, content-addressed KV-cache over the EG-187 HTTP surface, driven through the KG-2.306 EpistemicGraphKVBackend connector.

- **Intent verbs:** ask
- **REST route:** `/graph/kvcache`
- **MCP tags:** engine, engine_surface, graph-os, kvcache
- **Side effects:** 0/5 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `contains` → (no EG ledger match)
- `exists` → (no EG ledger match)
- `get` → (no EG ledger match)
- `put` → (no EG ledger match)
- `stats` → (no EG ledger match)

**Typed input:**

- `action` (string): get | put | contains | exists | stats
- `key` (string): Opaque block key (get/put/contains/exists).
- `value_b64` (string): Base64-encoded block bytes to store (put).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_learn`

**graph learn**

a pure-Rust KAN (Kolmogorov-Arnold) link-predictor over the resident graph, whose learned per-feature edge functions are THEMSELVES queryable KG nodes (interpretability, not raw accuracy).

- **Intent verbs:** ask
- **REST route:** `/graphlearn/fit`
- **MCP tags:** engine, engine_surface, graph-os, graphlearn, kan, link-prediction, neuro-symbolic
- **Side effects:** 2/2 actions matched an EG ledger Method; any_mutates=True; durability=['GraphRedb']; txn=['Atomic']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `fit` → EG `GraphLearnFit` (confidence 1.0)
- `predict` → EG `GraphLearnPredict` (confidence 1.0)

**Typed input:**

- `action` (string): Graph-learning action: 'fit' | 'predict'.
- `params_json` (string): JSON object of graph-learning kwargs, e.g. {"node_label":"Person","direction":"any","degree":4,"epochs":200,"writeback":true} (fit); {"model":{...},"node_label":"Person","top_k":20,"writeback":true} or {"model":{...},"node_label":"Person","candidate_pairs":[["a","b"],["c","d"]]} (predict).
- `graph` (string): Target graph (empty ⇒ deployment default).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_loops`

**graph loops**

The single entrypoint for long-running objectives (CONCEPT:AU-KG.research.these-properties-carry).

- **Intent verbs:** act, manage
- **REST route:** `/graph/loops`
- **MCP tags:** graph-os, loops, state
- **Side effects:** 0/9 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `cancel` → (no EG ledger match)
- `drive` → (no EG ledger match)
- `list` → (no EG ledger match)
- `prioritize` → (no EG ledger match)
- `review` → (no EG ledger match)
- `run` → (no EG ledger match)
- `specs` → (no EG ledger match)
- `state` → (no EG ledger match)
- `submit` → (no EG ledger match)

**Typed input:**

- `action` (string): submit|list|run|drive|cancel|prioritize|state|specs|review|placement_control
- `objective` (string): Objective text (submit).
- `kind` (string): research|develop|skill (submit).
- `loop_id` (string): Loop id (submit/cancel).
- `validation_cmd` (string): Shell command whose exit-0 completes a develop Loop.
- `end_state` (string): Human end-state (develop).
- `skill_ref` (string): Skill / skill-workflow name or id (skill Loop).
- `max_topics` (integer): Loops to advance per run.
- `limit` (integer): Max rows (list).
- `priority` (string): Priority bucket 0-3 or critical|high|normal|background (submit/prioritize).
- `spec_id` (string): SpecProposal id (review action).
- `decision` (string): approve|edit|reject — spec-review decision (review action).
- `status` (string): Filter SpecProposals by status (specs action): pending_review|approved|developing|published|reverted|rejected.
- `mine_discovery` (any): 'run' only: gate the discovery-flywheel mining stage (CONCEPT:AU-KG.evolution.mining-flywheel). None (default) falls back to config.kg_loop_mine_discovery (default True); explicit true/false overrides.
- `placement_scan_limit` (integer): 'placement_control' only: provenance row-scan cap for the placement mining pass.
- `placement_canary_tolerance` (number): 'placement_control' only: fraction the canary metric may regress by and still be promoted (SLO noise tolerance).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_memory`

**graph memory**

the engine's EG-318 memory surface: episodic→semantic memory, the spatial scene graph, and RL trajectories.

- **Intent verbs:** ask
- **REST route:** `/graph/memory`
- **MCP tags:** engine, engine_surface, graph-os, memory, scene, trajectory
- **Side effects:** 0/3 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `link` → (no EG ledger match)
- `recall` → (no EG ledger match)
- `store` → (no EG ledger match)

**Typed input:**

- `action` (string): Memory method: create_summary | consolidate | maintain | add_scene_object | world_transform | start_trajectory | append_step | discounted_return | get_* | ...
- `params_json` (string): JSON object of kwargs for the memory method, e.g. {"node_ids":["n1","n2"]}, {"object_id":"o1","transform":[...]}, {"trajectory_id":"t1","step":{"state":...,"action":...,"reward":1.0}}, or {"trajectory_id":"t1","gamma":0.99}.
- `graph` (string): Target graph (empty ⇒ deployment default).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_message`

**graph message**

bidirectional, cross-process, ordered message channel between an invoking agent and a spawned agent, over the epistemic-graph native channels.

- **Intent verbs:** act, ask
- **REST route:** `/graph/message`
- **MCP tags:** graph-os, messaging, orchestrate, query
- **Side effects:** 1/5 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Atomic']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `close` → (no EG ledger match)
- `history` → (no EG ledger match)
- `open` → (no EG ledger match)
- `receive` → (no EG ledger match)
- `send` → EG `SendMessage` (confidence 1.0)

**Typed input:**

- `action` (string): open | send | receive | history | close
- `channel_id` (string): Channel id (send/receive/history/close).
- `session_id` (string): Session id (open).
- `run_id` (string): Spawned run id (open).
- `sender` (string): Sender label (send).
- `payload` (string): Message text (send).
- `since` (integer): Cursor: messages already consumed (receive).
- `durable` (boolean): When True (send), also persist the message as a graph AgentMessage node so it survives engine restart and is replayable via action='history'.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_mine`

**graph mine**

the unified data-mining surface over the engine, compute-near-data (mining runs where the graph lives).

- **Intent verbs:** ask
- **REST route:** `/mining/associate`
- **MCP tags:** anomaly, clustering, data-mining, engine, engine_surface, graph-os, mining
- **Side effects:** 10/10 actions matched an EG ledger Method; any_mutates=True; durability=['GraphRedb', 'None']; txn=['Atomic', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `associate` → EG `MineAssociate` (confidence 1.0)
- `cluster` → EG `MineCluster` (confidence 1.0)
- `anomaly` → EG `MineAnomaly` (confidence 1.0)
- `classify_fit` → EG `MineClassifyFit` (confidence 1.0)
- `classify_predict` → EG `MineClassifyPredict` (confidence 1.0)
- `reduce` → EG `MineReduce` (confidence 1.0)
- `sequence` → EG `MineSequence` (confidence 1.0)
- `forecast` → EG `MineForecast` (confidence 1.0)
- `text` → EG `MineText` (confidence 1.0)
- `subgraph` → EG `MineSubgraph` (confidence 1.0)

**Typed input:**

- `action` (string): Mining action: 'associate' | 'cluster' | 'anomaly' | 'classify_fit' | 'classify_predict' | 'reduce' | 'sequence' | 'forecast' | 'text' | 'subgraph'.
- `params_json` (string): JSON object of mining kwargs, e.g. {"transactions":[["bread","milk"],["bread","butter"]],"min_support":0.5,"algorithm":"fpgrowth"} (associate); {"features":[[0,0],[10,10]],"algorithm":"dbscan","eps":1.0,"min_pts":2} or {"source":{"node_label":"Doc"},"algorithm":"kmedoids","k":3,"writeback":true} (cluster); {"values":[1,1,1,100],"algorithm":"zscore"} or {"source":{"node_label":"Metric"},"algorithm":"isoforest"} (anomaly); {"x":[[0,0],[10,10]],"y":[0,1],"algorithm":"logistic"} (classify_fit); {"model":{...},"x":[[0.1,0.1]]} (classify_predict); {"x":[[..]],"algorithm":"svd","n_components":2} or {"source":{"node_label":"Doc"},"algorithm":"umap","writeback":true} (reduce); {"sequences":[["login","browse","purchase"]],"min_support":0.5} or {"source":{"node_label":"Session"},"algorithm":"gsp","writeback":true} (sequence); {"values":[5,8,11,14],"algorithm":"arima","p":1,"d":1,"horizon":5} or {"values":[...],"algorithm":"holtwinters","period":12,"horizon":12} (forecast); {"docs":[["the","cat","sat"]],"algorithm":"tfidf"} or {"source":{"node_label":"Doc","field":"body"},"algorithm":"lda","k":5,"writeback":true} (text); {"min_support":0.1,"max_edges":2,"writeback":true} or {"label":"Concept","algorithm":"motif"} (subgraph).
- `graph` (string): Target graph (empty ⇒ deployment default).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_mine_deep`

**graph mine deep**

the deep-learning / heavy-Python family the engine core deliberately does NOT implement (no torch/GPU in the pure-Rust engine): this tool DISPATCHES to agents/data-science-mcp over MCP (the fleet call_tool_once connector, same one 'fleet.write_record' uses) and folds the result back into the KG as typed nodes (CONCEPT:AU-KG.mining.foldback-typed-nodes).

- **Intent verbs:** ask
- **REST route:** `/mining/deep/deep_forecast`
- **MCP tags:** data-science-mcp, deep-learning, delegation, engine, engine_surface, graph-os, mining
- **Side effects:** 0/5 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `deep_forecast` → (no EG ledger match)
- `deep_classify` → (no EG ledger match)
- `autoencoder_anomaly` → (no EG ledger match)
- `xgboost` → (no EG ledger match)
- `embed` → (no EG ledger match)

**Typed input:**

- `action` (string): Delegated mining action: 'deep_forecast' | 'deep_classify' | 'autoencoder_anomaly' | 'xgboost' | 'embed'.
- `params_json` (string): JSON object of kwargs, e.g. {"values":[5,8,11,14,18],"horizon":5,"lookback":3,"series_id":"metric:cpu","writeback":true} (deep_forecast); {"x":[[0,0],[10,10]],"y":[0,1],"epochs":100,"writeback":true} or {"source":{"node_label":"Doc","fields":["f1","f2"],"limit":200},"y":[0,1,...]} (deep_classify/xgboost); {"x":[[0,0],[0,1],[50,50]],"bottleneck":2,"writeback":true} or {"source":{"node_label":"Metric","fields":["v"],"limit":500}} (autoencoder_anomaly/embed).
- `graph` (string): Target graph (empty ⇒ deployment default).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_observe`

**graph observe**

Reason over the KG-native observability subgraph — traces, online-scores, assertion verdicts, generations, prompt versions — queries an opaque trace store can't do (CONCEPT:AU-KG.ingest.observability-queries-opik-cannot).

- **Intent verbs:** why
- **REST route:** `/graph/observe`
- **MCP tags:** analyze_suite, eval, graph-os, observe
- **Side effects:** 0/3 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `failure_cluster` → (no EG ledger match)
- `prompt_regression` → (no EG ledger match)
- `trace_rootcause` → (no EG ledger match)

**Typed input:**

- `action` (string): trace_rootcause | prompt_regression | failure_cluster
- `query` (string): Optional agent/capability filter (trace_rootcause).
- `top_k` (integer): Max rows/clusters.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_ontology`

**graph ontology**

Hosted-ontology lifecycle CRUD (CONCEPT:AU-KG.ontology.manage-arbitrary) — manage arbitrary OWL/RDF ontologies hosted in the running KG. action='load' (parse + SHACL-validate + register a .ttl/OWL fro

- **Intent verbs:** manage
- **REST route:** `/graph/ontology`
- **MCP tags:** graph-os, lifecycle, ontology
- **Side effects:** 1/11 actions matched an EG ledger Method; any_mutates=True; durability=['Outbox']; txn=['Atomic']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `activate` → (no EG ledger match)
- `deactivate` → (no EG ledger match)
- `delete` → (no EG ledger match)
- `get` → (no EG ledger match)
- `import_stardog` → (no EG ledger match)
- `list` → (no EG ledger match)
- `load` → (no EG ledger match)
- `publish_stardog` → EG `Publish` (confidence 1.0)
- `sync_packages` → (no EG ledger match)
- `update` → (no EG ledger match)
- `validate` → (no EG ledger match)

**Typed input:**

- `action` (string): load | list | get | update | delete | validate | activate | deactivate | sync_packages | publish_stardog | import_stardog.
- `source` (string): For load/update/validate: a .ttl/OWL file path, an HTTP(S) URL, or raw turtle/RDF text.
- `source_type` (string): How to read `source`: 'file' | 'url' | 'text' | 'auto' (sniff).
- `iri` (string): Ontology IRI (get/update/delete/activate/deactivate; optional override for load).
- `version` (string): Ontology version (defaults to '1.0.0' on load; omit on get/delete to target the newest).
- `serialize` (boolean): For action='get': also return the ontology re-serialized to turtle.
- `active_only` (boolean): For action='list': only ontologies currently active for reasoning.
- `drop_inferences` (boolean): For action='delete': also attempt to drop materialized inferences (engine-gap aware).
- `named_graph` (string): For publish_stardog/import_stardog: the Stardog named-graph URI to write to / read from (omit for the default graph).
- `overwrite` (boolean): For publish_stardog: REPLACE the target graph (clear-then-add) so an updated ontology updates the catalog instead of accumulating stale triples.
- `activate` (boolean): For import_stardog: activate the imported ontology for reasoning.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_ops_causal`

**graph ops causal**

Enterprise operations causal graph (Codex X-2): joins Langfuse traces -> agent/tool/model -> service -> deployment/container -> commit/merge-request -> incident/change -> capability/owner -> policy/control/evidence into one causal chain, and runs root-cause/blast-radius/change-risk/control-evidence analyses on it.

- **Intent verbs:** ask
- **REST route:** `/ops/causal`
- **MCP tags:** blast-radius, causal, graph-os, ops, ops_causal, root-cause
- **Side effects:** 0/5 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `root_cause` → (no EG ledger match)
- `blast_radius` → (no EG ledger match)
- `change_risk` → (no EG ledger match)
- `control_evidence` → (no EG ledger match)
- `join` → (no EG ledger match)

**Typed input:**

- `action` (string): root_cause | blast_radius | change_risk | control_evidence | join
- `node_id` (string): Seed node id: the failure/trace (root_cause), the change/commit (blast_radius, change_risk), or the control (control_evidence).
- `links_json` (string): JSON array of ops-causal edges: [{"source":..,"target":..,"rel_type":..,"strength":1.0,"observed_at":null}, ...] or [[source,rel_type,target], ...]. Empty + an active engine ⇒ load the neighborhood live from the KG around node_id (join, root_cause, blast_radius, control_evidence).
- `depth` (integer): Traversal depth bound.
- `max_results` (integer): Result cap (root_cause / blast_radius).
- `incident_history_json` (string): JSON array of {"node_id":..,"severity":0..1} historical incidents (change_risk).
- `now` (number): Unix seconds 'current time' for recency weighting (root_cause); 0 ⇒ no recency weighting.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_orchestrate`

**graph orchestrate**

Orchestrate multi-agent workflows, dispatch subagents, and manage execution loops.

- **Intent verbs:** act
- **REST route:** `/graph/orchestrate`
- **MCP tags:** analysis, graph-os, orchestrate
- **Side effects:** 1/34 actions matched an EG ledger Method; any_mutates=True; durability=['Outbox']; txn=['Atomic']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `assimilate` → (no EG ledger match)
- `compile_process` → (no EG ledger match)
- `compile_workflow` → (no EG ledger match)
- `computer_use` → (no EG ledger match)
- `consensus` → (no EG ledger match)
- `dispatch` → (no EG ledger match)
- `dispatch_workflow` → (no EG ledger match)
- `distill_skills` → (no EG ledger match)
- `enterprise_op` → (no EG ledger match)
- `execute_agent` → (no EG ledger match)
- `execute_workflow` → (no EG ledger match)
- `export_workflow` → (no EG ledger match)
- `failure_ingest` → (no EG ledger match)
- `finance_op` → (no EG ledger match)
- `grant_approval` → (no EG ledger match)
- `list_cron_jobs` → (no EG ledger match)
- `list_workflows` → (no EG ledger match)
- `loop_cycle` → (no EG ledger match)
- `ml_rlm_op` → (no EG ledger match)
- `optimize_component` → (no EG ledger match)
- `publish_proposal` → EG `Publish` (confidence 1.0)
- `request_approval` → (no EG ledger match)
- `rlm_benchmark` → (no EG ledger match)
- `rlm_optimize` → (no EG ledger match)
- `rlm_run` → (no EG ledger match)
- `run_org` → (no EG ledger match)
- `standardize` → (no EG ledger match)
- `start_debate` → (no EG ledger match)
- `status` → (no EG ledger match)
- `submit_risk_veto` → (no EG ledger match)
- `swarm` → (no EG ledger match)
- `synthesize_org` → (no EG ledger match)
- `trigger_cron_job` → (no EG ledger match)
- `workflow_status` → (no EG ledger match)

**Typed input:**

- `action` (string): Action to perform (dispatch, swarm, status, request_approval, grant_approval, execute_agent, computer_use, consensus, start_debate, submit_risk_veto, list_cron_jobs, trigger_cron_job, compile_workflow, compile_process, list_workflows, execute_workflow, export_workflow, synthesize_org, run_org, loop_cycle, assimilate, distill_skills, standardize, failure_ingest, publish_proposal, optimize_component, verify_action). 'verify_action' = pre-execution assurance check (CONCEPT:AU-OS.governance.assurance-state-machine-verifier) of a proposed ActionPolicy routing payload — task=<action kind>, dependencies=JSON {target,params,source,reason,actor_id} — returns the deterministic verdict (allowed/tier/reason/invariant/verify_ms) from ActionPolicy.evaluate() WITHOUT writing an audit/approval node, so a caller can self-check a payload before proposing it for real; the same invariants (role allowed-set, argument schema, state-machine precondition, reference existence) are enforced for real inside ActionPolicy.decide(). 'synthesize_org' = from a goal (in 'task'), the recruiter drafts an org chart (departments → roles) and staffs each role — reusing experienced :Employee staff grown by prior runs, else hiring a fresh template (CONCEPT:AU-ORCH.org.recruiter); optional dependencies JSON {domains:[...]}. 'run_org' = synthesize (or accept) an org, derive a :WorkItem dependency DAG, and run it over the existing orchestrator — independent items parallel, dependents wait, manager review/rework, human escalation on beyond-team blockers, and per-role experience accrual (CONCEPT:AU-ORCH.org.work-item-dag). 'computer_use' = run a GUI computer-use agent (Observe→Ground→Decide→Act) on a gui-sandbox desktop: provisions a sandbox on host=<inventory alias> (or drives an existing container_id=...), governed by ActionPolicy (workspace.computer_use), frames grounded in the KG via observe_screen (CONCEPT:AU-ORCH.execution.computer-use-agent). 'optimize_component' = run a DSPy optimization pass for an evolvable target (task=<system_prompt|tool_description|skill|extraction|concept_match|routing>, dependencies=optional JSON data: documents/labeled_pairs/traces) over the unified target registry + self-supervised optimizers; task='all'/'sweep' runs the propose-only sweep over all self-supervised targets — the on-demand twin of the KG_DSPY_OPTIMIZATION daemon tick (CONCEPT:AU-AHE.assimilation.empirical-parity-evidence-assimilation/3.40/3.44/3.45/3.46); 'loop_cycle' = advance the Loop engine one cycle (CONCEPT:AU-KG.research.these-properties-carry); 'distill_skills' = turn the mapped processes of ALL connected systems (egeria/leanix/aris/camunda) into propose-only atomic-skill + skill-workflow PROPOSALS, connector-agnostic over the ontology (add 'draft' to the task to also render reviewable SKILL.md staging artifacts) (CONCEPT:AU-KG.ontology.connector-agnostic-proposal/2.83); 'swarm' = one-shot goal→decompose→parallel-waves→verify→synthesize (CONCEPT:AU-ORCH.dispatch.kg-governed-agent-swarm); 'standardize' = enterprise standardization + consolidation recommendations (CONCEPT:AU-KG.ontology.populated-at-import-real-3); 'failure_ingest' = pull Langfuse failures → failure_gap topics → regression-gated remediation (CONCEPT:AU-AHE.harness.failure-evolution); 'compile_process' = compile a harvested BusinessProcess node (task=process node id, agent_name=optional workflow name) into an executable WorkflowDefinition with a REALIZES bridge edge (CONCEPT:AU-ORCH.planning.business-process-to-executable); 'publish_proposal' = one-shot evolution→branch bridge — publish a promoted proposal (task=proposal node id) as a reviewable local git branch through the ActionPolicy merge_promotion gate (CONCEPT:AU-AHE.harness.evolution-branch-bridge); 'rlm_benchmark' = run the long-context RLM benchmark (RLM vs vanilla vs compaction) for task=<s_niah|oolong|oolong_pairs|browsecomp_plus|longbench_codeqa>, dependencies=JSON {scales,cases_per_scale}, returning a paper-comparison scoreboard (CONCEPT:AU-AHE.rlm.long-context-benchmark).
- `task` (string): Task description or payload to dispatch.
- `job_id` (string): Job ID for checking status or granting approval.
- `approval_status` (string): Approval status (e.g., 'approved', 'rejected').
- `agent_name` (string): Name of the agent to execute.
- `max_steps` (integer): Maximum steps for agent execution.
- `dependencies` (string): JSON-encoded list of dependency job IDs.
- `completion_state` (string): Strict mathematical or semantic definition of when this workflow is considered done.
- `max_fan_out` (integer): Maximum number of parallel subagents to spawn during adversarial loop.
- `context` (string): CONCEPT:AU-ORCH.session.invoker-agent-handoff — curated context the invoking agent passes to the spawned agent (action='execute_agent'); injected into the spawned agent's prompt, budgeted to the model's context window.
- `budget_tokens` (integer): CONCEPT:AU-ORCH.session.invoker-agent-handoff — optional token budget the invoker grants the spawned agent (action='execute_agent'); enforced as a hard total-tokens limit. 0 = unbounded.
- `context_ref` (string): CONCEPT:AU-ORCH.session.invoker-agent-handoff — id of a persisted ContextBlob (from graph_context put) to hand to the spawned agent (action='execute_agent'); its content is resolved from the graph and injected. Use instead of inline 'context' for large/shared context.
- `allowed_tools` (string): CONCEPT:AU-ORCH.session.invoker-agent-handoff — comma-separated least-privilege tool allow-list for the spawned agent (action='execute_agent'); its tools/toolsets are filtered to ONLY these names. Empty = no restriction.
- `cred_ref` (string): CONCEPT:AU-ORCH.session.invoker-agent-handoff — REFERENCE (secret key, e.g. 'cred:{session}') to an ephemeral credential the invoker stored in the secrets backend; resolved to the spawned agent's auth_token at spawn (never logged). Use instead of passing raw secrets. Empty = none.
- `open_channel` (boolean): CONCEPT:AU-ORCH.session.session-anchored-collections-native — when True (action='execute_agent'), open a native bidirectional message channel for this run; the response JSON includes a 'channel_id' to talk to the spawned agent via graph_message(send/receive).
- `host` (string): CONCEPT:AU-ORCH.execution.computer-use-agent — for action='computer_use': inventory host alias to run the gui-sandbox on (over ssh:// docker/podman). Empty = local docker.
- `container_id` (string): CONCEPT:AU-ORCH.execution.computer-use-agent — for action='computer_use': drive an EXISTING gui-sandbox container by id instead of provisioning a fresh one.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_promql`

**graph promql**

query the engine's observability metrics with PromQL. action='instant' (a single evaluation at 'time', default now) or 'range' (over start..end at 'step').

- **Intent verbs:** ask
- **REST route:** `/graph/promql`
- **MCP tags:** engine, engine_surface, graph-os, metrics, observability
- **Side effects:** 0/2 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `instant` → (no EG ledger match)
- `range` → (no EG ledger match)

**Typed input:**

- `query` (string, required): A PromQL expression.
- `action` (string): instant | range
- `time` (string): Evaluation time (instant), RFC3339/unix.
- `start` (string): Range start (range).
- `end` (string): Range end (range).
- `step` (string): Range step, e.g. '30s' (range).
- `params_json` (string): JSON object of extra engine kwargs.
- `graph` (string): Target graph (empty ⇒ deployment default).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_query`

**graph query**

Execute a read-only Cypher query against the Knowledge Graph.

- **Intent verbs:** ask
- **REST route:** `/graph/query`
- **MCP tags:** graph-os, query
- **Side effects:** 1/1 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Atomic']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_query` → EG `CypherQuery` (confidence 0.5)

**Typed input:**

- `cypher` (string, required): A Cypher query string (read-only — no CREATE/MERGE/DELETE).
- `params` (string): JSON-encoded query parameters.
- `scope` (string): 'local' for the internal KG (Cypher), 'sql' to run read-only SQL over the KG + user tables via the engine's DataFusion surface (e.g. SELECT ... FROM nodes — CONCEPT:AU-KG.query.read-only-sql-over, same path as the pg-wire listener), 'sparql' to run a SPARQL 1.1 SELECT/ASK over the engine's RDF projection of the graph (CONCEPT:AU-KG.ingest.mirror-inbound), or 'federated' to query an external graph endpoint. For 'sql'/'sparql' the `cypher` arg carries the SQL/SPARQL string.
- `reference_id` (string): Required when scope='federated'. The ExternalGraphReference node ID.
- `as_of` (string): CONCEPT:AU-KG.query.as-of-instant-filter — optional ISO-8601 instant. When set, rows are filtered to those whose bi-temporal validity (valid_from <= as_of < valid_to) holds, answering 'what was true as of date T'.
- `target` (string): CONCEPT:AU-KG.backend.multi-connection-registry — named graph connection to query (default = primary). Use a registered connection name (e.g. 'prod-neo4j'), or 'all' (or a comma-separated list) to fan out the same query to several backends and get per-connection labeled results.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': 'CypherQuery', 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_reach`

**graph reach**

reach the user over a messaging backend (Telegram, Slack, Discord, ...).

- **Intent verbs:** ask
- **REST route:** `/graph/reach`
- **MCP tags:** graph-os, messaging, reach
- **Side effects:** 1/5 actions matched an EG ledger Method; any_mutates=False; durability=['None']; txn=['Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `last_channel` → (no EG ledger match)
- `list_channels` → EG `ListChannels` (confidence 1.0)
- `reach_user` → (no EG ledger match)
- `send` → (no EG ledger match)
- `status` → (no EG ledger match)

**Typed input:**

- `action` (string): reach_user | send | list_channels | last_channel | status
- `text` (string): Message text (reach_user/send).
- `platform` (string): Backend id, e.g. 'telegram' (send/list_channels).
- `channel_id` (string): Target channel/chat id (send).
- `user_id` (string): User id for routing (reach_user/last_channel).
- `thread_id` (string): Optional thread id (send).
- `reply_to_id` (string): Optional message id to reply to (send).
- `reason` (string): Why this message is being sent (audit trail).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_research`

**graph research**

Run the research/assimilation pipeline.

- **Intent verbs:** ask
- **REST route:** `/graph/research`
- **MCP tags:** analyze_suite, graph-os, research
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_research` → (no EG ledger match)

**Typed input:**

- `action` (string): synthesize | deep_extract | background_research | relevance_sweep | research_ingest | evolve_variants | track_citations | spawn_background
- `query` (string): Source / topic / artifact.
- `top_k` (integer): Complexity budget / result count.
- `node_id` (string): Optional node id.
- `depth` (integer): Optional depth.
- `target` (string): Optional target.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_runvcs`

**graph runvcs**

Agent-native run version-control (CONCEPT:AU-ORCH.runvcs.run-commit): fork, revert and review a LIVE agent run as content-addressed commits that bind its conversation + filesystem + process/event frontier into ONE exact world. action in 'list' (live run sessions), 'status' (a run's event/commit/message counts + log digest), 'commit' (snapshot messages+fs+events into one RunCommit — pass label), 'revert' (restore a run's files+process+messages to a commit — pass commit_id), 'fork' (branch a NEW run from a commit into a fresh workspace, parent untouched — pass commit_id), 'discard' (drop the uncommitted event delta), 'replay' (deterministically replay the CURRENT live run's event log — a recorded exchange stands in for the model — and verify reproduction).

- **Intent verbs:** act, manage
- **REST route:** `/graph/runvcs`
- **MCP tags:** fork, graph-os, revert, runvcs, state, twin
- **Side effects:** 2/7 actions matched an EG ledger Method; any_mutates=True; durability=['None']; txn=['Saga', 'Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `commit` → EG `Commit` (confidence 1.0)
- `discard` → (no EG ledger match)
- `fork` → EG `Fork` (confidence 1.0)
- `list` → (no EG ledger match)
- `replay` → (no EG ledger match)
- `revert` → (no EG ledger match)
- `status` → (no EG ledger match)

**Typed input:**

- `action` (string): list|status|commit|revert|fork|discard|replay|twin_capture|twin_replay|twin_counterfactual|twin_incident
- `run_id` (string): Target run session id (live-run actions) or run id to hydrate a twin from (twin_capture).
- `commit_id` (string): Target commit id (revert|fork).
- `label` (string): Commit label (commit).
- `twin` (string): JSON-serialized AgentDigitalTwin (CONCEPT:AU-ORCH.twin.agent-digital-twin) — the output of action='twin_capture'. Required by twin_replay/twin_counterfactual/twin_incident.
- `agent_name` (string): Agent name to stamp on the twin (twin_capture).
- `task` (string): Task description to stamp on the twin (twin_capture).
- `versions` (string): JSON VersionPins fields (model_id, model_provider, prompt_version_id, tool_versions, skill_versions, policy_version, policy_digest, catalog_epoch). For twin_capture: the pins this run executed under. For twin_counterfactual: the swapped pins to diff against the twin's recorded pins (reporting only — pass policy_overrides/model_responses to actually change the replay outcome).
- `outcome` (string): Outcome status to stamp on the twin (twin_capture; default 'succeeded').
- `persist` (boolean): twin_capture: best-effort persist the twin as a durable :AgentDigitalTwin KG node (no-op without a live engine).
- `policy_overrides` (string): JSON policy ruleset {version, defaults, rules} for twin_counterfactual — recompute every recorded decision under this swapped policy version and diff against what was originally decided.
- `model_responses` (string): JSON {request: alternate_response} for twin_counterfactual — substitute an alternate model/prompt response for a recorded model exchange and surface the resulting stream divergence.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_sandbox`

**graph sandbox**

Inspect and control the native warm-fork sandbox runtime (CONCEPT:AU-ORCH.sandbox.graph-sandbox-surface).

- **Intent verbs:** act, manage
- **REST route:** `/graph/sandbox`
- **MCP tags:** graph-os, sandbox, state, warm-fork
- **Side effects:** 0/3 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `reap` → (no EG ledger match)
- `status` → (no EG ledger match)
- `warm` → (no EG ledger match)

**Typed input:**

- `action` (string): status|reap|warm
- `rung` (string): Rung to warm (warm): forkserver|container_fork|...

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_schedules`

**graph schedules**

Inspect and control the unified scheduler (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent).

- **Intent verbs:** act, manage
- **REST route:** `/graph/schedules`
- **MCP tags:** graph-os, scheduler, state
- **Side effects:** 0/6 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `disable` → (no EG ledger match)
- `enable` → (no EG ledger match)
- `list` → (no EG ledger match)
- `prioritize` → (no EG ledger match)
- `run_now` → (no EG ledger match)
- `set_interval` → (no EG ledger match)

**Typed input:**

- `action` (string): list|enable|disable|prioritize|set_interval|run_now
- `name` (string): Schedule name (all but list).
- `priority` (string): Bucket 0-3 or critical|high|normal|background (prioritize).
- `interval_s` (number): New interval seconds (set_interval).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_search`

**graph search**

Search the Knowledge Graph using multiple strategies (hybrid, concept, analogy, memory, discover, dci).

- **Intent verbs:** ask
- **REST route:** `/graph/search`
- **MCP tags:** graph-os, query, search
- **Side effects:** 1/1 actions matched an EG ledger Method; any_mutates=False; durability=['None']; txn=['Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_search` → EG `SemanticSearch` (confidence 0.5)

**Typed input:**

- `query` (string, required): Natural language search query or concept ID.
- `mode` (string): Search strategy:
- 'hybrid': Semantic + keyword weighted search (default).
- 'hyde': Memory-first HyDE multi-query plan + dual threshold (CONCEPT:AU-KG.retrieval.self-correcting-second-pass).
- 'deep': Wide-recall single query at the 0.28 deep threshold.
- 'concept': Look up a CONCEPT:ID (e.g. 'AU-KG.query.vendor-agnostic-traversal', 'AU-ORCH.execution.inject-signal-board-observations').
- 'analogy': Find structurally similar concepts.
- 'memory': Search tiered memory (episodic/semantic/procedural).
- 'discover': Cross-reference query against all ingested content.
- 'dci': Direct Corpus Interaction.
- 'latent': Latent-topology hierarchical routing (CONCEPT:AU-KG.memory.auto-similarity-memory-graph).
- 'sira': Single-shot SIRA sparsity-aligned context.
- 'hard_negatives': Mine hard negatives for the query (CONCEPT:AU-KG.memory.auto-similarity-memory-graph).
- 'rerank': Hybrid semantic+keyword re-scoring of candidates.
- 'adore': Iterative query expansion with retrieval-grounded graded relevance feedback + training-free stopping (CONCEPT:AU-KG.query.adore-concept-expansion/2.87).
- 'chrono_ids': Attach an explicit temporal semantic ID (+recency bucket) to each result for generative retrieval (CONCEPT:AU-KG.query.chronoid-fits-residual-quantization).
- 'compiled': Policy-aware ``ContextCompiler`` bundle (CONCEPT:AU-KG.retrieval.context-compiler) — MMR-diversified, evidence/freshness-weighted, token-budgeted, policy-filtered context with citations + a proof graph, instead of the plain relevance-sorted text the other modes return.
- `top_k` (integer): Maximum results to return.
- `self_correct` (boolean): CONCEPT:AU-KG.retrieval.self-correcting-second-pass — run a self-correcting second retrieval pass at the deep threshold when the quality gate fails.
- `as_of` (string): Optional ISO-8601 instant. Pack-driven recency decay is measured relative to this time, enabling knowledge-state-as-of-date-D retrieval such as an academic literature state. Defaults to now (CONCEPT:EG-KG.compute.rust-native-training-loss).
- `target` (string): CONCEPT:AU-KG.backend.multi-connection-registry — named graph connection to search (default = primary). Use a registered connection name, or 'all' (or a comma-separated list) to fan out and get per-connection labeled results.
- `token_budget` (integer): mode='compiled' only: token budget the assembled bundle must fit inside (CONCEPT:AU-KG.retrieval.context-compiler). 0 uses the compiler's default budget.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': 'SemanticSearch', 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_search_synthesis`

**graph search synthesis**

Synthesize a shortcut-resistant deep-search task from the evidence graph, or diagnose realized search difficulty of solver trajectories (CONCEPT:AU-KG.retrieval.evidence-graph-workspace/2.71/2.72, AHE

- **Intent verbs:** ask
- **REST route:** `/graph/search-synthesis`
- **MCP tags:** graph-os, query, search, synthesis, training-data
- **Side effects:** 0/2 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `diagnose` → (no EG ledger match)
- `synthesize` → (no EG ledger match)

**Typed input:**

- `action` (string): 'synthesize': build an evidence subgraph around an answer entity and formulate + adversarially refine a question that forces multi-hop search (no exposed constants / single-clue / co-coverage shortcuts). 'diagnose': score solver trajectories with the FORT signatures (solving cost, answer hit time, prior-shortcut rate) + a search-heavy verdict.
- `answer_id` (string): action=synthesize — node id of the gold answer entity to build the task around.
- `hops` (integer): action=synthesize — evidence-graph BFS depth.
- `fanout` (integer): action=synthesize — max neighbors expanded per node.
- `min_trust` (number): action=synthesize — drop facts whose source_trust is below this.
- `max_per_source` (integer): action=synthesize — max clues allowed to share one evidence source before co-coverage trips.
- `root_popularity` (number): action=synthesize — 0..1 familiarity of the answer entity (high → prior-binding risk).
- `trajectories` (string): action=diagnose — JSON list of trajectories: [{"steps":[{"kind","observation","model_text"}],"answer_aliases":[...]}].

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_secret`

**graph secret**

Manage secrets (CONCEPT:AU-OS.identity.encrypted-secret-store) in the durable, engine-encrypted __secrets__ store (secret VALUES are sealed by the engine's encryption-at-rest; key NAMES + metadata stay queryable).

- **Intent verbs:** manage
- **REST route:** `/graph/secret`
- **MCP tags:** graph-os, secret, security
- **Side effects:** 0/4 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `delete` → (no EG ledger match)
- `get` → (no EG ledger match)
- `list` → (no EG ledger match)
- `set` → (no EG ledger match)

**Typed input:**

- `action` (string): set | get | list | delete
- `key` (string): Secret key (set/get/delete).
- `value` (string): Secret value (set).
- `metadata` (any): Optional non-secret metadata (set).
- `reason` (string): Why this mutation is happening (audit trail).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_sessions`

**graph sessions**

Manage durable sessions (action in 'list', 'get', 'delete', 'reply', 'cancel').

- **Intent verbs:** manage
- **REST route:** `/graph/sessions`
- **MCP tags:** graph-os, sessions, state
- **Side effects:** 0/5 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `cancel` → (no EG ledger match)
- `delete` → (no EG ledger match)
- `get` → (no EG ledger match)
- `list` → (no EG ledger match)
- `reply` → (no EG ledger match)

**Typed input:**

- `action` (string, required): Action: 'list', 'get', 'delete', 'reply', 'cancel'
- `session_id` (string): Target session ID
- `user_reply` (string): Reply content for 'reply' action

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_share`

**graph share**

Share a private node (CONCEPT:AU-KG.compute.data-is-private-its).

- **Intent verbs:** manage
- **REST route:** `/graph/share`
- **MCP tags:** graph-os, ontology, tenancy
- **Side effects:** 0/4 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `commons` → (no EG ledger match)
- `mark` → (no EG ledger match)
- `org` → (no EG ledger match)
- `private` → (no EG ledger match)

**Typed input:**

- `action` (string): 'org' share with my org | 'commons' promote to the shared commons graph | 'mark' attach a marking | 'private' restrict back to me.
- `node_id` (string): Id of the node to share.
- `marking` (string): Marking name (action='mark').

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_table`

**graph table**

mirror data into native engine SQL tables (DataFusion + pg-wire) and manage them.

- **Intent verbs:** ask, write
- **REST route:** `/graph/table`
- **MCP tags:** graph-os, ingestion, query, table
- **Side effects:** 0/6 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `create` → (no EG ledger match)
- `drop` → (no EG ledger match)
- `ingest` → (no EG ledger match)
- `list` → (no EG ledger match)
- `query` → (no EG ledger match)
- `rows` → (no EG ledger match)

**Typed input:**

- `action` (string): 'ingest' | 'rows' | 'create' | 'list' | 'drop' | 'query'.
- `source` (string): Registered connector key (action='ingest').
- `table` (string): Target SQL table name.
- `config_json` (string): JSON connector config (action='ingest').
- `columns_json` (string): JSON list of column names (action='create').
- `rows_json` (string): JSON list of row dicts (action='rows').
- `sql` (string): A read-only SELECT statement (action='query').
- `limit` (integer): Max rows to mirror (action='ingest').
- `replace` (boolean): Drop+recreate the table first (ingest/rows).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_traces`

**graph traces**

search or fetch distributed traces from the engine's observability surface. action='search' (filter by 'service'/'operation'/free-form 'query', capped by 'limit') or 'get' (a single 'trace_id').

- **Intent verbs:** ask
- **REST route:** `/graph/traces`
- **MCP tags:** engine, engine_surface, graph-os, observability, traces
- **Side effects:** 0/2 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `get` → (no EG ledger match)
- `search` → (no EG ledger match)

**Typed input:**

- `action` (string): search | get
- `trace_id` (string): Trace id (action='get').
- `service` (string): Service name filter (search).
- `operation` (string): Operation/span name filter (search).
- `query` (string): Free-form filter expression (search).
- `limit` (integer): Max traces to return (search).
- `params_json` (string): JSON object of extra engine kwargs.
- `graph` (string): Target graph (empty ⇒ deployment default).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_write`

**graph write**

Write nodes, relationships, or register external graphs to the Knowledge Graph.

- **Intent verbs:** write
- **REST route:** `/graph/write`
- **MCP tags:** graph-os, mutation, write, write_ingest
- **Side effects:** 4/14 actions matched an EG ledger Method; any_mutates=True; durability=['GraphRedb']; txn=['Atomic']
- **Cost:** (unmeasured)
- **Latency:** {'add_node': {'eg_method': 'AddNode', 'p50_ms': 0.187, 'p99_ms': 0.223, 'source': 'epistemic-graph/docs/benchmarks.md#results (2026-06-01, UDS, in-memory graph)', 'kind': 'measured'}}
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `add_edge` → EG `AddEdge` (confidence 1.0)
- `add_node` → EG `AddNode` (confidence 1.0)
- `bulk_ingest` → (no EG ledger match)
- `check_loop` → (no EG ledger match)
- `compare_and_set` → (no EG ledger match)
- `delete_edge` → EG `RemoveEdge` (confidence 1.0)
- `delete_node` → EG `RemoveNode` (confidence 1.0)
- `log_chat` → (no EG ledger match)
- `recall_media` → (no EG ledger match)
- `recall_memory` → (no EG ledger match)
- `register_execution` → (no EG ledger match)
- `register_external_graph` → (no EG ledger match)
- `store_memory` → (no EG ledger match)
- `submit_sdd` → (no EG ledger match)

**Typed input:**

- `action` (string, required): Action to perform (add_node, add_edge, delete_node, delete_edge, register_external_graph, bulk_ingest, compare_and_set, store_memory, recall_memory, recall_media, log_chat, submit_sdd, register_execution, check_loop). Use 'compare_and_set' for an ATOMIC conditional update — optimistic concurrency / safe concurrent graph-shaping: it applies 'updates' only if every field in 'conditions' still equals the node's current value (missing field reads as null), so two agents mutating the same node never lose each other's write (conditional state transitions, atomic reservations).
- `node_id` (string): The unique identifier for the node.
- `node_type` (string): The type or label of the node.
- `properties` (string): JSON-encoded dictionary of properties.
- `source_id` (string): The source node ID for an edge.
- `target_id` (string): The target node ID for an edge.
- `rel_type` (string): The relationship type for an edge.
- `endpoint_url` (string): URL for external graph registration.
- `graph_type` (string): Type of external graph (e.g., 'sparql', 'graphql').
- `agent_id` (string): ID of the agent performing the action.
- `nodes` (string): JSON-encoded list of nodes or tags for bulk operations.
- `target` (string): CONCEPT:AU-KG.backend.multi-connection-registry — named graph connection to write to (default = primary). Use a registered connection name, or 'all' (or a comma-separated list) to mirror the SAME write to several backends. Fan-out requires an explicit multi-target value; the default and a single named target stay single-write.
- `conditions` (object): For action='compare_and_set': field→expected-value the node must currently match for the update to apply (a missing field reads as null). e.g. {'status': 'pending'}.
- `updates` (object): For action='compare_and_set': field→new-value to merge into the node ONLY when every condition matches. e.g. {'status': 'claimed', 'owner': 'agent-7'}.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `graph_writeback`

**graph writeback**

Backfeed KG-derived knowledge into an external system-of-record (CONCEPT:EG-KG.storage.nonblocking-checkpoint/2.9). target='leanix'|'servicenow'|'erpnext'|'process'|'capability'. ops: inferences_json [{source,rel_type,target}] (relationships), enrichments_json [{node,patches,tag}], creations_json [{type,name}] (inventory CIs/items/fact sheets), retirements_json [{node}].

- **Intent verbs:** write
- **REST route:** `/graph/writeback`
- **MCP tags:** graph-os, ontology, writeback
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `graph_writeback` → (no EG ledger match)

**Typed input:**

- `target` (string): Write-back target: leanix | servicenow | erpnext | process | capability | (any registered sink).
- `action` (string): 'write' (default), 'proposals' (list queued high-stakes proposals), or 'approve' (apply proposal_id).
- `proposal_id` (string): For action='approve': the queued high-stakes proposal id to apply.
- `inferences_json` (string): JSON list of inferred edges [{source,rel_type,target}] to write as upstream relations.
- `enrichments_json` (string): JSON list of enrichments [{node, patches, tag}] onto existing records.
- `creations_json` (string): JSON list of new records [{type,name,...}] to create upstream (inventory CIs/items).
- `retirements_json` (string): JSON list [{node}] to retire/decommission upstream (highest risk).
- `process_ids_json` (string): For target=process: JSON list of process ids to narrow to.
- `inventory` (boolean): If true, collect the KG's reconciled inventory (infra/topology + LeanIX + TRM, deduped via ALIGNED_WITH) and create the items missing from the target CMDB/ERP.
- `findings` (boolean): If true, file the KG's risk findings (TRM TechnologyRisk: EOL/vuln) as issues in the target tracker (gitlab/github/plane). Pass project context via creations_json[0] or the route.
- `dry_run` (boolean): Preview proposed writes without mutating the system-of-record (default). Set false to apply.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `ingest_sessions`

**ingest sessions**

Ingest AI agent chat/session history into the usage store + KG (CONCEPT:AU-ECO.mcp.client-side-chat-session). 'collect' auto-detects installed agents on THIS host and parses their local logs (use when

- **Intent verbs:** write
- **REST route:** `/usage/ingest-sessions`
- **MCP tags:** graph-os, ingest, observability, write_ingest
- **Side effects:** 0/3 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `collect` → (no EG ledger match)
- `paths` → (no EG ledger match)
- `upload` → (no EG ledger match)

**Typed input:**

- `action` (string): collect | upload | paths
- `bundles_json` (string): For action=upload: JSON array of ParsedSessionBundle objects.
- `target_path` (string): For action=paths: JSON list or comma paths.
- `tenant_id` (string): Tenant scope for the rows.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `nl_query`

**nl query**

ask the Knowledge Graph in plain English, planned by agent-utilities' OWN configured fleet LLM (the local vLLM / provider the rest of AU uses) acting as the engine's NL planner.

- **Intent verbs:** ask
- **REST route:** `/graph/nl-query`
- **MCP tags:** graph-os, nl, query
- **Side effects:** 1/1 actions matched an EG ledger Method; any_mutates=False; durability=['None']; txn=['Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `nl_query` → EG `NlQuery` (confidence 1.0)

**Typed input:**

- `text` (string, required): The natural-language request to answer.
- `dialect` (string): 'auto' (model chooses, prefers uql) or 'uql'|'cypher'|'sql'|'sparql'.
- `schema_hint` (string): Optional extra schema/context hint to ground the planner.
- `execute` (boolean): When false, return only the generated query (preview/dry-run).
- `limit` (integer): Max result rows to return.
- `envelope` (string): 'raw' (default; byte-identical legacy shape) or 'bundle' (additionally wrap the result as an EvidenceBundle under `evidence_bundle`). Additive/opt-in.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': 'NlQuery', 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `object_edits`

**object edits**

Durable object-edit ledger (CONCEPT:AU-KG.ontology.edit-ledger-writeback): record a structured edit (property_set/link_add/link_remove/object_create/object_delete), revert an edit, or read per-object history / as_of snapshot.

- **Intent verbs:** ask
- **REST route:** `/object/edits`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/4 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `as_of` → (no EG ledger match)
- `history` → (no EG ledger match)
- `record` → (no EG ledger match)
- `revert` → (no EG ledger match)

**Typed input:**

- `action` (string): 'record' an edit | 'revert' an edit by id | 'history' per object | 'as_of' snapshot.
- `object_id` (string): Target object id (record/history/as_of).
- `edit_type` (string): property_set|link_add|link_remove|object_create|object_delete (for action='record').
- `properties_json` (string): JSON property map (record property_set/object_create).
- `link_target` (string): Link target id (record link_add/link_remove).
- `link_label` (string): Link label (record link_add/link_remove).
- `edit_id` (string): Edit id (action='revert').
- `ts` (number): Unix timestamp (action='as_of').
- `actor` (string): Acting principal recorded on the edit.
- `expect` (object): For action='record' edit_type='property_set': field→expected current value the object must still match for the set to apply (missing field reads as null). When non-empty the set goes through an atomic compare-and-set (CONCEPT:AU-KG.ontology.optimistic-concurrency-object-property): the edit is recorded ONLY if it wins; an unapplied set returns {'applied': false} and records nothing. Empty (default) = unconditional set, identical to prior behavior. e.g. {'status': 'pending'}.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `object_index`

**object index**

Object Index Lifecycle / Object Data Funnel (CONCEPT:AU-KG.ontology.batch-incremental-sync-live): batch/incremental sync of the live search index from source nodes, report staleness, or reindex stale

- **Intent verbs:** ask
- **REST route:** `/object/index`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/3 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `reindex` → (no EG ledger match)
- `status` → (no EG ledger match)
- `sync` → (no EG ledger match)

**Typed input:**

- `action` (string): 'sync' (batch rebuild) | 'reindex' (reconcile stale) | 'status' (live/tombstone counts).
- `nodes_json` (string): JSON list of source node mappings (sync/reindex).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `object_permissioning`

**object permissioning**

Fine-grained object permissioning (CONCEPT:AU-KG.ontology.redact-object-materialize-restricted): redact an object, materialize a restricted view, or attach a mandatory marking.

- **Intent verbs:** ask
- **REST route:** `/object/permissioning`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/3 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `mark` → (no EG ledger match)
- `redact` → (no EG ledger match)
- `restricted_view` → (no EG ledger match)

**Typed input:**

- `action` (string): 'redact' one object | 'restricted_view' an object set | 'mark' attach a marking.
- `objects_json` (string): JSON list of object dicts (restricted_view).
- `object_json` (string): JSON object dict (redact).
- `node_id` (string): Node id (action='mark').
- `marking` (string): Marking name (action='mark').
- `mask` (boolean): Mask withheld properties instead of dropping them.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `object_set`

**object set**

Object Set Service (CONCEPT:AU-KG.ontology.link-type-pivot/2.38): search/filter/search_around/pivot/aggregate and union/intersect/subtract over Foundry-style object sets.

- **Intent verbs:** ask
- **REST route:** `/object/set`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/9 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `aggregate` → (no EG ledger match)
- `from_ids` → (no EG ledger match)
- `intersect` → (no EG ledger match)
- `of_type` → (no EG ledger match)
- `pivot` → (no EG ledger match)
- `search` → (no EG ledger match)
- `search_around` → (no EG ledger match)
- `subtract` → (no EG ledger match)
- `union` → (no EG ledger match)

**Typed input:**

- `action` (string): of_type|from_ids|search|filter|search_around|pivot|aggregate|union|intersect|subtract.
- `type_or_interface` (string): Object type / interface (of_type).
- `ids_json` (string): JSON list of ids (from_ids / set algebra 'other').
- `query` (string): Search query (search).
- `link_type` (string): Link type (search_around/pivot); empty = any.
- `hops` (integer): Hop count (search_around).
- `direction` (string): out|in|both (search_around/pivot).
- `group_by` (string): Group-by property (pivot/aggregate).
- `metric` (string): count|sum|avg|min|max (aggregate).
- `field` (string): Numeric field (aggregate sum/avg/min/max).
- `limit` (integer): Result limit (search).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `ontology_derive`

**ontology derive**

Compute derived (function/cypher/sparql/embedding-backed) properties live at read time (CONCEPT:AU-KG.ontology.derived-property-registry).

- **Intent verbs:** ask
- **REST route:** `/ontology/derive`
- **MCP tags:** graph-os, ontology
- **Side effects:** 1/4 actions matched an EG ledger Method; any_mutates=False; durability=['None']; txn=['Snapshot']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `compute` → (no EG ledger match)
- `compute_all` → (no EG ledger match)
- `discover_extensions` → EG `Discover` (confidence 1.0)
- `list` → (no EG ledger match)

**Typed input:**

- `action` (string): 'list' declarations, 'compute' one property, 'compute_all', or 'discover_extensions' (propose ontology .ttl extensions from a text sample, CONCEPT:AU-KG.ontology.do-not-auto-merge).
- `object_json` (string): JSON object dict the property is computed for.
- `name` (string): Derived-property name for action='compute'.
- `object_type` (string): Optional object type for declaration resolution; the content/source type for action='discover_extensions'.
- `sample_text` (string): Representative document text for action='discover_extensions'.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `ontology_function`

**ontology function**

Typed, versioned ontology functions: list or invoke through the governed runtime (CONCEPT:AU-KG.ontology.default-runtime-bound-import).

- **Intent verbs:** ask
- **REST route:** `/ontology/function`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/2 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `invoke` → (no EG ledger match)
- `list` → (no EG ledger match)

**Typed input:**

- `action` (string): 'list' registered functions or 'invoke' one.
- `name` (string): Function name for action='invoke'.
- `params` (string): JSON-encoded typed input params.
- `version` (string): Optional pinned semver version.
- `actor` (string): Invoking actor id (recorded in the audit entry).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `ontology_interface`

**ontology interface**

Ontology interfaces: resolve implementers (targeting), check conformance, or emit OWL (CONCEPT:AU-KG.ontology.conformance-check).

- **Intent verbs:** ask
- **REST route:** `/ontology/interface`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/4 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `conforms` → (no EG ledger match)
- `implementers` → (no EG ledger match)
- `list` → (no EG ledger match)
- `owl` → (no EG ledger match)

**Typed input:**

- `action` (string): 'list' interfaces, 'implementers' (resolve an interface/type to concrete types), 'conforms' (check an object), 'owl', or 'explain_routing_eligibility' (X-4 WHY-eligible routing explanation).
- `name` (string): Interface or concrete type name.
- `object_json` (string): JSON object dict for action='conforms'.
- `registry` (string): Which interface registry: 'structural' (built-in shapes) or 'enterprise' (enterprise-standard contracts, CONCEPT:AU-KG.ontology.populated-at-import-real-3).
- `entity_id` (string): Candidate tool/agent node id (action='explain_routing_eligibility').
- `required_capability_type` (string): Required ontology capability type, e.g. 'TransportCapability' (action='explain_routing_eligibility').
- `tenant` (string): Required tenant scope, if any (action='explain_routing_eligibility').
- `policy_tags` (string): Comma-separated required policy tags, if any (action='explain_routing_eligibility').

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `ontology_leanix_sync`

**ontology leanix sync**

Discover the live LeanIX metamodel and mirror it natively as OWL/RDF: regenerates ontology_leanix.ttl (every fact sheet type, relation, field) and registers the types for OWL promotion (CONCEPT:AU-KG.

- **Intent verbs:** ask
- **REST route:** `/ontology/leanix-sync`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `ontology_leanix_sync` → (no EG ledger match)

**Typed input:**

- `dry_run` (boolean): Preview the generated ontology without writing (default). Set false to apply.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `ontology_link_materialize`

**ontology link materialize**

Reify a many-to-many ontology link as a (junction_node, edge_a, edge_b) triple and write it (CONCEPT:AU-KG.domains.trade-journal-bias-auditor).

- **Intent verbs:** ask
- **REST route:** `/ontology/link-materialize`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `types` → (no EG ledger match)

**Typed input:**

- `action` (string): 'types' to list link types, or 'materialize' a junction.
- `link_name` (string): The junction link type name, e.g. 'agent_skill'.
- `source_id` (string): Source endpoint node id.
- `target_id` (string): Target endpoint node id.
- `properties` (string): JSON-encoded junction (link) properties.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `ontology_property_types`

**ontology property types**

List the ontology property-type registry and resolve/validate a Palantir-style type ref (CONCEPT:AU-KG.ontology.ontology-property-types).

- **Intent verbs:** ask
- **REST route:** `/ontology/property-types`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/4 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `column_type` → (no EG ledger match)
- `describe` → (no EG ledger match)
- `list` → (no EG ledger match)
- `validate` → (no EG ledger match)

**Typed input:**

- `action` (string): 'list' all type names, 'describe' a type, 'column_type' a type's column DDL string, or 'validate' a value.
- `type_ref` (string): A type ref, e.g. 'array<string>' or 'vector<768>'.
- `value` (string): JSON-encoded value for action='validate'.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `ontology_sampling_profile`

**ontology sampling profile**

Task-aware LLM sampling profiles (CONCEPT:AU-ORCH.routing.sampling-profile-selection/KG-2.94): list/describe the per-task-class profiles, 'resolve' the profile that would be picked for a prompt/role,

- **Intent verbs:** ask
- **REST route:** `/ontology/sampling-profiles`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/6 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `describe` → (no EG ledger match)
- `evolve` → (no EG ledger match)
- `list` → (no EG ledger match)
- `owl` → (no EG ledger match)
- `resolve` → (no EG ledger match)
- `set` → (no EG ledger match)

**Typed input:**

- `action` (string): 'list' | 'describe' | 'resolve' | 'set' | 'evolve' | 'owl'.
- `task_class` (string): Task class for describe/set/evolve.
- `task_text` (string): Free-text prompt for action='resolve'.
- `role` (string): Functional role for action='resolve'.
- `profile_json` (string): JSON SamplingProfile dict for action='set'.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `ontology_value_types`

**ontology value types**

List/describe constrained ontology value types and validate or coerce a value (CONCEPT:AU-KG.ontology.value-type-shacl-load).

- **Intent verbs:** ask
- **REST route:** `/ontology/value-types`
- **MCP tags:** graph-os, ontology
- **Side effects:** 0/4 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `coerce` → (no EG ledger match)
- `describe` → (no EG ledger match)
- `list` → (no EG ledger match)
- `validate` → (no EG ledger match)

**Typed input:**

- `action` (string): 'list' | 'describe' | 'validate' | 'coerce'.
- `name` (string): The value-type name, e.g. 'EmailAddress'.
- `value` (string): JSON-encoded value for validate/coerce.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `quant`

**quant**

The Ultimate Quant System Tool.

- **Intent verbs:** ask
- **REST route:** `/quant`
- **MCP tags:** ontology
- **Side effects:** 1/14 actions matched an EG ledger Method; any_mutates=False; durability=['None']; txn=['None']
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `analyze` → (no EG ledger match)
- `balances` → (no EG ledger match)
- `cancel_order` → (no EG ledger match)
- `debate` → (no EG ledger match)
- `ensemble_predict` → (no EG ledger match)
- `fundamentals` → (no EG ledger match)
- `historical` → (no EG ledger match)
- `optimize` → (no EG ledger match)
- `order_book` → (no EG ledger match)
- `positions` → (no EG ledger match)
- `regime` → (no EG ledger match)
- `risk_metrics` → EG `Metrics` (confidence 1.0)
- `status` → (no EG ledger match)
- `submit_order` → (no EG ledger match)

**Typed input:**

- `domain` (string, required): 
- `action` (string, required): 
- `ticker` (string): 
- `asset_class` (string): 
- `period` (string): 
- `interval` (string): 
- `side` (string): 
- `quantity` (number): 
- `order_type` (string): 
- `price` (number): 
- `mode` (string): 
- `portfolio_id` (string): 
- `rounds` (integer): 

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `research_artifact`

**research artifact**

Agent-Native Research Artifacts over the one ontology-driven KG (CONCEPT:AU-KG.research.best-effort-lightweight-never/2.80). action in 'reason' (run OWL/RDF reasoning over the whole ecosystem and harv

- **Intent verbs:** manage
- **REST route:** `/research/artifact`
- **MCP tags:** graph-os, ontology, research, state
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `research_artifact` → (no EG ledger match)

**Typed input:**

- `action` (string): reason|compile|review|seal|capture|get|list|inquire
- `topic` (string): Topic to inquire into (inquire).
- `article_id` (string): Paper/article id (compile/review/get).
- `query` (string): Topic for 'reason' (reasoning is ecosystem-wide).
- `level` (string): Seal level: L1|L2|L3 (review).
- `text` (string): Event text (capture).
- `provenance` (string): capture provenance: user|ai_suggested|ai_executed|user_revised.
- `actor` (string): Originating actor id (capture).
- `event_type` (string): Force event type (capture).
- `target_codebase` (string): Codebase to ground claims against (compile).
- `limit` (integer): Max rows (list).
- `materialize` (boolean): Persist inquiry nodes (inquire).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `source_connector`

**source connector**

Document-source connectors (CONCEPT:AU-ECO.connector.document-source-framework–4.29, KG-2.59): list registered connectors, or run one (filesystem/web/rest/database/mcp:<package>/mcp_tool — mcp_tool dr

- **Intent verbs:** ask
- **REST route:** `/connector/source`
- **MCP tags:** connectors, ecosystem, graph-os, ontology
- **Side effects:** 0/2 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `list` → (no EG ledger match)
- `run` → (no EG ledger match)

**Typed input:**

- `action` (string): One of: 'list' (registered connector types), 'run' (build + ingest a connector).
- `source_type` (string): Connector type for 'run' (filesystem/web/rest/database/mcp:<package>/mcp_tool).
- `config` (object): Connector configuration dict for 'run' (e.g. {'root': '/docs'} or {'base_url': 'https://…'}).
- `connector_id` (string): Stable id for incremental checkpoint storage (optional).
- `contextual` (boolean): Enable contextual-retrieval enrichment (CONCEPT:AU-KG.enrichment.contextual-retrieval-enrichment).
- `incremental` (boolean): Use the connector's resumable poll (CONCEPT:AU-ECO.connector.incremental-poll-watermark) vs a full load.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `source_drain`

**source drain**

Watch a chunked async drain started by source_sync(mode='full') on a LARGE corpus (CONCEPT:AU-KG.ontology.single-source-full-drain). action='status' + drain_id returns cumulative progress (pages_done / items_seen / items_ingested) plus a live per-status breakdown of the drain's connector_drain :Task chain. action='list' lists registered chunked-drain sources.

- **Intent verbs:** write
- **REST route:** `/source/drain`
- **MCP tags:** graph-os, ingestion, ontology
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `source_drain` → (no EG ledger match)

**Typed input:**

- `action` (string): 'status' (progress for a drain_id) or 'list' (registered chunked sources).
- `drain_id` (string): The drain handle returned by source_sync(mode='full') (for action='status').

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `source_sync`

**source sync**

THE canonical connector→KG ingestion tool (CONCEPT:AU-KG.ingest.enterprise-source-extractor) — one entrypoint for every external source. source='leanix'|'camunda'|'servicenow'|'gitlab'|… (any registered hydration/materialize source), OR source='all' to sweep EVERY configured connector in one pass (the fleet-wide background-ingest sweep). mode='delta' (only changes since the watermark, default), 'full' (re-mirror all), or 'reconcile' (tombstone records deleted upstream).

- **Intent verbs:** write
- **REST route:** `/source/sync`
- **MCP tags:** graph-os, ingestion, ontology
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `source_sync` → (no EG ledger match)

**Typed input:**

- `source` (string): Registered source to sync (e.g. 'leanix', 'camunda', 'servicenow').
- `mode` (string): 'delta' (watermark poll), 'full' (re-mirror all), or 'reconcile' (tombstone deletions).
- `ids_json` (string): JSON list of record ids to narrow the sync (webhook delta).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `spec_ticket`

**spec ticket**

Link a KG SDD spec/feature to a Plane/Jira work item and make agents assignable (CONCEPT:AU-KG.ingest.enterprise-source-extractor). action='link' (push spec content onto the item, link it, assign to user/agent/act-as-user, comment) or 'pull' (read items assigned to a user — 'what do I own?').

- **Intent verbs:** write
- **REST route:** `/spec/ticket`
- **MCP tags:** graph-os, ontology, sdd, writeback
- **Side effects:** 0/1 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `spec_ticket` → (no EG ledger match)

**Typed input:**

- `action` (string): 'link' or 'pull'.
- `target` (string): 'plane' or 'jira'.
- `spec_json` (string): For action='link': the spec dict {feature_id,title,user_stories,...}.
- `issue_id` (string): The Plane work-item id / Jira issue key.
- `project_id` (string): Plane project id (for plane).
- `assignee` (string): Explicit user id to assign.
- `agent` (string): Agent id to assign (maps via AGENT_USER_MAP).
- `comment` (string): Optional comment to post.
- `user` (string): For action='pull': user whose items to read.
- `dry_run` (boolean): Preview without writing (default).

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---

### `usage_query`

**usage query**

Query usage/cost/observability analytics (CONCEPT:AU-ECO.mcp.usage-cost-observability-surface): token counts, cost, model/tool/skill/db-call usage, session browser, activity heatmap, full-text search, and Langfuse trace links.

- **Intent verbs:** write
- **REST route:** `/usage/query`
- **MCP tags:** graph-os, observability, usage, write_ingest
- **Side effects:** 0/11 actions matched an EG ledger Method; any_mutates=False; durability=[]; txn=[]
- **Cost/Latency:** unmeasured for this capability (no benchmark source)
- **Reliability:** (unmeasured — no live engine reward reachable at generation time)

**Does:**

- `activity` → (no EG ledger match)
- `by_agent` → (no EG ledger match)
- `by_model` → (no EG ledger match)
- `by_project` → (no EG ledger match)
- `search` → (no EG ledger match)
- `series` → (no EG ledger match)
- `session_detail` → (no EG ledger match)
- `sessions` → (no EG ledger match)
- `summary` → (no EG ledger match)
- `tools` → (no EG ledger match)
- `top_sessions` → (no EG ledger match)

**Typed input:**

- `action` (string): summary | by_model | by_project | by_agent | tools | activity | sessions | session_detail | top_sessions | search | traces | series
- `from_date` (string): ISO start (started_at >=).
- `to_date` (string): ISO end (started_at <=).
- `project` (string): Filter by project.
- `agent` (string): Filter by agent type.
- `model` (string): Filter by model.
- `origin` (string): ingested | runtime (omit for both).
- `tenant_id` (string): Tenant scope.
- `session_id` (string): For action=session_detail.
- `query` (string): For action=search (FTS).
- `limit` (integer): Row cap for list actions.

**Eligibility predicates:** eligible(candidate, required) = ontology_subsumption(candidate.capability_type, required) AND tenant_match(candidate.tenant, caller.tenant) AND policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by cosine(embedding) + reward_weight*(bandit_reward-0.5)

**Calibrated outcomes:** (empty — no live bandit reward reachable at generation time)

*Provenance: {'generator_version': '1.0.0', 'source_repo_au': 'agent-utilities', 'source_module_au': 'agent_utilities.mcp.kg_server', 'source_method_eg': None, 'eg_ledger_path': '/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md', 'eg_ledger_available': True, 'generated_at': '2026-07-11T22:23:17Z'}*

---
