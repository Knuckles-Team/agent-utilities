# Adjudication packet — AU-KG.compute

79 live concepts. The deterministic pass already decided 32 pointer(s) and 11 retirement(s) from module locality, git archaeology and id shape alone. Confirm or correct the items below, then write the decisions into .specify/triage/AU-KG.compute.yaml.

## Clusters — confirm ONE parent each; the members inherit it

### dockerhub-repositories  (18 concepts)
    agent_utilities/knowledge_graph/core/source_sync.py:3950 | # Ops / platform typed connectors (CONCEPT·AU-KG.compute.dockerhub-repositories–2.161) — server-configured, so the
    agent_utilities/knowledge_graph/core/source_sync.py:2606 | """Ingest DockerHub repositories as :Repository + :ContainerImage (CONCEPT·AU-KG.compute.dockerhub-repositories).
    members: audiobookshelf-libraries-books-authors, confluence-first-class-delta, connector-declared-page-drainer, dockerhub-repositories, firefly-iii-accounts-transactions, gitlab-api-gitlab-atlassian, gramps-web-people-families, home-assistant-states, homelab-rss-reader-as, jira-first-class-delta, langfuse-traces-observations, mcp-backed-dedicated-trackers, paperless-ngx-documents-correspondents, plane-first-class-delta, technitium-dns-zones-records, tunnel-manager-hosts, twenty-crm-people-companies, uptime-kuma-monitors

### offloaded-memory-write  (9 concepts)
    agent_utilities/knowledge_graph/core/engine_tasks.py:4359 | # CONCEPT·AU-KG.compute.offloaded-memory-write — a memory write offloaded from a SERVING process. The
    agent_utilities/mcp/tools/write_ingest_tools.py:1693 | # CONCEPT·AU-KG.compute.offloaded-memory-write); the host worker reads it back, parses and
    members: lane-bound-task, offloaded-memory-write, p99-latency-metric, per-channel-embedding-backfill, persistent-task-tracking, reactive-push, registered-edge-type, resolve, task-priority-tag

### graph-compute-engine  (6 concepts)
    agent_utilities/knowledge_graph/core/graph_compute.py:1 | # CONCEPT·AU-KG.compute.graph-compute-engine - High-Performance Graph Compute Engine
    .specify/design/kg-engine-native-compute/design.md:8 | CONCEPT·AU-KG.compute.graph-compute-engine ·
    members: graph-compute-engine, kg-2, tokio-service-layer, vector, when-exposes, when-exposes-native

### numpy-scipy-drop  (5 concepts)
    agent_utilities/numeric/__init__.py:10 | CONCEPT·AU-KG.compute.numpy-scipy-drop — **the hard numpy/scipy drop (Analytics Program P5 final).** numpy
    agent_utilities/harness/__init__.py:68 | # CONCEPT·AU-KG.compute.numpy-scipy-drop's compiled ``epistemic_graph.numeric`` kernel is
    members: executed-p2-p3-rollout, numeric-kernel, numpy-scipy-drop, surface-analytics-program, ufunc-method-surface

### surfaces-universal-latency-signal  (4 concepts)
    agent_utilities/core/model_capacity_autoscale.py:837 | """Drop all cached controllers (test isolation / config reload). CONCEPT·AU-KG.compute.surfaces-universal-latency-signal.
    agent_utilities/core/model_capacity_autoscale.py:766 | """Feed one observed call into a model's adaptive controller (CONCEPT·AU-KG.compute.surfaces-universal-latency-signal).
    members: concurrency-controller-sizing, pure-config-enumeration-fail, same-semantics-as, surfaces-universal-latency-signal

### http-route-graph  (3 concepts)
    agent_utilities/knowledge_graph/enrichment/routes.py:88 | (CONCEPT·AU-KG.compute.http-route-graph). ``service_id`` is the resolved ecosystem Service node id."""
    agent_utilities/mcp/kg_server.py:1701 | """REST twin of graph_analyze action=routes (CONCEPT·AU-KG.compute.http-route-graph): the HTTP route
    members: adr-crud, http-route-graph, world-model-forward-simulation

### first-class-action-conditioned  (2 concepts)
    agent_utilities/knowledge_graph/core/world_model.py:6 | CONCEPT·AU-KG.compute.first-class-action-conditioned — a first-class action-conditioned world model wrapping the graph's Markov transition kernel so an agent can roll an action pol
    tests/test_kg_2_67_world_model.py:1 | """Action-conditioned world model (CONCEPT·AU-KG.compute.first-class-action-conditioned).
    members: first-class-action-conditioned, reuse-model-latent

## Proposed RETIRE — the id names nothing (confirm or rescue) (11)

### data-is-private-its
    why: the id reads as a slugified prose fragment (data-is-private-its)
    agent_utilities/mcp/tools/ontology_tools.py:1981 | description="Share a private node (CONCEPT·AU-KG.compute.data-is-private-its). Data is private-to-its-owner by default; this is the explicit promotion path. action='org' shares wit
    agent_utilities/knowledge_graph/core/tenant_sharing.py:1 | # CONCEPT·AU-KG.compute.data-is-private-its - Hierarchical org-to-user data segmentation: private-by-default owner/scope markers over KG-2.58 tenant graphs, with in-place org shari

### kg-3
    why: the id is a bare legacy pillar reference (kg-3) — a citation of the old KG-N.NN numbering, not a name anyone chose
    agent_utilities/harness/world_model_task.py:4 | """World-model prediction as a SAI specialization domain (CONCEPT·AU-KG.compute.kg-3).
    agent_utilities/knowledge_graph/core/world_model.py:101 | """Learned parametric backend for :class:`WorldModel` (CONCEPT·AU-KG.compute.kg-3).

## UNDECIDED — the cheap signals ran out here (4)

### confluence-first-class-delta
    why: the marker text is truncated by the grammar ('# (``source_sync._DELTA_HANDLERS``, CONCEPT·AU-KG.compute.confluence-first-class-delta/2.1') — the id itself reads like a real name, so the marker text needs cleaning either way; decide whether the concept survives that cleanup
    agent_utilities/knowledge_graph/core/source_sync.py:2300 | """Full-mirror Confluence pages as ``:ConfluencePage`` Documents (CONCEPT·AU-KG.compute.confluence-first-class-delta).
    agent_utilities/core/config.py:3902 | """Confluence instances to mirror into the KG (CONCEPT·AU-KG.compute.confluence-first-class-delta). A JSON list of

### jira-first-class-delta
    why: the marker text is truncated by the grammar ('``_sync_plane``, CONCEPT·AU-KG.compute.jira-first-class-delta/2.125); this generic capabil') — the id itself reads like a real name, so the marker text needs cleaning either way; decide whether the concept survives that cleanup
    agent_utilities/knowledge_graph/core/hydration.py:722 | ``_sync_plane``, CONCEPT·AU-KG.compute.jira-first-class-delta/2.125); this generic capability is now only
    agent_utilities/knowledge_graph/core/source_sync.py:2083 | """Ingest Jira issues as typed issue/person/epic entities (CONCEPT·AU-KG.compute.jira-first-class-delta).

### spectral-cluster-navigator
    why: the marker text is truncated by the grammar ('CONCEPT·AU-KG.compute.spectral-cluster-navigator/2.15/2.34/2.35 — Topological Analysis Eng') — the id itself reads like a real name, so the marker text needs cleaning either way; decide whether the concept survives that cleanup
    agent_utilities/knowledge_graph/core/topological_analysis_engine.py:97 | """Global→local community retrieval (CONCEPT·AU-KG.compute.spectral-cluster-navigator, Deep GraphRAG).
    agent_utilities/knowledge_graph/core/model_display.py:18 | - CONCEPT·AU-KG.compute.spectral-cluster-navigator (AnalogyEngine): Finds similar model displays

## Proposed OWN DOCUMENT — is this really a decision? (32)

### built-ast-extended
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 3 marker site(s))
    agent_utilities/knowledge_graph/enrichment/pipeline.py:98 | # Extended-language tier (CONCEPT·AU-KG.compute.built-ast-extended; engine built with ast-extended).
    agent_utilities/knowledge_graph/core/gitlab_indexer.py:57 | # Extended-language tier (CONCEPT·AU-KG.compute.built-ast-extended).

### capability-abstraction
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/knowledge_graph/core/hydration.py:6 | Architecture (CONCEPT·AU-KG.compute.capability-abstraction — Capability Abstraction Layer):

### change-feed-subscription
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 5 marker site(s))
    agent_utilities/graph/reactive/engine_subscription.py:4 | """CONCEPT·AU-KG.compute.change-feed-subscription — Engine change-feed subscription primitive for poll→push reactivity.
    agent_utilities/orchestration/fleet_autoscaler.py:439 | CONCEPT·AU-KG.compute.change-feed-subscription — the poll→push seam for autoscaling: instead of waiting for

### code-intelligence-tools
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 7 marker site(s))
    agent_utilities/tools/code_intelligence_tools.py:1 | """CONCEPT·AU-KG.compute.code-intelligence-tools — Code-Intelligence Tools: the SWE agent's grounding surface.
    agent_utilities/knowledge_graph/ingestion/gpu_slot_scheduler.py:1 | """Single-GPU-slot inference scheduler (CONCEPT·AU-KG.compute.code-intelligence-tools).

### config-keyed-embedder-client
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 6 marker site(s))
    agent_utilities/core/embedding_utilities.py:268 | """Construct a fresh embedding client (the un-cached path, CONCEPT·AU-KG.compute.config-keyed-embedder-client).
    agent_utilities/core/embedding_utilities.py:219 | # CONCEPT·AU-KG.compute.config-keyed-embedder-client — return the cached client for this exact resolved config

### cross-pillar-synergy
    why: a singleton: no sibling shares its source footprint or introducing commit (8 source file(s), 36 marker site(s))
    agent_utilities/knowledge_graph/retrieval/hybrid_retriever.py:495 | CONCEPT·AU-KG.compute.cross-pillar-synergy: Provides the structural vector to match novel relation topologies
    agent_utilities/knowledge_graph/memory/strategies/heavy_thinking_cache.py:22 | - CONCEPT·AU-KG.compute.cross-pillar-synergy (Hypergraphs): EncPI metadata on trajectory interactions

### dockerhub-repositories
    why: the head of a 18-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/knowledge_graph/core/source_sync.py:3950 | # Ops / platform typed connectors (CONCEPT·AU-KG.compute.dockerhub-repositories–2.161) — server-configured, so the
    agent_utilities/knowledge_graph/core/source_sync.py:2606 | """Ingest DockerHub repositories as :Repository + :ContainerImage (CONCEPT·AU-KG.compute.dockerhub-repositories).

### engine-surface-manifest
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 9 marker site(s))
    agent_utilities/mcp/tools/engine_tools.py:46 | CONCEPT·AU-KG.compute.engine-surface-manifest — Engine surface manifest (client-introspection source of truth)
    scripts/gen_graphos_manifest.py:343 | # Never silently emit a shrunken manifest (CONCEPT·AU-KG.compute.engine-surface-manifest) — a

### epistemic-operations-protocol
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/protocols/epistemic_operations/__init__.py:7 | CONCEPT·AU-KG.compute.epistemic-operations-protocol — one current-only operations

### event-driven-sync
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/knowledge_graph/core/kafka_graph_sync.py:6 | CONCEPT·AU-KG.compute.event-driven-sync — Event-Driven Graph Synchronization

### first-class-action-conditioned
    why: the head of a 2-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/knowledge_graph/core/world_model.py:6 | CONCEPT·AU-KG.compute.first-class-action-conditioned — a first-class action-conditioned world model wrapping the graph's Markov transition kernel so an agent can roll an action pol
    tests/test_kg_2_67_world_model.py:1 | """Action-conditioned world model (CONCEPT·AU-KG.compute.first-class-action-conditioned).

### first-class-reasoner-paradigm
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 2 marker site(s))
    agent_utilities/knowledge_graph/core/reasoner.py:6 | CONCEPT·AU-KG.compute.first-class-reasoner-paradigm — a first-class Reasoner paradigm abstraction with an outcome-learning router that selects a reasoning paradigm per task by capa
    tests/test_kg_2_68_reasoner_router.py:1 | """Pluggable reasoning paradigms + outcome-learning router (CONCEPT·AU-KG.compute.first-class-reasoner-paradigm).

### first-class-rss-atom
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 4 marker site(s))
    agent_utilities/automation/feed_sources.py:300 | """Tombstone a registered feed by its url/key (CONCEPT·AU-KG.compute.first-class-rss-atom). Best-effort."""
    agent_utilities/automation/feed_sources.py:225 | The long-missing "presets→KG" wiring (CONCEPT·AU-KG.compute.first-class-rss-atom): a feed is a first-class

### graph-builder
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 2 marker site(s))
    agent_utilities/graph/builder.py:516 | # Engine-only (CONCEPT·AU-KG.compute.graph-builder): the registry graph persists as
    agent_utilities/graph/builder.py:524 | "agent registry graph (CONCEPT·AU-KG.compute.graph-builder)"

### http-route-graph
    why: the head of a 3-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/knowledge_graph/enrichment/routes.py:88 | (CONCEPT·AU-KG.compute.http-route-graph). ``service_id`` is the resolved ecosystem Service node id."""
    agent_utilities/mcp/kg_server.py:1701 | """REST twin of graph_analyze action=routes (CONCEPT·AU-KG.compute.http-route-graph): the HTTP route

### inductive-knowledge-hypergraphs
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/knowledge_graph/core/hypergraph.py:5 | CONCEPT·AU-KG.compute.inductive-knowledge-hypergraphs: Inductive Knowledge Hypergraphs

### interactive-lane-floor
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 10 marker site(s))
    agent_utilities/core/resource_priority.py:25 | (CONCEPT·AU-KG.compute.interactive-lane-floor) reserves a worker floor that non-interactive lanes can never
    agent_utilities/knowledge_graph/core/task_lanes.py:139 | # CONCEPT·AU-KG.compute.interactive-lane-floor — INTERACTIVE lanes: latency-sensitive work that must ALWAYS

### lane-soft-timeout
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 3 marker site(s))
    agent_utilities/knowledge_graph/core/task_lanes.py:175 | """The soft execution-timeout bound (seconds) for ``lane`` (CONCEPT·AU-KG.compute.lane-soft-timeout)."""
    agent_utilities/knowledge_graph/core/task_lanes.py:180 | """The soft execution-timeout bound (seconds) for a task TYPE (CONCEPT·AU-KG.compute.lane-soft-timeout).

### model-display-optimization
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 8 marker site(s))
    agent_utilities/knowledge_graph/core/model_display.py:4 | """CONCEPT·AU-KG.compute.model-display-optimization — Model Display Optimization.
    agent_utilities/models/imodel.py:39 | # Display strategy enumeration (CONCEPT·AU-KG.compute.model-display-optimization)

### native-sparql-owl-shacl
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 21 marker site(s))
    agent_utilities/knowledge_graph/pipeline/phases/shacl_gate.py:114 | (``get_rdf`` -- one N-Triples round-trip over the live graph, CONCEPT·AU-KG.compute.native-sparql-owl-shacl); when no
    agent_utilities/knowledge_graph/core/owl_bridge.py:1211 | # case). CONCEPT·AU-KG.compute.native-sparql-owl-shacl (over the engine projection, engine concept KG-2.240).

### numpy-scipy-drop
    why: the head of a 5-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/numeric/__init__.py:10 | CONCEPT·AU-KG.compute.numpy-scipy-drop — **the hard numpy/scipy drop (Analytics Program P5 final).** numpy
    agent_utilities/harness/__init__.py:68 | # CONCEPT·AU-KG.compute.numpy-scipy-drop's compiled ``epistemic_graph.numeric`` kernel is

### offloaded-memory-write
    why: the head of a 9-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/knowledge_graph/core/engine_tasks.py:4359 | # CONCEPT·AU-KG.compute.offloaded-memory-write — a memory write offloaded from a SERVING process. The
    agent_utilities/mcp/tools/write_ingest_tools.py:1693 | # CONCEPT·AU-KG.compute.offloaded-memory-write); the host worker reads it back, parses and

### positional-interaction-encoding
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 2 marker site(s))
    agent_utilities/graph/verification.py:1109 | # CONCEPT·AU-KG.compute.positional-interaction-encoding: Compute positional interaction encoding for structural generalization
    agent_utilities/graph/verification.py:1043 | Leverages CONCEPT·AU-KG.compute.positional-interaction-encoding (Inductive Knowledge Hypergraphs) to map derived tactics to

### priority-class-propagation
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 5 marker site(s))
    agent_utilities/observability/correlation.py:86 | # CONCEPT·AU-KG.compute.priority-class-propagation — carry the resource PriorityClass so a spawned/child agent,
    agent_utilities/observability/correlation.py:117 | # CONCEPT·AU-KG.compute.priority-class-propagation — restore the inherited resource priority for the block, so

### single-dropped-connection
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 3 marker site(s))
    agent_utilities/knowledge_graph/core/engine_breaker.py:254 | # Adaptive transient-retry (CONCEPT·AU-KG.compute.single-dropped-connection). A single dropped connection
    tests/unit/knowledge_graph/test_engine_breaker.py:238 | counting a breaker failure — so a blip never cascades (CONCEPT·AU-KG.compute.single-dropped-connection)."""

### structural-fingerprint
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 4 marker site(s))
    agent_utilities/knowledge_graph/core/fingerprint.py:30 | See docs/pillars/architecture_c4.md §CONCEPT·AU-KG.compute.structural-fingerprint
    agent_utilities/knowledge_graph/core/fingerprint.py:4 | """CONCEPT·AU-KG.compute.structural-fingerprint — Structural Fingerprint Engine.

### surfaces-universal-latency-signal
    why: the head of a 4-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/core/model_capacity_autoscale.py:837 | """Drop all cached controllers (test isolation / config reload). CONCEPT·AU-KG.compute.surfaces-universal-latency-signal.
    agent_utilities/core/model_capacity_autoscale.py:766 | """Feed one observed call into a model's adaptive controller (CONCEPT·AU-KG.compute.surfaces-universal-latency-signal).

### symbol-blast-radius
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 2 marker site(s))
    agent_utilities/knowledge_graph/core/blast_radius.py:6 | CONCEPT·AU-KG.compute.symbol-blast-radius — Symbol Blast Radius Analyzer

### topological-analogy
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/knowledge_graph/core/analogy_engine.py:4 | CONCEPT·AU-KG.compute.topological-analogy — Topological Analogy Engine

### topological-mincut-partitioning
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 3 marker site(s))
    agent_utilities/knowledge_graph/core/topological_partition.py:3 | CONCEPT·AU-KG.compute.topological-mincut-partitioning — Mincut Partitioning
    agent_utilities/knowledge_graph/core/topological_partition.py:24 | CONCEPT·AU-KG.compute.topological-mincut-partitioning

### user-override-prompt-library
    why: a singleton: no sibling shares its source footprint or introducing commit (7 source file(s), 12 marker site(s))
    agent_utilities/agent/registry_builder.py:131 | """Compute the namespaced ``PromptNode`` id for a prompt (CONCEPT·AU-KG.compute.user-override-prompt-library).
    agent_utilities/core/schedule_engine.py:500 | # engine-native O(1) compare-and-set (CONCEPT·AU-KG.compute.user-override-prompt-library) — NOT a write-Cypher

### workspace-attention-scoring
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 2 marker site(s))
    agent_utilities/graph/executor.py:295 | # Source 2: MemoryRetriever historical proficiency (soft dependency on CONCEPT·AU-KG.compute.workspace-attention-scoring)
    agent_utilities/graph/executor.py:990 | # CONCEPT·AU-KG.compute.workspace-attention-scoring — WorkspaceAttention scoring for specialist priority
