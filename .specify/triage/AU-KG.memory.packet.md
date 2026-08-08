# Adjudication packet — AU-KG.memory

35 live concepts. The deterministic pass already decided 3 pointer(s) and 1 retirement(s) from module locality, git archaeology and id shape alone. Confirm or correct the items below, then write the decisions into .specify/triage/AU-KG.memory.yaml.

## Clusters — confirm ONE parent each; the members inherit it

### decay-scanner-merge  (2 concepts)
    agent_utilities/knowledge_graph/memory/cli.py:370 | # hygiene (CONCEPT·AU-KG.memory.decay-scanner-merge — decay scanner + semantic merge)
    agent_utilities/knowledge_graph/memory/cli.py:173 | """Run a memory-hygiene pass (CONCEPT·AU-KG.memory.decay-scanner-merge)."""
    members: decay-scanner-merge, targeted-add-update-delete

### ground-truth-preamble-declaring  (2 concepts)
    agent_utilities/knowledge_graph/memory/memory_engine.py:1092 | # CONCEPT·AU-KG.memory.ground-truth-preamble-declaring -- authoritative (durable, injected) memory outranks advisory hints.
    agent_utilities/knowledge_graph/memory/memory_engine.py:945 | # CONCEPT·AU-KG.memory.ground-truth-preamble-declaring -- Ground-Truth preamble declaring authoritative sources up front.
    members: ground-truth-preamble-declaring, team-startup-context

### ingestion-serving-separation  (2 concepts)
    agent_utilities/knowledge_graph/core/engine_memory.py:339 | CONCEPT·AU-KG.memory.ingestion-serving-separation — Ingestion/serving plane separation: in a SERVING role (any
    members: ingestion-serving-separation, parammem

### live-data-science-mcp  (2 concepts)
    agent_utilities/knowledge_graph/memory/weights_distillation.py:716 | # CONCEPT·AU-KG.memory.live-data-science-mcp — LIVE data-science-mcp train dispatch + TrainingJob status poll
    agent_utilities/knowledge_graph/memory/weights_distillation.py:152 | # Hard wall-clock bound on the live data-science-mcp dispatch (CONCEPT·AU-KG.memory.live-data-science-mcp).
    members: live-data-science-mcp, memory-weights-distillation-export

### provides-real-ephemeral-one  (2 concepts)
    agent_utilities/knowledge_graph/memory/timeseries/__init__.py:11 | artifact in prod and the test fixture (CONCEPT·AU-KG.memory.provides-real-ephemeral-one) provides a real ephemeral one, so an
    agent_utilities/knowledge_graph/memory/timeseries/engine_backend.py:19 | test fixture (CONCEPT·AU-KG.memory.provides-real-ephemeral-one) provides a real ephemeral one, so an unreachable
    members: provides-real-ephemeral-one, time-series-lives-one

## Proposed RETIRE — the id names nothing (confirm or rescue) (1)

### parammem
    why: the id is a single generic noun — it names a subject area, not a choice
    agent_utilities/knowledge_graph/core/engine_memory.py:320 | # CONCEPT·AU-KG.memory.parammem — Research: MEMO Survey (2504.01990v2), ParamMem (2604.27707v1)
    agent_utilities/knowledge_graph/core/engine_memory.py:566 | # Context budget compaction (CONCEPT·AU-KG.memory.parammem — Research: 2604.20874v1)

## UNDECIDED — the cheap signals ran out here (3)

### ahe-record-this-base
    why: the marker text is truncated by the grammar ('# Decentralized per-agent memory + exploit/explore bandit (CONCEPT·AU-KG.memory.ahe-record') — the id itself reads like a real name, so the marker text needs cleaning either way; decide whether the concept survives that cleanup
    agent_utilities/harness/__init__.py:129 | # Decentralized per-agent memory + exploit/explore bandit (CONCEPT·AU-KG.memory.ahe-record-this-base/AHE-3.33)
    agent_utilities/harness/agentic_evolution_engine.py:382 | # CONCEPT·AU-KG.memory.ahe-record-this-base / AHE-3.33 — record this base's winners as reusable

### knowledge-currency
    why: the marker exists only in prose/doc files — nothing in the shipped tree realises it, which is usually a retirement but occasionally a real decision recorded only in prose
    agent_utilities/skills/graph-query-and-explanation/SKILL.md:69 | 1. **`explain_provenance_by_ids`** (CONCEPT·AU-KG.memory.knowledge-currency / Seam 1) — take any

### memory-weights-distillation-export
    why: the marker text is truncated by the grammar ('"""Action-core for ``graph_analyze action=distill_memory`` (CONCEPT·AU-KG.memory.memory-we') — the id itself reads like a real name, so the marker text needs cleaning either way; decide whether the concept survives that cleanup
    agent_utilities/knowledge_graph/memory/weights_distillation.py:958 | """Action-core for ``graph_analyze action=distill_memory`` (CONCEPT·AU-KG.memory.memory-weights-distillation-export/2.318).
    agent_utilities/knowledge_graph/memory/weights_distillation.py:105 | # The typed data-science-mcp hand-off CONTRACT (CONCEPT·AU-KG.memory.memory-weights-distillation-export). The export side

## Proposed OWN DOCUMENT — is this really a decision? (28)

### anti-collapse
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/knowledge_graph/memory/optimization_engine.py:1201 | CONCEPT·AU-KG.memory.anti-collapse — Anti-collapse now lives in

### auto-similarity-memory-graph
    why: a singleton: no sibling shares its source footprint or introducing commit (16 source file(s), 85 marker site(s))
    agent_utilities/knowledge_graph/retrieval/capabilities-power.json:21450 | "description": "Search strategy:\n- 'hybrid': Semantic + keyword weighted search (default).\n- 'hyde': Memory-first HyDE multi-query plan + dual threshold (CONCEPT·AU-KG.retrieval.
    agent_utilities/knowledge_graph/retrieval/capabilities-power.json:21403 | "description": "Search strategy:\n- 'hybrid': Semantic + keyword weighted search (default).\n- 'hyde': Memory-first HyDE multi-query plan + dual threshold (CONCEPT·AU-KG.retrieval.

### background-learning-engine
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 11 marker site(s))
    agent_utilities/knowledge_graph/memory/learning_engine.py:39 | CONCEPT·AU-KG.memory.background-learning-engine (+ memory-os typed-extraction enhancement, ClaudioDrews/memory-os@a4ca094,
    agent_utilities/knowledge_graph/memory/learning_engine.py:64 | CONCEPT·AU-KG.memory.background-learning-engine — mirrors Quarq's learn-time relative→absolute rule (agent.py:3114-3161):

### checkpoint-worthiness-scoring
    why: a singleton: no sibling shares its source footprint or introducing commit (6 source file(s), 26 marker site(s))
    agent_utilities/mcp/tools/engine_surface_tools.py:249 | #: (CONCEPT·AU-KG.memory.checkpoint-worthiness-scoring). It MUST outlive a single tool
    agent_utilities/observability/gateway_metrics.py:548 | "Checkpoint-worthiness verdicts (CONCEPT·AU-KG.memory.checkpoint-worthiness-scoring) "

### decay-scanner-merge
    why: the head of a 2-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/knowledge_graph/memory/cli.py:370 | # hygiene (CONCEPT·AU-KG.memory.decay-scanner-merge — decay scanner + semantic merge)
    agent_utilities/knowledge_graph/memory/cli.py:173 | """Run a memory-hygiene pass (CONCEPT·AU-KG.memory.decay-scanner-merge)."""

### drive-one-agent-native
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 10 marker site(s))
    agent_utilities/knowledge_graph/memory/lifecycle.py:176 | """AU-side driver for the engine agent-native-memory primitives (CONCEPT·AU-KG.memory.drive-one-agent-native).
    agent_utilities/knowledge_graph/memory/distillation.py:4 | loop (CONCEPT·AU-KG.memory.drive-one-agent-native, :mod:`agent_utilities.knowledge_graph.memory.lifecycle`)

### episodic-procedural-memory-distillation
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 8 marker site(s))
    agent_utilities/knowledge_graph/memory/distillation.py:425 | """Distill one recurring cluster into a procedural artifact (CONCEPT·AU-KG.memory.episodic-procedural-memory-distillation).
    agent_utilities/knowledge_graph/memory/distillation.py:517 | """CLI/daemon/lifecycle entry for one distillation cycle (CONCEPT·AU-KG.memory.episodic-procedural-memory-distillation).

### generation-scoped-selective-reward
    why: a singleton: no sibling shares its source footprint or introducing commit (4 source file(s), 14 marker site(s))
    agent_utilities/knowledge_graph/retrieval/capabilities-power.json:14205 | "one_line": "Record a human correction so the brain learns: correction_type 'outcome' adjusts an entity's reward, 'rule' persists a durable governance/voice/source rule consulted a
    agent_utilities/knowledge_graph/retrieval/capability_index.py:143 | # CONCEPT·AU-KG.memory.generation-scoped-selective-reward — generation-scoped selective reward erasure for memory maintenance.

### ground-truth-preamble-declaring
    why: the head of a 2-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/knowledge_graph/memory/memory_engine.py:1092 | # CONCEPT·AU-KG.memory.ground-truth-preamble-declaring -- authoritative (durable, injected) memory outranks advisory hints.
    agent_utilities/knowledge_graph/memory/memory_engine.py:945 | # CONCEPT·AU-KG.memory.ground-truth-preamble-declaring -- Ground-Truth preamble declaring authoritative sources up front.

### ingestion-serving-separation
    why: the head of a 2-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/knowledge_graph/core/engine_memory.py:339 | CONCEPT·AU-KG.memory.ingestion-serving-separation — Ingestion/serving plane separation: in a SERVING role (any

### kv-checkpoint-resource
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 31 marker site(s))
    agent_utilities/mcp/tools/engine_surface_tools.py:1595 | """Thin verb over :class:`~agent_utilities.kvcache.KVCheckpointStore` (CONCEPT·AU-KG.memory.kv-checkpoint-resource)
    agent_utilities/kvcache/checkpoint.py:311 | "[CONCEPT·AU-KG.memory.kv-checkpoint-resource] txn conflict for %s (not committed), compensating unref",

### layered-project-context
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/prompting/builder.py:6 | CONCEPT·AU-KG.memory.layered-project-context — Project-Aware Context

### live-data-science-mcp
    why: the head of a 2-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/knowledge_graph/memory/weights_distillation.py:716 | # CONCEPT·AU-KG.memory.live-data-science-mcp — LIVE data-science-mcp train dispatch + TrainingJob status poll
    agent_utilities/knowledge_graph/memory/weights_distillation.py:152 | # Hard wall-clock bound on the live data-science-mcp dispatch (CONCEPT·AU-KG.memory.live-data-science-mcp).

### live-refreshable-artifact-models
    why: a singleton: no sibling shares its source footprint or introducing commit (6 source file(s), 11 marker site(s))
    agent_utilities/knowledge_graph/live_artifacts/models.py:1 | """CONCEPT·AU-KG.memory.live-refreshable-artifact-models — Live Refreshable Artifact models + bounded-JSON + safe interpolation.
    agent_utilities/knowledge_graph/live_artifacts/store.py:1 | """CONCEPT·AU-KG.memory.live-refreshable-artifact-models — Live Artifact store (in-memory index + optional KG persistence).

### mementified-context
    why: a singleton: no sibling shares its source footprint or introducing commit (5 source file(s), 26 marker site(s))
    agent_utilities/knowledge_graph/memory/memento_compressor.py:612 | # ── Semantic-boundary segmentation (CONCEPT·AU-KG.memory.mementified-context MEM-3, paper §Stage 1-3) ────────────────────
    agent_utilities/knowledge_graph/memory/memento_compressor.py:17 | * **Judge-refine loop** (CONCEPT·AU-KG.memory.mementified-context / paper §Stage 4) — a compressor→judge→recompress cycle.

### memento-compress-evict
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/capabilities/composition.py:99 | # CONCEPT·AU-KG.memory.memento-compress-evict — the live block-compress-evict sawtooth

### memory-lifecycle-manager
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 2 marker site(s))
    agent_utilities/harness/evolving_memory.py:78 | """Persist, reconcile, and resolve typed memory records (CONCEPT·AU-KG.memory.memory-lifecycle-manager)."""
    agent_utilities/harness/evolving_memory.py:6 | CONCEPT·AU-KG.memory.memory-lifecycle-manager — Unified Memory Manager

### observational-memory-bridge
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/core/paths.py:229 | CONCEPT·AU-KG.memory.observational-memory-bridge — Observational Memory Bridge

### persistent-self-model-owl
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 3 marker site(s))
    agent_utilities/knowledge_graph/retrieval/memory_retriever.py:4 | """CONCEPT·AU-KG.memory.persistent-self-model-owl — Persistent Self-Model with OWL Integration.
    agent_utilities/knowledge_graph/retrieval/memory_retriever.py:24 | See docs/pillars/architecture_c4.md §CONCEPT·AU-KG.memory.persistent-self-model-owl

### provides-real-ephemeral-one
    why: the head of a 2-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/knowledge_graph/memory/timeseries/__init__.py:11 | artifact in prod and the test fixture (CONCEPT·AU-KG.memory.provides-real-ephemeral-one) provides a real ephemeral one, so an
    agent_utilities/knowledge_graph/memory/timeseries/engine_backend.py:19 | test fixture (CONCEPT·AU-KG.memory.provides-real-ephemeral-one) provides a real ephemeral one, so an unreachable

### refresh-per-session-memento
    why: a singleton: no sibling shares its source footprint or introducing commit (4 source file(s), 13 marker site(s))
    agent_utilities/knowledge_graph/memory/session_memento_cache.py:33 | """Process-local LRU of ``{source -> (mementos, fetched_at)}`` (CONCEPT·AU-KG.memory.refresh-per-session-memento).
    agent_utilities/messaging/router.py:792 | # CONCEPT·AU-KG.memory.refresh-per-session-memento — refresh the per-session memento cache in this SAME background

### refresh-service
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/knowledge_graph/live_artifacts/refresh.py:1 | """CONCEPT·AU-KG.memory.refresh-service — Refresh service: re-derive artifact data from the KG; preserve prior on failure.

### rlm-memory-extraction
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/knowledge_graph/memory/rlm_memory.py:3 | CONCEPT·AU-KG.memory.rlm-memory-extraction — RLM Memory Extraction Signature

### semantic-response-cache
    why: a singleton: no sibling shares its source footprint or introducing commit (6 source file(s), 17 marker site(s))
    agent_utilities/models/knowledge_graph.py:555 | # CONCEPT·AU-KG.memory.semantic-response-cache — a trace's semantic-cache decision (hit/miss + the full key
    agent_utilities/models/knowledge_graph.py:196 | # A semantic-cache hit/miss decision (CONCEPT·AU-KG.memory.semantic-response-cache) — the key components

### tiered-memory-caching
    why: a singleton: no sibling shares its source footprint or introducing commit (28 source file(s), 158 marker site(s))
    agent_utilities/graph/routing/strategies/fast_path.py:1 | """R1 — Fast-path / adaptive model routing (CONCEPT·AU-KG.memory.tiered-memory-caching, widened CONCEPT·AU-ORCH.routing.original-rule-was-far).
    agent_utilities/models/knowledge_graph.py:376 | # Company Operations (CONCEPT·AU-KG.research.research-pipeline-runner, CONCEPT·AU-KG.memory.tiered-memory-caching)

### unified-memory-crud-core
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 4 marker site(s))
    agent_utilities/mcp/tools/engine_surface_tools.py:883 | # CONCEPT·AU-KG.memory.unified-memory-crud-core — the unified memory-CRUD core. graph_memory's recall/store/link
    agent_utilities/mcp/tools/engine_surface_tools.py:2050 | # CONCEPT·AU-KG.memory.unified-memory-crud-core — unified memory-CRUD short-circuit: recall/store/link go

### universal-knowledge-assimilation
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 4 marker site(s))
    agent_utilities/knowledge_graph/kb/knowledge_classifier.py:4 | CONCEPT·AU-KG.memory.universal-knowledge-assimilation — Universal Knowledge Assimilation
    agent_utilities/knowledge_graph/kb/x_workflows.py:177 | "CONCEPT·AU-KG.memory.universal-knowledge-assimilation",

### working-set-eviction
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/knowledge_graph/core/working_set_manager.py:6 | CONCEPT·AU-KG.memory.working-set-eviction — Working Set Eviction & Memory Management
