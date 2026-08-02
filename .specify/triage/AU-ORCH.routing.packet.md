# Adjudication packet — AU-ORCH.routing

31 live concepts. The deterministic pass already decided 4 pointer(s) and 0 retirement(s) from module locality, git archaeology and id shape alone. Confirm or correct the items below, then write the decisions into .specify/triage/AU-ORCH.routing.yaml.

## Clusters — confirm ONE parent each; the members inherit it

### mcp-child-error-unwrap  (3 concepts)
    agent_utilities/orchestration/agent_runner.py:236 | CONCEPT·AU-ORCH.routing.mcp-child-error-unwrap — when a remote MCP child fails, anyio wraps the real cause
    agent_utilities/graph/_router_impl.py:1472 | # CONCEPT·AU-ORCH.routing.mcp-child-error-unwrap — an expert step that fails by calling a remote MCP tool
    members: filtered-specialist-injection, mcp-child-error-unwrap, transition-state-checkpoint

### route-inference-parameters  (3 concepts)
    agent_utilities/graph/adaptive_agent_router.py:94 | # CONCEPT·AU-ORCH.routing.route-inference-parameters — the inference-parameter bundle for this route, picked from the
    members: route-inference-parameters, topological-routing, uno-orchestra-derived

### altitude-description  (2 concepts)
    agent_utilities/messaging/router.py:575 | # CONCEPT·AU-ORCH.routing.altitude-description — describe the actual altitude: a focused-tools turn runs the
    members: altitude-description, chat-budget-routing

### single-router-edge  (2 concepts)
    agent_utilities/graph/builder.py:786 | # CONCEPT·AU-ORCH.routing.single-router-edge — the router has a SINGLE outgoing edge (→ dispatcher). It must NOT
    members: single-router-edge, structural-build-reuse

## UNDECIDED — the cheap signals ran out here (4)

### chat-budget-routing
    why: the marker text is truncated by the grammar ('# a slow round (CONCEPT·AU-ORCH.routing.chat-budget-routing/4.74). The real cause, if any,') — the id itself reads like a real name, so the marker text needs cleaning either way; decide whether the concept survives that cleanup
    agent_utilities/messaging/router.py:1391 | CONCEPT·AU-ORCH.routing.chat-budget-routing — the run uses the ``chat`` execution profile, so each LLM round is
    agent_utilities/messaging/router.py:1492 | # endpoint (CONCEPT·AU-ORCH.routing.chat-budget-routing) — surface the graceful message. Only a non-timeout

### fallback-logic
    why: the marker exists only in prose/doc files — nothing in the shipped tree realises it, which is usually a retirement but occasionally a real decision recorded only in prose
    docs/examples/graph-os-mcp-examples.md:123 | "target": "CONCEPT·AU-ORCH.routing.fallback-logic",

### sampling-profile-selection
    why: the marker text is truncated by the grammar ('"one_line": "Task-aware LLM sampling profiles (CONCEPT·AU-ORCH.routing.sampling-profile-se') — the id itself reads like a real name, so the marker text needs cleaning either way; decide whether the concept survives that cleanup
    agent_utilities/agent/sampling_profile.py:5 | CONCEPT·AU-ORCH.routing.sampling-profile-selection — Task-aware per-call LLM sampling that selects and threads a SamplingProfile of temperature top_p top_k min_p repetition_penalty
    agent_utilities/knowledge_graph/retrieval/capabilities-power.json:25827 | "one_line": "Task-aware LLM sampling profiles (CONCEPT·AU-ORCH.routing.sampling-profile-selection/KG-2.94): list/describe the per-task-class profiles, 'resolve' the profile that wo

### structural-build-reuse
    why: the marker text is truncated by the grammar ('**P1 — non-blocking:** ✅ **DONE** (CONCEPT·AU-ORCH.routing.structural-build-reuse/1.65) — ') — the id itself reads like a real name, so the marker text needs cleaning either way; decide whether the concept survives that cleanup
    agent_utilities/graph/builder.py:458 | # CONCEPT·AU-ORCH.routing.structural-build-reuse — cache the built graph TOPOLOGY per routing-config. The topology +
    agent_utilities/graph/builder.py:478 | # Warm graph hit (CONCEPT·AU-ORCH.routing.structural-build-reuse): reuse the structural topology + node registry

## Proposed OWN DOCUMENT — is this really a decision? (23)

### adaptive-role-routing
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 3 marker site(s))
    agent_utilities/core/model_factory.py:189 | # CONCEPT·AU-ORCH.routing.adaptive-role-routing — route adaptively from the learned per-role confidence
    agent_utilities/core/model_router.py:4 | """Adaptive local-LLM router (CONCEPT·AU-ORCH.routing.adaptive-role-routing).

### altitude-description
    why: the head of a 2-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/messaging/router.py:575 | # CONCEPT·AU-ORCH.routing.altitude-description — describe the actual altitude: a focused-tools turn runs the

### conductor-per-step-model
    why: a singleton: no sibling shares its source footprint or introducing commit (5 source file(s), 19 marker site(s))
    agent_utilities/models/model_registry.py:73 | CONCEPT·AU-ORCH.routing.conductor-per-step-model — resolved at runtime through :meth:`ModelRegistry.pick_for_task`,
    agent_utilities/graph/executor.py:1808 | # CONCEPT·AU-ORCH.routing.conductor-per-step-model — honor a Conductor-assigned per-step model_id (ctx.inputs is

### confidence-gated-routing-log
    why: a singleton: no sibling shares its source footprint or introducing commit (2 source file(s), 7 marker site(s))
    agent_utilities/models/model_registry.py:474 | # ── CONCEPT·AU-ORCH.routing.confidence-gated-routing-log tier helpers ────────────────────────────────────────────
    agent_utilities/models/model_registry.py:33 | # Ordered tier list for CONCEPT·AU-ORCH.routing.confidence-gated-routing-log confidence-gated routing helpers.

### confidence-signal-forwarding
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 4 marker site(s))
    agent_utilities/core/resource_optimizer.py:142 | for CONCEPT·AU-ORCH.routing.confidence-signal-forwarding confidence-gated routing. Otherwise falls back to the
    agent_utilities/core/resource_optimizer.py:165 | # CONCEPT·AU-ORCH.routing.confidence-signal-forwarding — Forward confidence signal when available

### depth-tiered-sampling
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/rlm/repl.py:658 | # CONCEPT·AU-ORCH.routing.depth-tiered-sampling — depth-tiered sampling. The root is the strong reasoner

### emergent-specialization-discovery-pass
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 2 marker site(s))
    agent_utilities/graph/specialization_discovery.py:6 | CONCEPT·AU-ORCH.routing.emergent-specialization-discovery-pass — an emergent-specialization discovery pass that clusters the failing or expensive task stream and proposes a new spe
    tests/unit/graph/test_orch_collective.py:2 | coordination (CONCEPT·AU-ORCH.routing.virtual-agent-economy-task, CONCEPT·AU-ORCH.routing.emergent-specialization-discovery-pass, CONCEPT·AU-ORCH.dispatch.hierarchical-coordination

### functional-role-resolution
    why: a singleton: no sibling shares its source footprint or introducing commit (3 source file(s), 7 marker site(s))
    agent_utilities/core/model_factory.py:476 | # CONCEPT·AU-ORCH.routing.functional-role-resolution — resolve a functional role (planner/generator/learner/judge)
    agent_utilities/core/model_factory.py:166 | CONCEPT·AU-ORCH.routing.functional-role-resolution. Loads the registry from ``config.model_registry_path`` (kept

### kg-specialist-estimation
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/graph/subagent_patterns.py:247 | CONCEPT·AU-ORCH.routing.kg-specialist-estimation — KG-Driven Specialist Estimation

### load-shedding-backoff
    why: a singleton: no sibling shares its source footprint or introducing commit (4 source file(s), 14 marker site(s))
    agent_utilities/core/model_circuit_breaker.py:1 | """Per-model-endpoint circuit breaker + capacity-aware backpressure (CONCEPT·AU-ORCH.routing.load-shedding-backoff).
    agent_utilities/core/model_circuit_breaker.py:271 | """Return (creating if needed) the cached breaker for ``model`` (CONCEPT·AU-ORCH.routing.load-shedding-backoff).

### mcp-child-error-unwrap
    why: the head of a 3-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/orchestration/agent_runner.py:236 | CONCEPT·AU-ORCH.routing.mcp-child-error-unwrap — when a remote MCP child fails, anyio wraps the real cause
    agent_utilities/graph/_router_impl.py:1472 | # CONCEPT·AU-ORCH.routing.mcp-child-error-unwrap — an expert step that fails by calling a remote MCP tool

### model-fallback-chain
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/capabilities/model_fallback.py:16 | Two pieces, deliberately kept separate (CONCEPT·AU-ORCH.routing.model-fallback-chain):

### offload-sync-roundtrip
    why: a singleton: no sibling shares its source footprint or introducing commit (6 source file(s), 27 marker site(s))
    agent_utilities/orchestration/agent_runner.py:829 | # CONCEPT·AU-ORCH.routing.offload-sync-roundtrip — ``_resolve_agent_from_kg`` runs synchronous backend round-trips;
    agent_utilities/graph/_router_impl.py:141 | # CONCEPT·AU-ORCH.routing.offload-sync-roundtrip — the pre-LLM discovery below is several SYNCHRONOUS engine

### optional-role-override
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/core/config.py:4941 | """CONCEPT·AU-ORCH.routing.optional-role-override — optional role→{tier,tags} overrides for planner/generator/

### original-rule-was-far
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 5 marker site(s))
    agent_utilities/graph/routing/strategies/fast_path.py:1 | """R1 — Fast-path / adaptive model routing (CONCEPT·AU-KG.memory.tiered-memory-caching, widened CONCEPT·AU-ORCH.routing.original-rule-was-far).
    agent_utilities/graph/routing/strategies/fast_path.py:121 | """Return True if ``query`` should take the single-round fast path (CONCEPT·AU-ORCH.routing.original-rule-was-far).

### rejected-candidate-provenance
    why: a singleton: no sibling shares its source footprint or introducing commit (6 source file(s), 17 marker site(s))
    agent_utilities/models/knowledge_graph.py:1658 | """One model-routing decision's chosen model + its rejected alternatives (CONCEPT·AU-ORCH.routing.rejected-candidate-provenance).
    agent_utilities/models/model_registry.py:136 | """One candidate's score/features from a routing decision (CONCEPT·AU-ORCH.routing.rejected-candidate-provenance).

### resolve-body-single-canonical
    why: a singleton: no sibling shares its source footprint or introducing commit (9 source file(s), 18 marker site(s))
    agent_utilities/orchestration/action_policy.py:100 | # auto+notify so it's auditable but frictionless; a dispatch (CONCEPT·AU-ORCH.routing.resolve-body-single-canonical)
    scripts/gen_prompt_schema.py:4 | CONCEPT·AU-ORCH.routing.resolve-body-single-canonical. The ``StructuredPrompt`` model is the single source of truth

### role-specialized-model-routing
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/mcp/tools/analysis_tools.py:3934 | # ── CONCEPT·AU-ORCH.routing.role-specialized-model-routing: Role-Specialized Model Routing ──

### route-inference-parameters
    why: the head of a 3-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/graph/adaptive_agent_router.py:94 | # CONCEPT·AU-ORCH.routing.route-inference-parameters — the inference-parameter bundle for this route, picked from the

### route-outcome-feedback
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 1 marker site(s))
    agent_utilities/knowledge_graph/adaptation/feedback.py:374 | # CONCEPT·AU-ORCH.routing.route-outcome-feedback — a model-route outcome also trains the adaptive router's

### single-router-edge
    why: the head of a 2-concept cluster — the siblings are proposed as pointers at this one, so this is the decision that has to be written down
    agent_utilities/graph/builder.py:786 | # CONCEPT·AU-ORCH.routing.single-router-edge — the router has a SINGLE outgoing edge (→ dispatcher). It must NOT

### topology-escalation-policy
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 2 marker site(s))
    agent_utilities/graph/reasoning/policy.py:6 | CONCEPT·AU-ORCH.routing.topology-escalation-policy
    docs/architecture/reasoning-graph-topologies.md:9 | > `CONCEPT·AU-ORCH.routing.topology-escalation-policy` (cheapest-adequate-first router) ·

### virtual-agent-economy-task
    why: a singleton: no sibling shares its source footprint or introducing commit (1 source file(s), 2 marker site(s))
    agent_utilities/orchestration/agent_market.py:6 | CONCEPT·AU-ORCH.routing.virtual-agent-economy-task — a virtual-agent-economy task allocator that runs a capability-gated second-price auction so a collective self-organizes who doe
    tests/unit/graph/test_orch_collective.py:2 | coordination (CONCEPT·AU-ORCH.routing.virtual-agent-economy-task, CONCEPT·AU-ORCH.routing.emergent-specialization-discovery-pass, CONCEPT·AU-ORCH.dispatch.hierarchical-coordination
