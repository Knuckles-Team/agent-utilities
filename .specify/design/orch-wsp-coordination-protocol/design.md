# Design Document: Coordination is an explicit architectural layer, not implicit in agent-to-agent messages

CONCEPT:AU-ORCH.execution.coordination-protocol-metadata ·
CONCEPT:AU-ORCH.execution.coordination-negotiation

> `agent_utilities/graph/coordination.py` (primary — `CoordinationLayer` and
> its supporting models). Pointer call site in
> `agent_utilities/protocols/a2a_graph_skill.py`.

## Decision — a pluggable, KG-observed `CoordinationLayer` sits between team composition and graph execution

`CONCEPT:AU-ORCH.execution.coordination-protocol-metadata`

The module docstring cites the design's source directly: research paper
2605.03310v1, "Coordination as an Architectural Layer for LLM-Based
Multi-Agent Systems." Its key insight, quoted in the module: *"Coordination
in LLM-based multi-agent systems should be treated as an explicit
architectural layer"* rather than being implicit in agent-to-agent
communication (`coordination.py:1-24`). The module implements that
principle with three parts: a declarative `CoordinationProtocol` (protocol
type — consensus/voting/delegation/handoff/broadcast, `ProtocolType`,
93-103 — plus convergence criterion, round/timeout limits, `116-145`); a
`CoordinationLayer` that *selects* a protocol from five `BUILTIN_PROTOCOLS`
(178-220) via a deterministic heuristic on agent count and execution mode,
overridable by historical KG success rates
(`select_protocol()`/`_lookup_best_protocol()`, 280-367, querying
`CoordinationTrace` nodes and requiring at least 3 data points before
trusting history); and `apply_protocol()` (371-449), which computes a
per-protocol-type `quality_score` forecast and returns a `CoordinationResult`
that `log_coordination_trace()` (453-505) persists back to the KG as a
`CoordinationTrace` node — closing the loop that `_lookup_best_protocol`
reads from.

**The rejected alternative** is the status quo the cited paper argues
against and the module docstring names explicitly: implicit coordination,
where agents negotiate turn-taking, agreement, and authority ad hoc through
their own prompts/messages inside whatever execution path happens to run
them. That loses because there is no single place to observe, tune, or
learn from "how well did N agents coordinate on this task type" — no
`CoordinationTrace` to query, and swapping a team from voting to consensus
would mean touching every call site's prompting instead of one protocol
selection. The chosen design also draws an explicit boundary the docstring
states outright: `apply_protocol()` "is the synchronous coordination step
that occurs *before* graph execution begins... the actual coordination
happens in the graph nodes themselves — this sets up the protocol"
(371-385). The layer decides and forecasts; it does not itself coordinate.

### Pointer — `CONCEPT:AU-ORCH.execution.coordination-negotiation`

Grounded at `agent_utilities/protocols/a2a_graph_skill.py:218-259`
(`CoordinatedPlannerSkill.run()`). The decision: an A2A-exposed graph skill
negotiates its coordination protocol — `coordination_layer.select_protocol()`
then `apply_protocol()` (218-229) — **before** invoking `execute_graph()`,
and always writes the resulting `CoordinationTrace` afterward on both paths:
on success, right after the graph result comes back (line 248); on failure,
`coord_result.converged` is explicitly flipped to `False` and the exception
text captured into `coord_result.metadata["error"]` before the (still
attempted) trace log (254-259), so a failed coordination round is itself a
KG-visible data point rather than a silently dropped one. This is the one
call site where `CoordinationLayer` protocol negotiation is wired directly
into an *inbound A2A JSON-RPC* skill (`skill_id="coordinated_planner"`,
line 165) rather than into the internal `AgentOrchestrationEngine.synthesize_team()`
path — the "an external caller's request negotiates a protocol before
graph execution runs" instance of the coordination-protocol-metadata
decision above, not a second coordination mechanism.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/coordination.py`,
  `agent_utilities/protocols/a2a_graph_skill.py`,
  `agent_utilities/models/knowledge_graph.py` (`TeamComposition.coordination_protocol`
  metadata field, populated with this layer's output).
- **Backward Compatible**: Yes — protocol selection falls back to the
  deterministic heuristic whenever KG history is absent or under 3 samples.
- **Known weak point**: `_lookup_best_protocol()`'s KG query and
  `log_coordination_trace()`'s KG write are both wrapped in broad
  `except Exception` handlers that only `logger.debug` on failure
  (`coordination.py:364-365`, `503-505`) — a KG outage silently drops both
  the historical-selection signal and the trace being written, degrading to
  the static heuristic with no visible error.
