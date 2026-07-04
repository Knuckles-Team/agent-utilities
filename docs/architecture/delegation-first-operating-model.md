# The delegation-first operating model — local LLM + graph-os do the work; the harness orchestrates + resolves exceptions

> The architecture-doc form of the canonical `AGENTS.md` discipline
> *"Delegate to the KG + graph-os — you are the orchestrator + exception-resolver"*.
> Concepts: KG-2.296 (`:ToolCall` / `RunTrace` provenance) · AU-ORCH.scheduling.resource-priority-edict/1.99
> (resource-priority edict) · AU-ECO.mcp.full-api-mcp-surface (full engine MCP/REST surface) · ORCH-1.95/96/97
> (the execution seam) · ORCH-1.100/1.101 (the `agent-utilities-expert`) · AU-KG.retrieval.kg-4
> (cross-layer troubleshooting) · AU-KG.research.these-properties-carry / AU-KG.research.evolutionstate-live-surface-per/2.291 (the Loop engine + evolution
> state) · AU-OS.config.autonomous-spec-develop-off (the spec review-veto gate) · AU-AHE.optimization.telemetry-optimization (the hardening loop).

## The principle

The platform is built so that **the local LLM + graph-os do the work**, and Claude / the
harness **orchestrates and resolves exceptions**. The standing default is **delegate as
much as possible; reserve direct action for what the autonomous system genuinely cannot do
yet** — the trajectory is to orchestrate *off the harness* over time.

This is not a style preference; it is the design center of the whole system. Every
sub-system documented elsewhere — the execution seam, the expert agent, the troubleshoot
provider, the engine surface, the resource-priority edict — exists to make delegation both
**capable** (the local model can actually run the work against real tools) and **safe**
(every run is fully visible, steerable, and non-starving).

Your job as the harness becomes two things:

1. **Orchestrate** — decompose a goal, dispatch it to graph-os / the local LLM, and
   **steer** it (query live `EvolutionState`, the `:ToolCall` / `RunTrace` provenance,
   reprioritize, approve/veto).
2. **Resolve exceptions** — when a delegated run fails, returns a wrong or ungrounded
   answer, or the system couldn't self-troubleshoot, **that** is your job: read the
   `RunTrace` / `:ToolCall` to see exactly what the local LLM did, find **why**, fix the
   gap, and re-delegate.

## The three delegation routes

Before doing anything yourself, route it to one of three delegation paths.

### 1. Understand code → the KG, never grep first

To learn how an area works, where a symbol is used, or what a change impacts, query the
**code KG first**:

```text
graph_analyze action=code_context  query="<area/symbol/question>"  target=how|usage|impact
```

(REST `POST /graph/analyze/code-context`, CONCEPT:AU-KG.retrieval.synthesized-cited-answer.) `how` returns a definition +
what it calls + owning CONCEPT + docs + routes; `usage` returns callers (`file:line`) +
near-clones + the cross-repo usage view (AU-KG.retrieval.every-usage-published-symbol); `impact` returns transitive callers
(blast radius) + git change-coupling. Read only the few `file:line`s you must **edit**,
not to understand — then close the loop with `graph_feedback correction_type=reads_avoided`
(AU-AHE.evaluation.reads-avoided-feedback) so the retriever learns which answers replace a read. If an area is uningested,
`source_sync source=all mode=delta` first, then fall back to grep. (See
[`codebase-context.md`](codebase-context.md).)

### 2. Do a task → `graph_orchestrate` on the local LLM

Hand a task an ingested skill / workflow / agent can already do to the local model:

```text
graph_orchestrate action=execute_agent    agent=agent-utilities-expert  task="<ecosystem task>"
graph_orchestrate action=execute_agent    agent=<ingested-skill>        task="…"
graph_orchestrate action=execute_workflow name=<workflow>               …
```

The **`agent-utilities-expert`** is the default delegate for ecosystem work — a native,
KG-bound, dispatchable persona that grounds its answers in graph-os instead of
hallucinating (see [`agent-utilities-expert.md`](agent-utilities-expert.md)). For a task
that maps cleanly onto one known capability, dispatch that ingested skill / workflow
directly. Either way the **execution seam**
([`orchestration-execution-seam.md`](orchestration-execution-seam.md)) runs it on the
local vLLM against real MCP tools and writes full provenance. Reach the rest of the
~58-server fleet via the `engine_<domain>` surface + the multiplexer meta-tools
(`find_tools` / `load_tools`).

### 3. Evolve / manage the ecosystem → the loop engine + review-veto

Drive the **Loop engine** (`graph_loops` / `LoopController`, AU-KG.research.these-properties-carry) and the evolution
flywheel + the AU-AHE.optimization.telemetry-optimization hardening loop — and **review** their proposals rather than
hand-doing what the flywheel produces. The research→assimilation→distill→develop pipeline
turns research/concepts into first-class, queryable `:SpecProposal` nodes (AU-KG.research.close-distill-develop-seam) that
enter the promotion pipeline **only after the AU-OS.config.autonomous-spec-develop-off spec-review checkpoint approves them**
— propose-and-hold (veto) is the default. Read live `EvolutionState` (`graph_loops
action=state`, AU-KG.research.evolutionstate-live-surface-per) and the saturation gauge (AU-KG.research.saturation-gauge-aggregates-four) to know which direction is
exhausted, and approve/veto rather than implement by hand.

```mermaid
flowchart TD
    H["Claude / harness<br/>(orchestrator + exception-resolver)"]
    H -->|understand code| KG["graph_analyze code_context<br/>(KG-2.134/135) → cited answer"]
    H -->|do a task| EX["graph_orchestrate execute_agent/execute_workflow<br/>on agent-utilities-expert / an ingested skill (local LLM)"]
    H -->|evolve/manage| LOOP["graph_loops + evolution flywheel<br/>:SpecProposal → AU-OS.config.autonomous-spec-develop-off review-veto"]
    EX --> ENGINE["engine_&lt;domain&gt; surface (AU-ECO.mcp.full-api-mcp-surface)<br/>+ multiplexer meta-tools"]
    EX --> PROV[":ToolCall / RunTrace provenance (KG-2.296)<br/>+ run_id handle (AU-ORCH.execution.rich-result-wrapper)"]
    PROV -->|query: what did it do?| H
    PROV -->|on failure| TS["troubleshoot provider (AU-KG.retrieval.kg-4)<br/>read RunTrace → find why"]
    TS --> FIX["fix the gap: missing skill / unbound tool /<br/>prompt / ingestion"]
    FIX -->|re-delegate + harden (AU-AHE.optimization.telemetry-optimization)| EX
    EDICT["resource-priority edict (AU-ORCH.scheduling.resource-priority-edict/99)<br/>orchestration outranks ingestion"] -.guards.- EX
    EDICT -.guards.- H
```

## The full engine surface available to delegates (AU-ECO.mcp.full-api-mcp-surface)

agent-utilities **is** the native API/MCP layer for the Rust epistemic-graph engine. The
curated high-level `graph_*` / `ontology_*` / `object_*` tools cover the synthesized,
agent-facing operations; **AU-ECO.mcp.full-api-mcp-surface** adds complete **1:1 coverage of the engine's
low-level capability surface** so no engine method is reachable only from a Python import.
A delegate (or the harness) reaches it through the multiplexer like any other tool.

- **One action-routed MCP tool per engine domain** — `engine_<domain>` — each a thin
  generic dispatcher that resolves the engine client and calls
  `getattr(client.<domain>, action)(**params_json)`, with its REST twin `/engine/<domain>`
  registered in the **same** call (surface-parity gate stays green). Implementation:
  `agent_utilities/mcp/tools/engine_tools.py` → `register_engine_tools`.
- **Drift-free by introspection (AU-KG.compute.engine-surface-manifest)** — the action set per domain is *discovered*
  by introspecting the `epistemic_graph` client sub-client classes (the 19 sub-clients =
  `nodes`, `edges`, `graph`, `analytics`, `lifecycle`, `reasoning`, `ledger`, `channels`,
  `tenants`, `resharding`, `consensus`, `finance`, `datascience`, `query`, `txn`,
  `timeseries`, `rdf`, `streaming`, `blob` — ~222 methods). A new engine method shows up
  automatically once the client wraps it; no hand-maintained list to rot.
- **Verbose 1:1 surface** — `MCP_TOOL_MODE=verbose`/`both` emits one
  `engine_<domain>_<method>` tool per method (generated from `ENGINE_DOMAINS` by the
  graph-os verbose builder / `gen_graphos_manifest`); the default condensed mode keeps the
  one action-routed tool per domain.

This is why the delegation-first model can push **heavy compute to the engine** (vector
similarity, ANN, graph algorithms, ML math, finance) instead of writing an O(N) loop in
Python: the full engine is one MCP call away. Python orchestrates; the engine computes.

## What makes delegation safe and non-starving

### Provenance — `:ToolCall` / `RunTrace` (KG-2.296)

Every delegated run writes its provenance to the epistemic-graph: a run-level `RunTrace`
(`trace:<run_id>`) and a first-class `:ToolCall` node per tool call the local LLM made,
linked `(:RunTrace)-[:MADE_TOOL_CALL]->(:ToolCall)`, capturing `tool_name`, `server`,
secret-redacted `args`, `result_preview`, `error`, `status`, and `sequence`. The MCP
`execute_agent` / `execute_workflow` surfaces return a **`run_id`** handle (AU-ORCH.execution.rich-result-wrapper) so a
delegation is trackable. This is the keystone: **full visibility + steerability is
guaranteed by design** — query a run over graph-os and see exactly which tools the local
LLM called, with what args, and what came back. Mechanics in
[`orchestration-execution-seam.md`](orchestration-execution-seam.md).

### The resource-priority edict (AU-ORCH.scheduling.resource-priority-edict/1.99)

Interactive / orchestration work **outranks** background ingestion, so your orchestration
is never starved by the system's own ingestion. The engine keeps a **reserved interactive
read lane** so orchestration reads aren't blocked under a write-storm, and admission tags
delegated runs ORCHESTRATION/INTERACTIVE so they are never stuck behind ingestion
enrichment. See [`resource-priority-edict.md`](resource-priority-edict.md). Without this,
"delegate everything" would let a re-ingest sweep starve the very control plane you
orchestrate through; with it, delegation scales.

## The exception-resolution loop

When a delegated run fails, is ungrounded, or the system couldn't self-troubleshoot —
**that is the harness's job.** The loop:

1. **Read the provenance.** Pull the run's `RunTrace` + `:ToolCall` chain
   (`graph_query MATCH (t:RunTrace {id:'trace:<run_id>'})-[:MADE_TOOL_CALL]->(tc:ToolCall)
   RETURN … ORDER BY tc.sequence`), or drive the **`troubleshoot` provider**
   (`graph_analyze action=explain target="troubleshoot:run" node_id=<run_id>`,
   [`troubleshooting.md`](troubleshooting.md)) to trace across every layer (app →
   container → system → host → cross-cutting).
2. **Find why.** The first failing `:ToolCall` (wrong tool, bad args, unbound tool, a
   `502`/`exit 137` traced down the stack) is usually the root.
3. **Fix the gap.** A missing or weak skill, an unbound tool, a prompt, missing
   data/ingestion — fix the one thing that made the local LLM fail.
4. **Re-delegate**, and **harden** so the system self-handles that case next time: add the
   skill, fix the tool binding, harden the prompt, capture the gotcha via `graph_feedback`
   — the AU-AHE.optimization.telemetry-optimization hardening loop. Every exception you resolve should make the autonomous
   system one step more autonomous.

The goal is to orchestrate **completely off the harness**: the local LLM + graph-os handle
the work, you handle the shrinking set of exceptions, and each exception you resolve
shrinks that set further.

## Related docs

- [`orchestration-execution-seam.md`](orchestration-execution-seam.md) — how an ingested
  capability becomes an executed-by-a-local-LLM run with full `:ToolCall` provenance.
- [`agent-utilities-expert.md`](agent-utilities-expert.md) — the default KG-bound delegate
  for ecosystem work.
- [`troubleshooting.md`](troubleshooting.md) — the cross-layer diagnose provider for the
  resolve-exceptions half.
- [`resource-priority-edict.md`](resource-priority-edict.md) — why orchestration is never
  starved by ingestion.
- [`codebase-context.md`](codebase-context.md) — the code-KG (`code_context`) that serves
  delegation route 1.
