# KG‑2.10 — OWL‑Driven Orchestration & Agent Synthesis (stream "S4")

> The KG stops merely *describing* the system and starts *composing* it: evolve/
> create prompts from identified needs, synthesize agents (right prompt + tools +
> skills) for a goal, assemble hierarchical agent teams, and evolve the pydantic‑ai
> graph + skill‑workflow orchestration — all driven by OWL relationships over the
> enriched graph. This is the core of KG‑driven orchestration. Builds on KG‑2.8/2.9
> (entities, concepts, capability cards, cross‑links) + `capability_index`/
> `designate()` + `graph_orchestrate`.

## Ontology (relationships the OWL layer defines)
Entities: `Agent`, `Team`, `Prompt`, `Tool`, `Skill`, `Workflow`, `Goal`,
`A2AAgentCard`, plus existing `Code`/`Concept`/`Feature`/`Incident`/…
Edges: `HAS_PROMPT`, `USES_TOOL`, `HAS_SKILL`, `ORCHESTRATES` (Workflow→step),
`MEMBER_OF_TEAM`, `REPORTS_TO` (team hierarchy), `SOLVES` (Agent/Team→Goal),
`EVOLVED_FROM` (Prompt/Agent lineage), `EXPOSES_SKILL` (A2AAgentCard→capability),
`DELEGATES_TO` (Agent→A2A agent). Promote these in `owl_bridge` so the reasoner
infers reachable capabilities, team coverage, and gaps.

## Capabilities
1. **Prompt evolution/creation** — find needs/low‑hanging fruit from KG signals:
   reward write‑back (Plan‑08), failing/low‑value tasks, features/concepts with no
   covering prompt, `needs_work` tests, distilled enhancement candidates. LLM
   drafts or evolves a `Prompt` (stored with `EVOLVED_FROM` lineage); A/B via
   reward signal. Reuses the KG‑2.8 distill pattern (gather→rank→LLM→write).
2. **Agent synthesis** — given a `Goal`/problem: `designate()`/semantic search the
   KG for relevant Tools, Skills, Prompts; LLM composes an `Agent` spec (system
   prompt + tool set + skills) grounded in those; emit an agent package /
   `TeamConfig`. Validate tools exist (USES_TOOL edges resolve to real MCP tools).
3. **Team construction + hierarchy** — decompose a Goal into sub‑goals; synthesize
   specialized agents per sub‑goal; wire `MEMBER_OF_TEAM` + `REPORTS_TO` into a
   hierarchy (lead/orchestrator → specialists). Output a runnable team config.
4. **Pydantic‑ai graph + workflow evolution** — identify the current pydantic‑ai
   graph (the orchestration nodes/edges) from the codebase + runtime traces;
   propose changes to execution/orchestration flow expressed as agent‑skill
   **skill‑workflows** (via skill‑workflow‑builder); validate against the KG.
5. **A2A agent‑card ingestion** — ingest external A2A agent cards → `A2AAgentCard`
   nodes (name, capabilities/skills, endpoint, auth) + `EXPOSES_SKILL` edges, so
   synthesis can `DELEGATE_TO` external agents alongside local ones. *(This is the
   immediately‑parallelizable extractor — built first via the registry pattern.)*

## How it composes
Same machinery: A2A cards ingest via a registry extractor (like the enterprise
sources); agents/teams/prompts are typed nodes with capability cards + embeddings;
synthesis is the distill pattern (KG query → rank → LLM compose → write artifact);
OWL inference fills reachability/coverage/gaps. Existing assets to reuse:
`agent-spawner`/`agent-builder`/`skill-workflow-builder` skills,
`kg-delegation-router` (graph_orchestrate execute_agent/execute_workflow),
`capability_index`/`facade.designate`, autonomous‑contribution TeamConfigs.

## Sub‑streams
- **S4a (parallel now):** A2A agent‑card extractor (`extractors/a2a.py` +
  `ontology_a2a.ttl` + test) — self‑registering, fakes only.
- **S4b:** orchestration ontology (`ontology_orchestration.ttl`) + owl_bridge edges
  + Agent/Team/Prompt/Workflow node writers.
- **S4c:** synthesis engine (`enrichment/synthesize.py`): `synthesize_agent(goal)`,
  `synthesize_team(goal)`, `evolve_prompts()` — distill pattern, LLM injectable.
- **S4d:** pydantic‑ai graph discovery + workflow‑evolution proposals.
