---
name: agent-utilities-self-evolution
skill_type: skill
description: >-
  Evolve the agent-utilities repository by implementing the latest research concepts
  against it. Orchestrates research-scanner → kg-ingest → comparative-analysis
  → SDD implementer into a unified pipeline. Queries the KG for hot topics, finds
  relevant papers, ingests them, runs comparative analysis, and generates SDD plans
  with constitution-mandated artifact requirements baked in. Can run as a background
  daemon (every 60 minutes) via the KG engine's evolution cycle.
  Triggers on "evolve agent-utilities", "run evolution pipeline", "find and implement
  research", "research-driven development", "auto-evolve", "evolution scan".
  Also includes AST-based wiring sweep for concept traceability, dead code detection,
  and import graph analysis. Triggers on "wiring sweep", "concept audit",
  "dead code analysis", "traceability check", "code health".
license: MIT
tags: [evolution, research, pipeline, automation, agent-workflow, sdd, wiring, audit]
metadata:
  author: Genius
  version: '0.3.0'
---

# Agent Utilities Evolution Skill

Autonomous research-driven development and code health pipeline for evolving
the `agent-utilities` codebase. Combines research discovery with code-enhancer
analysis domains tailored for the 5-pillar KG-driven architecture.

## Overview

This skill orchestrates **7 capabilities** into a unified evolution pipeline:

### Research Pipeline
1. **Topic Detection** — Query the KG for hot topics, unresolved concepts, and high-scoring
   unimplemented research findings
2. **Research Scanner** — Find new papers matching those topics via `scholarx` MCP
3. **Knowledge Graph Ingest** — Ingest discovered papers and codebases into the KG
4. **Comparative Analysis** — Analyze ingested items against `agent-utilities` for gaps
   and feature extraction
5. **SDD Plan Generation** — Create implementation plans with constitution-mandated artifacts

### Code Health Pipeline (adapted from code-enhancer)
6. **Wiring Sweep** — AST-based import graph, dead code, concept traceability analysis
   (see `scripts/wiring_sweep.py`)
7. **Architecture Review** — C4 compliance, pillar concept coverage, hot-path integration
   audit tailored to the 5-pillar architecture (ORCH, KG, AHE, ECO, OS)

### Code-Enhancer Domains (agent-utilities-specific)

The following analysis domains from the `code-enhancer` skill are natively integrated,
tailored for the agent-utilities 5-pillar architecture:

| Domain | What It Checks | agent-utilities Specialization |
|--------|---------------|-------------------------------|
| **Concept Traceability** | 1:1:1 Code→Tests→Docs coverage for all `CONCEPT:X-Y.Z` tags | Validates against `docs/concept_map.md` registry; checks pillar doc pages |
| **Architecture Review** | C4 component compliance, SOLID principles, deep modules | Validates all 5 pillars have matching C4 diagrams in `architecture_c4.md`; checks mixin MRO integrity |
| **Dependency Audit** | `pyproject.toml` deps, version currency | Checks `pydantic-ai`, `pydantic-graph`, `kuzu`, `mcp` version alignment |
| **Test Coverage** | Concept-tagged pytest coverage, FIRST rubric | Validates every concept has at least 1 tagged test; checks async test patterns |
| **Engineering Heuristics** | Deep module analysis, duplication, naming | Checks for mixin anti-patterns, proper `properties=dict(...)` API usage, consistent logging |
| **Security Analysis** | Secret exposure, guardrail coverage | Validates `OS-5.1` security concepts, checks env var exposure in MCP configs |
| **Documentation Governance** | README, AGENTS.md, /docs completeness | Validates pillar summaries reference all registered concepts |

## Architecture

```mermaid
flowchart TD
    A[Start: User or Cron Trigger] --> B{Topics Specified?}
    B -->|No| C[Query KG for Hot Topics]
    B -->|Yes| D[Use User Topics]
    C --> E{Enough Topics?}
    E -->|No| F[Ask Clarifying Questions]
    F --> D
    E -->|Yes| D
    D --> G[research-scanner: Find Papers]
    G --> H[kg-ingest: Ingest Papers]
    H --> I[comparative-analysis: Analyze vs agent-utilities]
    I --> J[Generate SDD Plan Items]
    J --> K[Mark Topics in KG with SDD Plan ID]
    K --> L[Constitution Artifact Mandate Check]
    L --> M{Auto-Execute SDD?}
    M -->|No| N[Present Plan for Review]
    M -->|Yes| O[Execute SDD Implementer]
```

## Background Daemon

The evolution pipeline runs as a background daemon (`KG-Evolution-Daemon`) in the
`agent-utilities` engine task manager. It triggers every **60 minutes** and:

1. Queries the KG for unresolved `ResearchTopic` / `Concept` nodes not yet addressed
2. Runs a lightweight research scan for those topics
3. Ingests any new highly-relevant papers
4. Runs comparative analysis to extract actionable features
5. Logs evolution cycle results as `EvolutionCycle` nodes in the KG

The daemon is enabled by default and can be configured via `KG_EVOLUTION_INTERVAL`
environment variable (seconds, default: 3600).

## Graph-Native Assimilation Engine (CONCEPT:AU-KG.query.vendor-agnostic-traversal)

The comparative-analysis → plan-generation work is now a **graph-compute pipeline**
(`agent_utilities/knowledge_graph/assimilation/`), not per-source LLM reading. After
sources are ingested, run the assimilation pass — directly or via MCP:

```
graph_orchestrate(action="assimilate")                 # dedup → gap → synergy → rank
graph_orchestrate(action="assimilate", task="synthesize")  # + propose grounded SDD plans
graph_orchestrate(action="assimilate", task="force")       # ignore the idempotency watermark
```

Stages (all graph operations, LLM only at plan synthesis):
1. **dedup** — collapse the same capability across paper/library/our-code → `SIMILAR_TO` + `SUPERSEDES`.
2. **gap** (`auto_satisfy`) — match features vs our `Concept`s → `SATISFIED_BY`; `open_features()` excludes anything already built (the "stop rediscovering" filter).
3. **synergy** — cross-pillar community detection → `HAS_SYNERGY_WITH` bundles.
4. **rank** — leverage = `source_count × (1+centrality)`.
5. **plan synthesis** — grounded SDD proposals from each top gap's KG neighborhood.

**Idempotency:** a per-cycle **state watermark** ((id, status, content_hash) of feature/source
nodes) skips the pass when nothing changed; ingest is content-addressed (`canonical_source_id`
collapses arxiv/DOI/URL/path duplicates; `content_fingerprint` per-item skip). Cost grows with the
*delta*, not the *corpus*. On implementation, `ledger.close_out` writes
`DERIVED_FROM_RESEARCH` + `ASSIMILATED_INTO` so the feature is permanently excluded.

**Multi-source:** ingest docs (PRD/BRD/SOW/tasks → `Requirement`) and chat/SDD (→ `Decision`) via
`assimilation.ingest`, alongside papers, OSS libraries, and our ~62 repos — all into one graph.

The background daemon (golden-loop tick) runs this pass each cycle automatically. See
`.specify/specs/ecosystem-evolution/PHASE0_IMPLEMENTATION_PLAN.md`.

## Mining Flywheel (CONCEPT:AU-KG.evolution.mining-flywheel)

The loop engine (`LoopController.run_one_cycle`, `agent_utilities/knowledge_graph/
research/loop_controller.py`) runs a `mine_discovery` stage after `reason` and
before `synthesize` each cycle (gated by `config.kg_loop_mine_discovery`, default
**ON**; override per-call via `graph_loops(action="run", mine_discovery=...)`).
It calls the engine's `graph_mine`/`graph_learn` surfaces directly (through the
same `_invoke()` boundary those MCP tools use, so it degrades to an empty/no-op
result on a no-mining engine build) and feeds:

- **Step 5 (SDD Plan Generation) — WIRED.** Association-rule mining
  (`graph_mine action=associate`) over each `Capability` node's outbound
  neighborhood (`SATISFIED_BY`/`RELATES_TO` concept edges, `DERIVED_FROM_RESEARCH`
  article edges, `HAS_SYNERGY_WITH` sibling-capability edges) discovers
  concept-co-occurrence rules — "capabilities that share concept A + B usually
  also relate to concept/capability Z" — and writes them back as `:AssociationRule`
  nodes for plan generation to read as candidate implementations.
- **Step 1 (Topic Detection) — PARTIALLY WIRED.** A coverage-anomaly pass
  (`graph_mine action=anomaly`, z-score over each Capability's covered-concept
  count) flags divergent/under-implemented capabilities as `:Anomaly` nodes — a
  real, queryable topic source today. `graph_learn` fit→predict over `Concept`
  nodes similarly writes `:PredictedEdge` nodes suggesting missing concept↔concept
  relations. **Aspirational** (not implemented in this pass): clustering
  papers/concepts by embedding+citation graph to *prioritize* the topic queue by
  research theme — no code wires cluster output into topic ranking yet.

All write-back is **propose-only**: it materializes typed KG facts for a human/
agent to review via `graph_query`/`graph_loops(action="state")`; it never creates
an SDD plan, edits code, or triggers a merge by itself (no `GovernedAutoMerger`
wiring). See `agent_utilities/knowledge_graph/research/loop_controller.py`
(`_run_mine_discovery` + the three `_mine_*` helpers) and
`tests/unit/knowledge_graph/test_loop_controller.py` for the concrete call
sequence and scope cuts (documented in the method docstring).

## Execution Steps

### Step 1: Topic Detection (KG-First)

Query the Knowledge Graph for research topics that haven't been addressed yet:

```
Use mcp_agent-utilities-kg_kg_query with:

cypher: "MATCH (c) WHERE (c:ConceptNode OR c:Concept)
         AND NOT exists { MATCH (c)-[:ADDRESSED_BY]->(:SDDPlan) }
         RETURN c.id AS id, c.name AS name, c.description AS description
         ORDER BY c.name LIMIT 15"
```

**Fallback topic sources** (if no unresolved concepts found):
1. Extract from `Concept` nodes with pillar tags (ORCH, KG, AHE, ECO, OS)
2. Mine from recent comparative analysis gaps via `kg_search`
3. Check `RELEVANCE_SCORED` edges for high-scoring unimplemented papers

If fewer than 3 topics are found from any source, **ask clarifying questions**:

### Step 2: Clarifying Questions (if needed)

| Question | Default | When to Ask |
|----------|---------|-------------|
| Target research areas | KG-derived topics | Always if no topics found |
| arxiv categories to scan | cs.AI, cs.MA, cs.CL, cs.SE | If user doesn't specify |
| Papers per scan | All RSS Feed Available | If user wants to cap it |
| Target codebase | agent-utilities | If user doesn't specify |
| Auto-execute SDD? | No (generate plan only) | Always ask before execution |

### Step 3: Research Scan

Delegate to the `research-scanner` skill:

1. Use the detected topics to build search queries
2. Search via `mcp_scholarx_sx_search` with action `recent` for last 7 days
3. Also run `search` action with topic keywords for broader coverage
4. Score papers using `dynamic_scorer.py` from the research-scanner skill:

```bash
python /home/apps/workspace/agent-packages/skills/universal-skills/universal_skills/research/research-scanner/scripts/dynamic_scorer.py \
    --papers papers.json \
    --min-score 3.0 \
    --output top_papers.json
```

### Step 4: Download & Ingest

1. Download top-scoring papers via `mcp_scholarx_sx_storage` with action `bulk_download`
2. Ingest the downloaded PDFs via `mcp_agent-utilities-kg_kg_ingest`
3. Monitor ingestion progress via `mcp_agent-utilities-kg_kg_jobs`

### Step 5: Comparative Analysis

Run comparative analysis against the target codebase (default: `agent-utilities`):

1. Use `mcp_agent-utilities-kg_kg_analyze` with action `relevance_sweep` to score
   all ingested items against `agent-utilities`
2. Query rankings via `mcp_agent-utilities-kg_kg_analyze` with action `relevance_rankings`
3. For top-ranked items, run `deep_extract` to get structured feature recommendations

### Step 6: SDD Plan Generation

Generate an SDD implementation plan incorporating:

1. All feature recommendations from comparative analysis
2. **Constitution-mandated artifacts** (ALWAYS include these):
   - `/docs` updates
   - `AGENTS.md` updates
   - `CHANGELOG.md` entries
   - `README.md` updates
   - `.specify/` sync
   - C4 architecture diagrams
   - Pytests for all new functionality
3. Cross-reference with existing SDD plans to avoid duplication

### Step 7: Topic Tracking

After plan generation, mark topics as addressed in the KG:

```
Use mcp_agent-utilities-kg_kg_write with:

action: "upsert_node"
node_type: "SDDPlan"
properties: {"id": "<plan_id>", "created_at": "<timestamp>", "status": "proposed"}

Then for each topic:
action: "create_edge"
source_id: "<topic_id>"
target_id: "<plan_id>"
rel_type: "ADDRESSED_BY"
```

## Constitution Enforcement

Before finalizing any SDD plan, **verify against the constitution**:

```
Use mcp_agent-utilities-kg_kg_inspect with view: "constitution"
```

Cross-check that the plan includes ALL 7 mandatory post-modification artifacts.
A plan that omits any artifact is **INVALID** and must be revised.

## Evolution Cycle Node

Each evolution cycle is logged in the KG for tracking:

```
Use mcp_agent-utilities-kg_kg_write with:

action: "upsert_node"
node_type: "EvolutionCycle"
properties: {
    "id": "evo_cycle_<timestamp>",
    "triggered_by": "daemon|user",
    "topics_scanned": <count>,
    "papers_found": <count>,
    "papers_ingested": <count>,
    "recommendations_generated": <count>,
    "sdd_plan_id": "<plan_id or null>",
    "created_at": "<timestamp>"
}
```

## Dynamic Scorer Topic Export

The `dynamic_scorer.py` in `research-scanner` auto-detects topics from the KG.
This skill extends that by also **exporting** the detected topics back to the KG
as `ResearchTopic` nodes for tracking:

```
For each detected topic:
Use mcp_agent-utilities-kg_kg_write with:

action: "upsert_node"
node_type: "ResearchTopic"
properties: {
    "id": "topic_<slug>",
    "name": "<topic_name>",
    "source": "dynamic_scorer",
    "detected_at": "<timestamp>",
    "pillar": "<ORCH|KG|AHE|ECO|OS>"
}
```

## Wiring Sweep — AST-Based Codebase Analysis

The evolution skill includes a standalone AST-based analysis tool for detecting
dead code, concept traceability gaps, and import graph anomalies.

### Usage

```bash
# Full sweep with markdown report
python scripts/wiring_sweep.py /path/to/agent-utilities

# JSON output for CI integration
python scripts/wiring_sweep.py /path/to/agent-utilities --json

# Write report to file
python scripts/wiring_sweep.py /path/to/agent-utilities --output report.md
```

The script path is (shipped inside this package-owned skill):
```
agent_utilities/skills/agent-utilities-self-evolution/scripts/wiring_sweep.py
```

### What It Analyzes

| Analysis | Description |
|----------|-------------|
| **Import Graph** | Builds module-to-module dependency edges from all `import` / `from ... import` statements |
| **Orphan Modules** | Finds `.py` files never imported by any non-test, non-init module |
| **Concept Traceability** | Cross-references `CONCEPT:X-Y.Z` tags across code → tests → docs |
| **Dead Definitions** | Functions/classes defined but never referenced in any other file |
| **Health Score** | 0-100 composite score (40pt concept coverage + 25pt orphans + 25pt dead code + 10pt syntax) |

### Health Score Grading

| Score | Grade | Meaning |
|-------|-------|---------|
| 80-100 | 🟢 | Healthy — all concepts wired, minimal dead code |
| 60-79 | 🟡 | Needs attention — some gaps in traceability or orphaned modules |
| 0-59 | 🔴 | Critical — significant wiring gaps or dead code |

### CI Integration

The script exits with code 1 if the health score is below 60, making it suitable
for use as a pre-commit check or CI gate:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: wiring-sweep
      name: Wiring Sweep
      entry: python scripts/wiring_sweep.py . --json
      language: system
      pass_filenames: false
```

### Triggers

When the user says any of:
- "wiring sweep", "run wiring sweep"
- "concept audit", "concept traceability check"
- "dead code analysis", "find dead code"
- "traceability check"
- "code health", "check code health"

Execute the sweep:

```bash
python /path/to/agent_utilities/skills/agent-utilities-self-evolution/scripts/wiring_sweep.py \
    /home/apps/workspace/agent-packages/agent-utilities \
    --output /tmp/wiring_sweep_report.md
```

Then present the markdown report to the user and highlight:
1. Any concepts with `missing_tests` or `missing_docs`
2. Orphan modules with high line counts (likely unmigrated code)
3. Dead definitions that may need cleanup

## Ecosystem Standardization

The evolution pipeline also governs cross-project standardization across the
agent-packages ecosystem. This ensures all 36+ agents maintain consistent
structure, documentation, and concept traceability.

### CONCEPT ID Bridge Strategy

All agent-packages projects connect to agent-utilities via `CONCEPT:AU-ECO.messaging.native-backend-abstraction`
(Unified Toolkit Ingestion). Each project has:

- A unique CONCEPT prefix (e.g., `PORT-` for portainer-agent, `SNOW-` for servicenow-api)
- A `docs/concepts.md` registry mapping local concepts to cross-project references
- Tool descriptions annotated with CONCEPT IDs for KG ingestion

**Prefix Registry** (39 unique prefixes, zero collisions):

| Prefix | Project | | Prefix | Project |
|--------|---------|---|--------|--------|
| `ABOX` | archivebox-api | | `ARR` | arr-mcp |
| `ATL` | atlassian-agent | | `AU` | agent-utilities |
| `CMGR` | container-manager-mcp | | `DSCI` | data-science-mcp |
| `DOCDB` | documentdb-mcp | | `GENIUS` | genius-agent |
| `GH` | github-agent | | `GL` | gitlab-api |
| `HASS` | home-assistant-agent | | `JELLYFIN` | jellyfin-mcp |
| `LF` | langfuse-agent | | `LIX` | leanix-agent |
| `LM` | listmonk-api | | `MEAL` | mealie-mcp |
| `MDLD` | media-downloader | | `MSFT` | microsoft-agent |
| `NC` | nextcloud-agent | | `OC` | owncast-agent |
| `PA` | postiz-agent | | `PLANE` | plane-agent |
| `PORT` | portainer-agent | | `QBT` | qbittorrent-agent |
| `RM` | repository-manager | | `SNOW` | servicenow-api |
| `SRX` | searxng-mcp | | `SX` | scholarx |
| `SYS` | systems-manager | | `STIRLINGPDF` | stirlingpdf-agent |
| `TUI` | agent-terminal-ui | | `TUN` | tunnel-manager |
| `UKA` | uptime-kuma-agent | | `VEC` | vector-mcp |
| `WEBUI` | agent-webui | | `WGER` | wger-agent |
| `ANSIBLE` | ansible-tower-mcp | | | |

### Cross-Project Synergy Mapping

When ingesting any agent-package codebase via the KG, the following synergies
are automatically detected and mapped:

1. **ECO-4.0 → All agents**: Every agent's MCP tools are discoverable via unified ingestion
2. **ORCH-1.2 → All agents**: Confidence-gated routing applies to all MCP tool domains
3. **OS-5.* → All agents**: Security, scheduling, guardrails are inherited from agent-utilities
4. **KG-2.* → All agents**: Knowledge graph concepts bridge all codebase ingestion

### Standardization Audit Checklist

As part of each evolution cycle, verify:

- [ ] Every agent has `docs/concepts.md` with unique prefix
- [ ] No CONCEPT prefix collisions across ecosystem
- [ ] All `auth.py` use standard env var patterns (`_URL`, `_TOKEN`, `_SSL_VERIFY`)
- [ ] All agents have `CHANGELOG.md`
- [ ] No `legacy_readme.md` files remain in docs/
- [ ] `mcp/` subdirectory standard is documented (existing agents migrated over time)

## References

- [research-scanner](../research-scanner/SKILL.md) — Paper discovery and scoring
- [comparative-analysis](../comparative-analysis/SKILL.md) — Feature extraction
- [kg-ingest](../../automation/kg-ingest/SKILL.md) — Bulk ingestion
- [sdd-implementer](../../development/sdd-implementer/SKILL.md) — Task execution
