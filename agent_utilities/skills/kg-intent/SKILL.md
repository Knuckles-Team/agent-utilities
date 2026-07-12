---
name: kg-intent
skill_type: skill
description: >-
  The graph-os intent surface — six verb-tools (ask/find/write/act/manage/why) that
  collapse the ~95-tool condensed graph-os surface behind a small, fixed schema for
  small/cheap-LLM profiles. Use when MCP_TOOL_MODE=intent is active, when a caller/model
  should route by natural-language INTENT rather than naming a granular tool directly, when
  asked "what's the intent surface", "how do ask/find/write/act/manage/why work", "when
  should I use condensed vs intent mode", or "reclaim/load a tool from the intent surface".
license: MIT
tags: [graph-os, meta, intent, routing, cpd, small-model]
tier: meta
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-intent — the intent surface (Seam 8)

graph-os's granular surface has ~95 `graph_*`/`engine_*`/`ontology_*`/`object_*`/
`source_*` tools. Past a point, more tools *lowers* LLM tool-selection accuracy — worst
for Haiku/local/small-context models — and blows past the ~100-128 client tool cap. So
the **intent surface** (`MCP_TOOL_MODE=intent`) is now graph-os's **DEFAULT** profile: it
still registers every granular tool exactly as before (nothing removed, REST +
`_execute_tool` unaffected), but additionally gates them from the default session tool
list and fronts the whole surface with six tiny, fixed-schema verb-tools. The granular
tools are loaded on demand via `load_tools`; set `MCP_TOOL_MODE=condensed`/`verbose`/`both`
to expose them eagerly instead. This is a **meta** skill — it documents the
resolver/dispatcher mechanism itself, not one verb.

## The six intent verbs

| Verb | Resolves to (examples) | Use for |
|---|---|---|
| `ask` | `graph_query`, `graph_search`, `graph_analyze`, `nl_query`, `ask_data`, `graph_explain`, ... | Any natural-language READ/analysis question. |
| `find` | every verb, unfiltered — capability DISCOVERY across the whole surface (+ fleet-wide when a multiplexer is attached) | "What tool can do X?" when you don't know the verb either. |
| `write` | `graph_write`, `graph_ingest`, `graph_writeback`, `source_sync`, `graph_etl`, ... | Ingest/mutate/persist intents. |
| `act` | `graph_orchestrate`, `graph_loops`, `graph_goals`, `graph_sandbox`, `graph_bus`, ... | Execute/orchestrate/schedule intents. |
| `manage` | `graph_configure`, `graph_secret`, `graph_sessions`, `graph_kvcache`, `graph_ontology`, ... PLUS the load/unload lifecycle (below) | Configure/admin intents, and reclaiming tool-list context. |
| `why` | `graph_explain`, `graph_evaluate`, `graph_observe`, ... | Explain a decision/belief/change — including the routing decision itself. |

Each is called the same way: `<verb>(intent="<natural language>", hints_json="{...}",
execute=true)`.

- **`intent`** — the natural-language request. For read-shaped tools with a single
  free-text parameter (e.g. `graph_search`'s `query`), this alone is enough — zero hints
  needed.
- **`hints_json`** — optional structured JSON forwarded to the resolved tool
  (`{"node_id": "..."}` for a write) OR `{"tool": "graph_write"}` to **pin** an exact
  tool, bypassing ranking entirely (score `1.0`) — the same escape hatch `load_tools`
  provides, usable inline.
- **`execute`** — `false` returns only the routing decision (a dry-run/preview) without
  calling the underlying tool.

Every call returns `{"result", "routing", "executed"}` (or `{"error", "routing"}` on a
dispatch failure) — `routing` carries `chosen_tool`/`action`, `matched_terms`, `why` (a
plain-English justification), `alternatives` considered, `capability_source` (whether
the ranking came from the generated Capability Power Descriptor set or the pre-CPD
lexical fallback), and `calibrated_outcome_reward` (the learned reward-EMA for this
verb+tool pairing — see **Learning loop** below).

## How resolution works

1. **CPD-backed ranking** (`CONCEPT:AU-ECO.mcp.intent-surface-cpd-ranking`). Each candidate is scored
   against its own generated Capability Power Descriptor (`docs/capabilities-power.json`
   — `one_line`/`examples`/`does[]` action names) when one exists; a tool without a CPD
   entry yet falls back per-capability to a lexical score over its docstring + the
   hand-curated verb table — never an error, never a silent gap.
2. **Learning loop** (`CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning`). Every dispatch's
   success/failure feeds the SAME durable-bandit reward-EMA mechanism the rest of the
   platform already shares (`OutcomeRouter` over `CapabilityIndex.record_outcome`/
   `reward_of` — no second learner) keyed `verb:tool`. A capability that keeps failing
   under a verb sinks in the ranking; one that keeps succeeding rises. `find(...)` never
   dispatches, so it never records an outcome — only `ask`/`write`/`act`/`manage`/`why`
   do.
3. **Resolution cache** (`CONCEPT:AU-ECO.mcp.intent-surface-resolution-cache`). A repeated
   `(verb, intent, hints, top_k)` is served from a small bounded in-process cache instead
   of re-ranking from scratch, until the tool surface/CPD is regenerated OR a fresh
   outcome is recorded (either bumps the cache's internal generation counters, busting
   exactly the affected entries).

## Responsible tool usage — the load → use → unload lifecycle

Reclaiming context is a **`manage`** concern, not a 7th verb:

```jsonc
// Pull ONE granular tool into this session's tool list, then call it directly.
manage(intent="load the query tool", hints_json='{"action": "load", "tools": ["graph_query"]}')
graph_query(query="MATCH (n:Concept) RETURN n.id LIMIT 5")

// One-shot: auto-retract right after its next call — doesn't linger in a long session.
manage(intent="", hints_json='{"action": "load", "tools": ["engine_blob"], "auto_unload": true}')

// Retract explicitly by tool / whole server / toolset tag.
manage(intent="", hints_json='{"action": "unload", "tools": ["graph_query"]}')
```

Nothing is destroyed by `unload` — `load` brings the tool straight back at any time.

## Condensed vs. intent — when to use which

- **`MCP_TOOL_MODE=intent`** (the **default**): state the intent via one of the six verbs
  above and the resolver picks the tool and shows its work; or `load_tools`/pin via
  `hints_json={"tool": ...}` to reach an EXACT granular tool when you already know which one
  you want — you are never forced through ranking. Best for small/cheap/local models and
  any session where the ~100-tool cap or tool-selection accuracy matters (i.e. the default).
- **`MCP_TOOL_MODE=condensed`/`verbose`/`both`** (opt-in): a capable model with enough
  context that wants every granular `graph_*`/`engine_*` tool exposed eagerly — call it
  directly. Every `kg-*` skill besides this one documents that direct path.

Both profiles reach the exact same ~95 granular tools and identical REST routes — this
skill is about **which front door** an LLM session uses, never about lost capability.

## Verify

```bash
MCP_TOOL_MODE=intent python -c "
from agent_utilities.mcp.tools import intent_tools
print(intent_tools.resolve_intent('ask', 'search the knowledge graph', top_k=3))
"
```

Related: every wrapped granular tool documents its OWN `kg-*` skill (e.g. `kg-query`,
`kg-write`, `kg-orchestrate`) — each carries a "Condensed intent-surface note" pointing
back here. `kg-mux-use` is the analogous meta-skill for the FLEET-wide `load_tools`/
`find_tools` mechanism this skill's `manage` lifecycle shortcut reuses.
