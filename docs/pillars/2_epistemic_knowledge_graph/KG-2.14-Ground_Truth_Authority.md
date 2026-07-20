# Ground-Truth Context Authority (CONCEPT:AU-KG.memory.ground-truth-preamble-declaring)

## Overview

Ground-Truth Context Authority makes injected memory **authoritative**: the startup context now
opens with a Ground-Truth Hierarchy preamble telling the agent to treat the injected memory below
as already-known fact and **stop re-fetching/rediscovering** it ("memory-zero behavior").
Assimilated from memory-os Layer 7 (`ClaudioDrews/memory-os@a4ca094, layers/07-ground-truth.md`),
made structural — authority is a graph-grounded tier on each chunk, not an unverifiable prompt
assertion. Extends **KG-2.1** (Tiered Memory & Context).

## How it works

- **Authority tier.** Each `StartupChunk` carries a `source_authority` ∈
  {advisory, standard, authoritative}. Durable, curated memory (profile facts, preferences,
  identity, active goals, team conventions, layered project rules) is *authoritative*; transient
  recall hints are *advisory*. `_authority_for(source, heading)` classifies; `_chunk_priority`
  applies an `AUTHORITY_BOOST` so authoritative memory outranks hints in the budgeted payload.
- **Ground-Truth preamble.** `_build_authority_preamble` emits a block naming the authoritative
  sources present and instructing the agent: do not re-fetch/re-search/re-derive what is already
  injected; injected memory wins over prior assumptions; only this-turn runtime tool output
  outranks it. The preamble is budget-reserved so it never crowds out content.
- **Graph-grounded ranking.** Authority composes with KG-2.11 bi-temporal validity and KG-2.6 trust
  so the hierarchy reflects real provenance, not a flat rule.

## Key files / API

| Piece | Location |
|---|---|
| Authority machinery | `knowledge_graph/memory/memory_engine.py` (`StartupChunk.source_authority`, `_authority_for`, `AUTHORITY_BOOST`, `_build_authority_preamble`, `StartupContextBuilder.build_payload`) |

## Wiring (≤3 hops)

`agent-utilities-memory context` (cmd_context), MCP `graph` context action, and
`MemoryEngine.build_startup_context` → `build_payload` (2 hops).

## Research provenance

memory-os Layer 7 Ground Truth Hierarchy — `layers/07-ground-truth.md`, `icarus/hooks.py` (verified).
