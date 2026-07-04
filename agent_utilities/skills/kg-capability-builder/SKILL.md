---
name: kg-capability-builder
description: >-
  End-to-end meta-recipe for adding a new capability to the platform — from an
  engine crate, to an engine_<domain>/graph_* verb + REST route, to the wrapping
  kg-* skill that makes it discoverable. Use when building a new graph-os
  capability end-to-end ("add a new engine domain", "expose a new verb", "wire a
  capability from Rust to skill", "why is my new verb an orphan / uncovered").
license: MIT
tags: [graph-os, meta, capability, builder, surface-parity]
tier: meta
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Capability Builder (engine crate → verb → route → skill)

The end-to-end meta-skill for adding a capability to the platform so it is
reachable, documented, and discoverable — with no drift between the layers.
This is a **meta** skill: it orchestrates a build across four layers and wraps
no single verb.

## The four layers (build in order)

1. **Engine crate (Rust).** Implement the capability in the epistemic-graph
   engine and expose it as a wire `Method` in `crates/eg-types/src/protocol.rs`.
   The pure-Python `epistemic_graph` client mirrors the wire protocol 1:1, so a
   new method surfaces as a coroutine on a sub-client (`.nodes`, `.blob`,
   `.finance`, …). That client is the source of truth for "what the engine can do".

2. **MCP verb + REST route (agent-utilities).** agent-utilities is the native
   API/MCP layer for the engine.
   - **Low-level:** a new engine method is auto-discovered by
     `engine_tools._discover_domains()` (client introspection, CONCEPT:KG-2.278),
     so it appears under the domain's `engine_<domain>` action-routed tool with
     no hand-edit. A brand-new *domain* needs an entry in `_DOMAIN_CLASSES` +
     `_DOMAIN_BLURB`. Its REST twin `/engine/<domain>` is registered in lockstep
     (`ACTION_TOOL_ROUTES`, CONCEPT:ECO-4.99).
   - **High-level:** for a synthesized, agent-facing operation, add a curated
     `graph_*` / `ontology_*` / `object_*` tool and register its
     `ACTION_TOOL_ROUTES` REST route in the same call so the surface-parity gate
     stays green (tool ⇄ REST route legs).

3. **REST route.** Confirmed by (2) — every tool has a `POST /graph/<...>` or
   `POST /engine/<...>` twin; the parity gate fails if one is missing.

4. **Wrapping kg-* skill.** Author a `kg-*` `SKILL.md` so operators can discover
   the verb. The `kg-coverage-doctor` (third parity leg) enforces this:
   - Slug rule: `kg-<x>` ⇒ `graph_<x>` (1:1, no config).
   - `wraps: [verb, ...]` when the skill fronts several verbs (e.g. each
     `kg-modality-*` wraps its `engine_*` domains).
   - `tier: meta|surface` when the skill is not a verb wrapper (exempt from
     coverage + orphan checks).
   Set `tier:` on every skill.

## Verify (close the loop)

```bash
# 1. surface parity — tool ⇄ REST route ⇄ skill coverage
python -m agent_utilities.mcp.skill_coverage         # 0 uncovered, 0 orphans
pytest tests/unit/test_gateway_mcp_parity.py

# 2. regenerate the action manifest from the client-introspection source of truth
python scripts/gen_graphos_manifest.py
```

A new verb that ships without a skill shows as **uncovered**; a skill whose slug
or `wraps` points at a non-existent verb shows as an **orphan** — fix both before
merge. Add a verb to `INTENTIONALLY_UNSKILLED` only with a written justification.

## Related skills

- `kg-mux-extend` — register a child MCP server (a different kind of extension).
- `kg-mux-use` — discover + mount the new tool once it exists.
- `agent-package-builder` / `mcp-builder` — scaffold a whole new package.
