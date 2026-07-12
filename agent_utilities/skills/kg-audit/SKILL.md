---
name: kg-audit
skill_type: skill
description: >-
  Tamper-evident audit ledger (G23): verifies the engine's hash-chained
  durable-mutation audit log and reconstructs "what happened to entity X" from
  the KG's own :ToolCall provenance. Use for "verify the audit chain", "is our
  audit log tamper-evident", "has anything mutated entity X and in what
  order", "reverse-index the tool calls that touched this node".
license: MIT
tags: [graph-os, audit, governance, provenance, tamper-evident]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-audit

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_audit` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_audit"])` once per session (as below), then proceed exactly as documented; or (2) call the `ask` intent verb with the same natural-language request — the resolver routes to `graph_audit` for you and returns the result plus a routing justification (`graph_audit` has no hand-curated verb entry yet, so it resolves via the universal `ask` fallback). Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tool eagerly instead.

`graph_audit` (CONCEPT:AU-KG.audit.hash-chain-verify) exposes the engine's mature Rust
hash-chained audit log (`epistemic-graph/src/audit.rs` — every durable mutation already
chains into a per-graph SHA-256 hash chain, redb `AUDIT` table) to Python/MCP — no new
audit mechanism, just the first Python/MCP wrapper over it. Two actions:

- **`verify`** — cryptographically walks the default graph's audit chain and reports
  whether every entry's stored hash still matches its recomputed link (tamper
  evidence). Returns `{"ok": true, ...}` when intact, or `first_broken_seq` when the
  chain is broken. Degrades cleanly with a clear error when the connected engine
  build/config doesn't support it (no `security` cargo feature compiled in, or no
  durable redb persist dir configured) — that's a corresponding gap in the
  `epistemic-graph` repo, not this one.
- **`for_target`** — the entity-anchored reverse index (the KG-side half of G23):
  every `:ToolCall` that acted on a given `target_id`, in call order, plus a
  best-effort `verify()` snapshot alongside it (`Orchestrator.get_tool_calls_for_target`).
  Requires `target_id`.

## Invoke
- **MCP:** `load_tools(tools=["graph_audit"])`, then `graph_audit(action="verify")`.
- **REST twin:** `POST /audit` with `{"action": "for_target", "target_id": "commit:abc123"}`.

## Example
```
graph_audit(action="verify")
// -> {"surface": "audit", "action": "verify", "available": true, "ok": true, "entries": 412, ...}

graph_audit(action="for_target", target_id="commit:abc123")
// -> {"surface": "audit", "action": "for_target", "target_id": "commit:abc123",
//     "tool_calls": [...], "verify": {"ok": true, ...}}
```

## Honest limitations
- `verify`'s coverage is exactly what the connected engine build exposes — a build
  without the `security` cargo feature, or one with no durable redb persist dir
  configured, returns `available: false` with an explanatory error rather than a
  false "ok".
- `for_target` only sees `:ToolCall` provenance the orchestrator has already
  recorded — it does not itself instrument callers that bypass the Orchestrator.
- `kg-compliance`'s `posture` action already joins this skill's `verify()` output
  with governance node counts for an auditor-facing rollup — reach for that skill
  when the ask is "our compliance posture", not just the raw chain check.

## Delegation
If graph-os is reachable, prefer `graph_orchestrate(action="execute_workflow")` for a
recurring audit-chain check or a bulk `for_target` sweep rather than hand-driving the
calls yourself.
