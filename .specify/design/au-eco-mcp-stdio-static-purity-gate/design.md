# Design Document: Static stdio JSON-RPC purity gate

CONCEPT:AU-ECO.mcp.stdio-static-purity-gate

> `agent_utilities/mcp/server_factory.py` (the runtime fd-diversion that makes
> `mcp.run()`'s stdio transport survive a stray `print()`),
> `scripts/check_no_stdout_writes.py` (the static gate this document is
> about), wired as the `check-no-stdout-writes` pre-commit hook.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-OS.governance.fail-closed-degraded-read` | fail-closed gate family | low | OS |

### Extension Analysis

- **Primary Extension Point**: none — this is the first static-analysis gate
  for the stdio JSON-RPC transport specifically.
- **Extension Strategy**: new, narrowly-scoped guardrail.
- **New Concept Required?**: Yes.

## Problem

`mcp.run()`'s stdio transport (`stdio_server()`) treats stdout as a
JSON-RPC-only wire, byte for byte. A stray `print()` (or anything else that
writes to `sys.stdout`) in the served MCP surface corrupts the frame the
client is mid-parse of — silently, and only for a client connected over
stdio, never over HTTP, which makes it easy to miss in ad hoc testing.
`server_factory.py`'s runtime fd-diversion (redirect fd 1 to fd 2 for the
lifetime of `stdio_server()`, restored in a `finally`) is the *runtime* net,
but it only helps code that runs **after** `mcp.run()` has claimed the
descriptor — it does nothing for a `print()` that executes during import,
engine bootstrap, or a co-service thread started moments earlier.

## Decision

Catch the offending pattern at authoring time instead of relying solely on
the runtime net: `scripts/check_no_stdout_writes.py` statically walks the
served MCP surface for `print()`/`sys.stdout.write()` and fails the fast
pre-commit tier (`check-no-stdout-writes`) on any match. This is a
**static, always-run** gate — not a runtime assertion — because the failure
mode it defends against (a corrupted stdio frame observed only by a stdio
client, hours later) is exactly the kind of defect a test suite run over
HTTP transport would never surface.

## Wire-First

Wired as the `check-no-stdout-writes` local pre-commit hook
(`.pre-commit-config.yaml`, `always_run: true`), so it runs on every commit
regardless of which files changed — matching the same rationale documented
for its siblings `check-no-env-sprawl`/`env-var-drift`/`lane-guard`.
