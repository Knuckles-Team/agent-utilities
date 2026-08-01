# Design Document: AgentDeps.workspace is untyped `Any` to keep the model layer out of the runtime layer

CONCEPT:AU-ORCH.execution.developer-workspace-runtime

> `agent_utilities/models/agent.py:27-32` (the `AgentDeps.workspace` field).

## Decision — the workspace handle is typed `Any`, not `DevWorkspace`, on the shared per-agent dependency object

`AgentDeps.workspace` is defined with its own inline comment naming the
decision explicitly: "the developer-workspace runtime (OS-5.33) the SWE
agent acts in. A `DevWorkspace` handle; None for non-SWE agents. Kept `Any`
to avoid importing the runtime package into the model layer"
(`agent.py:29-32`). `AgentDeps` itself (`agent.py:9-32`) lives in
`agent_utilities/models/` — a layer imported broadly and directly across the
codebase: `agent/factory.py`, `patterns/manual_test_tool.py`,
`patterns/tdd.py`, `sdd/orchestrator.py`, and (separately) five modules
under `tools/` — `pattern_tools.py`, `kg_evolution_tools.py`,
`swe_workspace_tools.py`, `code_intelligence_tools.py`,
`kg_share_tools.py` — all import `AgentDeps` from the models layer for
unrelated fields (`session_id`, `model_id`, `mcp_toolsets`, …). `DevWorkspace`
itself is defined in `agent_utilities/runtime/workspace.py`, a separate,
heavier package (container exec drivers, action dispatch, provenance
mirroring).

**The rejected alternative** — import `DevWorkspace` directly and type the
field as `DevWorkspace | None` — is the obviously more type-safe choice, and
it loses for a layering reason the field's own comment states outright: it
would force `models/agent.py`, and transitively every module above that
only needs `AgentDeps` for fields that have nothing to do with a workspace,
to pull in the entire `runtime` package. `Any` is the accepted cost:
`AgentDeps.workspace` forfeits static type-checking on that one field — no
autocomplete or mypy/pyright error if a caller assigns the wrong object — in
exchange for keeping `models/` a leaf layer that `runtime/` (and every other
package) can depend on, never the other way around.

## Risk Assessment

- **Blast Radius**: `agent_utilities/models/agent.py` (the field
  declaration itself); every consumer that reads `deps.workspace`
  (`orchestration/swe_agent.py`, `orchestration/computer_use_agent.py`,
  `tools/swe_workspace_tools.py`, `tools/computer_use_tools.py`, etc.)
  relies on a runtime check rather than a type-checked contract.
- **Backward Compatible**: Yes.
- **Known weak point**: because the field is `Any`, a caller that assigns
  the wrong object into `.workspace` is only caught at first use (an
  `AttributeError`, or a silent no-op for tools that `getattr(..., None)`
  guard it) — never at construction time and never by the type checker.
  This is the trade-off the field's own source comment documents as
  deliberate, not an oversight.
