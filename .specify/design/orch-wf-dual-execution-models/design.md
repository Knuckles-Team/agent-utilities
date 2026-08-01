# Design Document: Agent discovery collapses to two execution shapes — remote A2A peer, or unified local specialist

CONCEPT:AU-ORCH.execution.dual-execution-models

> `agent_utilities/agent/discovery.py:36-53` (`discover_agents`).

## Decision — every discovered agent is either a remote A2A peer or a single "unified specialist," never one of three separately-typed local shapes

`discover_agents` reads the KG-backed discovery registry and branches on
exactly one boolean question: is this agent `agent_type == "a2a"`? If so, it
is described as `type: "remote_a2a"` with its endpoint URL — a genuinely
different execution shape, since it requires a network call to another
agent's own A2A server rather than local dispatch. Otherwise, it is
described as `type: "specialist"`, with the inline comment stating the
history directly: **"Unified specialist (was prompt, mcp, or specialist)"**
— three formerly-distinct local agent kinds (a bare prompt agent, an
MCP-tool-bound agent, and a "specialist") have been collapsed into one
representation carrying `skills`, `mcp_server`, and `tools` as optional
fields on the same shape, rather than three separate dict layouts a caller
would need to branch on.

**The rejected alternative is the prior three-way local split the comment
names** — `prompt`/`mcp`/`specialist` as distinct discovery result shapes.
That was the actual previous state of this code, not a hypothetical: a
caller of `discover_agents` had to know which of three dict layouts it was
looking at to read the right fields. Collapsing them to one `"specialist"`
shape means every LOCAL agent, regardless of whether it's prompt-only,
MCP-tool-bound, or a full specialist, is described identically by the
discovery layer — the actual execution-time difference between those three
(which fields are populated: `tools`, `mcp_server`, both, or neither) is data
on a uniform shape, not a type tag a caller has to switch on. The one
execution-shape distinction retained is the one that actually matters for a
CALLER deciding how to reach the agent: local dispatch vs. remote A2A
network call.

## Risk Assessment

- **Blast Radius**: `agent_utilities/agent/discovery.py` — a read-only
  discovery/description function; it does not itself dispatch execution.
- **Backward Compatible**: Yes — the comment documents a collapse that
  already happened; this doc records the resulting invariant (two output
  shapes, not one of several).
- **Known weak point**: the two-way split is enforced only at the point
  `discover_agents` builds its output dict, by convention, not by a shared
  typed schema (`DiscoveredSpecialist` vs. the returned plain dicts differ) —
  a caller pattern-matching on the dict's `type` key rather than checking for
  the presence of `url` vs. `mcp_server`/`tools` could still accidentally
  reintroduce a three-way branch downstream without anything catching it.
