# Design Document: Cross-specialist observations travel through a shared board, not direct messages

CONCEPT:AU-ORCH.execution.inject-signal-board-observations

> Definition: `agent_utilities/graph/state.py:442-450` (`GraphState.signal_board`).
> Emission: `agent_utilities/capabilities/adversarial_verifier.py:146-155`.
> Injection: `agent_utilities/graph/executor.py:900-916`,
> `agent_utilities/graph/_router_impl.py:1116-1134`. Biomimicry naming source:
> `agent_utilities/knowledge_graph/orchestration/engine_query.py:1702-1705`.
> 13 source files, 58 marker sites carry this id, but most are module-docstring
> tags on unrelated `graph/` package files (see "Marker noise" below) — the
> decision itself lives in the four sites cited above.

## The real decision

`GraphState.signal_board` (`graph/state.py:442-450`) is a lightweight pub/sub
map, `dict[str, list[str]]`, scoped to one graph execution session:

> *"Lightweight pub/sub within the graph execution session. Specialists emit
> signals (e.g., `dependency_gap`, `security_concern`) via `emit_signal()` and
> the dispatcher injects relevant signals into subsequent specialist system
> prompts."*

The engine's own biomimicry keyword table names the pattern explicitly
(`knowledge_graph/orchestration/engine_query.py:1702-1705`):

```
"stigmergy": {
    "analogy": "indirect coordination via environment",
    "domain": "signal_boards",
},
```

**Stigmergy** — agents coordinate indirectly by modifying a shared environment
that later agents read, rather than by addressing each other directly. That is
exactly the shape here: `adversarial_verifier.py:146-155` emits findings by
writing into `state.signal_board[signal_type]` (capped at the top 5 findings
per verification pass); a later specialist never receives them as a
message — the **dispatcher** (`_router_impl.py:1116-1134`) and the
**executor** (`executor.py:900-916`) independently read `ctx.state.signal_board`
and fold it into the next specialist's system prompt as an
"OBSERVATIONS FROM PRIOR SPECIALISTS" section, or emit it as a
`signal_board_context` UI event.

## The rejected alternative — unbounded injection, and direct specialist-to-specialist messaging

Two bounds are stated directly in the injection code and are the load-bearing
part of the decision, not incidental limits:

**1. Unbounded fan-in into every downstream prompt was rejected.**
`executor.py:903-905` caps injection at 3 messages per signal type and 10
lines total, with the comment *"Limit injection to avoid prompt bloat"* right
at the call site. A signal board with no cap degrades into every specialist's
prompt silently growing with every prior specialist's output — the same
"oversized-context" failure class the execution-budget-caps decision
(`CONCEPT:AU-ORCH.execution.execution-budget-caps`) guards against at the
token-limit layer; this is the equivalent guard at the *shape* of what gets
injected, not the token count of the wire.

**2. Direct specialist-to-specialist addressing was rejected in favour of an
environment-mediated channel.** The board is typed by *signal category*
(`dependency_gap`, `security_concern`, `quality_gap`), not by sender/recipient
pair — a specialist that emits a signal has no addressee and no expectation of
a reply; a specialist that reads the board has no notion of *which* prior
specialist a given signal came from beyond the `[adversarial]`-style tag
prefix convention used in `adversarial_verifier.py:154`. This decouples
specialist count and topology from the coordination mechanism: adding a new
specialist type never requires teaching existing specialists a new message
format or address, because nothing addresses anything — every reader consumes
the same typed, capped digest regardless of how many writers produced it or in
what order.

## Marker noise

Most of this concept's 58 marker sites are **module-docstring tags on other
`graph/` package files** — `graph/hsm.py:6`, `graph/lifecycle.py:6`,
`graph/models.py:2`, and `graph/adaptive_agent_router.py:755,870` — none of
which discuss the signal board; they read as a bulk "this file lives in the
signal-board-adjacent execution package" tag applied during ingestion rather
than four more instances of this decision. `docs/guides/emergent-architecture.md`
reuses the same id across several unrelated table rows (e.g. labelling the
`VARIANT_OF` KG edge type), which is stale/mismapped documentation, not
evidence. Neither is cited above as grounding; the decision is fully grounded
in the four code sites at the top of this document.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/state.py`,
  `agent_utilities/graph/executor.py`, `agent_utilities/graph/_router_impl.py`,
  `agent_utilities/capabilities/adversarial_verifier.py`.
- **Backward Compatible**: Yes — `signal_board` defaults to an empty dict; a
  run with no emitters is a no-op at every injection site.
- **Known weak point**: the board is a flat `dict[str, list[str]]` with no
  emitter identity or timestamp — two specialists in the same wave writing the
  same `signal_type` are indistinguishable in the injected digest, and
  `graph/state.py:674`'s fork-handling appends a `"state_fork"` entry into the
  same untyped structure, so the channel has no schema evolution path beyond
  "another string in a list."
