# Design Document: Inject only the top-N most relevant specialists into the router prompt, selected by KG hybrid search — not the full discovered roster

CONCEPT:AU-ORCH.routing.filtered-specialist-injection

> Realised by `agent_utilities/graph/_router_impl.py:503-509` (the
> `get_relevant_specialists` call) and `:548-549`
> (`format_specialist_step_info`), with the retrieval helper at
> `agent_utilities/core/config.py:6535-6564`. The formatter lives in
> `agent_utilities/graph/routing/strategies/optimization.py`.

## Decision — treat the router prompt's specialist roster as a retrieval problem with a fixed budget, rather than an enumeration

The router needs to know which specialists exist in order to route to them. The
straightforward implementation puts the full discovered specialist list into
the router's prompt.

This decision caps that list at the top 7 most relevant specialists, selected
by KG hybrid search against the task (`config.py:6535-6564`), and then filters
further on pheromone/telemetry signals before formatting
(`_router_impl.py:503-549`).

**The rejected alternative is injecting the unfiltered roster**, and the code's
own framing names why it lost: *filtered specialist injection for prompt bloat
reduction*. The roster is discovered at runtime and grows with the fleet, so
its size is not a property anyone controls — every specialist added anywhere in
the system silently lengthens every router prompt. That has three costs that
compound: tokens on every single routed turn, latency proportional to the
prompt, and — the one that actually degrades quality — dilution, since a
correct choice among 7 candidates is an easier discrimination for the model
than the same choice among many dozens.

Choosing a *fixed* budget of 7 rather than a proportional or adaptive one keeps
the router's prompt cost constant as the fleet grows, which is the property
that makes fleet growth safe.

**Evidence strength, stated honestly:** the rejected alternative here is
attested by the in-code framing and by the structure of the change itself.
Unlike most decisions in this domain, no commit message elaborating the
trade-off was recoverable — the pickaxe search on this code bottoms out in a
large squashed documentation-hardening commit with no descriptive body. The
"before" state (unfiltered injection) is inferred from the code's own
description of what it is doing and why, not from a quoted prior implementation.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/_router_impl.py`,
  `agent_utilities/core/config.py`,
  `agent_utilities/graph/routing/strategies/optimization.py`.
- **Backward Compatible**: Yes mechanically; the router's *choice set* is
  narrower, which is a behaviour change in the routing outcome.
- **Known weak point**: this makes routing quality depend on retrieval quality.
  A specialist that hybrid search fails to surface for a task is not merely
  ranked low — it is absent from the prompt, so the router cannot select it at
  all and there is no signal that it was omitted. The top-7 cap is a
  hardcoded constant with no measurement behind it, and nothing tracks how often
  the correct specialist fell outside the cut.
