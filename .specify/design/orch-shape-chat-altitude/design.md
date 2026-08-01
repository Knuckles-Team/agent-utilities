# Design Document: A chat turn gets a bounded node-timeout altitude, and the shape's own budget — not a fixed transport timeout — decides inline-vs-deferred delivery

CONCEPT:AU-ORCH.execution.chat-profile-timeouts ·
CONCEPT:AU-ORCH.execution.passthrough-identity ·
CONCEPT:AU-ORCH.execution.rich-result-wrapper

> `agent_utilities/orchestration/execution_profile.py` (module docstring,
> `reply_budget_s`, `is_interactive`), `agent_utilities/messaging/router.py`,
> `agent_utilities/orchestration/agent_runner.py`.

## Decision — a `"chat"` `ExecutionProfile` bounds every node round to a small fraction of the messaging reply budget, so a degraded backend fails ONE fast attempt instead of stalling the whole turn

`CONCEPT:AU-ORCH.execution.chat-profile-timeouts`

The module docstring (`execution_profile.py:1-25`) names the concrete bug
this fixes: both chat turns and multi-step tasks historically used the same
`DEFAULT_GRAPH_ROUTER_TIMEOUT`/`DEFAULT_GRAPH_VERIFIER_TIMEOUT` (300s each).
On a degraded backend, the FIRST router round of a chat turn alone could
stall for the full 300s — far above the messaging reply budget — which then
killed the run and triggered a SECOND slow LLM call via the plain-chat
fallback (measured: >90s total for what should have been a fast failure).
`CHAT_NODE_TIMEOUT_S = 12.0` bounds each sequential LLM round in the `"chat"`
profile so even several rounds stay inside the reply budget, and
`resolve_execution_profile` additionally shrinks the node budget further if
the deployment's configured `MESSAGING_REPLY_TIMEOUT` is set even lower — a
single round still fits regardless of how the reply budget is tuned.

**The rejected alternative is the one long timeout for every entrypoint.**
It's the simpler design (one constant, no per-entrypoint branching) and it's
what shipped first — the docstring is describing an actual prior-state bug,
not a hypothetical. It loses because "task" and "chat" genuinely want
different altitudes: a task may legitimately run several specialist rounds
each worth waiting the long default for, while a chat turn must answer inside
a human-scale budget or the transport-level reply timeout kills it anyway —
uniformly long timeouts don't fail fast for chat, and uniformly short ones
would truncate a legitimate task.

### Pointer — `CONCEPT:AU-ORCH.execution.passthrough-identity`

Two related but distinct facets share this concept id:

**(1) Shape-driven inline-vs-deferred delivery** —
`execution_profile.py:139-176` (`reply_budget_s`, `is_interactive`) and
`agent_utilities/messaging/router.py:492-593`. The dynamic per-job
`reply_budget_s` (derived from which shape stages are enabled: direct=25s,
focused-tools=90+40×servers capped at 180s, a full turn adds discovery/
resolve/verify/expert-execution time) decides BOTH how long a turn should
reasonably take AND whether the messaging transport answers it INLINE
(`shape.is_interactive`, budget ≤ 50s) or acknowledges immediately and
delivers the result as a background follow-up. The router's own comment
frames this as a passthrough: "the transport stays thin — the core shape
makes the call; here we only render it for this medium" — i.e. the shape's
decision passes through to the transport layer unchanged, rather than the
transport re-deciding timing on its own terms. The rejected alternative is a
single fixed reply timeout applied uniformly: wrong in both directions
(too-long a wait for a trivial turn on a degraded backend, and a premature
cutoff of a legitimate multi-agent tool turn that genuinely needs the
external tool round-trip time).

**(2) Prompt-only universal entrypoints bypass KG agent resolution
entirely** — `agent_utilities/orchestration/agent_runner.py:170-174, 836`.
`_PASSTHROUGH_AGENTS = frozenset({"messaging-assistant"})` names identities
that must flow through the full multi-agent graph AS THEMSELVES, never
resolved against the KG as if they were a named specialist. Resolving one is
explicitly called out as both wasteful (a multi-second semantic search,
measured ~21s) and actively WRONG (it mis-binds the universal messaging
assistant to an unrelated tag via `prepare_messages`). The rejected
alternative — resolving every `agent_name` including these — is not merely
slower, it produces an incorrect binding, which is why this is a hard
exemption (`_PASSTHROUGH_AGENTS` membership) rather than a soft
optimization.

Both facets share the same underlying idea — an identity or a decision
"passes through" a layer unchanged rather than being re-interpreted by
it — which is why the codebase reuses one concept id for what are, at the
code level, two separate mechanisms in two different files.

### Pointer — `CONCEPT:AU-ORCH.execution.rich-result-wrapper`

`agent_utilities/orchestration/agent_runner.py:1729-1745`. When a caller opts
into the richer result wrapper (`return_mermaid=True`, the MCP
`execute_agent` path), the `run_id` is ALWAYS surfaced — the handle needed to
query that run's `RunTrace`/`:ToolCall` provenance (KG-2.296) over graph-os,
and the prerequisite for async/streaming/steering later. Internal callers
(`return_mermaid=False`) keep the bare-string contract bit-for-bit unchanged.
The rejected alternative is surfacing `run_id` conditionally or only on
error — that would leave a caller that opted into rich output with no
trackable handle for a SUCCESSFUL run, exactly when a caller would want to
look up what actually happened (tool calls made, timing) rather than just
trusting the text answer.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/execution_profile.py`,
  `agent_utilities/messaging/router.py`,
  `agent_utilities/orchestration/agent_runner.py`.
- **Backward Compatible**: Yes — `"task"` profile behavior (long timeouts,
  bare-string internal contract) is byte-for-byte unchanged; only the `"chat"`
  profile and the opt-in rich wrapper get new behavior.
- **Known weak point**: `_PASSTHROUGH_AGENTS` is a hardcoded single-entry
  frozenset — a new universal prompt-only entrypoint added later must
  remember to register itself here, or it silently pays the same wasteful
  and potentially-mis-binding KG resolution `messaging-assistant` was
  explicitly exempted from.
