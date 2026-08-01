# Design Document: An agent's task-start default is querying the code KG, not grepping the tree

CONCEPT:AU-KG.retrieval.task-start-kg-priming

> `agent_utilities/orchestration/agent_runner.py:889-895,2498,2666`.

## Decision — every agent run is primed with the KG's synthesized view of the task's code area, off the event loop, before the model sees the task

`agent_runner.py:889-895` names the default directly: "Prime the KG's
synthesized view of the task's code area — the task-start 'query the code KG
before you grep' default." `_prime_code_context` runs off the asyncio loop,
best-effort, and is skipped on the lightweight chat profile — it composes
with the sibling per-session memento priming step immediately above it
(`_prime_recent_mementos`, step 2b) which was moved off the loop for the same
reason: "the priming used to run inline in `_build_execution_config`," which
blocked the synchronous backend round-trip on the async reply path.

**The rejected alternative is the default an agent would otherwise fall back
to**: grep-then-read across the tree to build situational understanding of a
task's code area from scratch, on every run, with no memory of what the KG
already knows about that area (call graph, prior enrichment, related
`CONCEPT:` markers). Priming with `code_context` (the synthesized, cited
answer decision — see
`.specify/design/kgr-code-intelligence-cited-answers/design.md`) means the
model starts a task already grounded in the KG's existing understanding of the
relevant area, rather than re-deriving it turn one from raw file contents. The
priming being **best-effort and off-loop** is itself a rejected-alternative
choice: an earlier, in-line implementation ran synchronously inside
`_build_execution_config`, which is exactly the blocking-the-loop failure
mode `_prime_recent_mementos`'s neighboring comment documents as already
having caused production trouble elsewhere in this same function.

## Risk Assessment

- **Blast Radius**: `agent_runner.py` (`_prime_code_context`,
  `_build_execution_config`), any execution profile that isn't `chat`.
- **Backward Compatible**: Yes — best-effort priming that degrades to no
  prime on failure or on the chat profile; it does not change the task
  contract.
- **Known weak point**: "best-effort" priming that silently returns nothing
  on failure means a degraded KG read at task start is indistinguishable, to
  the model receiving the primed context, from "the KG genuinely knows
  nothing about this area yet" — there is no separate signal distinguishing
  the two.
