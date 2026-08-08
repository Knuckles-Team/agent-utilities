# Design Document: A chat turn runs under the chat execution profile, and a backend timeout is answered with a static degraded message — never a second LLM call to the same sick endpoint

CONCEPT:AU-ORCH.routing.chat-budget-routing

> Realised by `agent_utilities/messaging/router.py:1050-1063`
> (`_is_backend_timeout`) and `:1370-1541` (`_graph_agent_reply` — profile
> selection at `:1444-1459`, timeout handling at `:1490-1526`, fallback
> suppression at `:1538`). Introduced by commit `b794e6af`
> ("perf(orchestration): chat execution profile + non-blocking reply path
> (P0/P1)").

## Decision — bound the per-round budget by execution profile, and make the failure path cheaper than the success path

Two things were wrong with a chat turn's cost profile, and this decision fixes
both with one rule.

First, budget. `_graph_agent_reply` runs the universal graph agent with
`execution_profile="chat"` (`:1444-1459`), which bounds each LLM round to the
chat budget (~12s) rather than inheriting the 300s task-profile default. A
conversational turn that has not answered in 12s is not going to become a good
answer at 300s; it is going to hold a messaging session open until the user
gives up.

Second — and this is the substantive part — the failure path. The rejected
alternative is named in the commit as *"the double-LLM tax"*: on failure, the
prior code always issued a second, full plain-chat LLM call to the same
endpoint, which the commit says *"pushed a single turn past 90s."* The flaw is
that the retry is aimed at the endpoint that just failed. If the first call
timed out because the backend is saturated or down, the second call is the
worst possible response: it doubles load on a degraded endpoint, doubles the
user's wait, and then usually fails too.

So the rule is discriminating rather than blanket. `_is_backend_timeout`
(`:1050-1063`) classifies the failure. On a genuine backend/LLM timeout the
router returns a graceful, *static* degraded message — no second model call at
all (`:1538`). Plain-chat fallback still fires, but only for non-timeout
failures: structural or delegation errors, where the backend is healthy and a
simpler request genuinely might succeed. The retry was not deleted; it was
restricted to the case where it can actually help.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/router.py`; the execution-shape
  computation it consumes lives in
  `agent_utilities/orchestration/execution_profile.py`.
- **Backward Compatible**: Behaviourally no, and deliberately so — a chat turn
  that previously ran up to 300s per round now stops at the chat budget, and a
  timed-out turn now returns a fixed message where it previously returned a
  second-attempt answer. Both changes are the point of the commit.
- **Known weak point**: the whole rule rests on `_is_backend_timeout`
  correctly distinguishing "the backend is sick" from "this particular request
  was too hard". A misclassification in one direction wastes a call on a dying
  endpoint; in the other it returns a canned degraded message to a user whose
  request would have succeeded on a retry. The classifier is a heuristic over
  exception shape, not a health signal from the endpoint itself.
