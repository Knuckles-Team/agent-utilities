# Design Document: Task allocation clears through a capability-gated second-price auction, so bidding your true cost is the dominant strategy

CONCEPT:AU-ORCH.routing.virtual-agent-economy-task

> Realised by `agent_utilities/orchestration/agent_market.py:1-89`
> (`MarketAllocator`, `Bid`). Introduced by commit `21069306` ("feat:
> multi-agent collective — market, specialization, hierarchical coord"),
> operationalising §5.4 "Virtual Agent Economies" of the collective-intelligence
> paper the commit cites.

## Decision — price and scarcity decide who does the work, and the clearing rule is second-price specifically to make honest bidding rational

The module docstring states the prior condition: *"AU's allocation was
centrally pushed by static role strings or keyword matching, with no notion of
price, scarcity or self-organization."* Static assignment has no way to express
that an agent is currently busy, that a task is unusually cheap for one
specialist, or that two agents could both do it but one is much better placed.

`MarketAllocator` replaces the push with a clearing mechanism. Agents bid;
each bid's cost is discounted by that agent's calibrated confidence, so a
confident agent effectively underbids an uncertain one at the same nominal
cost. Bids are capability-gated first — an agent that cannot do the task is
excluded before price is considered, so the market cannot allocate work to
someone cheap and unqualified.

**The rejected alternative inside the auction design is a first-price
(pay-your-bid) clearing rule, and the reason it lost is the substantive part of
this decision.** Under first-price, the rational move is to shade your bid —
bid above your true cost and hope to still win. Every agent shades, the bids
stop meaning anything, and the allocator is choosing between numbers that no
longer represent cost. Second-price (Vickrey) clearing — the winner is the
lowest bidder but pays the second-lowest bid — makes truthful bidding a
dominant strategy, which is the property the docstring calls out. The
allocator's inputs are only useful if they are honest, and second-price is what
makes honesty the self-interested choice rather than an assumption.

Static role-based push was rejected as the outer alternative because it cannot
express scarcity at all: it produces the same assignment whether the chosen
agent is idle or saturated.

## Status — designed and tested, not yet load-bearing

This must not be read as describing live production behaviour.
`git grep` shows `MarketAllocator` and `Bid` are referenced only from
`tests/unit/graph/test_orch_collective.py`. The module's own docstring is
candid that *"the production hook sits behind the ORCH-1.45 dispatch seam"* —
the seam exists, the market is not wired into it. The decision and its
rationale are real and the mechanism is implemented and tested; what has not
happened is adoption on the live dispatch path.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/agent_market.py` only,
  today. Wiring it into the ORCH-1.45 dispatch seam would extend the radius to
  every dispatched task.
- **Backward Compatible**: Yes, trivially — nothing in production calls it.
- **Known weak point**: the incentive guarantee is only as good as the cost
  and confidence inputs. Second-price makes truthful bidding dominant *given*
  an agent's true cost, but an agent whose calibrated confidence is
  systematically overstated wins bids it should lose, and the auction has no
  mechanism to penalise that — the calibration loop lives elsewhere. Until the
  production hook is wired, none of this is exercised against real bidders.
