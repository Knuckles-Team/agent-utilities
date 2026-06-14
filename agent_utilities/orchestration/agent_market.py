#!/usr/bin/python
from __future__ import annotations

"""Market-based task allocation for an agent collective.

CONCEPT:ORCH-1.51 — a virtual-agent-economy task allocator that runs a capability-gated second-price auction so a collective self-organizes who does what via price signals instead of static role routing, surfacing scarcity and value

The paper (§5.4) names *Virtual Agent Economies* — collectives that emerge from market
dynamics where "price signals coordinate vast numbers of AGI agents" — as a path to
collective superintelligence. AU's allocation was centrally pushed by static role strings
or keyword matching, with no notion of price, scarcity or self-organization. This adds the
decentralized lever: capability-matched agents bid a cost (a token/compute budget,
discounted by their calibrated confidence) for a task, and a second-price (Vickrey)
auction clears it — truthful bidding is a dominant strategy, the clearing price is a
scarcity signal, and the collective allocates work without a central planner. Pure and
backend-agnostic; the production hook sits behind the ORCH-1.45 dispatch seam.
"""

from dataclasses import dataclass, field


@dataclass
class Bid:
    """One agent's bid to perform a task."""

    agent_id: str
    cost: float  # the agent's asked cost (lower = cheaper); e.g. token/compute budget
    capabilities: frozenset[str] = field(default_factory=frozenset)
    confidence: float = 1.0  # calibrated success confidence in (0, 1]; discounts cost

    @property
    def effective_cost(self) -> float:
        """Confidence-adjusted cost — a more reliable agent is effectively cheaper."""
        return self.cost / max(1e-6, self.confidence)


@dataclass
class Allocation:
    """The cleared allocation: who won and the scarcity-signalling clearing price."""

    winner: str | None
    clearing_price: float
    bidders: int
    reason: str = ""


class MarketAllocator:
    """Capability-gated second-price (Vickrey) auction over agent bids."""

    def allocate(
        self, required_capabilities: frozenset[str] | set[str], bids: list[Bid]
    ) -> Allocation:
        """Clear one task to the lowest effective-cost capable bidder.

        Eligible bidders must provide every required capability. The winner is the
        lowest effective cost; it is charged the *second*-lowest effective cost (the
        Vickrey clearing price), which makes truthful bidding dominant and exposes the
        marginal scarcity of the capability. With one eligible bidder the clearing price
        is its own cost; with none the task is unallocated.
        """
        req = set(required_capabilities)
        eligible = sorted(
            (b for b in bids if req <= set(b.capabilities)),
            key=lambda b: b.effective_cost,
        )
        if not eligible:
            return Allocation(None, 0.0, 0, "no capability-matched bidder")
        winner = eligible[0]
        clearing = (
            eligible[1].effective_cost if len(eligible) > 1 else winner.effective_cost
        )
        return Allocation(
            winner=winner.agent_id,
            clearing_price=round(clearing, 6),
            bidders=len(eligible),
            reason="second-price clear"
            if len(eligible) > 1
            else "single eligible bidder",
        )

    def scarcity(self, allocations: list[Allocation]) -> float:
        """Mean clearing price across recent allocations — a capability scarcity index.

        Persistently high clearing prices for a capability signal scarcity, which a
        scaler (OS-5.29) can read to add capacity for that capability.
        """
        cleared = [a.clearing_price for a in allocations if a.winner is not None]
        return sum(cleared) / len(cleared) if cleared else 0.0
