"""Multi-agent collective: market allocation, emergent specialization, hierarchical
coordination (CONCEPT:ORCH-1.51, ORCH-1.52, ORCH-1.53).

The SCALE stage of the self-improving substrate: a market self-organizes who does what,
specialization is discovered from the task stream, and coordination federates so a
collective scales past small N.
"""

from __future__ import annotations

import pytest

from agent_utilities.graph.hierarchical_coordination import (
    HierarchicalCoordinator,
    majority,
    mean,
)
from agent_utilities.graph.specialization_discovery import SpecializationDiscovery
from agent_utilities.orchestration.agent_market import Bid, MarketAllocator

pytestmark = pytest.mark.concept("ORCH-1.51")


class TestMarket:
    def test_second_price_clears_to_cheapest_capable(self):
        bids = [
            Bid("a", cost=10, capabilities=frozenset({"code"})),
            Bid("b", cost=6, capabilities=frozenset({"code"})),
            Bid("c", cost=8, capabilities=frozenset({"code"})),
        ]
        alloc = MarketAllocator().allocate({"code"}, bids)
        assert alloc.winner == "b" and alloc.clearing_price == 8.0  # 2nd-lowest

    def test_capability_gating(self):
        bids = [
            Bid("a", cost=1, capabilities=frozenset({"vision"})),  # wrong cap
            Bid("b", cost=9, capabilities=frozenset({"code"})),
        ]
        alloc = MarketAllocator().allocate({"code"}, bids)
        assert alloc.winner == "b" and alloc.bidders == 1

    def test_no_eligible_bidder(self):
        alloc = MarketAllocator().allocate({"code"}, [Bid("a", 1, frozenset({"x"}))])
        assert alloc.winner is None

    def test_confidence_discounts_cost(self):
        bids = [
            Bid("reliable", cost=10, capabilities=frozenset({"c"}), confidence=1.0),
            Bid(
                "flaky", cost=6, capabilities=frozenset({"c"}), confidence=0.5
            ),  # eff 12
        ]
        alloc = MarketAllocator().allocate({"c"}, bids)
        assert alloc.winner == "reliable"  # 10 < 12 effective

    def test_scarcity_index(self):
        a = MarketAllocator()
        allocs = [
            a.allocate(
                {"c"}, [Bid("x", 4, frozenset({"c"})), Bid("y", 6, frozenset({"c"}))]
            )
        ]
        assert a.scarcity(allocs) == 6.0


@pytest.mark.concept("ORCH-1.52")
class TestSpecialization:
    def test_proposes_specialist_for_uncovered_niche(self):
        # a tight cluster of 3 tasks pointing in a direction no archetype covers
        tasks = [("t1", [0.0, 1.0]), ("t2", [0.01, 1.0]), ("t3", [0.0, 0.99])]
        archetypes = [[1.0, 0.0]]  # orthogonal ⇒ uncovered
        props = SpecializationDiscovery(min_cluster=3, coverage_floor=0.5).discover(
            tasks, archetypes
        )
        assert len(props) == 1 and props[0].size == 3 and props[0].coverage < 0.5

    def test_no_proposal_when_archetype_covers(self):
        tasks = [("t1", [0.0, 1.0]), ("t2", [0.0, 1.0]), ("t3", [0.0, 1.0])]
        archetypes = [[0.0, 1.0]]  # already covered
        assert SpecializationDiscovery(min_cluster=3).discover(tasks, archetypes) == []

    def test_no_proposal_for_small_cluster(self):
        tasks = [("t1", [0.0, 1.0]), ("t2", [0.0, 1.0])]  # only 2 < min_cluster
        assert SpecializationDiscovery(min_cluster=3).discover(tasks, []) == []


@pytest.mark.concept("ORCH-1.53")
class TestHierarchicalCoordination:
    def test_federated_majority(self):
        # 3 neighborhoods → reps [x, y, y] → unambiguous majority y
        votes = {
            "a": "x",
            "b": "x",
            "c": "y",
            "d": "y",
            "e": "y",
            "f": "y",
            "g": "y",
            "h": "x",
            "i": "y",
        }
        nbhd = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        out = HierarchicalCoordinator(majority).coordinate(votes, nbhd)
        assert out["consensus"] == "y" and len(out["representatives"]) == 3

    def test_interaction_is_bounded(self):
        votes = {str(i): "v" for i in range(100)}
        nbhd = [[str(i) for i in range(j, j + 10)] for j in range(0, 100, 10)]
        out = HierarchicalCoordinator().coordinate(votes, nbhd)
        assert (
            out["interaction_bound"] == 10 and out["global_bound"] == 100
        )  # 10 << 100

    def test_mean_aggregator(self):
        votes = {"a": 1.0, "b": 3.0, "c": 5.0, "d": 7.0}
        out = HierarchicalCoordinator(mean).coordinate(votes, [["a", "b"], ["c", "d"]])
        assert out["consensus"] == pytest.approx(4.0)  # mean([2.0, 6.0])

    def test_recursive_autopartition(self):
        votes = {str(i): "yes" for i in range(50)}
        out = HierarchicalCoordinator(majority).coordinate_recursive(votes, fanout=8)
        assert (
            out["consensus"] == "yes"
            and out["depth"] >= 2
            and out["global_bound"] == 50
        )
