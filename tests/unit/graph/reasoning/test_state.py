"""Unit tests for the shared reasoning-state substrate."""

from agent_utilities.graph.reasoning.state import (
    NodeKind,
    ReasoningState,
    ThoughtNode,
)


def test_children_of_supports_multiple_parents_merge_provenance():
    """A GoT aggregate node with two parent_ids is itself the merge record."""
    state = ReasoningState(topology="got")
    state.add_node(ThoughtNode("a", NodeKind.GENERATE, "thought a"))
    state.add_node(ThoughtNode("b", NodeKind.GENERATE, "thought b"))
    state.add_node(
        ThoughtNode("merged", NodeKind.AGGREGATE, "merged", parent_ids=["a", "b"])
    )

    assert state.children_of("a") == ["merged"]
    assert state.children_of("b") == ["merged"]
    assert state.nodes["merged"].parent_ids == ["a", "b"]


def test_path_to_root_walks_primary_edge_only():
    state = ReasoningState(topology="rap")
    state.add_node(ThoughtNode("root", NodeKind.ROOT, "q"))
    state.add_node(ThoughtNode("mid", NodeKind.GENERATE, "m", parent_ids=["root"]))
    state.add_node(ThoughtNode("leaf", NodeKind.GENERATE, "l", parent_ids=["mid"]))

    assert state.path_to_root("leaf") == ["leaf", "mid", "root"]


def test_path_to_root_never_infinite_loops_on_malformed_cycle():
    state = ReasoningState(topology="rap")
    state.add_node(ThoughtNode("a", NodeKind.GENERATE, "a", parent_ids=["b"]))
    state.add_node(ThoughtNode("b", NodeKind.GENERATE, "b", parent_ids=["a"]))

    path = state.path_to_root("a")
    assert path == ["a", "b"]


def test_append_rationale_is_append_only():
    state = ReasoningState(topology="cot")
    state.append_rationale("step 1 summary")
    state.append_rationale("step 2 summary")
    assert state.rationale_summary == ["step 1 summary", "step 2 summary"]


def test_memento_round_trip_preserves_full_state():
    state = ReasoningState(topology="tot_bfs")
    state.add_node(ThoughtNode("n0", NodeKind.ROOT, "q", value=1.0))
    state.add_node(
        ThoughtNode("n1", NodeKind.GENERATE, "c", parent_ids=["n0"], value=0.7)
    )
    state.frontier = ["n1"]
    state.visited = {"n0"}
    state.append_rationale("summary")
    state.step = 3

    restored = ReasoningState.from_memento(state.to_memento())

    assert restored.topology == "tot_bfs"
    assert set(restored.nodes) == {"n0", "n1"}
    assert restored.nodes["n1"].value == 0.7
    assert restored.nodes["n1"].parent_ids == ["n0"]
    assert restored.frontier == ["n1"]
    assert restored.visited == {"n0"}
    assert restored.rationale_summary == ["summary"]
    assert restored.step == 3


def test_thought_node_uct_unvisited_is_infinite():
    node = ThoughtNode("n", NodeKind.GENERATE, "c")
    assert node.uct(parent_visits=5, c=1.4) == float("inf")


def test_thought_node_uct_rewards_higher_mean_reward():
    high = ThoughtNode("high", NodeKind.GENERATE, "c", visits=4, total_reward=3.6)
    low = ThoughtNode("low", NodeKind.GENERATE, "c", visits=4, total_reward=0.4)
    assert high.uct(parent_visits=10, c=1.4) > low.uct(parent_visits=10, c=1.4)
