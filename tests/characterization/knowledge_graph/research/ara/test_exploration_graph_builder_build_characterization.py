"""Characterization tests for ``ExplorationGraphBuilder.build`` (CX-AU-09).

CCN 11 at time of writing
(``agent_utilities/knowledge_graph/research/ara/exploration.py``). These
tests pin the OBSERVED, black-box behaviour before any decomposition: exact
node ordering across all five input categories, the id-numbering scheme
(a single counter shared across every call, NOT per-kind), the None-vs-empty-
list equivalence, and the root-parent defaulting.

Per the two-commit discipline, this file must be added and pass GREEN
against the UNMODIFIED ``exploration.py`` before any refactor commit, and
must not change during the refactor commit that follows.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.ara.exploration import (
    ExplorationGraphBuilder,
)


def test_build_root_only_with_no_optional_inputs() -> None:
    traj = ExplorationGraphBuilder("art-1").build("Root question?")
    assert len(traj.nodes) == 1
    root = traj.nodes[0]
    assert root.id == "exploration_node:art-1:question:1"
    assert root.kind == "question"
    assert root.text == "Root question?"
    assert root.parent_id == ""
    assert traj.root_id == root.id


def test_build_none_and_empty_list_inputs_are_equivalent() -> None:
    traj_none = ExplorationGraphBuilder("a").build("Q")
    traj_empty = ExplorationGraphBuilder("a").build(
        "Q",
        decisions=[],
        experiments=[],
        results=[],
        failure_clusters=[],
        matcher_rejects=[],
    )
    assert len(traj_none.nodes) == len(traj_empty.nodes) == 1


def test_build_orders_all_five_categories() -> None:
    # OBSERVED: the id counter is a SINGLE monotonic sequence shared across
    # every call to self._nid -- NOT a per-kind counter. The root consumes
    # n=1, so the first decision is n=2, not n=1.
    builder = ExplorationGraphBuilder("art-2")
    traj = builder.build(
        "Question text",
        decisions=["d0", "d1"],
        experiments=["e0"],
        results=["r0"],
        failure_clusters=["fc0"],
        matcher_rejects=["mr0"],
    )
    kinds = [n.kind for n in traj.nodes]
    assert kinds == [
        "question",
        "decision",
        "decision",
        "experiment",
        "result",
        "dead_end",
        "pivot",
    ]
    ids = [n.id for n in traj.nodes]
    assert ids == [
        "exploration_node:art-2:question:1",
        "exploration_node:art-2:decision:2",
        "exploration_node:art-2:decision:3",
        "exploration_node:art-2:experiment:4",
        "exploration_node:art-2:result:5",
        "exploration_node:art-2:dead_end:6",
        "exploration_node:art-2:pivot:7",
    ]
    texts = [n.text for n in traj.nodes]
    assert texts == ["Question text", "d0", "d1", "e0", "r0", "fc0", "mr0"]


def test_build_every_non_root_node_parents_to_the_root_by_default() -> None:
    traj = ExplorationGraphBuilder("art-3").build(
        "Q", decisions=["d"], failure_clusters=["fc"], matcher_rejects=["mr"]
    )
    root_id = traj.root_id
    for n in traj.nodes:
        if n.id != root_id:
            assert n.parent_id == root_id


def test_build_dead_ends_and_pivots_run_through_text_of() -> None:
    # dicts and plain strings both resolve to human text via _text_of.
    traj = ExplorationGraphBuilder("art-4").build(
        "Q",
        failure_clusters=[{"summary": "recall stalled at 502"}],
        matcher_rejects=["unrelated: kafka queue"],
    )
    dead_end = next(n for n in traj.nodes if n.kind == "dead_end")
    pivot = next(n for n in traj.nodes if n.kind == "pivot")
    assert dead_end.text == "recall stalled at 502"
    assert pivot.text == "unrelated: kafka queue"


def test_build_returns_a_trajectory_rooted_at_the_first_node() -> None:
    traj = ExplorationGraphBuilder("art-5").build("Q", results=["r0"])
    assert traj.root_id == traj.nodes[0].id
    assert traj.nodes[0].kind == "question"
