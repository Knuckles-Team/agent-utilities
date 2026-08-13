"""CONCEPT:AU-KG.backend.company-brain-write-guard

Regression test for B-20: ``CompanyBrain.pre_commit_validate`` used to call
``base_graph.fork()`` unconditionally, which raises a bare ``AttributeError`` for any
``base_graph`` that doesn't implement ``fork()`` -- which is every graph object this
repo can actually construct today (the engine dispatches ``Method::Fork`` server-side,
but the Python client has never bound it). This module does NOT depend on the compiled
``epistemic_graph._epistemic_graph`` extension (unlike ``test_company_brain.py``, which
``pytest.importorskip``s it and is therefore always skipped in this environment) --
it proves the *guard* fires with a specific, named, actionable error instead of an
AttributeError, using a plain stand-in object.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.company_brain import (
    CompanyBrain,
    GraphForkUnavailableError,
)


class _GraphWithoutFork:
    """Stands in for any graph object that has no ``fork()`` -- true of every concrete
    graph type this repo can construct today (see module docstring)."""


class _GraphWithFork:
    """Stands in for a graph object that DOES implement ``fork()`` (a future bound
    client, or a test double), proving the guard does not block a capable graph."""

    def __init__(self) -> None:
        self.forked = False

    def fork(self) -> _GraphWithFork:
        self.forked = True
        return self

    def has_node(self, node_id: str) -> bool:
        return False

    def add_node(self, node_id: str, props_str: str) -> None:
        return None

    def add_edge(self, src: str, tgt: str, props_str: str) -> None:
        return None

    def get_nodes(self) -> list[tuple[str, str]]:
        return []

    def get_edges(self) -> list[tuple[str, str, str]]:
        return []


def test_pre_commit_validate_fails_loudly_not_with_attributeerror_when_fork_is_missing():
    """The B-20 path must never surface a bare AttributeError."""
    brain = CompanyBrain()

    with pytest.raises(GraphForkUnavailableError) as excinfo:
        brain.pre_commit_validate(
            base_graph=_GraphWithoutFork(),
            proposed_node=("agent:test", {"type": "Agent", "name": "Test"}),
        )

    message = str(excinfo.value)
    assert "fork()" in message
    assert "Method::Fork" in message
    assert "_GraphWithoutFork" in message


def test_pre_commit_validate_does_not_raise_attributeerror_when_fork_is_missing():
    """Explicitly pin the failure MODE: AttributeError must never leak from this path."""
    brain = CompanyBrain()

    try:
        brain.pre_commit_validate(base_graph=_GraphWithoutFork())
    except GraphForkUnavailableError:
        pass
    except AttributeError:  # pragma: no cover - exactly what this test guards against
        pytest.fail(
            "pre_commit_validate() leaked a bare AttributeError instead of raising "
            "the named GraphForkUnavailableError"
        )


def test_pre_commit_validate_calls_fork_when_the_graph_supports_it():
    """A graph object that DOES implement fork() is used, not rejected."""
    brain = CompanyBrain()
    graph = _GraphWithFork()

    report = brain.pre_commit_validate(
        base_graph=graph,
        proposed_node=("agent:test", {"type": "Agent", "name": "Test"}),
    )

    assert graph.forked is True
    assert "conforms" in report
