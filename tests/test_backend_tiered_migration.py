"""Retired-``tiered``-backend migration.

The engine-authority consolidation removed L0/L1/L2/L3 tiering. A deployment whose
GRAPH_BACKEND env / config still says ``tiered`` must keep booting by mapping forward
to the self-contained engine authority (``epistemic_graph``) — not crash with
"Unknown graph backend type: 'tiered'".
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends import create_backend


def test_tiered_maps_to_epistemic_graph(monkeypatch):
    monkeypatch.setenv("GRAPH_BACKEND", "tiered")
    backend = create_backend()
    assert backend is not None
    assert "Epistemic" in type(backend).__name__


def test_explicit_tiered_arg_maps_too():
    backend = create_backend(backend_type="tiered")
    assert backend is not None
    assert "Epistemic" in type(backend).__name__
