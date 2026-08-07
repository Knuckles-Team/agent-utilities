"""GraphComputeEngine.query_unified — client version-compat for
``reorder_filter_selectivity`` (D-W2X-4).

CONCEPT:AU-KG.compute.graph-compute-engine — client-version compat for the
engine's one-costed-round-trip unified plan (retargeted off the retired
``AU-KG.compute.kg-2`` bare legacy-numbering citation, D-CDOC-1/D-CIP-16: the
decision these tests actually exercise, "computation happens in the engine in
one round-trip", is documented at
``.specify/design/kg-engine-native-compute/design.md`` under this id).

The installed ``epistemic_graph`` client's ``query.unified()`` may predate
``reorder_filter_selectivity`` under the frozen ``au 2.0.0`` / ``eg 2.23.1``
pin -- passing it unconditionally TypeErrors against an older client. These
tests construct a bare ``GraphComputeEngine`` (no real engine connection,
matching ``test_graph_compute_audit.py``'s pattern) with a fake ``_client``
whose ``query.unified`` either does or doesn't accept the kwarg.
"""

from __future__ import annotations

import inspect

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def _bare_engine(unified_fn) -> GraphComputeEngine:
    eng = GraphComputeEngine.__new__(GraphComputeEngine)

    class _Query:
        pass

    class _Client:
        pass

    client = _Client()
    client.query = _Query()
    client.query.unified = unified_fn
    eng._client = client
    return eng


@pytest.fixture(autouse=True)
def _no_epistemic_attach(monkeypatch):
    # Keep these tests focused on the reorder_filter_selectivity dispatch —
    # not the separate light-epistemic-attach layer, which needs its own
    # explain_provenance_by_ids wiring. Empty rows + light_default=False means
    # should_attach_epistemic_columns short-circuits False (no rows to be
    # contested), so attach_epistemic_columns is never reached.
    from agent_utilities.core.config import config as app_config

    monkeypatch.setattr(app_config, "epistemic_light_default", False, raising=False)


def test_passes_reorder_kwarg_when_the_client_accepts_it():
    calls = []

    def unified(plan, reorder_filter_selectivity=None):
        calls.append((plan, reorder_filter_selectivity))
        return []

    eng = _bare_engine(unified)
    eng.query_unified([{"Scan": {"label": "X"}}], reorder_filter_selectivity=0.5)

    assert calls == [([{"Scan": {"label": "X"}}], 0.5)]


def test_falls_back_without_the_kwarg_when_the_client_predates_it():
    calls = []

    def unified(plan):  # no reorder_filter_selectivity in this signature at all
        calls.append(plan)
        return []

    eng = _bare_engine(unified)
    # Must NOT raise TypeError('unexpected keyword argument') -- this is the
    # exact bug D-W2X-4 named, reproduced live against the currently-installed
    # epistemic_graph client.
    eng.query_unified([{"Scan": {"label": "X"}}], reorder_filter_selectivity=0.5)

    assert calls == [[{"Scan": {"label": "X"}}]]


def test_older_client_signature_genuinely_lacks_the_kwarg():
    # Sanity check on the introspection this fix relies on: a plain
    # single-arg callable's signature really doesn't carry the parameter.
    def unified(plan):
        return []

    assert "reorder_filter_selectivity" not in inspect.signature(unified).parameters
