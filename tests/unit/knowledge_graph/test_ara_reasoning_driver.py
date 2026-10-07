"""OntologyReasoningDriver — reasoning-as-engine harvest (CONCEPT:AU-KG.research.best-effort-lightweight-never).

43197d7c6 ("refactor: move semantic authority to epistemic graph") rewrote
this driver onto EG-native, read-only OwlReason: it no longer runs a local
OWL bridge, downfeeds inferred edges into the graph, or promotes cross-domain
findings into research topics/loops (see reasoning_driver.py's own module
docstring — "ARA's former cross-domain edge harvest therefore remains
unavailable until EG exposes that exact result contract"). Every code path
through ``extrapolate()`` now ends in a non-empty ``error`` — even a fully
consistent, well-formed native reasoning result reports "unsupported",
by design — so the former harvest-success test (asserting new inferred
edges/topics) tested a capability that no longer exists here at all, and was
deleted rather than updated.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.ara import OntologyReasoningDriver


class _Graph:
    def __init__(self, *, owl_reason=None):
        self._owl_reason = owl_reason

    def owl_reason(self, class_base: str):
        if self._owl_reason is None:
            raise AssertionError("owl_reason not configured for this test")
        return self._owl_reason(class_base)


class _Engine:
    def __init__(self, *, owl_reason=None):
        self.graph = _Graph(owl_reason=owl_reason)
        self.backend = None


def test_consistent_result_still_reports_harvest_unsupported():
    """Even a fully consistent, well-formed native result is not a harvest:
    OwlReason is read-only and returns no inferred property edges, so every
    successful call still reports the documented "unsupported" error."""

    def _owl_reason(class_base: str):
        assert class_base == "http://agent-utilities.dev/ontology#"
        return {
            "consistent": True,
            "schema_digests": ["sha256:abc"],
            "subclasses": [1, 2],
            "direct_subclasses": [1],
            "instances": [1, 2, 3],
        }

    eng = _Engine(owl_reason=_owl_reason)
    h = OntologyReasoningDriver(eng).extrapolate()

    assert h.inferred_edges == []
    assert h.new_topics == []
    assert h.stats == {
        "mode": "read_only",
        "consistent": True,
        "schema_digests": ["sha256:abc"],
        "subclass_entailments": 2,
        "direct_subclass_entailments": 1,
        "instance_entailments": 3,
    }
    assert "unsupported" in h.error


def test_inconsistent_result_reports_the_specific_reason():
    eng = _Engine(owl_reason=lambda class_base: {"consistent": False})
    h = OntologyReasoningDriver(eng).extrapolate()
    assert h.error == "native OwlReason did not prove committed ontology consistency"


def test_missing_schema_digests_reports_the_specific_reason():
    eng = _Engine(owl_reason=lambda class_base: {"consistent": True})
    h = OntologyReasoningDriver(eng).extrapolate()
    assert h.error == "native OwlReason omitted committed GraphSchema digests"


def test_reasoning_failure_is_best_effort():
    def _boom(class_base: str):
        raise RuntimeError("no owl backend")

    eng = _Engine(owl_reason=_boom)
    h = OntologyReasoningDriver(eng).extrapolate()
    assert h.error and not h.inferred_edges  # degrades, never raises into the loop


def test_no_graph_yields_error_harvest():
    class _E:
        graph = None

    h = OntologyReasoningDriver(_E()).extrapolate()
    assert h.error == "native GraphComputeEngine.owl_reason is unavailable"
