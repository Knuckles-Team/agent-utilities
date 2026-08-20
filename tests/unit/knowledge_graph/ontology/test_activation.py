"""Unit tests for ontology integrity-policy activation
(CONCEPT:AU-KG.ontology.integrity-bootstrap, ledger NE-152).

The live defect: the engine's RDF write guard (``EG-KG.ontology.rdf-update-
guard``) rejects EVERY ``AddTriples`` against an ontology graph until that
graph has a registered SHACL/ICV integrity policy. These tests exercise
:mod:`agent_utilities.knowledge_graph.ontology.activation` in isolation with
fake engine doubles — no live engine, no rdflib parsing required — proving:
ordering (policy registered + verified before content triples), restart
idempotence (bootstrap run twice is a no-op success), exact binding (a wrong
tenant/graph/policy digest is rejected, never silently reconfigured), bounded
retry/backoff with no log spam, and that every outcome is recorded for
``readiness.py`` to read.
"""

from __future__ import annotations

import logging

import pytest

from agent_utilities.knowledge_graph.ontology import activation


@pytest.fixture(autouse=True)
def _clean_activation_state():
    activation.reset_activation_state_for_tests()
    yield
    activation.reset_activation_state_for_tests()


ALT_SHAPES_TTL = """@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix icv: <urn:agent-utilities:ontology-integrity-policy#> .
icv:AltShape a sh:NodeShape .
"""


class _OrderedFakeGc:
    """Records every ``icv_configure``/``add_triples`` call, in order, so
    ordering can be asserted precisely."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def icv_configure(self, shapes, *, graph=None, mode="enforce"):
        self.calls.append(("icv_configure", graph))
        return True

    def add_triples(self, turtle=None, ntriples=None):
        self.calls.append(("add_triples", turtle))
        return {"triples": 0 if not turtle else 5}


class _FlakyGc:
    """``icv_configure`` fails transiently ``fail_times`` times, then
    succeeds — models "engine starting / shard moving"."""

    def __init__(self, fail_times: int) -> None:
        self._fail_times = fail_times
        self.icv_call_count = 0
        self.probe_call_count = 0

    def icv_configure(self, shapes, *, graph=None, mode="enforce"):
        self.icv_call_count += 1
        if self.icv_call_count <= self._fail_times:
            raise ConnectionError("engine starting")
        return True

    def add_triples(self, turtle=None, ntriples=None):
        if not turtle:
            self.probe_call_count += 1
        return {"triples": 0}


class _Bare:
    """No RDF/ICV surface at all."""


# ── requirement 1: ordering ────────────────────────────────────────────────


def test_policy_is_registered_and_verified_before_content_triples_load():
    gc = _OrderedFakeGc()
    activation.ensure_ontology_graph_activated(
        gc, tenant=None, graph_name="g-order", ontology_turtle=""
    )
    # Simulate the chokepoint's subsequent real-content load, exactly like
    # ``lifecycle._load_axioms`` does after activation succeeds.
    gc.add_triples(turtle="@prefix ex: <urn:ex#> .\nex:Dog a <urn:ex#Class> .")

    kinds = [c[0] for c in gc.calls]
    assert kinds[0] == "icv_configure", "policy must be registered FIRST"
    assert kinds[1] == "add_triples" and gc.calls[1][1] == "", (
        "the zero-triple readback/verify probe must run before any real load"
    )
    assert kinds[2] == "add_triples" and gc.calls[2][1] != "", (
        "real ontology content must only load AFTER activation succeeds"
    )


# ── requirement 3: restart idempotence ─────────────────────────────────────


def test_rerunning_bootstrap_twice_is_a_noop_success():
    gc = _OrderedFakeGc()
    first = activation.ensure_ontology_graph_activated(
        gc, tenant="acme", graph_name="g-idem", ontology_turtle="v1"
    )
    assert first.idempotent is False
    icv_calls_after_first = sum(1 for c in gc.calls if c[0] == "icv_configure")
    assert icv_calls_after_first == 1

    second = activation.ensure_ontology_graph_activated(
        gc, tenant="acme", graph_name="g-idem", ontology_turtle="v1"
    )
    assert second.idempotent is True
    icv_calls_after_second = sum(1 for c in gc.calls if c[0] == "icv_configure")
    assert icv_calls_after_second == icv_calls_after_first, (
        "a rerun must not perform a second IcvConfigure registration"
    )
    assert second.policy_digest == first.policy_digest
    assert second.graph == first.graph == "g-idem"


def test_new_ontology_content_version_over_same_graph_is_not_a_mismatch():
    """Loading a NEW ontology version over an already-activated graph
    (lifecycle.update()) must succeed idempotently -- only the POLICY digest
    is bound, never the content digest."""
    gc = _OrderedFakeGc()
    first = activation.ensure_ontology_graph_activated(
        gc, tenant=None, graph_name="g-version", ontology_turtle="v1 content"
    )
    second = activation.ensure_ontology_graph_activated(
        gc, tenant=None, graph_name="g-version", ontology_turtle="v2 DIFFERENT content"
    )
    assert second.idempotent is True
    # the recorded ontology_digest is provenance for the FIRST activation only
    assert second.ontology_digest == first.ontology_digest
    assert first.ontology_digest == activation._digest("v1 content")


# ── requirement 2: exact binding ───────────────────────────────────────────


def test_wrong_policy_digest_is_rejected_not_silently_reconfigured():
    gc = _OrderedFakeGc()
    activation.ensure_ontology_graph_activated(
        gc, tenant="acme", graph_name="g-policy", ontology_turtle="v1"
    )
    icv_calls_before = sum(1 for c in gc.calls if c[0] == "icv_configure")

    with pytest.raises(activation.OntologyPolicyBindingMismatchError):
        activation.ensure_ontology_graph_activated(
            gc,
            tenant="acme",
            graph_name="g-policy",
            ontology_turtle="v1",
            shapes_ttl=ALT_SHAPES_TTL,
        )
    icv_calls_after = sum(1 for c in gc.calls if c[0] == "icv_configure")
    assert icv_calls_after == icv_calls_before, (
        "a mismatched policy must never be silently registered"
    )


def test_wrong_tenant_is_rejected():
    gc = _OrderedFakeGc()
    activation.ensure_ontology_graph_activated(
        gc, tenant="tenant-a", graph_name="g-tenant", ontology_turtle="v1"
    )
    with pytest.raises(activation.OntologyPolicyBindingMismatchError):
        activation.ensure_ontology_graph_activated(
            gc, tenant="tenant-b", graph_name="g-tenant", ontology_turtle="v1"
        )


# ── requirement 4: bounded retry/backoff, no log spam ──────────────────────


def test_bounded_retry_recovers_from_transient_failures(caplog):
    gc = _FlakyGc(fail_times=2)
    with caplog.at_level(
        logging.WARNING, logger="agent_utilities.knowledge_graph.ontology.activation"
    ):
        record = activation.ensure_ontology_graph_activated(
            gc,
            tenant=None,
            graph_name="g-retry-ok",
            ontology_turtle="x",
            max_attempts=5,
            base_delay_s=0.001,
            max_delay_s=0.002,
            time_ceiling_s=5.0,
        )
    assert record.attempts == 3
    assert gc.icv_call_count == 3
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, (
        "only the FIRST failure may be logged, never every attempt"
    )


def test_bounded_retry_gives_up_after_max_attempts_with_typed_error(caplog):
    gc = _FlakyGc(fail_times=999)
    with caplog.at_level(
        logging.WARNING, logger="agent_utilities.knowledge_graph.ontology.activation"
    ):
        with pytest.raises(activation.OntologyActivationTimeoutError):
            activation.ensure_ontology_graph_activated(
                gc,
                tenant=None,
                graph_name="g-retry-fail",
                ontology_turtle="x",
                max_attempts=3,
                base_delay_s=0.001,
                max_delay_s=0.002,
                time_ceiling_s=5.0,
            )
    assert gc.icv_call_count == 3, "must stop at the attempt ceiling"
    warning_count = sum(1 for r in caplog.records if r.levelno == logging.WARNING)
    error_count = sum(1 for r in caplog.records if r.levelno == logging.ERROR)
    assert warning_count == 1, "no per-attempt log spam"
    assert error_count == 1, "exactly one final give-up log line"


def test_no_engine_rdf_icv_surface_raises_typed_error():
    with pytest.raises(activation.OntologyActivationError):
        activation.ensure_ontology_graph_activated(
            _Bare(), tenant=None, graph_name="g-bare", ontology_turtle="x"
        )


# ── requirement 5: readiness state recording ───────────────────────────────


def test_status_is_none_until_activation_is_attempted():
    assert activation.get_activation_status("g-untouched") is None


def test_status_is_ready_after_successful_activation():
    gc = _OrderedFakeGc()
    activation.ensure_ontology_graph_activated(
        gc, tenant=None, graph_name="g-status-ok", ontology_turtle="x"
    )
    status = activation.get_activation_status("g-status-ok")
    assert status is not None
    assert status["state"] == "ready"


def test_status_is_unavailable_after_timeout_giveup():
    gc = _FlakyGc(fail_times=999)
    with pytest.raises(activation.OntologyActivationTimeoutError):
        activation.ensure_ontology_graph_activated(
            gc,
            tenant=None,
            graph_name="g-status-fail",
            ontology_turtle="x",
            max_attempts=2,
            base_delay_s=0.001,
            max_delay_s=0.001,
            time_ceiling_s=5.0,
        )
    status = activation.get_activation_status("g-status-fail")
    assert status is not None
    assert status["state"] == "unavailable"
    assert status["reason"] == "activation_timeout"


def test_status_is_unavailable_after_binding_mismatch():
    gc = _OrderedFakeGc()
    activation.ensure_ontology_graph_activated(
        gc, tenant="acme", graph_name="g-status-mismatch", ontology_turtle="v1"
    )
    with pytest.raises(activation.OntologyPolicyBindingMismatchError):
        activation.ensure_ontology_graph_activated(
            gc,
            tenant="someone-else",
            graph_name="g-status-mismatch",
            ontology_turtle="v1",
        )
    status = activation.get_activation_status("g-status-mismatch")
    assert status["state"] == "unavailable"
    assert status["reason"] == "policy_binding_mismatch"
