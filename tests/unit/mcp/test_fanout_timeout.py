"""Per-target timeout for graph fan-out (CONCEPT:AU-KG.backend.multi-connection-registry follow-up).

A slow/hung backend must not stall a ``target='all'`` fan-out: ``fanout_execute``
runs each target concurrently under a per-target wall-clock budget, returning the
fast results and marking the slow one as a timeout error (partial success) — instead
of the old sequential loop that blocked on the slowest backend.
"""

from __future__ import annotations

import time

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    CypherEngineError,
)
from agent_utilities.knowledge_graph.core.engine_breaker import EngineCircuitOpenError
from agent_utilities.mcp.kg_server import fanout_error_label, fanout_execute

pytestmark = pytest.mark.concept("AU-KG.backend.multi-connection-registry")


def _fast(name, _engine):
    return f"ok:{name}"


def test_fast_targets_all_return():
    results, errors = fanout_execute([("a", None), ("b", None)], _fast, timeout=5)
    assert results == {"a": "ok:a", "b": "ok:b"}
    assert errors == {}


def test_slow_target_times_out_while_others_return():
    def fn(name, _engine):
        if name == "slow":
            time.sleep(10)
            return "never"
        return f"ok:{name}"

    started = time.time()
    results, errors = fanout_execute(
        [("fast1", None), ("slow", None), ("fast2", None)], fn, timeout=1
    )
    elapsed = time.time() - started

    assert elapsed < 4  # did NOT wait 10s for the slow target
    assert results == {"fast1": "ok:fast1", "fast2": "ok:fast2"}
    assert errors["slow"] == "target_timeout"


def test_raising_target_is_captured_without_exception_text(caplog):
    def fn(name, _engine):
        if name == "bad":
            raise RuntimeError("credential-and-endpoint-material")
        return "ok"

    results, errors = fanout_execute([("ok1", None), ("bad", None)], fn, timeout=5)
    assert results == {"ok1": "ok"}
    assert errors["bad"] == "target_operation_failed"
    assert "credential-and-endpoint-material" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_empty_entries():
    assert fanout_execute([], _fast, timeout=5) == ({}, {})


# BUG-048 — a per-target breaker-open must be labeled distinctly from a
# per-target rejection. `fanout_execute` (and the `graph_query` "primary
# target" direct-call mirror in query_tools.py that deliberately bypasses its
# concurrency machinery but must match its error handling) previously reduced
# EVERY raised exception to the SAME "target_operation_failed" string,
# outside the OperationResult/OperationError schema `public_error_payload`
# already fixed — a breaker trip on one fan-out target was indistinguishable
# from a genuinely rejected query on that target, the same defect BUG-048
# fixed at the single-target boundary, just at a second, un-audited site.


def test_breaker_open_target_is_labeled_degraded_not_operation_failed(caplog):
    def fn(name, _engine):
        if name == "bad":
            raise EngineCircuitOpenError("breaker open, cooldown=15s")
        return "ok"

    results, errors = fanout_execute([("ok1", None), ("bad", None)], fn, timeout=5)
    assert results == {"ok1": "ok"}
    assert errors["bad"] == "target_degraded"
    assert errors["bad"] != "target_operation_failed"
    assert "breaker open, cooldown=15s" not in caplog.text


def test_cypher_engine_error_wrapping_a_breaker_trip_is_labeled_degraded():
    """The native-Cypher sanitization boundary wraps the breaker's exception
    `from None` before it reaches fan-out — this must still classify as
    retryable-degraded via the preserved `error_type` class name, not the
    (deliberately discarded) message/cause."""

    def fn(name, _engine):
        if name == "bad":
            raise CypherEngineError(
                "MATCH (n) RETURN n", "read", EngineCircuitOpenError("breaker open")
            )
        return "ok"

    results, errors = fanout_execute([("ok1", None), ("bad", None)], fn, timeout=5)
    assert results == {"ok1": "ok"}
    assert errors["bad"] == "target_degraded"


def test_cypher_engine_error_wrapping_a_rejection_stays_operation_failed():
    """The other half of the collapse: a genuine query rejection must NOT be
    upgraded to "target_degraded" just because it crossed the same
    sanitization boundary — the two must not collapse in either direction."""

    def fn(name, _engine):
        if name == "bad":
            raise CypherEngineError(
                "MATCH (n) RETURN n", "read", ValueError("unknown property 'foo'")
            )
        return "ok"

    results, errors = fanout_execute([("ok1", None), ("bad", None)], fn, timeout=5)
    assert results == {"ok1": "ok"}
    assert errors["bad"] == "target_operation_failed"


def test_fanout_error_label_matches_error_surface_classification():
    """`fanout_error_label` is the single place `fanout_execute` and the
    `graph_query` primary-target mirror both call — direct unit coverage of
    the classification itself, independent of the concurrency machinery."""
    assert (
        fanout_error_label(EngineCircuitOpenError("breaker open")) == "target_degraded"
    )
    assert fanout_error_label(ConnectionError("connect refused")) == "target_degraded"
    assert fanout_error_label(ValueError("bad predicate")) == "target_operation_failed"
    assert (
        fanout_error_label(
            CypherEngineError("Q", "read", EngineCircuitOpenError("open"))
        )
        == "target_degraded"
    )
    assert (
        fanout_error_label(CypherEngineError("Q", "read", ValueError("bad")))
        == "target_operation_failed"
    )
