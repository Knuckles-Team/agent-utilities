"""GraphComputeEngine.audit_verify (G23, audit-trail closure).

CONCEPT:AU-KG.audit.hash-chain-verify

Mirrors ``test_graph_compute_rdf_ops.py``'s pattern: construct a bare
``GraphComputeEngine`` without running ``__init__`` and stub ``_send_wire`` so
no real engine connection is needed.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def _engine_with_send_wire(fn):
    eng = GraphComputeEngine.__new__(GraphComputeEngine)
    eng._send_wire = fn  # type: ignore[assignment]
    return eng


def test_audit_verify_returns_clean_report():
    report = {
        "graph": "__commons__",
        "ok": True,
        "entries": 42,
        "first_broken_seq": None,
        "detail": "chain verified",
    }
    eng = _engine_with_send_wire(lambda method, payload=None: dict(report))
    out = eng.audit_verify()
    assert out == report


def test_audit_verify_calls_the_raw_audit_verify_wire_method():
    seen = {}

    def fake_send_wire(method, payload=None):
        seen["method"] = method
        seen["payload"] = payload
        return {"graph": "g", "ok": True, "entries": 0, "first_broken_seq": None, "detail": ""}

    eng = _engine_with_send_wire(fake_send_wire)
    eng.audit_verify()
    assert seen["method"] == "AuditVerify"
    assert seen["payload"] is None


def test_audit_verify_surfaces_tampered_chain():
    tampered = {
        "graph": "g",
        "ok": False,
        "entries": 3,
        "first_broken_seq": 3,
        "detail": "hash-chain break at seq 3 (entry mutated or chain altered)",
    }
    eng = _engine_with_send_wire(lambda method, payload=None: dict(tampered))
    out = eng.audit_verify()
    assert out["ok"] is False
    assert out["first_broken_seq"] == 3


def test_audit_verify_raises_when_engine_build_lacks_support():
    """No 'security' feature / no durable redb dir ⇒ raises, so callers degrade
    cleanly (see agent_utilities.mcp.tools.audit_tools._verify) instead of
    silently trusting an unverified log."""

    def fake_send_wire(method, payload=None):
        raise RuntimeError(
            "AuditVerify requires a durable redb backend (no persist dir configured)"
        )

    eng = _engine_with_send_wire(fake_send_wire)
    with pytest.raises(RuntimeError, match="AuditVerify"):
        eng.audit_verify()
