"""EH-410: security-audit feeds enter the graph pseudonymized (operator ruling)."""

from __future__ import annotations

import pytest

from agent_utilities.security.audit_pseudonym import (
    AuditFieldPolicy,
    AuditPseudonymizer,
    AuditPseudonymUnavailable,
    ip_network,
)

KEY = AuditPseudonymizer(b"test-only-audit-key")
POLICY = AuditFieldPolicy(
    keep=("type", "time", "details"),
    identities=("userId", "auth.entity_id"),
    secret_paths=("request.path",),
    ips=("ipAddress",),
    whole_ips=("value",),
)
EVENT = {
    "type": "LOGIN",
    "time": 1700000000000,
    "userId": "8c2d0c1e-alice",
    "ipAddress": "203.0.113.77",
    "auth": {"entity_id": "entity-42", "client_token": "hmac-sha256:abc"},
    "request": {"path": "/secret/data/apps/graph-os", "data": {"password": "s3cret"}},
    "details": {"username": "alice"},
    "value": "198.51.100.9",
}


def test_identities_and_paths_are_keyed_and_ips_truncated() -> None:
    out = KEY.record(EVENT, POLICY)
    assert out["type"] == "LOGIN" and out["time"] == 1700000000000
    assert out["userId"].startswith("apseud_identity_")
    assert out["auth_entity_id"] == KEY.identity("entity-42")
    assert out["request_path"] == KEY.secret_path("secret/data/apps/graph-os")
    assert out["ipAddress"] == "203.0.113.0/24"
    assert out["value"] == "198.51.100.9", "a ban target is kept whole"
    rendered = repr(out)
    for leaked in ("alice", "s3cret", "entity-42", "client_token", "graph-os", ".77"):
        assert leaked not in rendered, leaked
    assert "details" not in out, "nested bodies never pass the allowlist"


def test_pseudonyms_are_stable_per_key_and_differ_across_keys() -> None:
    other = AuditPseudonymizer(b"another-deployment")
    assert KEY.identity("alice") == KEY.identity("alice")
    assert KEY.identity("alice") != other.identity("alice")
    assert KEY.identity("alice") != KEY.secret_path("alice")
    assert KEY.identity("") == ""


def test_ip_truncation_covers_v4_v6_and_garbage() -> None:
    assert ip_network("10.1.2.3") == "10.1.2.0/24"
    assert ip_network("2001:db8:abcd:12::1") == "2001:db8:abcd::/48"
    assert ip_network("not-an-ip") == ""


def test_no_key_refuses_the_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUDIT_PSEUDONYM_HMAC_KEY_REF", raising=False)
    with pytest.raises(AuditPseudonymUnavailable):
        AuditPseudonymizer.from_settings()
