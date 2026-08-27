"""Focused consumer fixtures for verified ClusterMembers/session bounds.

These fixtures deliberately stop at the AU consumer seam.  They do not start
an engine, exercise transport sockets, or stand in for the epistemic-graph
client's signature verification; the real client is responsible for that
cryptographic check before AU consumes a response.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.cluster_discovery import (
    ClusterDiscoveryError,
    ClusterDiscoveryRejected,
    ClusterDiscoveryStale,
    ClusterTopologyAuthority,
)
from agent_utilities.knowledge_graph.core.transport_lifecycle import (
    EngineDrainingError,
    TransportDrainGate,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _context(tenant: str = "tenant-a") -> dict[str, str]:
    return {"tenant": tenant, "principal": "principal-a", "agent_id": "agent-a"}


def _answer(
    context: dict[str, str],
    *,
    endpoint: str = "tls://graph-a.example:9443",
    membership_epoch: int = 4,
    placement_epoch: int = 7,
    certificate_rotation_epoch: int = 2,
    certificate_id: str | None = "cert-a",
    not_after_ms: int | None = None,
) -> dict[str, Any]:
    now_ms = 1_700_000_000_000
    not_after_ms = not_after_ms if not_after_ms is not None else now_ms + 300_000
    member = {
        "node_id": 1,
        "member_identity": _digest("member-a"),
        "role": "leader",
        "client_endpoint": endpoint,
        "tls_name": "graph-a.example" if endpoint.startswith("tls://") else None,
        "health": "healthy",
        "certificate": {
            "id": certificate_id,
            "rotation_epoch": certificate_rotation_epoch,
            "not_before_ms": now_ms - 300_000,
            "not_after_ms": not_after_ms,
        },
    }
    return {
        "schema_version": 1,
        "cluster_id": _digest("cluster-a"),
        "epoch": membership_epoch,
        "membership_epoch": membership_epoch,
        "placement_epoch": placement_epoch,
        "leader": {"group_id": 1, "node_id": 1},
        "leaders": [{"group_id": 1, "node_id": 1}],
        "groups": [{"group_id": 1, "leader_id": 1, "members": [member]}],
        "auth_binding": {
            "tenant_digest": _digest(context["tenant"]),
            "principal_digest": _digest(context["principal"]),
            "agent_digest": _digest(context["agent_id"]),
        },
        "signature": "hmac-sha256:" + "a" * 64,
    }


class _Topology:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls = 0

    def members(self, **_: Any) -> Any:
        self.calls += 1
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


class _Client:
    def __init__(self, answer: Any, context: dict[str, str]) -> None:
        self.cluster_topology = _Topology(answer)
        self._context = context

    def _effective_verified_context(self) -> dict[str, str]:
        return self._context


def test_verified_snapshot_selects_leader_and_is_cached_within_bound() -> None:
    monotonic = iter([10.0, 10.5, 10.5])
    context = _context()
    client = _Client(_answer(context), context)
    authority = ClusterTopologyAuthority(
        max_age_s=5.0,
        clock_skew_s=1.0,
        monotonic=lambda: next(monotonic),
        wall_clock_ms=lambda: 1_700_000_000_000,
    )

    first = authority.read(client, verified_context=context)
    second = authority.read(client, verified_context=context)

    assert first.endpoint_for_group(1).client_endpoint == "tls://graph-a.example:9443"
    assert second is first
    assert client.cluster_topology.calls == 1


def test_wrong_context_is_rejected_without_last_good_fallback() -> None:
    context = _context()
    client = _Client(_answer(context), context)
    authority = ClusterTopologyAuthority(
        max_age_s=30.0,
        monotonic=lambda: 10.0,
        wall_clock_ms=lambda: 1_700_000_000_000,
    )
    authority.read(client, verified_context=context)
    client.cluster_topology.answer = _answer(_context("tenant-b"))

    with pytest.raises(ClusterDiscoveryRejected):
        authority.read(client, verified_context=context, force_refresh=True)


def test_transport_failure_uses_only_current_last_good_snapshot() -> None:
    context = _context()
    client = _Client(_answer(context), context)
    now = [10.0]
    authority = ClusterTopologyAuthority(
        max_age_s=5.0,
        monotonic=lambda: now[0],
        wall_clock_ms=lambda: 1_700_000_000_000,
    )
    first = authority.read(client, verified_context=context)
    client.cluster_topology.answer = ConnectionError("leader unavailable")
    retry = authority.read(client, verified_context=context, force_refresh=True)
    assert retry is first

    now[0] = 16.0
    with pytest.raises(ClusterDiscoveryError):
        authority.read(client, verified_context=context, force_refresh=True)


def test_remote_plaintext_member_and_expired_certificate_fail_closed() -> None:
    context = _context()
    authority = ClusterTopologyAuthority(
        monotonic=lambda: 10.0,
        wall_clock_ms=lambda: 1_700_000_000_000,
    )
    plaintext = _Client(
        _answer(context, endpoint="tcp://graph-a.example:9443"), context
    )
    with pytest.raises(ClusterDiscoveryRejected):
        authority.read(plaintext, verified_context=context)

    expired = _Client(_answer(context, not_after_ms=1_699_999_999_000), context)
    with pytest.raises(ClusterDiscoveryStale):
        ClusterTopologyAuthority(
            monotonic=lambda: 10.0,
            wall_clock_ms=lambda: 1_700_000_000_000,
        ).read(expired, verified_context=context)


def test_certificate_identity_change_requires_a_new_rotation_epoch() -> None:
    context = _context()
    client = _Client(
        _answer(context, certificate_rotation_epoch=2, certificate_id="cert-a"),
        context,
    )
    authority = ClusterTopologyAuthority(
        monotonic=lambda: 10.0,
        wall_clock_ms=lambda: 1_700_000_000_000,
    )
    authority.read(client, verified_context=context)

    client.cluster_topology.answer = _answer(
        context,
        certificate_rotation_epoch=2,
        certificate_id="cert-b",
    )
    with pytest.raises(ClusterDiscoveryRejected):
        authority.read(client, verified_context=context, force_refresh=True)


def test_transport_drain_stops_admission_and_reports_timeout_explicitly() -> None:
    gate = TransportDrainGate()
    with gate.admit():
        status = gate.begin(0.0)
        assert status.state == "draining"
        assert status.timed_out is True
        assert status.active_requests == 1
        with pytest.raises(EngineDrainingError):
            with gate.admit():
                pass
    assert gate.status().active_requests == 0


def _parse_direct(authority: ClusterTopologyAuthority, answer: Any, *, context: dict[str, str], prior: Any = None) -> Any:
    return authority._parse(
        answer,
        verified_context=context,
        client_context=None,
        expected_cluster_id=None,
        min_membership_epoch=None,
        min_placement_epoch=None,
        prior=prior,
    )


def test_parse_rejects_malformed_top_level_shape() -> None:
    """CXA-AU-03-04 characterization: envelope-section rejection (missing field)."""
    authority = ClusterTopologyAuthority(
        monotonic=lambda: 10.0, wall_clock_ms=lambda: 1_700_000_000_000
    )
    context = _context()
    answer = _answer(context)
    del answer["signature"]
    with pytest.raises(ClusterDiscoveryRejected):
        _parse_direct(authority, answer, context=context)


def test_parse_rejects_binding_mismatch() -> None:
    """CXA-AU-03-04 characterization: binding-section rejection (digest mismatch)."""
    authority = ClusterTopologyAuthority(
        monotonic=lambda: 10.0, wall_clock_ms=lambda: 1_700_000_000_000
    )
    context = _context()
    answer = _answer(context)
    answer["auth_binding"]["tenant_digest"] = _digest("someone-else")
    with pytest.raises(ClusterDiscoveryRejected):
        _parse_direct(authority, answer, context=context)


def test_parse_rejects_group_with_unknown_leader() -> None:
    """CXA-AU-03-04 characterization: groups-section rejection (leader not a member)."""
    authority = ClusterTopologyAuthority(
        monotonic=lambda: 10.0, wall_clock_ms=lambda: 1_700_000_000_000
    )
    context = _context()
    answer = _answer(context)
    answer["groups"][0]["leader_id"] = 99
    with pytest.raises(ClusterDiscoveryRejected):
        _parse_direct(authority, answer, context=context)


def test_parse_rejects_leaders_list_inconsistent_with_groups() -> None:
    """CXA-AU-03-04 characterization: leaders-section rejection (top-level/group mismatch)."""
    authority = ClusterTopologyAuthority(
        monotonic=lambda: 10.0, wall_clock_ms=lambda: 1_700_000_000_000
    )
    context = _context()
    answer = _answer(context)
    answer["leaders"] = []
    with pytest.raises(ClusterDiscoveryRejected):
        _parse_direct(authority, answer, context=context)


def test_parse_rejects_malformed_signature() -> None:
    """CXA-AU-03-04 characterization: signature-section rejection (bad shape)."""
    authority = ClusterTopologyAuthority(
        monotonic=lambda: 10.0, wall_clock_ms=lambda: 1_700_000_000_000
    )
    context = _context()
    answer = _answer(context)
    answer["signature"] = "not-a-real-signature"
    with pytest.raises(ClusterDiscoveryRejected):
        _parse_direct(authority, answer, context=context)


def test_parse_rejects_epoch_moving_backwards_against_prior() -> None:
    """CXA-AU-03-04 characterization: prior-section rejection (epoch regression)."""
    authority = ClusterTopologyAuthority(
        monotonic=lambda: 10.0, wall_clock_ms=lambda: 1_700_000_000_000
    )
    context = _context()
    prior_answer = _answer(context, membership_epoch=4, placement_epoch=7)
    prior = _parse_direct(authority, prior_answer, context=context)
    regressed = _answer(context, membership_epoch=3, placement_epoch=7)
    with pytest.raises(ClusterDiscoveryRejected):
        _parse_direct(authority, regressed, context=context, prior=prior)


def test_parse_accepts_well_formed_answer() -> None:
    """CXA-AU-03-04 characterization: full-pipeline positive path returns a snapshot."""
    authority = ClusterTopologyAuthority(
        monotonic=lambda: 10.0, wall_clock_ms=lambda: 1_700_000_000_000
    )
    context = _context()
    answer = _answer(context)
    snapshot = _parse_direct(authority, answer, context=context)
    assert snapshot.cluster_id == answer["cluster_id"]
    assert snapshot.membership_epoch == 4
    assert len(snapshot.groups) == 1
