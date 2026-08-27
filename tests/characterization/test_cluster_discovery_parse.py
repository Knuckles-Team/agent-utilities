"""Characterization tests for ``ClusterTopologyAuthority._parse``
(CX-AU-01, CCN 82 in ``cluster_discovery.py``).

``_parse`` is a single security-critical validation pipeline over a signed
``ClusterMembers`` response: shape/schema checks, cluster-identity binding,
epoch consistency, auth-context matching, a nested groups/members parse with
per-member field + certificate validation, leader consistency, signature
format, and (when a ``prior`` snapshot is supplied) monotonicity checks
against it. The refactor this pins is extracting each of those stages into a
named private method called in sequence — no validation RULE changes, only
where each rule lives. Table-driven: one row per rejection reason, each
constructed to trip exactly that check and no other, which both documents
the pipeline's real behaviour (including which of two similarly-worded
"leader is inconsistent" messages fires for which malformed input) and
serves as its own known-bad proof — a row that failed to raise, or raised
the wrong reason, would fail the test immediately. A handful of the
trickiest invariants (leader-consistency precedence, prior-snapshot
monotonicity, certificate liveness) get dedicated tests instead of table
rows because they need a differently-shaped baseline.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from agent_utilities.knowledge_graph.core import cluster_discovery
from agent_utilities.knowledge_graph.core.cluster_discovery import (
    ClusterDiscoveryRejected,
    ClusterDiscoverySnapshot,
    ClusterDiscoveryStale,
    ClusterTopologyAuthority,
    _digest,
)

NOW_MS = 1_700_000_000_000
VERIFIED_CONTEXT = {"tenant": "t", "principal": "p", "agent_id": "a"}


def _authority(**kwargs: Any) -> ClusterTopologyAuthority:
    return ClusterTopologyAuthority(
        wall_clock_ms=lambda: NOW_MS, monotonic=lambda: 1_000.0, **kwargs
    )


def _member(
    node_id: int,
    role: str,
    *,
    endpoint: str = "tcp://127.0.0.1:9000",
    tls_name: str | None = None,
    health: str = "healthy",
    cert_id: str | None = None,
    rotation_epoch: int = 0,
    not_before_ms: int | None = None,
    not_after_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "member_identity": _digest(f"member-{node_id}"),
        "role": role,
        "client_endpoint": endpoint,
        "tls_name": tls_name,
        "health": health,
        "certificate": {
            "id": cert_id,
            "rotation_epoch": rotation_epoch,
            "not_before_ms": NOW_MS - 1_000 if not_before_ms is None else not_before_ms,
            "not_after_ms": (
                NOW_MS + 1_000_000_000 if not_after_ms is None else not_after_ms
            ),
        },
    }


def _valid_answer() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cluster_id": _digest("cluster-1"),
        "epoch": 5,
        "membership_epoch": 5,
        "placement_epoch": 3,
        "leader": {"group_id": 0, "node_id": 1},
        "leaders": [{"group_id": 0, "node_id": 1}],
        "groups": [
            {
                "group_id": 0,
                "leader_id": 1,
                "members": [_member(1, "leader"), _member(2, "follower")],
            }
        ],
        "auth_binding": {
            "tenant_digest": _digest("t"),
            "principal_digest": _digest("p"),
            "agent_digest": _digest("a"),
        },
        "signature": "hmac-sha256:" + "0" * 64,
    }


def _parse(answer: Any, **overrides: Any) -> ClusterDiscoverySnapshot:
    kwargs: dict[str, Any] = dict(
        verified_context=VERIFIED_CONTEXT,
        client_context=None,
        expected_cluster_id=None,
        min_membership_epoch=None,
        min_placement_epoch=None,
        prior=None,
    )
    kwargs.update(overrides)
    return _authority()._parse(answer, **kwargs)


def test_golden_path_produces_a_verified_snapshot():
    snapshot = _parse(_valid_answer())
    assert snapshot.cluster_id == _digest("cluster-1")
    assert snapshot.membership_epoch == 5
    assert snapshot.placement_epoch == 3
    assert snapshot.tenant_digest == _digest("t")
    assert len(snapshot.groups) == 1
    group_id, members, leader_id = snapshot.groups[0]
    assert group_id == 0
    assert leader_id == 1
    assert {m.node_id for m in members} == {1, 2}
    leader = next(m for m in members if m.node_id == 1)
    assert leader.role == "leader"


# ---------------------------------------------------------------------------
# Table-driven rejection paths: (mutator, expected substring in the message)
# ---------------------------------------------------------------------------


def _mut_not_a_mapping(a):
    return ["not", "a", "mapping"]


def _mut_missing_field(a):
    del a["epoch"]
    return a


def _mut_extra_field(a):
    a["bogus"] = 1
    return a


def _mut_bad_schema_version(a):
    a["schema_version"] = 2
    return a


def _mut_schema_version_is_bool(a):
    a["schema_version"] = True
    return a


def _mut_cluster_id_not_digest(a):
    a["cluster_id"] = "not-a-digest"
    return a


def _mut_epoch_alias_mismatch(a):
    a["epoch"] = 6
    return a


def _mut_epoch_negative(a):
    a["epoch"] = -1
    a["membership_epoch"] = -1
    return a


def _mut_binding_wrong_keys(a):
    a["auth_binding"] = {"tenant_digest": _digest("t")}
    return a


def _mut_binding_not_digest(a):
    a["auth_binding"]["tenant_digest"] = "nope"
    return a


def _mut_binding_mismatch(a):
    a["auth_binding"]["tenant_digest"] = _digest("someone-else")
    return a


def _mut_groups_not_list(a):
    a["groups"] = {}
    return a


def _mut_too_many_groups(a):
    a["groups"] = a["groups"] * (cluster_discovery._MAX_GROUPS + 1)
    return a


def _mut_group_wrong_keys(a):
    a["groups"][0] = {"group_id": 0}
    return a


def _mut_duplicate_groups(a):
    a["groups"].append(copy.deepcopy(a["groups"][0]))
    return a


def _mut_members_not_list(a):
    a["groups"][0]["members"] = {}
    return a


def _mut_member_wrong_keys(a):
    a["groups"][0]["members"][0] = {"node_id": 1}
    return a


def _mut_duplicate_members(a):
    a["groups"][0]["members"].append(copy.deepcopy(a["groups"][0]["members"][0]))
    return a


def _mut_member_identity_not_digest(a):
    a["groups"][0]["members"][0]["member_identity"] = "nope"
    return a


def _mut_role_invalid(a):
    a["groups"][0]["members"][0]["role"] = "supervisor"
    return a


def _mut_endpoint_bad_scheme(a):
    a["groups"][0]["members"][0]["client_endpoint"] = "http://127.0.0.1:9000"
    return a


def _mut_tls_endpoint_no_name(a):
    a["groups"][0]["members"][0]["client_endpoint"] = "tls://example.internal:9000"
    a["groups"][0]["members"][0]["tls_name"] = None
    return a


def _mut_health_invalid(a):
    a["groups"][0]["members"][0]["health"] = "on-fire"
    return a


def _mut_certificate_wrong_keys(a):
    a["groups"][0]["members"][0]["certificate"] = {"id": None}
    return a


def _mut_certificate_id_too_large(a):
    a["groups"][0]["members"][0]["certificate"]["id"] = "x" * 600
    return a


def _mut_rotation_without_id(a):
    a["groups"][0]["members"][0]["certificate"]["rotation_epoch"] = 1
    a["groups"][0]["members"][0]["certificate"]["id"] = None
    return a


def _mut_certificate_validity_inverted(a):
    a["groups"][0]["members"][0]["certificate"]["not_before_ms"] = NOW_MS + 10
    a["groups"][0]["members"][0]["certificate"]["not_after_ms"] = NOW_MS - 10
    return a


def _mut_leaders_inconsistent(a):
    a["leaders"] = [{"group_id": 0, "node_id": 2}]
    return a


def _mut_signature_bad_prefix(a):
    a["signature"] = "hmac-sha1:" + "0" * 64
    return a


def _mut_signature_bad_length(a):
    a["signature"] = "hmac-sha256:" + "0" * 10
    return a


def _mut_signature_non_hex(a):
    a["signature"] = "hmac-sha256:" + ("z" * 64)
    return a


REJECTION_TABLE: list[tuple[Any, str]] = [
    (_mut_not_a_mapping, "not a mapping"),
    (_mut_missing_field, "unexpected fields"),
    (_mut_extra_field, "unexpected fields"),
    (_mut_bad_schema_version, "schema version is unsupported"),
    (_mut_schema_version_is_bool, "schema version is unsupported"),
    (_mut_cluster_id_not_digest, "cluster identity is malformed"),
    (_mut_epoch_alias_mismatch, "epoch alias does not match"),
    (_mut_epoch_negative, "epoch"),
    (_mut_binding_wrong_keys, "auth binding is malformed"),
    (_mut_binding_not_digest, "auth binding is malformed"),
    (_mut_binding_mismatch, "request context does not match"),
    (_mut_groups_not_list, "groups exceed bounds"),
    (_mut_too_many_groups, "groups exceed bounds"),
    (_mut_group_wrong_keys, "group entry is malformed"),
    (_mut_duplicate_groups, "duplicate groups"),
    (_mut_members_not_list, "members is not a list"),
    (_mut_member_wrong_keys, "member entry is malformed"),
    (_mut_duplicate_members, "duplicate members"),
    (_mut_member_identity_not_digest, "member identity is malformed"),
    (_mut_role_invalid, "member role is invalid"),
    (_mut_endpoint_bad_scheme, "must use tcp:// or tls://"),
    (_mut_tls_endpoint_no_name, "has no server name"),
    (_mut_health_invalid, "member health is invalid"),
    (_mut_certificate_wrong_keys, "certificate metadata is malformed"),
    (_mut_certificate_id_too_large, "certificate id is too large"),
    (_mut_rotation_without_id, "certificate rotation requires an id"),
    (_mut_certificate_validity_inverted, "certificate validity is inverted"),
    (_mut_leaders_inconsistent, "leaders are inconsistent"),
    (_mut_signature_bad_prefix, "signature is malformed"),
    (_mut_signature_bad_length, "signature is malformed"),
    (_mut_signature_non_hex, "signature is malformed"),
]


@pytest.mark.parametrize(
    "mutator,expected_substring",
    REJECTION_TABLE,
    ids=[m.__name__ for m, _ in REJECTION_TABLE],
)
def test_malformed_response_is_rejected(mutator, expected_substring):
    answer = mutator(_valid_answer())
    with pytest.raises(ClusterDiscoveryRejected, match=expected_substring):
        _parse(answer)


# ---------------------------------------------------------------------------
# Dedicated tests: invariants that need a differently-shaped baseline
# ---------------------------------------------------------------------------


def test_group_level_leader_inconsistency_is_rejected():
    """``leader_id`` points at a node that exists but is not role=leader."""
    answer = _valid_answer()
    answer["groups"][0]["members"][1]["role"] = "leader"  # both node 1 and 2 "leader"
    # node_id 1 is still leader_id; make it a follower so leader_id no longer
    # resolves to a member with role == "leader".
    answer["groups"][0]["members"][0]["role"] = "follower"
    with pytest.raises(ClusterDiscoveryRejected, match="leader is inconsistent"):
        _parse(answer)


def test_top_level_leader_field_inconsistency_is_rejected():
    """``leaders`` (per-group) is consistent, but the top-level ``leader``
    convenience field disagrees with it."""
    answer = _valid_answer()
    answer["leader"] = {"group_id": 0, "node_id": 999}
    with pytest.raises(ClusterDiscoveryRejected, match="leader is inconsistent"):
        _parse(answer)


def test_no_verified_context_is_rejected():
    with pytest.raises(ClusterDiscoveryRejected, match="no verified context"):
        _parse(_valid_answer(), verified_context=None, client_context=None)


def test_incomplete_verified_context_is_rejected():
    with pytest.raises(ClusterDiscoveryRejected, match="context is incomplete"):
        _parse(
            _valid_answer(),
            verified_context={"tenant": "t", "principal": "", "agent_id": "a"},
        )


def test_expected_cluster_id_mismatch_is_rejected():
    with pytest.raises(
        ClusterDiscoveryRejected, match="belongs to a different cluster"
    ):
        _parse(_valid_answer(), expected_cluster_id=_digest("other-cluster"))


def test_malformed_expected_cluster_id_is_rejected():
    with pytest.raises(
        ClusterDiscoveryRejected, match="expected cluster identity is malformed"
    ):
        _parse(_valid_answer(), expected_cluster_id="not-a-digest")


def test_stale_membership_epoch_is_rejected():
    with pytest.raises(ClusterDiscoveryRejected, match="membership snapshot is stale"):
        _parse(_valid_answer(), min_membership_epoch=6)


def test_stale_placement_epoch_is_rejected():
    with pytest.raises(ClusterDiscoveryRejected, match="placement snapshot is stale"):
        _parse(_valid_answer(), min_placement_epoch=4)


def test_members_exceeding_the_bound_are_rejected(monkeypatch):
    monkeypatch.setattr(cluster_discovery, "_MAX_MEMBERS", 1)
    with pytest.raises(ClusterDiscoveryRejected, match="members exceed bounds"):
        _parse(_valid_answer())  # 2 members > cap of 1


def test_certificate_not_yet_valid_is_stale():
    answer = _valid_answer()
    answer["groups"][0]["members"][0]["certificate"]["not_before_ms"] = (
        NOW_MS + 1_000_000
    )
    with pytest.raises(ClusterDiscoveryStale, match="not valid yet"):
        _parse(answer)


def test_certificate_expired_is_stale():
    answer = _valid_answer()
    answer["groups"][0]["members"][0]["certificate"]["not_before_ms"] = (
        NOW_MS - 2_000_000
    )
    answer["groups"][0]["members"][0]["certificate"]["not_after_ms"] = (
        NOW_MS - 1_000_000
    )
    with pytest.raises(ClusterDiscoveryStale, match="has expired"):
        _parse(answer)


def _snapshot_from(answer: dict[str, Any]) -> ClusterDiscoverySnapshot:
    return _parse(answer)


def test_prior_cluster_identity_change_is_rejected():
    prior = _snapshot_from(_valid_answer())
    answer = _valid_answer()
    answer["cluster_id"] = _digest("a-different-cluster")
    # A differently-clustered response must still parse as internally valid
    # up to the identity check itself; only bump the digest, keep everything
    # else self-consistent.
    with pytest.raises(ClusterDiscoveryRejected, match="cluster identity changed"):
        _parse(answer, prior=prior)


def test_prior_epoch_moving_backwards_is_rejected():
    prior = _snapshot_from(_valid_answer())
    answer = _valid_answer()
    answer["epoch"] = 4
    answer["membership_epoch"] = 4
    with pytest.raises(ClusterDiscoveryRejected, match="epoch moved backwards"):
        _parse(answer, prior=prior)


def test_prior_certificate_rotation_epoch_moving_backwards_is_rejected():
    first = _valid_answer()
    first["groups"][0]["members"][0]["certificate"]["rotation_epoch"] = 2
    first["groups"][0]["members"][0]["certificate"]["id"] = "cert-a"
    prior = _snapshot_from(first)

    second = _valid_answer()
    second["epoch"] = 6
    second["membership_epoch"] = 6
    second["groups"][0]["members"][0]["certificate"]["rotation_epoch"] = 1
    second["groups"][0]["members"][0]["certificate"]["id"] = "cert-a"
    with pytest.raises(
        ClusterDiscoveryRejected, match="certificate epoch moved backwards"
    ):
        _parse(second, prior=prior)


def test_prior_certificate_id_change_without_rotation_is_rejected():
    first = _valid_answer()
    first["groups"][0]["members"][0]["certificate"]["rotation_epoch"] = 2
    first["groups"][0]["members"][0]["certificate"]["id"] = "cert-a"
    prior = _snapshot_from(first)

    second = _valid_answer()
    second["epoch"] = 6
    second["membership_epoch"] = 6
    second["groups"][0]["members"][0]["certificate"]["rotation_epoch"] = 2  # same
    second["groups"][0]["members"][0]["certificate"]["id"] = "cert-b"  # changed!
    with pytest.raises(
        ClusterDiscoveryRejected,
        match="certificate changed without a rotation epoch",
    ):
        _parse(second, prior=prior)


def test_prior_snapshot_with_consistent_rotation_is_accepted():
    """The three prior-snapshot checks above must not fire on a legitimate,
    monotonically-advancing response — this is the guard against an
    over-eager rewrite of the prior-snapshot block rejecting everything."""
    first = _valid_answer()
    first["groups"][0]["members"][0]["certificate"]["rotation_epoch"] = 2
    first["groups"][0]["members"][0]["certificate"]["id"] = "cert-a"
    prior = _snapshot_from(first)

    second = _valid_answer()
    second["epoch"] = 6
    second["membership_epoch"] = 6
    second["groups"][0]["members"][0]["certificate"]["rotation_epoch"] = 3  # advanced
    second["groups"][0]["members"][0]["certificate"]["id"] = "cert-b"
    snapshot = _parse(second, prior=prior)
    assert snapshot.membership_epoch == 6
