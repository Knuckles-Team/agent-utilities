# CONCEPT:AU-KG.sharding.cluster-members-discovery-session — Verified
# engine-authoritative topology consumption and bounded route continuity.
"""Verified epistemic-graph cluster-member discovery for AU consumers.

The native engine is the sole authority for the address of a placed Raft
group.  ``GRAPH_SERVICE_ENDPOINTS`` is only a bounded bootstrap contact list;
``GRAPH_RAFT_GROUP_ENDPOINTS`` is deliberately *not* consulted here.  The
epistemic-graph client verifies the signed ``ClusterMembers`` response before
this module sees it.  This module adds the consumer-side bounds that are
specific to a long-lived AU process: context binding, certificate freshness,
last-good expiry, monotonic epochs, and deterministic leader-first selection.

The cache is intentionally process-local and bounded by ``max_age_s``.  It is
not a second membership authority and it is never serialized as durable
session state.  A restart therefore requires a fresh authenticated snapshot;
an AU process may not claim continuity merely because an old endpoint was
present in configuration.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

__all__ = [
    "ClusterDiscoveryError",
    "ClusterDiscoveryRejected",
    "ClusterDiscoveryStale",
    "ClusterDiscoverySnapshot",
    "ClusterMember",
    "ClusterTopologyAuthority",
]

_SCHEMA_VERSION = 1
_MAX_GROUPS = 1_024
_MAX_MEMBERS = 4_096
_MAX_FIELD_BYTES = 4 * 1_024
_MAX_CERTIFICATE_ID_BYTES = 512
_MAX_U64 = (1 << 64) - 1
_DEFAULT_MAX_AGE_S = 30.0
_MAX_MAX_AGE_S = 300.0
_DEFAULT_CLOCK_SKEW_S = 5.0
_DIGEST_PREFIX = "sha256:"
_DIGEST_LENGTH = len(_DIGEST_PREFIX) + 64


class ClusterDiscoveryError(RuntimeError):
    """Base error for a missing or unusable verified topology snapshot."""


class ClusterDiscoveryRejected(ClusterDiscoveryError):
    """A response was malformed, cross-context, stale, or otherwise unsafe."""


class ClusterDiscoveryStale(ClusterDiscoveryError):
    """The last verified snapshot is older than the configured continuity bound."""


def _digest(value: str) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _DIGEST_LENGTH
        and value.startswith(_DIGEST_PREFIX)
        and all(
            character in "0123456789abcdefABCDEF"
            for character in value[len(_DIGEST_PREFIX) :]
        )
    )


def _bounded_text(value: Any, *, field: str, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > _MAX_FIELD_BYTES
    ):
        raise ClusterDiscoveryRejected(f"ClusterMembers.{field} is malformed")
    if any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in value
    ):
        raise ClusterDiscoveryRejected(
            f"ClusterMembers.{field} contains unsafe characters"
        )
    return value


def _non_negative_int(value: Any, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_U64
    ):
        raise ClusterDiscoveryRejected(f"ClusterMembers.{field} is malformed")
    return value


def _endpoint_is_bounded(endpoint: Any) -> str:
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or len(endpoint.encode("utf-8")) > _MAX_FIELD_BYTES
    ):
        raise ClusterDiscoveryRejected("ClusterMembers.client_endpoint is malformed")
    if not (endpoint.startswith("tcp://") or endpoint.startswith("tls://")):
        raise ClusterDiscoveryRejected(
            "ClusterMembers.client_endpoint must use tcp:// or tls://"
        )
    if any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in endpoint
    ):
        raise ClusterDiscoveryRejected(
            "ClusterMembers.client_endpoint contains unsafe characters"
        )
    parsed = urlsplit(endpoint)
    host = parsed.hostname
    if (
        not host
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ClusterDiscoveryRejected(
            "ClusterMembers.client_endpoint contains an authority escape"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ClusterDiscoveryRejected(
            "ClusterMembers.client_endpoint has an invalid port"
        ) from exc
    if port is None or not 1 <= port <= 65_535:
        raise ClusterDiscoveryRejected(
            "ClusterMembers.client_endpoint has an invalid port"
        )
    # AU's native transport permits plaintext TCP only for loopback.  Keeping
    # that rule here prevents a discovery response from becoming an insecure
    # remote fallback before the transport layer gets a chance to reject it.
    if endpoint.startswith("tcp://"):
        try:
            loopback = ipaddress.ip_address(host).is_loopback or host.casefold() in {
                "localhost",
                "localhost.",
            }
        except ValueError:
            loopback = False
        if not loopback:
            raise ClusterDiscoveryRejected("remote discovered endpoints require tls://")
    return endpoint


@dataclass(frozen=True)
class ClusterMember:
    """One engine-authoritative member usable for client traffic."""

    group_id: int
    node_id: int
    member_identity: str
    role: str
    client_endpoint: str
    tls_name: str | None
    health: str
    certificate_id: str | None
    certificate_rotation_epoch: int
    certificate_not_before_ms: int | None
    certificate_not_after_ms: int | None

    @property
    def certificate_key(self) -> tuple[str | None, int]:
        return self.certificate_id, self.certificate_rotation_epoch

    def assert_certificate_current(
        self,
        *,
        now_ms: int,
        clock_skew_s: float,
    ) -> None:
        skew_ms = int(max(0.0, clock_skew_s) * 1_000)
        if (
            self.certificate_not_before_ms is not None
            and self.certificate_not_before_ms > now_ms + skew_ms
        ):
            raise ClusterDiscoveryStale(
                f"certificate for member {self.node_id} is not valid yet"
            )
        if (
            self.certificate_not_after_ms is not None
            and self.certificate_not_after_ms <= now_ms - skew_ms
        ):
            raise ClusterDiscoveryStale(
                f"certificate for member {self.node_id} has expired"
            )
        if self.client_endpoint.startswith("tls://") and not self.tls_name:
            raise ClusterDiscoveryRejected(
                f"TLS member {self.node_id} has no verified server name"
            )


@dataclass(frozen=True)
class ClusterDiscoverySnapshot:
    """A verified, bounded topology snapshot with an explicit local expiry."""

    cluster_id: str
    membership_epoch: int
    placement_epoch: int
    observed_monotonic: float
    observed_wall_ms: int
    tenant_digest: str
    principal_digest: str
    agent_digest: str
    groups: tuple[tuple[int, tuple[ClusterMember, ...], int | None], ...]
    max_age_s: float
    clock_skew_s: float

    @property
    def expires_at_monotonic(self) -> float:
        return self.observed_monotonic + self.max_age_s

    @property
    def certificate_epoch(self) -> int:
        return max(
            (
                member.certificate_rotation_epoch
                for _, members, _ in self.groups
                for member in members
            ),
            default=0,
        )

    def assert_current(
        self,
        *,
        now_monotonic: float | None = None,
        now_wall_ms: int | None = None,
        min_membership_epoch: int | None = None,
        min_placement_epoch: int | None = None,
    ) -> None:
        current_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        if current_monotonic >= self.expires_at_monotonic:
            raise ClusterDiscoveryStale(
                "ClusterMembers snapshot exceeded its freshness bound"
            )
        if (
            min_membership_epoch is not None
            and self.membership_epoch < min_membership_epoch
        ):
            raise ClusterDiscoveryStale("ClusterMembers membership epoch is stale")
        if (
            min_placement_epoch is not None
            and self.placement_epoch < min_placement_epoch
        ):
            raise ClusterDiscoveryStale("ClusterMembers placement epoch is stale")
        wall_ms = int(time.time() * 1_000) if now_wall_ms is None else int(now_wall_ms)
        for _, members, _ in self.groups:
            for member in members:
                member.assert_certificate_current(
                    now_ms=wall_ms,
                    clock_skew_s=self.clock_skew_s,
                )

    def endpoint_for_group(self, group_id: int) -> ClusterMember:
        """Return the healthy leader, then a healthy follower, for ``group_id``."""
        self.assert_current()
        group = next((entry for entry in self.groups if entry[0] == group_id), None)
        if group is None:
            raise ClusterDiscoveryRejected(f"ClusterMembers has no group {group_id}")
        _, members, leader_id = group
        by_id = {member.node_id: member for member in members}
        ordered: list[ClusterMember] = []
        if leader_id is not None and leader_id in by_id:
            ordered.append(by_id[leader_id])
        ordered.extend(
            member
            for member in members
            if member.node_id != leader_id and member.role == "follower"
        )
        for member in ordered:
            if member.health == "healthy":
                return member
        raise ClusterDiscoveryStale(
            f"group {group_id} has no healthy discovered member"
        )


class ClusterTopologyAuthority:
    """Bounded consumer of the engine client's already-verified topology RPC."""

    def __init__(
        self,
        *,
        max_age_s: float = _DEFAULT_MAX_AGE_S,
        clock_skew_s: float = _DEFAULT_CLOCK_SKEW_S,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock_ms: Callable[[], int] | None = None,
    ) -> None:
        try:
            rendered_age = float(max_age_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("cluster discovery max_age_s must be finite") from exc
        if not 0.001 <= rendered_age <= _MAX_MAX_AGE_S:
            raise ValueError("cluster discovery max_age_s is out of bounds")
        try:
            rendered_skew = float(clock_skew_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("cluster discovery clock_skew_s must be finite") from exc
        if not 0.0 <= rendered_skew <= rendered_age:
            raise ValueError("cluster discovery clock_skew_s is out of bounds")
        self.max_age_s = rendered_age
        self.clock_skew_s = rendered_skew
        self._monotonic = monotonic
        self._wall_clock_ms = wall_clock_ms or (lambda: int(time.time() * 1_000))
        self._lock = threading.RLock()
        self._last_good: dict[tuple[str, str, str, str], ClusterDiscoverySnapshot] = {}

    @property
    def last_good(self) -> ClusterDiscoverySnapshot | None:
        with self._lock:
            return next(reversed(self._last_good.values()), None)

    def last_good_for(
        self,
        *,
        verified_context: Mapping[str, Any],
        expected_cluster_id: str | None = None,
    ) -> ClusterDiscoverySnapshot | None:
        """Return only the cache entry bound to one verified context.

        ``last_good`` remains a diagnostic convenience, but route-change
        detection must never compare one tenant/principal's epochs with
        another's.  Callers selecting a live endpoint use this context-bound
        accessor instead.
        """
        if not isinstance(verified_context, Mapping):
            return None
        if any(
            not str(verified_context.get(key, "") or "").strip()
            for key in ("tenant", "principal", "agent_id")
        ):
            return None
        key = (
            _digest(str(verified_context.get("tenant", ""))),
            _digest(str(verified_context.get("principal", ""))),
            _digest(str(verified_context.get("agent_id", ""))),
            str(expected_cluster_id or ""),
        )
        with self._lock:
            return self._last_good.get(key)

    def invalidate(self) -> None:
        with self._lock:
            self._last_good.clear()

    @staticmethod
    def _client_context(client: Any) -> Mapping[str, Any] | None:
        for candidate in (client, getattr(client, "_client", None)):
            method = getattr(candidate, "_effective_verified_context", None)
            if callable(method):
                context = method()
                return context if isinstance(context, Mapping) else None
        return None

    @classmethod
    def context_for(cls, client: Any) -> Mapping[str, Any] | None:
        """Expose the native client's already-verified context to consumers."""
        return cls._client_context(client)

    @staticmethod
    def _validate_shape_and_schema(answer: Any) -> None:
        if not isinstance(answer, Mapping):
            raise ClusterDiscoveryRejected("ClusterMembers response is not a mapping")
        required = {
            "schema_version",
            "cluster_id",
            "epoch",
            "membership_epoch",
            "placement_epoch",
            "leader",
            "leaders",
            "groups",
            "auth_binding",
            "signature",
        }
        if set(answer) != required:
            raise ClusterDiscoveryRejected(
                "ClusterMembers response has unexpected fields"
            )
        if answer["schema_version"] != _SCHEMA_VERSION or isinstance(
            answer["schema_version"], bool
        ):
            raise ClusterDiscoveryRejected(
                "ClusterMembers schema version is unsupported"
            )

    @staticmethod
    def _validate_cluster_identity(
        answer: Mapping[str, Any], expected_cluster_id: str | None
    ) -> str:
        cluster_id = answer["cluster_id"]
        if not _is_digest(cluster_id):
            raise ClusterDiscoveryRejected(
                "ClusterMembers cluster identity is malformed"
            )
        if expected_cluster_id is not None and not _is_digest(expected_cluster_id):
            raise ClusterDiscoveryRejected("expected cluster identity is malformed")
        if expected_cluster_id is not None and cluster_id != expected_cluster_id:
            raise ClusterDiscoveryRejected(
                "ClusterMembers belongs to a different cluster"
            )
        return cluster_id

    @staticmethod
    def _parse_epochs(
        answer: Mapping[str, Any],
        min_membership_epoch: int | None,
        min_placement_epoch: int | None,
    ) -> tuple[int, int]:
        epoch = _non_negative_int(answer["epoch"], field="epoch")
        membership_epoch = _non_negative_int(
            answer["membership_epoch"], field="membership_epoch"
        )
        placement_epoch = _non_negative_int(
            answer["placement_epoch"], field="placement_epoch"
        )
        if epoch != membership_epoch:
            raise ClusterDiscoveryRejected(
                "ClusterMembers epoch alias does not match membership_epoch"
            )
        if min_membership_epoch is not None and membership_epoch < min_membership_epoch:
            raise ClusterDiscoveryRejected(
                "ClusterMembers membership snapshot is stale"
            )
        if min_placement_epoch is not None and placement_epoch < min_placement_epoch:
            raise ClusterDiscoveryRejected("ClusterMembers placement snapshot is stale")
        return membership_epoch, placement_epoch

    @staticmethod
    def _validate_auth_binding(
        answer: Mapping[str, Any],
        verified_context: Mapping[str, Any] | None,
        client_context: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        binding = answer["auth_binding"]
        if (
            not isinstance(binding, Mapping)
            or set(binding)
            != {
                "tenant_digest",
                "principal_digest",
                "agent_digest",
            }
            or not all(_is_digest(binding[key]) for key in binding)
        ):
            raise ClusterDiscoveryRejected("ClusterMembers auth binding is malformed")
        expected_context = (
            verified_context if verified_context is not None else client_context
        )
        if expected_context is None:
            # The engine client normally verifies this before returning.  A
            # client/fake without a verified-context seam is not an authority
            # source for a live AU process, so reject it instead of guessing.
            raise ClusterDiscoveryRejected(
                "ClusterMembers client has no verified context"
            )
        ClusterTopologyAuthority._validate_context_matches_binding(
            expected_context, binding
        )
        return binding

    @staticmethod
    def _validate_context_matches_binding(
        expected_context: Mapping[str, Any], binding: Mapping[str, Any]
    ) -> None:
        if any(
            not str(expected_context.get(key, "") or "").strip()
            for key in ("tenant", "principal", "agent_id")
        ):
            raise ClusterDiscoveryRejected(
                "ClusterMembers verified request context is incomplete"
            )
        expected_binding = {
            "tenant_digest": _digest(str(expected_context.get("tenant", ""))),
            "principal_digest": _digest(str(expected_context.get("principal", ""))),
            "agent_digest": _digest(str(expected_context.get("agent_id", ""))),
        }
        if binding != expected_binding:
            raise ClusterDiscoveryRejected(
                "ClusterMembers request context does not match"
            )

    def _parse_certificate(
        self, certificate: Any
    ) -> tuple[str | None, int, int | None, int | None]:
        if not isinstance(certificate, Mapping) or set(certificate) != {
            "id",
            "rotation_epoch",
            "not_before_ms",
            "not_after_ms",
        }:
            raise ClusterDiscoveryRejected(
                "ClusterMembers certificate metadata is malformed"
            )
        certificate_id, certificate_rotation_epoch = self._parse_certificate_identity(
            certificate
        )
        certificate_not_before_ms = certificate["not_before_ms"]
        certificate_not_after_ms = certificate["not_after_ms"]
        for value, name in (
            (certificate_not_before_ms, "certificate.not_before_ms"),
            (certificate_not_after_ms, "certificate.not_after_ms"),
        ):
            if value is not None:
                _non_negative_int(value, field=name)
        if (
            certificate_not_before_ms is not None
            and certificate_not_after_ms is not None
            and certificate_not_before_ms > certificate_not_after_ms
        ):
            raise ClusterDiscoveryRejected("certificate validity is inverted")
        return (
            certificate_id,
            certificate_rotation_epoch,
            certificate_not_before_ms,
            certificate_not_after_ms,
        )

    @staticmethod
    def _parse_certificate_identity(
        certificate: Mapping[str, Any],
    ) -> tuple[str | None, int]:
        certificate_id = _bounded_text(
            certificate["id"], field="certificate.id", required=False
        )
        if (
            certificate_id is not None
            and len(certificate_id.encode("utf-8")) > _MAX_CERTIFICATE_ID_BYTES
        ):
            raise ClusterDiscoveryRejected("ClusterMembers certificate id is too large")
        certificate_rotation_epoch = _non_negative_int(
            certificate["rotation_epoch"], field="certificate.rotation_epoch"
        )
        if certificate_rotation_epoch > 0 and certificate_id is None:
            raise ClusterDiscoveryRejected("certificate rotation requires an id")
        return certificate_id, certificate_rotation_epoch

    def _parse_member(
        self, raw_member: Any, group_id: int, seen_members: set[int]
    ) -> ClusterMember:
        if not isinstance(raw_member, Mapping) or set(raw_member) != {
            "node_id",
            "member_identity",
            "role",
            "client_endpoint",
            "tls_name",
            "health",
            "certificate",
        }:
            raise ClusterDiscoveryRejected("ClusterMembers member entry is malformed")
        node_id = _non_negative_int(raw_member["node_id"], field="node_id")
        if node_id in seen_members:
            raise ClusterDiscoveryRejected("ClusterMembers contains duplicate members")
        seen_members.add(node_id)
        member_identity = raw_member["member_identity"]
        if not _is_digest(member_identity):
            raise ClusterDiscoveryRejected(
                "ClusterMembers member identity is malformed"
            )
        role = raw_member["role"]
        if role not in {"leader", "follower", "learner"}:
            raise ClusterDiscoveryRejected("ClusterMembers member role is invalid")
        endpoint = _endpoint_is_bounded(raw_member["client_endpoint"])
        tls_name = _bounded_text(
            raw_member["tls_name"], field="tls_name", required=False
        )
        if endpoint.startswith("tls://") and tls_name is None:
            raise ClusterDiscoveryRejected(
                "TLS ClusterMembers endpoint has no server name"
            )
        health = raw_member["health"]
        if health not in {"healthy", "degraded", "unknown"}:
            raise ClusterDiscoveryRejected("ClusterMembers member health is invalid")
        (
            certificate_id,
            certificate_rotation_epoch,
            certificate_not_before_ms,
            certificate_not_after_ms,
        ) = self._parse_certificate(raw_member["certificate"])
        member = ClusterMember(
            group_id=group_id,
            node_id=node_id,
            member_identity=member_identity,
            role=role,
            client_endpoint=endpoint,
            tls_name=tls_name,
            health=health,
            certificate_id=certificate_id,
            certificate_rotation_epoch=certificate_rotation_epoch,
            certificate_not_before_ms=certificate_not_before_ms,
            certificate_not_after_ms=certificate_not_after_ms,
        )
        member.assert_certificate_current(
            now_ms=self._wall_clock_ms(),
            clock_skew_s=self.clock_skew_s,
        )
        return member

    def _parse_groups(
        self, groups_raw: Any
    ) -> list[tuple[int, tuple[ClusterMember, ...], int | None]]:
        if not isinstance(groups_raw, list) or len(groups_raw) > _MAX_GROUPS:
            raise ClusterDiscoveryRejected("ClusterMembers groups exceed bounds")
        parsed_groups: list[tuple[int, tuple[ClusterMember, ...], int | None]] = []
        seen_groups: set[int] = set()
        total_members = 0
        for raw_group in groups_raw:
            group_id, leader_id, members_raw = self._parse_group_header(
                raw_group, seen_groups
            )
            members: list[ClusterMember] = []
            seen_members: set[int] = set()
            for raw_member in members_raw:
                member = self._parse_member(raw_member, group_id, seen_members)
                members.append(member)
                total_members += 1
                if total_members > _MAX_MEMBERS:
                    raise ClusterDiscoveryRejected(
                        "ClusterMembers members exceed bounds"
                    )
            ClusterTopologyAuthority._validate_group_leader(members, leader_id)
            parsed_groups.append((group_id, tuple(members), leader_id))
        return parsed_groups

    @staticmethod
    def _parse_group_header(
        raw_group: Any, seen_groups: set[int]
    ) -> tuple[int, int | None, list[Any]]:
        if not isinstance(raw_group, Mapping) or set(raw_group) != {
            "group_id",
            "leader_id",
            "members",
        }:
            raise ClusterDiscoveryRejected("ClusterMembers group entry is malformed")
        group_id = _non_negative_int(raw_group["group_id"], field="group_id")
        if group_id in seen_groups:
            raise ClusterDiscoveryRejected("ClusterMembers contains duplicate groups")
        seen_groups.add(group_id)
        leader_id = raw_group["leader_id"]
        if leader_id is not None:
            leader_id = _non_negative_int(leader_id, field="leader_id")
        members_raw = raw_group["members"]
        if not isinstance(members_raw, list):
            raise ClusterDiscoveryRejected("ClusterMembers members is not a list")
        return group_id, leader_id, members_raw

    @staticmethod
    def _validate_group_leader(
        members: list[ClusterMember], leader_id: int | None
    ) -> None:
        if leader_id is None:
            return
        leader = next(
            (member for member in members if member.node_id == leader_id), None
        )
        if leader is None or leader.role != "leader":
            raise ClusterDiscoveryRejected("ClusterMembers leader is inconsistent")

    @staticmethod
    def _validate_leaders(
        answer: Mapping[str, Any],
        parsed_groups: list[tuple[int, tuple[ClusterMember, ...], int | None]],
    ) -> None:
        leaders = answer["leaders"]
        expected_leaders = [
            {"group_id": group_id, "node_id": leader_id}
            for group_id, _, leader_id in parsed_groups
            if leader_id is not None
        ]
        if leaders != expected_leaders:
            raise ClusterDiscoveryRejected("ClusterMembers leaders are inconsistent")
        expected_leader = expected_leaders[0] if expected_leaders else None
        if answer["leader"] != expected_leader:
            raise ClusterDiscoveryRejected("ClusterMembers leader is inconsistent")

    @staticmethod
    def _validate_signature(answer: Mapping[str, Any]) -> None:
        signature = answer["signature"]
        if (
            not isinstance(signature, str)
            or not signature.startswith("hmac-sha256:")
            or len(signature) != len("hmac-sha256:") + 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in signature[len("hmac-sha256:") :]
            )
        ):
            raise ClusterDiscoveryRejected("ClusterMembers signature is malformed")

    @staticmethod
    def _validate_against_prior(
        prior: ClusterDiscoverySnapshot | None,
        cluster_id: str,
        membership_epoch: int,
        placement_epoch: int,
        parsed_groups: list[tuple[int, tuple[ClusterMember, ...], int | None]],
    ) -> None:
        if prior is None:
            return
        if cluster_id != prior.cluster_id:
            raise ClusterDiscoveryRejected("ClusterMembers cluster identity changed")
        if (
            membership_epoch < prior.membership_epoch
            or placement_epoch < prior.placement_epoch
        ):
            raise ClusterDiscoveryRejected("ClusterMembers epoch moved backwards")
        prior_members = {
            (member.group_id, member.node_id): member
            for _, members, _ in prior.groups
            for member in members
        }
        for _, group_members, _ in parsed_groups:
            for member in group_members:
                ClusterTopologyAuthority._validate_member_certificate_against_prior(
                    member, prior_members.get((member.group_id, member.node_id))
                )

    @staticmethod
    def _validate_member_certificate_against_prior(
        member: ClusterMember, old: ClusterMember | None
    ) -> None:
        if old is None:
            return
        if member.certificate_rotation_epoch < old.certificate_rotation_epoch:
            raise ClusterDiscoveryRejected(
                "ClusterMembers certificate epoch moved backwards"
            )
        if (
            member.certificate_rotation_epoch == old.certificate_rotation_epoch
            and member.certificate_id != old.certificate_id
        ):
            raise ClusterDiscoveryRejected(
                "ClusterMembers certificate changed without a rotation epoch"
            )

    def _parse(
        self,
        answer: Any,
        *,
        verified_context: Mapping[str, Any] | None,
        client_context: Mapping[str, Any] | None,
        expected_cluster_id: str | None,
        min_membership_epoch: int | None,
        min_placement_epoch: int | None,
        prior: ClusterDiscoverySnapshot | None,
    ) -> ClusterDiscoverySnapshot:
        """Verify and parse one signed ``ClusterMembers`` response.

        Split into named per-stage validators (CX-AU-01, CCN 82 -> see the
        ``_validate_*``/``_parse_*`` methods above), called in the same
        order the original single function checked them in. Each stage's
        validation RULES are unchanged, byte-identical logic, only WHERE
        they live moved.
        """
        self._validate_shape_and_schema(answer)
        cluster_id = self._validate_cluster_identity(answer, expected_cluster_id)
        membership_epoch, placement_epoch = self._parse_epochs(
            answer, min_membership_epoch, min_placement_epoch
        )
        binding = self._validate_auth_binding(answer, verified_context, client_context)
        parsed_groups = self._parse_groups(answer["groups"])
        self._validate_leaders(answer, parsed_groups)
        self._validate_signature(answer)
        self._validate_against_prior(
            prior, cluster_id, membership_epoch, placement_epoch, parsed_groups
        )
        return ClusterDiscoverySnapshot(
            cluster_id=cluster_id,
            membership_epoch=membership_epoch,
            placement_epoch=placement_epoch,
            observed_monotonic=self._monotonic(),
            observed_wall_ms=self._wall_clock_ms(),
            tenant_digest=str(binding["tenant_digest"]),
            principal_digest=str(binding["principal_digest"]),
            agent_digest=str(binding["agent_digest"]),
            groups=tuple(parsed_groups),
            max_age_s=self.max_age_s,
            clock_skew_s=self.clock_skew_s,
        )

    def read(
        self,
        client: Any,
        *,
        verified_context: Mapping[str, Any] | None = None,
        expected_cluster_id: str | None = None,
        min_membership_epoch: int | None = None,
        min_placement_epoch: int | None = None,
        force_refresh: bool = False,
    ) -> ClusterDiscoverySnapshot:
        """Read or safely reuse one verified snapshot.

        Only transport failures may use a still-current last-good snapshot.
        Verification/context/epoch/certificate failures are never hidden by a
        cache fallback; this distinction prevents a wrong-cluster response from
        being silently treated as a temporary outage.
        """
        client_context = self._client_context(client)
        effective_context = (
            verified_context if verified_context is not None else client_context
        )
        if effective_context is None:
            raise ClusterDiscoveryRejected(
                "ClusterMembers client has no verified request context"
            )
        if any(
            not str(effective_context.get(key, "") or "").strip()
            for key in ("tenant", "principal", "agent_id")
        ):
            raise ClusterDiscoveryRejected(
                "ClusterMembers client has an incomplete verified request context"
            )
        context_key = (
            _digest(str(effective_context.get("tenant", ""))),
            _digest(str(effective_context.get("principal", ""))),
            _digest(str(effective_context.get("agent_id", ""))),
            str(expected_cluster_id or ""),
        )
        with self._lock:
            prior = self._last_good.get(context_key)
        if not force_refresh and prior is not None:
            try:
                prior.assert_current(
                    now_monotonic=self._monotonic(),
                    now_wall_ms=self._wall_clock_ms(),
                    min_membership_epoch=min_membership_epoch,
                    min_placement_epoch=min_placement_epoch,
                )
                if (
                    expected_cluster_id is not None
                    and prior.cluster_id != expected_cluster_id
                ):
                    raise ClusterDiscoveryRejected(
                        "cached ClusterMembers belongs to a different cluster"
                    )
                return prior
            except ClusterDiscoveryError as exc:
                # Best-effort: a stale or foreign cached snapshot simply falls
                # through to the live authority below. The cause is still worth
                # surfacing -- silently discarding it left operators unable to
                # tell a routine expiry from a persistent cluster-id mismatch.
                logger.info("cached ClusterMembers snapshot rejected: %s", exc)

        topology = getattr(client, "cluster_topology", None)
        members = getattr(topology, "members", None)
        if not callable(members):
            raise ClusterDiscoveryError(
                "engine client has no verified ClusterMembers discovery authority"
            )
        try:
            answer = members(
                expected_cluster_id=expected_cluster_id,
                min_membership_epoch=min_membership_epoch,
                min_placement_epoch=min_placement_epoch,
            )
        except (ConnectionError, OSError, TimeoutError) as exc:
            if prior is not None:
                try:
                    prior.assert_current(
                        now_monotonic=self._monotonic(),
                        now_wall_ms=self._wall_clock_ms(),
                        min_membership_epoch=min_membership_epoch,
                        min_placement_epoch=min_placement_epoch,
                    )
                    if (
                        expected_cluster_id is None
                        or prior.cluster_id == expected_cluster_id
                    ):
                        logger.warning(
                            "ClusterMembers transport unavailable; using bounded last-good snapshot"
                        )
                        return prior
                except ClusterDiscoveryError as stale_exc:
                    # The transport is already down AND the last-good snapshot is
                    # unusable, so this is the path to a hard failure. Warn with
                    # the real reason before raising, or the operator only ever
                    # sees "discovery is unavailable" with no way to tell a stale
                    # snapshot from a cluster-identity mismatch.
                    logger.warning(
                        "ClusterMembers transport unavailable and the last-good "
                        "snapshot is not usable: %s",
                        stale_exc,
                    )
            raise ClusterDiscoveryError(
                "ClusterMembers discovery is unavailable"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ClusterDiscoveryRejected(
                "engine rejected the ClusterMembers discovery response"
            ) from exc
        snapshot = self._parse(
            answer,
            verified_context=verified_context,
            client_context=client_context,
            expected_cluster_id=expected_cluster_id,
            min_membership_epoch=min_membership_epoch,
            min_placement_epoch=min_placement_epoch,
            prior=prior,
        )
        with self._lock:
            self._last_good[context_key] = snapshot
        return snapshot
