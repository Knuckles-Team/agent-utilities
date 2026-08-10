"""Cross-host concept reservation authority.

CONCEPT:AU-OS.governance.cross-host-concept-reservation-authority

The repository-local allocator remains the compatibility projection for
``docs/concept_reservations.d``.  It is deliberately *not* the authority for
separate hosts: a git common directory and a filesystem lock stop being shared
at that boundary.  This module defines the graph-os boundary that owns global
concept uniqueness and supplies a fail-closed adapter over the engine's existing
native ``CreateNodeIfAbsent`` and ``CompareAndSetNodeFields`` primitives.

There is intentionally no file, JSON, process-lock, or generic Cypher fallback
here.  A canonical reservation node is keyed by the complete concept ID:
``CreateNodeIfAbsent`` chooses the first durable claimant and
``CompareAndSetNodeFields`` fences lifecycle/reclaim transitions.  A caller
that cannot reach those native primitives receives
:class:`AuthorityUnavailable` instead of an unsafe local success.

Request-key idempotency is scoped to that canonical concept node: a repeated
key/fingerprint for the same concept replays the node, while the same key on a
different concept uses a different node.  This module does not claim a
globally atomic secondary request-key index.

``FixtureConceptReservationAuthority`` is useful for deterministic unit tests
and contract examples only.  It advertises ``authoritative = False`` and must
never be selected by a deployment as a cross-host authority.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol, Self

from agent_utilities.governance.concept_hierarchy import (
    is_valid_domain,
    parse_okf_id,
)
from agent_utilities.security.persistence_privacy import persistence_reference

__all__ = [
    "AuthorityUnavailable",
    "ConceptNamespacePolicy",
    "ConceptReservationAuthority",
    "ConceptReservationConflict",
    "ConceptReservationError",
    "ConceptReservationFenceConflict",
    "ConceptReservationIdUnavailable",
    "ConceptReservationNotFound",
    "ConceptReservationRecord",
    "ConceptReservationRequest",
    "ConceptReservationService",
    "ConceptReservationState",
    "ConceptReservationUnauthorized",
    "ConceptReservationVisibility",
    "FixtureConceptReservationAuthority",
    "NativeConceptReservationAuthority",
    "NativeConceptReservationPort",
    "ProjectionReconciliation",
    "RESERVATION_NODE_LABEL",
    "RESERVATION_NODE_PREFIX",
    "reconcile_projection",
    "reservation_request",
    "reserve_next_numeric",
]


SCHEMA_VERSION = "1"
RESERVATION_NODE_LABEL = "ConceptReservation"
RESERVATION_NODE_PREFIX = "concept-reservation:"
_REFERENCE_RE = re.compile(r"^pref_[a-z0-9_]+_[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_LIST_LIMIT = 1000
_MAX_CURSOR_BYTES = 512
_MAX_READ_PAGES = 1024
_MAX_READ_RECORDS = 10_000


class ConceptReservationError(ValueError):
    """Base error for invalid concept reservation input or lifecycle state."""


class ConceptReservationConflict(ConceptReservationError):
    """A request key or concept id already names different immutable input."""


class ConceptReservationFenceConflict(ConceptReservationConflict):
    """A lifecycle mutation used a stale owner or fencing token."""


class ConceptReservationIdUnavailable(ConceptReservationConflict):
    """A candidate ID is owned by a different request or terminal claim."""


class ConceptReservationUnauthorized(PermissionError):
    """The verified tenant/owner cannot inspect or mutate this reservation."""


class ConceptReservationNotFound(ConceptReservationError):
    """The requested reservation is not present in the authority."""


class AuthorityUnavailable(RuntimeError):
    """The native cross-host authority is unavailable; callers must defer."""


class ConceptReservationState(StrEnum):
    """Durable claim lifecycle states.

    ``TOMBSTONED`` is terminal even when a repository later removes the marker:
    external visibility makes ID reuse unsafe forever.
    """

    RESERVED = "reserved"
    MATERIALIZED = "materialized"
    LANDED = "landed"
    RELEASED = "released"
    EXPIRED = "expired"
    TOMBSTONED = "tombstoned"


class ConceptReservationVisibility(StrEnum):
    """The strongest visibility reached by a claim."""

    PRIVATE = "private"
    FRAGMENT = "fragment"
    REPOSITORY = "repository"
    EXTERNAL = "external"


_VISIBILITY_RANK = {
    ConceptReservationVisibility.PRIVATE: 0,
    ConceptReservationVisibility.FRAGMENT: 1,
    ConceptReservationVisibility.REPOSITORY: 2,
    ConceptReservationVisibility.EXTERNAL: 3,
}


def _strongest_visibility(
    *values: ConceptReservationVisibility,
) -> ConceptReservationVisibility:
    """Return the highest visibility reached by a lifecycle request."""

    return max(values, key=_VISIBILITY_RANK.__getitem__)


def _validate_cursor(cursor: object, field_name: str = "cursor") -> str | None:
    """Validate a native cursor before handing it to a graph client.

    Cursors are opaque node IDs, but they are still control input.  A bounded
    size and the same printable-string rule used for references prevent a bad
    backend or caller from turning a paginated read into an unbounded loop.
    """

    if cursor is None:
        return None
    if not isinstance(cursor, str) or not cursor or cursor.strip() != cursor:
        raise ConceptReservationError(f"{field_name} is invalid")
    if len(cursor.encode("utf-8")) > _MAX_CURSOR_BYTES or any(
        ord(char) < 0x20 for char in cursor
    ):
        raise ConceptReservationError(f"{field_name} is invalid")
    return cursor


def _nonblank(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConceptReservationError(f"{field_name} must be a non-blank string")
    if any(ord(char) < 0x20 for char in value):
        raise ConceptReservationError(
            f"{field_name} must not contain control characters"
        )
    return value


def _reference(value: object, field_name: str) -> str:
    rendered = _nonblank(value, field_name)
    if not _REFERENCE_RE.fullmatch(rendered):
        raise ConceptReservationError(
            f"{field_name} must be a non-reversible persistence reference"
        )
    return rendered


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _parse_time(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ConceptReservationError(
                f"{field_name} is not an ISO-8601 timestamp"
            ) from exc
    else:
        raise ConceptReservationError(f"{field_name} is required")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _refs(values: Iterable[object] | None, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise ConceptReservationError(f"{field_name} must be a sequence")
    return tuple(sorted({_reference(item, field_name) for item in values}))


@dataclass(frozen=True, slots=True)
class ConceptNamespacePolicy:
    """Allowed namespace/range rules evaluated by the authority transaction.

    ``namespace`` is the exact OKF prefix below the pillar, for example
    ``AU-OS.governance``.  Optional numeric bounds apply to a final ``-N``
    suffix, allowing a deployment to partition an explicitly numeric range
    without imposing numeric identifiers on the existing semantic grammar.
    ``concept_prefixes`` can further partition a namespace by semantic prefix.
    """

    namespace: str
    range_start: int | None = None
    range_end: int | None = None
    concept_prefixes: tuple[str, ...] = ()
    policy_version: str = "1"

    def __post_init__(self) -> None:
        namespace = _nonblank(self.namespace, "namespace")
        policy_version = _nonblank(self.policy_version, "policy_version")
        if self.range_start is not None and self.range_start < 0:
            raise ConceptReservationError("namespace range bounds are invalid")
        if self.range_end is not None and self.range_end < 0:
            raise ConceptReservationError("namespace range bounds are invalid")
        if (
            self.range_start is not None
            and self.range_end is not None
            and self.range_end < self.range_start
        ):
            raise ConceptReservationError("namespace range bounds are invalid")
        prefixes = tuple(
            _nonblank(prefix, "concept_prefixes") for prefix in self.concept_prefixes
        )
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "policy_version", policy_version)
        object.__setattr__(self, "concept_prefixes", tuple(sorted(set(prefixes))))

    def accepts(self, concept_id: str) -> bool:
        if not concept_id.startswith(self.namespace + "."):
            return False
        tail = concept_id[len(self.namespace) + 1 :]
        if self.concept_prefixes and not any(
            tail == prefix
            or tail.startswith(prefix + ".")
            or tail.startswith(prefix + "-")
            for prefix in self.concept_prefixes
        ):
            return False
        if self.range_start is None and self.range_end is None:
            return True
        match = re.search(r"-(\d+)(?:\.|$)", tail)
        if match is None:
            return False
        value = int(match.group(1))
        return (self.range_start is None or value >= self.range_start) and (
            self.range_end is None or value <= self.range_end
        )


def _ranges_overlap(
    left: ConceptNamespacePolicy, right: ConceptNamespacePolicy
) -> bool:
    """Return whether two optional inclusive numeric ranges can intersect."""

    left_start = left.range_start if left.range_start is not None else 0
    right_start = right.range_start if right.range_start is not None else 0
    left_end = left.range_end
    right_end = right.range_end
    if left_end is not None and right_start > left_end:
        return False
    if right_end is not None and left_start > right_end:
        return False
    return True


def _prefixes_overlap(
    left: ConceptNamespacePolicy, right: ConceptNamespacePolicy
) -> bool:
    """Return whether semantic prefixes can accept one common concept tail."""

    if not left.concept_prefixes or not right.concept_prefixes:
        return True

    def covers(prefix: str, candidate: str) -> bool:
        return candidate == prefix or candidate.startswith((prefix + ".", prefix + "-"))

    return any(
        covers(left_prefix, right_prefix) or covers(right_prefix, left_prefix)
        for left_prefix in left.concept_prefixes
        for right_prefix in right.concept_prefixes
    )


def _policies_overlap(
    left: ConceptNamespacePolicy, right: ConceptNamespacePolicy
) -> bool:
    """Conservatively identify policies that could claim one concept ID."""

    # A nested namespace is also ambiguous when the parent policy has no
    # semantic partition.  Rejecting potential overlap is safer than selecting
    # one policy based on version-string or insertion order.
    same_namespace = left.namespace == right.namespace
    nested_namespace = left.namespace.startswith(
        right.namespace + "."
    ) or right.namespace.startswith(left.namespace + ".")
    if not (same_namespace or nested_namespace):
        return False
    return _ranges_overlap(left, right) and _prefixes_overlap(left, right)


def _validate_policy_set(
    policies: Sequence[ConceptNamespacePolicy],
) -> tuple[ConceptNamespacePolicy, ...]:
    normalized = tuple(policies)
    if any(not isinstance(policy, ConceptNamespacePolicy) for policy in normalized):
        raise ConceptReservationError("authority policies are malformed")
    for index, policy in enumerate(normalized):
        for other in normalized[index + 1 :]:
            if _policies_overlap(policy, other):
                raise ConceptReservationError(
                    "authority policies overlap ambiguously; partition namespace, "
                    "range, or semantic prefix explicitly"
                )
    return normalized


@dataclass(frozen=True, slots=True)
class ConceptReservationRequest:
    """Immutable input accepted by the native reservation authority.

    All identity-bearing fields are already opaque references.  Use
    :func:`reservation_request` at an application boundary rather than placing
    raw tenant, owner, repository, lane, or request-key values in a durable
    graph payload.
    """

    tenant_ref: str
    concept_id: str
    namespace: str
    repository_ref: str
    lane_ref: str
    owner_ref: str
    request_key_ref: str
    purpose_digest: str
    design_ref: str | None
    created_at: datetime
    expires_at: datetime
    range_start: int | None = None
    range_end: int | None = None
    policy_version: str = ""
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parsed = parse_okf_id(self.concept_id)
        if not is_valid_domain(parsed.pillar, parsed.domain):
            raise ConceptReservationError(
                f"domain {parsed.domain!r} is not registered for pillar {parsed.pillar!r}"
            )
        _reference(self.tenant_ref, "tenant_ref")
        _reference(self.repository_ref, "repository_ref")
        _reference(self.lane_ref, "lane_ref")
        _reference(self.owner_ref, "owner_ref")
        _reference(self.request_key_ref, "request_key_ref")
        if self.design_ref is not None:
            _reference(self.design_ref, "design_ref")
        if not _DIGEST_RE.fullmatch(self.purpose_digest):
            raise ConceptReservationError("purpose_digest must be a SHA-256 digest")
        if self.range_start is not None and self.range_start < 0:
            raise ConceptReservationError("range_start must be non-negative")
        if self.range_end is not None and (
            self.range_end < 0
            or (self.range_start is not None and self.range_end < self.range_start)
        ):
            raise ConceptReservationError("range_end is invalid")
        if self.policy_version:
            _nonblank(self.policy_version, "policy_version")
        if self.expires_at <= self.created_at:
            raise ConceptReservationError("expires_at must be after created_at")
        namespace = _nonblank(self.namespace, "namespace")
        if not self.concept_id.startswith(namespace + "."):
            raise ConceptReservationError(
                "namespace does not contain the requested concept id"
            )
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(
            self, "provenance_refs", _refs(self.provenance_refs, "provenance_refs")
        )

    @property
    def immutable_fingerprint(self) -> str:
        """Digest of every field that an idempotency key cannot change."""

        # Creation/expiry timestamps are server-owned replay metadata. A client
        # retry may rebuild the request a few seconds later; changing those
        # values must not turn an otherwise identical idempotency request into
        # a conflict. The native transaction retains the first committed times.
        return _digest(self.to_wire(include_times=False))

    def to_wire(self, *, include_times: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tenant_ref": self.tenant_ref,
            "concept_id": self.concept_id,
            "namespace": self.namespace,
            "repository_ref": self.repository_ref,
            "lane_ref": self.lane_ref,
            "owner_ref": self.owner_ref,
            "request_key_ref": self.request_key_ref,
            "purpose_digest": self.purpose_digest,
            "design_ref": self.design_ref,
            "range_start": self.range_start,
            "range_end": self.range_end,
            "policy_version": self.policy_version,
            "provenance_refs": list(self.provenance_refs),
        }
        if include_times:
            payload.update(
                created_at=_iso(self.created_at),
                expires_at=_iso(self.expires_at),
            )
        return payload


def reservation_request(
    concept_id: str,
    *,
    tenant_id: str,
    repository: str,
    lane: str,
    owner: str,
    request_key: str,
    purpose: str,
    design_doc: str | None = None,
    ttl_seconds: int = 86_400,
    namespace: str | None = None,
    range_start: int | None = None,
    range_end: int | None = None,
    policy_version: str = "",
    provenance: Sequence[str] | None = None,
    now: datetime | None = None,
) -> ConceptReservationRequest:
    """Build a privacy-safe request from application-level values."""

    parsed = parse_okf_id(concept_id)
    if not is_valid_domain(parsed.pillar, parsed.domain):
        raise ConceptReservationError(
            f"domain {parsed.domain!r} is not registered for pillar {parsed.pillar!r}"
        )
    if ttl_seconds <= 0:
        raise ConceptReservationError("ttl_seconds must be positive")
    created = _parse_time(now or datetime.now(UTC), "now")
    tenant_ref = persistence_reference("concept_tenant", tenant_id)
    return ConceptReservationRequest(
        tenant_ref=tenant_ref,
        concept_id=concept_id,
        namespace=namespace or f"{parsed.slug}-{parsed.pillar}.{parsed.domain}",
        repository_ref=persistence_reference("concept_repository", repository),
        lane_ref=persistence_reference("concept_lane", lane),
        owner_ref=persistence_reference("concept_owner", owner),
        request_key_ref=persistence_reference(
            "concept_request", request_key, namespace=tenant_ref
        ),
        purpose_digest=hashlib.sha256(
            _nonblank(purpose, "purpose").encode("utf-8")
        ).hexdigest(),
        design_ref=(
            persistence_reference("concept_design", design_doc) if design_doc else None
        ),
        created_at=created,
        expires_at=created + timedelta(seconds=ttl_seconds),
        range_start=range_start,
        range_end=range_end,
        policy_version=policy_version,
        provenance_refs=tuple(provenance or ()),
    )


def reserve_next_numeric(
    authority: ConceptReservationAuthority,
    request: ConceptReservationRequest,
    *,
    concept_prefix: str,
    range_start: int,
    range_end: int,
) -> ConceptReservationRecord:
    """Probe a bounded deterministic numeric range through native create-once.

    Each candidate is a separate canonical node, so unrelated namespaces/ranges
    do not share a process lock.  Concurrent clients converge on one winner per
    candidate through the authority's existing ``CreateNodeIfAbsent`` primitive.
    The bound is deliberately explicit to prevent an accidental unbounded scan.
    """

    prefix = _nonblank(concept_prefix, "concept_prefix")
    if range_start < 0 or range_end < range_start:
        raise ConceptReservationError("numeric allocation range is invalid")
    if range_end - range_start > 100_000:
        raise ConceptReservationError("numeric allocation range is too large")
    for number in range(range_start, range_end + 1):
        candidate = replace(
            request,
            concept_id=f"{request.namespace}.{prefix}-{number}",
        )
        try:
            return authority.reserve(candidate)
        except ConceptReservationIdUnavailable:
            continue
    raise ConceptReservationConflict("numeric concept allocation range is exhausted")


@dataclass(frozen=True, slots=True)
class ConceptReservationRecord:
    """Durable authoritative claim returned by graph-os."""

    reservation_id: str
    request: ConceptReservationRequest
    state: ConceptReservationState
    visibility: ConceptReservationVisibility
    fence: int
    created_at: datetime
    expires_at: datetime
    transitioned_at: datetime
    materialized_at: datetime | None = None
    landed_at: datetime | None = None
    released_at: datetime | None = None
    expired_at: datetime | None = None
    tombstoned_at: datetime | None = None

    def __post_init__(self) -> None:
        _reference(self.reservation_id, "reservation_id")
        if not isinstance(self.state, ConceptReservationState):
            raise ConceptReservationError("record state is invalid")
        if not isinstance(self.visibility, ConceptReservationVisibility):
            raise ConceptReservationError("record visibility is invalid")
        if self.fence < 1:
            raise ConceptReservationError("fence must be positive")
        if self.expires_at <= self.created_at:
            raise ConceptReservationError("record expiry must follow creation")
        if self.transitioned_at < self.created_at:
            raise ConceptReservationError("transition timestamp precedes creation")
        times = {
            "materialized_at": self.materialized_at,
            "landed_at": self.landed_at,
            "released_at": self.released_at,
            "expired_at": self.expired_at,
            "tombstoned_at": self.tombstoned_at,
        }
        if any(
            value is not None and value < self.created_at for value in times.values()
        ):
            raise ConceptReservationError("lifecycle timestamp precedes creation")
        previous_time = self.created_at
        for value in times.values():
            if value is None:
                continue
            if value > self.transitioned_at or value < previous_time:
                raise ConceptReservationError("lifecycle timestamps are not monotonic")
            previous_time = value
        state = self.state
        if state is ConceptReservationState.RESERVED and (
            self.visibility is not ConceptReservationVisibility.PRIVATE
            or any(value is not None for value in times.values())
        ):
            raise ConceptReservationError(
                "reserved record has advanced lifecycle fields"
            )
        if state is ConceptReservationState.MATERIALIZED and (
            _VISIBILITY_RANK[self.visibility]
            < _VISIBILITY_RANK[ConceptReservationVisibility.FRAGMENT]
            or self.materialized_at is None
            or any(
                value is not None
                for value in (
                    self.landed_at,
                    self.released_at,
                    self.expired_at,
                    self.tombstoned_at,
                )
            )
        ):
            raise ConceptReservationError("materialized record is inconsistent")
        if state is ConceptReservationState.LANDED and (
            _VISIBILITY_RANK[self.visibility]
            < _VISIBILITY_RANK[ConceptReservationVisibility.REPOSITORY]
            or self.materialized_at is None
            or self.landed_at is None
            or self.tombstoned_at is not None
        ):
            raise ConceptReservationError("landed record is inconsistent")
        if state is ConceptReservationState.RELEASED and (
            _VISIBILITY_RANK[self.visibility]
            >= _VISIBILITY_RANK[ConceptReservationVisibility.REPOSITORY]
            or self.released_at is None
            or self.landed_at is not None
            or self.tombstoned_at is not None
        ):
            raise ConceptReservationError("released record is externally visible")
        if state is ConceptReservationState.EXPIRED and (
            _VISIBILITY_RANK[self.visibility]
            >= _VISIBILITY_RANK[ConceptReservationVisibility.REPOSITORY]
            or self.expired_at is None
            or self.landed_at is not None
            or self.tombstoned_at is not None
        ):
            raise ConceptReservationError("expired record is externally visible")
        if state is ConceptReservationState.TOMBSTONED and (
            self.visibility is not ConceptReservationVisibility.EXTERNAL
            or self.tombstoned_at is None
        ):
            raise ConceptReservationError(
                "tombstoned record must be externally visible"
            )

    @property
    def concept_id(self) -> str:
        return self.request.concept_id

    @property
    def tenant_ref(self) -> str:
        return self.request.tenant_ref

    @property
    def owner_ref(self) -> str:
        return self.request.owner_ref

    @property
    def immutable_fingerprint(self) -> str:
        return self.request.immutable_fingerprint

    def to_wire(self) -> dict[str, Any]:
        value = self.request.to_wire()
        value.update(
            {
                "reservation_id": self.reservation_id,
                "state": self.state.value,
                "visibility": self.visibility.value,
                "fence": self.fence,
                "transitioned_at": _iso(self.transitioned_at),
                "materialized_at": _iso(self.materialized_at)
                if self.materialized_at
                else None,
                "landed_at": _iso(self.landed_at) if self.landed_at else None,
                "released_at": _iso(self.released_at) if self.released_at else None,
                "expired_at": _iso(self.expired_at) if self.expired_at else None,
                "tombstoned_at": _iso(self.tombstoned_at)
                if self.tombstoned_at
                else None,
            }
        )
        return value

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> Self:
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ConceptReservationError(
                "concept reservation schema_version is unsupported"
            )
        request = ConceptReservationRequest(
            tenant_ref=_reference(value.get("tenant_ref"), "tenant_ref"),
            concept_id=_nonblank(value.get("concept_id"), "concept_id"),
            namespace=_nonblank(value.get("namespace"), "namespace"),
            repository_ref=_reference(value.get("repository_ref"), "repository_ref"),
            lane_ref=_reference(value.get("lane_ref"), "lane_ref"),
            owner_ref=_reference(value.get("owner_ref"), "owner_ref"),
            request_key_ref=_reference(value.get("request_key_ref"), "request_key_ref"),
            purpose_digest=_nonblank(value.get("purpose_digest"), "purpose_digest"),
            design_ref=(
                _reference(value.get("design_ref"), "design_ref")
                if value.get("design_ref") is not None
                else None
            ),
            created_at=_parse_time(value.get("created_at"), "created_at"),
            expires_at=_parse_time(value.get("expires_at"), "expires_at"),
            range_start=value.get("range_start"),
            range_end=value.get("range_end"),
            policy_version=str(value.get("policy_version") or ""),
            provenance_refs=_refs(value.get("provenance_refs"), "provenance_refs"),
        )
        try:
            state = ConceptReservationState(str(value.get("state")))
            visibility = ConceptReservationVisibility(str(value.get("visibility")))
        except ValueError as exc:
            raise ConceptReservationError(
                "concept reservation state is invalid"
            ) from exc
        fence = value.get("fence")
        if not isinstance(fence, int) or isinstance(fence, bool):
            raise ConceptReservationError("concept reservation fence is invalid")
        return cls(
            reservation_id=_reference(value.get("reservation_id"), "reservation_id"),
            request=request,
            state=state,
            visibility=visibility,
            fence=fence,
            created_at=_parse_time(value.get("created_at"), "created_at"),
            expires_at=_parse_time(value.get("expires_at"), "expires_at"),
            transitioned_at=_parse_time(
                value.get("transitioned_at"), "transitioned_at"
            ),
            materialized_at=(
                _parse_time(value.get("materialized_at"), "materialized_at")
                if value.get("materialized_at")
                else None
            ),
            landed_at=(
                _parse_time(value.get("landed_at"), "landed_at")
                if value.get("landed_at")
                else None
            ),
            released_at=(
                _parse_time(value.get("released_at"), "released_at")
                if value.get("released_at")
                else None
            ),
            expired_at=(
                _parse_time(value.get("expired_at"), "expired_at")
                if value.get("expired_at")
                else None
            ),
            tombstoned_at=(
                _parse_time(value.get("tombstoned_at"), "tombstoned_at")
                if value.get("tombstoned_at")
                else None
            ),
        )


class ConceptReservationAuthority(Protocol):
    """Authority port used by the service and Repository Manager adapter."""

    authoritative: bool

    def reserve(
        self, request: ConceptReservationRequest
    ) -> ConceptReservationRecord: ...

    def get(
        self, reservation_id: str, *, tenant_ref: str
    ) -> ConceptReservationRecord: ...

    def list(
        self,
        *,
        tenant_ref: str,
        namespace: str | None = None,
        state: ConceptReservationState | None = None,
        concept_prefix: str | None = None,
        limit: int = _MAX_LIST_LIMIT,
        cursor: str | None = None,
    ) -> tuple[list[ConceptReservationRecord], str | None]: ...

    def transition(
        self,
        reservation_id: str,
        *,
        tenant_ref: str,
        owner_ref: str,
        expected_fence: int,
        target: ConceptReservationState,
        visibility: ConceptReservationVisibility | None = None,
    ) -> ConceptReservationRecord: ...


class NativeConceptReservationPort(Protocol):
    """Existing sync engine primitives required by the native adapter.

    ``GraphComputeEngine`` exposes the first two methods directly.  The
    adapter also accepts an epistemic-graph sync client (``engine.nodes``) and
    the backend's equivalent wrappers, which keeps this seam usable from the
    graph-os MCP process without adding a second engine command.
    """

    def create_node_if_absent(
        self, node_id: str, properties: dict[str, Any]
    ) -> bool: ...

    def compare_and_set_node_fields(
        self,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
    ) -> bool: ...

    def get_node_properties(self, node_id: str) -> Mapping[str, Any] | None: ...

    def list_nodes_by_label(
        self, label: str, limit: int, *, after: str | None = None
    ) -> Sequence[tuple[str, Mapping[str, Any]]]: ...


def _reservation_node_id(concept_id: str) -> str:
    """Return the single global graph identity for one canonical concept ID."""

    return f"{RESERVATION_NODE_PREFIX}{concept_id}"


def _new_record(
    request: ConceptReservationRequest,
    *,
    fence: int = 1,
    reservation_id: str | None = None,
    state: ConceptReservationState = ConceptReservationState.RESERVED,
    visibility: ConceptReservationVisibility = ConceptReservationVisibility.PRIVATE,
) -> ConceptReservationRecord:
    now = request.created_at
    return ConceptReservationRecord(
        reservation_id=reservation_id
        or persistence_reference(
            "concept_reservation",
            f"{request.tenant_ref}:{request.request_key_ref}:{request.concept_id}",
        ),
        request=request,
        state=state,
        visibility=visibility,
        fence=fence,
        created_at=now,
        expires_at=request.expires_at,
        transitioned_at=now,
    )


def _transition_map() -> dict[ConceptReservationState, set[ConceptReservationState]]:
    return {
        ConceptReservationState.RESERVED: {
            ConceptReservationState.MATERIALIZED,
            ConceptReservationState.RELEASED,
            ConceptReservationState.EXPIRED,
            ConceptReservationState.TOMBSTONED,
        },
        ConceptReservationState.MATERIALIZED: {
            ConceptReservationState.LANDED,
            ConceptReservationState.RELEASED,
            ConceptReservationState.EXPIRED,
            ConceptReservationState.TOMBSTONED,
        },
        ConceptReservationState.LANDED: {ConceptReservationState.TOMBSTONED},
        ConceptReservationState.RELEASED: {ConceptReservationState.TOMBSTONED},
        ConceptReservationState.EXPIRED: {ConceptReservationState.TOMBSTONED},
        ConceptReservationState.TOMBSTONED: set(),
    }


class NativeConceptReservationAuthority:
    """Durable authority built from epistemic-graph's existing native writes.

    One graph node, keyed by :func:`_reservation_node_id`, is the authority for
    one complete concept ID.  ``CreateNodeIfAbsent`` arbitrates the first
    claimant.  Reclaim and lifecycle transitions use a single CAS whose
    conditions include the current reservation, owner, state, and fence.  The
    node identity is global within the configured authoritative graph; every
    host must therefore use the same graph placement/tenant authority.
    """

    authoritative = True

    def __init__(
        self,
        engine: NativeConceptReservationPort,
        *,
        policies: Sequence[ConceptNamespacePolicy] = (),
    ) -> None:
        self._engine = engine
        self._policies = _validate_policy_set(policies)

    def _method(self, name: str) -> Any:
        direct = getattr(self._engine, name, None)
        if callable(direct):
            return direct
        nodes = getattr(self._engine, "nodes", None)
        aliases = {
            "create_node_if_absent": "create_if_absent",
            "compare_and_set_node_fields": "compare_and_set",
            "get_node_properties": "properties",
            "list_nodes_by_label": "list_by_label",
        }
        candidate = getattr(nodes, aliases.get(name, name), None)
        if callable(candidate):
            return candidate
        client = getattr(self._engine, "_client", None)
        nodes = getattr(client, "nodes", None)
        candidate = getattr(nodes, aliases.get(name, name), None)
        if callable(candidate):
            return candidate
        graph = getattr(self._engine, "_graph", None)
        candidate = getattr(graph, name, None)
        if callable(candidate):
            return candidate
        raise AuthorityUnavailable(
            f"epistemic-graph lacks required native primitive {name}"
        )

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            result = self._method(name)(*args, **kwargs)
            if inspect.isawaitable(result):
                raise AuthorityUnavailable(
                    "native concept authority requires a synchronous graph client"
                )
            return result
        except AuthorityUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed at authority boundary
            raise AuthorityUnavailable(f"native graph primitive {name} failed") from exc

    def _check_policy(self, request: ConceptReservationRequest) -> None:
        if not self._policies:
            raise AuthorityUnavailable(
                "native concept authority has no authority-owned namespace policy"
            )
        for policy in self._policies:
            if policy.accepts(request.concept_id):
                return
        raise ConceptReservationError(
            f"concept id {request.concept_id!r} is outside the namespace/range policy"
        )

    def _normalize_request(
        self, request: ConceptReservationRequest
    ) -> ConceptReservationRequest:
        """Bind caller input to the authority-owned policy version/range."""

        self._check_policy(request)
        matches = [
            policy for policy in self._policies if policy.accepts(request.concept_id)
        ]
        if len(matches) != 1:
            raise ConceptReservationError(
                "authority policy selection is ambiguous for the concept id"
            )
        policy = matches[0]
        if (
            request.namespace != policy.namespace
            or (
                request.range_start is not None
                and request.range_start != policy.range_start
            )
            or (request.range_end is not None and request.range_end != policy.range_end)
            or (
                request.policy_version
                and request.policy_version != policy.policy_version
            )
        ):
            raise ConceptReservationConflict(
                "request namespace/range/policy differs from authority policy"
            )
        return replace(
            request,
            namespace=policy.namespace,
            range_start=policy.range_start,
            range_end=policy.range_end,
            policy_version=policy.policy_version,
        )

    def _record_properties(self, record: ConceptReservationRecord) -> dict[str, Any]:
        value = record.to_wire()
        value["node_type"] = RESERVATION_NODE_LABEL
        value["immutable_fingerprint"] = record.immutable_fingerprint
        return value

    def _parse_properties(self, value: Mapping[str, Any]) -> ConceptReservationRecord:
        if value.get("node_type") != RESERVATION_NODE_LABEL:
            raise AuthorityUnavailable("native reservation node has an invalid label")
        try:
            record = ConceptReservationRecord.from_wire(value)
            if value.get("immutable_fingerprint") != record.immutable_fingerprint:
                raise ConceptReservationError(
                    "native reservation immutable fingerprint is inconsistent"
                )
            return record
        except (ConceptReservationError, TypeError, ValueError) as exc:
            raise AuthorityUnavailable("native reservation node is malformed") from exc

    def _load_node(self, node_id: str) -> ConceptReservationRecord | None:
        value = self._call("get_node_properties", node_id)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise AuthorityUnavailable(
                "native reservation node properties are malformed"
            )
        record = self._parse_properties(value)
        if node_id != _reservation_node_id(record.concept_id):
            raise AuthorityUnavailable(
                "native reservation node identity is inconsistent"
            )
        return record

    def _list_nodes(
        self, *, limit: int, after: str | None
    ) -> Sequence[tuple[str, Mapping[str, Any]]]:
        if not 1 <= limit <= _MAX_LIST_LIMIT:
            raise ConceptReservationError("native label page limit is invalid")
        after = _validate_cursor(after, "native cursor")
        if after is None:
            try:
                value = self._call(
                    "list_nodes_by_label", RESERVATION_NODE_LABEL, limit, after=None
                )
            except AuthorityUnavailable:
                # Older AU GraphComputeEngine wrappers expose a bounded label
                # read without a cursor.  Keep the first page usable, but do
                # not pretend it can paginate a large authority.
                for fallback in ("get_nodes_by_label", "nodes_by_label"):
                    try:
                        value = self._call(fallback, RESERVATION_NODE_LABEL, limit)
                        break
                    except AuthorityUnavailable:
                        continue
                else:
                    raise AuthorityUnavailable(
                        "native graph has no bounded reservation label read"
                    )
        else:
            value = self._call(
                "list_nodes_by_label", RESERVATION_NODE_LABEL, limit, after=after
            )
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise AuthorityUnavailable("native reservation label page is malformed")
        if len(value) > limit:
            raise AuthorityUnavailable(
                "native reservation label page exceeded its limit"
            )
        rows: list[tuple[str, Mapping[str, Any]]] = []
        previous_node_id: str | None = None
        for row in value:
            if not isinstance(row, Sequence) or len(row) != 2:
                raise AuthorityUnavailable("native reservation label row is malformed")
            node_id, props = row
            if not isinstance(node_id, str) or not isinstance(props, Mapping):
                raise AuthorityUnavailable("native reservation label row is malformed")
            try:
                _validate_cursor(node_id, "native node id")
            except ConceptReservationError as exc:
                raise AuthorityUnavailable(
                    "native reservation node id is malformed"
                ) from exc
            if previous_node_id is not None and node_id <= previous_node_id:
                raise AuthorityUnavailable(
                    "native reservation label page is not strictly ordered"
                )
            previous_node_id = node_id
            rows.append((node_id, props))
        return rows

    def _find_reservation(
        self, reservation_id: str, tenant_ref: str
    ) -> tuple[str, ConceptReservationRecord]:
        cursor: str | None = None
        seen_cursors: set[str] = set()
        scanned = 0
        for _page_number in range(_MAX_READ_PAGES):
            rows = self._list_nodes(limit=_MAX_LIST_LIMIT, after=cursor)
            scanned += len(rows)
            if scanned > _MAX_READ_RECORDS:
                raise AuthorityUnavailable(
                    "native reservation lookup exceeded its record bound"
                )
            for node_id, props in rows:
                record = self._parse_properties(props)
                if node_id != _reservation_node_id(record.concept_id):
                    raise AuthorityUnavailable(
                        "native reservation node identity is inconsistent"
                    )
                if record.reservation_id != reservation_id:
                    continue
                if record.tenant_ref != tenant_ref:
                    raise ConceptReservationUnauthorized(
                        "concept reservation belongs to another tenant"
                    )
                return node_id, record
            if len(rows) < _MAX_LIST_LIMIT:
                break
            next_cursor = _validate_cursor(rows[-1][0], "native cursor")
            if (
                next_cursor is None
                or (cursor is not None and next_cursor <= cursor)
                or next_cursor in seen_cursors
            ):
                raise AuthorityUnavailable(
                    "native reservation lookup cursor did not advance"
                )
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        else:
            raise AuthorityUnavailable(
                "native reservation lookup exceeded its page bound"
            )
        raise ConceptReservationNotFound("concept reservation was not found")

    def reserve(self, request: ConceptReservationRequest) -> ConceptReservationRecord:
        request = self._normalize_request(request)
        node_id = _reservation_node_id(request.concept_id)
        candidate = _new_record(request)
        created = self._call(
            "create_node_if_absent", node_id, self._record_properties(candidate)
        )
        if created:
            return candidate
        for _attempt in range(32):
            current = self._load_node(node_id)
            if current is None:
                raise AuthorityUnavailable(
                    "native create lost but its reservation node is not readable"
                )
            if current.request.request_key_ref == request.request_key_ref:
                if current.immutable_fingerprint != request.immutable_fingerprint:
                    raise ConceptReservationConflict(
                        "request key was reused with different immutable reservation input"
                    )
                return current
            if current.state not in {
                ConceptReservationState.RELEASED,
                ConceptReservationState.EXPIRED,
            }:
                raise ConceptReservationIdUnavailable(
                    f"concept id is already reserved: {request.concept_id}"
                )
            if (
                _VISIBILITY_RANK[current.visibility]
                >= _VISIBILITY_RANK[ConceptReservationVisibility.REPOSITORY]
            ):
                raise ConceptReservationIdUnavailable(
                    f"externally visible concept id is never reusable: {request.concept_id}"
                )

            # Reclaim is a fenced CAS on the one canonical concept node. A
            # second caller may win between the point read and this CAS; the
            # bounded loop rereads and classifies that winner.
            reclaimed = replace(candidate, fence=current.fence + 1)
            if self._call(
                "compare_and_set_node_fields",
                node_id,
                {
                    "node_type": RESERVATION_NODE_LABEL,
                    "reservation_id": current.reservation_id,
                    "state": current.state.value,
                    "fence": current.fence,
                    "immutable_fingerprint": current.immutable_fingerprint,
                },
                self._record_properties(reclaimed),
            ):
                return reclaimed
        raise AuthorityUnavailable(
            "native concept reservation contention exceeded retry bound"
        )

    def get(self, reservation_id: str, *, tenant_ref: str) -> ConceptReservationRecord:
        reservation = _reference(reservation_id, "reservation_id")
        tenant = _reference(tenant_ref, "tenant_ref")
        _node_id, record = self._find_reservation(reservation, tenant)
        return record

    def list(
        self,
        *,
        tenant_ref: str,
        namespace: str | None = None,
        state: ConceptReservationState | None = None,
        concept_prefix: str | None = None,
        limit: int = _MAX_LIST_LIMIT,
        cursor: str | None = None,
    ) -> tuple[list[ConceptReservationRecord], str | None]:
        if not 1 <= limit <= _MAX_LIST_LIMIT:
            raise ConceptReservationError("limit is outside the bounded range")
        tenant = _reference(tenant_ref, "tenant_ref")
        native_cursor = _validate_cursor(cursor)
        page_limit = min(_MAX_LIST_LIMIT, max(limit * 2, 32))
        if cursor == "":
            raise ConceptReservationError("cursor is invalid")
        rows_out: list[ConceptReservationRecord] = []
        seen_cursors: set[str] = set()
        scanned = 0
        for _page_number in range(_MAX_READ_PAGES):
            page = self._list_nodes(limit=page_limit, after=native_cursor)
            scanned += len(page)
            if scanned > _MAX_READ_RECORDS:
                raise AuthorityUnavailable(
                    "native reservation list exceeded its record bound"
                )
            last_scanned: str | None = None
            for node_id, props in page:
                last_scanned = node_id
                record = self._parse_properties(props)
                if record.tenant_ref != tenant:
                    continue
                if namespace and record.request.namespace != namespace:
                    continue
                if state and record.state is not state:
                    continue
                if concept_prefix and not record.concept_id.startswith(concept_prefix):
                    continue
                rows_out.append(record)
                if len(rows_out) == limit:
                    break
            if len(rows_out) == limit:
                # The page is intentionally wider than the requested filtered
                # result.  Advance only past the last node actually inspected;
                # using page[-1] here would silently skip uninspected matches.
                if len(page) < page_limit and last_scanned == page[-1][0]:
                    return rows_out, None
                next_cursor = _validate_cursor(last_scanned, "native cursor")
            elif len(page) < page_limit:
                return rows_out, None
            else:
                next_cursor = _validate_cursor(page[-1][0], "native cursor")
            if (
                next_cursor is None
                or (native_cursor is not None and next_cursor <= native_cursor)
                or next_cursor in seen_cursors
            ):
                raise AuthorityUnavailable(
                    "native reservation list cursor did not advance"
                )
            seen_cursors.add(next_cursor)
            native_cursor = next_cursor
            if len(rows_out) == limit:
                return rows_out, native_cursor
        raise AuthorityUnavailable("native reservation list exceeded its page bound")

    def transition(
        self,
        reservation_id: str,
        *,
        tenant_ref: str,
        owner_ref: str,
        expected_fence: int,
        target: ConceptReservationState,
        visibility: ConceptReservationVisibility | None = None,
    ) -> ConceptReservationRecord:
        if expected_fence < 1:
            raise ConceptReservationFenceConflict("expected_fence must be positive")
        reservation = _reference(reservation_id, "reservation_id")
        tenant = _reference(tenant_ref, "tenant_ref")
        owner = _reference(owner_ref, "owner_ref")
        node_id, current = self._find_reservation(reservation, tenant)
        requested_visibility = _strongest_visibility(
            current.visibility, visibility or current.visibility
        )
        if (
            target is not ConceptReservationState.TOMBSTONED
            and _VISIBILITY_RANK[requested_visibility]
            >= _VISIBILITY_RANK[ConceptReservationVisibility.REPOSITORY]
        ):
            # Visibility is monotonic. A release/expiry request cannot erase
            # repository/external evidence; preserve it as a tombstone.
            target = ConceptReservationState.TOMBSTONED
        # A retry may arrive with the pre-CAS fence after another caller has
        # already committed this exact transition.  Check that idempotent case
        # before rejecting the now-stale expected fence, but never let another
        # owner observe a successful retry.
        if (
            current.owner_ref == owner
            and current.state is target
            and current.fence == expected_fence + 1
        ):
            return current
        if current.owner_ref != owner or current.fence != expected_fence:
            raise ConceptReservationFenceConflict("reservation owner or fence is stale")
        if (
            target is ConceptReservationState.EXPIRED
            and datetime.now(UTC) < current.expires_at
        ):
            raise ConceptReservationConflict("reservation has not reached its expiry")
        allowed = _transition_map().get(current.state, set())
        if target not in allowed:
            raise ConceptReservationConflict(
                f"cannot transition {current.state.value} to {target.value}"
            )
        now = max(current.transitioned_at, datetime.now(UTC))
        next_visibility = requested_visibility
        if target is ConceptReservationState.MATERIALIZED:
            next_visibility = _strongest_visibility(
                next_visibility, ConceptReservationVisibility.FRAGMENT
            )
        elif target is ConceptReservationState.LANDED:
            next_visibility = _strongest_visibility(
                next_visibility, ConceptReservationVisibility.REPOSITORY
            )
        elif target is ConceptReservationState.TOMBSTONED:
            next_visibility = ConceptReservationVisibility.EXTERNAL
        next_record = replace(
            current,
            state=target,
            visibility=next_visibility,
            fence=current.fence + 1,
            transitioned_at=now,
            materialized_at=(
                now
                if target is ConceptReservationState.MATERIALIZED
                else current.materialized_at
            ),
            landed_at=(
                now if target is ConceptReservationState.LANDED else current.landed_at
            ),
            released_at=(
                now
                if target is ConceptReservationState.RELEASED
                else current.released_at
            ),
            expired_at=(
                now if target is ConceptReservationState.EXPIRED else current.expired_at
            ),
            tombstoned_at=(
                now
                if target is ConceptReservationState.TOMBSTONED
                else current.tombstoned_at
            ),
        )
        applied = self._call(
            "compare_and_set_node_fields",
            node_id,
            {
                "node_type": RESERVATION_NODE_LABEL,
                "reservation_id": reservation,
                "tenant_ref": tenant,
                "owner_ref": owner,
                "state": current.state.value,
                "fence": expected_fence,
                "immutable_fingerprint": current.immutable_fingerprint,
            },
            self._record_properties(next_record),
        )
        if applied:
            return next_record
        _node_id, latest = self._find_reservation(reservation, tenant)
        if (
            latest.owner_ref == owner
            and latest.state is target
            and latest.fence == expected_fence + 1
        ):
            return latest
        raise ConceptReservationFenceConflict("reservation owner or fence is stale")


class FixtureConceptReservationAuthority:
    """Thread-safe policy fixture; never a production/global authority.

    The lock makes this object deterministic for unit tests.  It does not make
    independent processes or hosts coordinate, which is why ``authoritative``
    is explicitly false and the native adapter refuses to fall back to it.
    """

    authoritative = False

    def __init__(self, policies: Sequence[ConceptNamespacePolicy] = ()) -> None:
        self._policies = _validate_policy_set(policies)
        self._records: dict[str, ConceptReservationRecord] = {}
        self._by_request: dict[tuple[str, str, str], str] = {}
        self._lock = threading.RLock()

    def _check_policy(self, request: ConceptReservationRequest) -> None:
        if not self._policies:
            return
        if not any(policy.accepts(request.concept_id) for policy in self._policies):
            raise ConceptReservationError(
                f"concept id {request.concept_id!r} is outside the namespace/range policy"
            )

    def _normalize_request(
        self, request: ConceptReservationRequest
    ) -> ConceptReservationRequest:
        self._check_policy(request)
        if not self._policies:
            return request
        matches = [
            policy for policy in self._policies if policy.accepts(request.concept_id)
        ]
        if len(matches) != 1:
            raise ConceptReservationError(
                "authority policy selection is ambiguous for the concept id"
            )
        policy = matches[0]
        if (
            request.namespace != policy.namespace
            or (
                request.range_start is not None
                and request.range_start != policy.range_start
            )
            or (request.range_end is not None and request.range_end != policy.range_end)
            or (
                request.policy_version
                and request.policy_version != policy.policy_version
            )
        ):
            raise ConceptReservationConflict(
                "request namespace/range/policy differs from authority policy"
            )
        return replace(
            request,
            namespace=policy.namespace,
            range_start=policy.range_start,
            range_end=policy.range_end,
            policy_version=policy.policy_version,
        )

    def _require(
        self, reservation_id: str, tenant_ref: str
    ) -> ConceptReservationRecord:
        record = self._records.get(reservation_id)
        if record is None:
            raise ConceptReservationNotFound("concept reservation was not found")
        if record.tenant_ref != tenant_ref:
            raise ConceptReservationUnauthorized(
                "concept reservation belongs to another tenant"
            )
        return record

    def reserve(self, request: ConceptReservationRequest) -> ConceptReservationRecord:
        with self._lock:
            request = self._normalize_request(request)
            # Idempotency is scoped to the same canonical concept node.  Do
            # not imply a globally atomic request-key index that this fixture
            # or the native graph protocol does not maintain.
            request_key = (
                request.tenant_ref,
                request.request_key_ref,
                request.concept_id,
            )
            previous_id = self._by_request.get(request_key)
            if previous_id:
                previous = self._records[previous_id]
                if previous.immutable_fingerprint != request.immutable_fingerprint:
                    raise ConceptReservationConflict(
                        "request key was reused with different immutable reservation input"
                    )
                return previous
            for previous in self._records.values():
                if previous.concept_id == request.concept_id and previous.state not in {
                    ConceptReservationState.RELEASED,
                    ConceptReservationState.EXPIRED,
                }:
                    raise ConceptReservationIdUnavailable(
                        f"concept id is already reserved: {request.concept_id}"
                    )
            now = request.created_at
            record = ConceptReservationRecord(
                reservation_id=persistence_reference(
                    "concept_reservation",
                    f"{request.tenant_ref}:{request.request_key_ref}:{request.concept_id}",
                ),
                request=request,
                state=ConceptReservationState.RESERVED,
                visibility=ConceptReservationVisibility.PRIVATE,
                fence=1,
                created_at=now,
                expires_at=request.expires_at,
                transitioned_at=now,
            )
            self._records[record.reservation_id] = record
            self._by_request[request_key] = record.reservation_id
            return record

    def get(self, reservation_id: str, *, tenant_ref: str) -> ConceptReservationRecord:
        with self._lock:
            return self._require(
                _reference(reservation_id, "reservation_id"),
                _reference(tenant_ref, "tenant_ref"),
            )

    def list(
        self,
        *,
        tenant_ref: str,
        namespace: str | None = None,
        state: ConceptReservationState | None = None,
        concept_prefix: str | None = None,
        limit: int = _MAX_LIST_LIMIT,
        cursor: str | None = None,
    ) -> tuple[list[ConceptReservationRecord], str | None]:
        if not 1 <= limit <= _MAX_LIST_LIMIT:
            raise ConceptReservationError("limit is outside the bounded range")
        if cursor:
            try:
                offset = int(cursor)
            except ValueError as exc:
                raise ConceptReservationError("cursor is invalid") from exc
            if offset < 0:
                raise ConceptReservationError("cursor is invalid")
        else:
            offset = 0
        tenant = _reference(tenant_ref, "tenant_ref")
        with self._lock:
            rows = [
                record
                for record in self._records.values()
                if record.tenant_ref == tenant
            ]
            if namespace:
                rows = [
                    record for record in rows if record.request.namespace == namespace
                ]
            if state:
                rows = [record for record in rows if record.state is state]
            if concept_prefix:
                rows = [
                    record
                    for record in rows
                    if record.concept_id.startswith(concept_prefix)
                ]
            rows.sort(key=lambda record: record.reservation_id)
            page = rows[offset : offset + limit]
            next_cursor = str(offset + limit) if offset + limit < len(rows) else None
            return page, next_cursor

    def transition(
        self,
        reservation_id: str,
        *,
        tenant_ref: str,
        owner_ref: str,
        expected_fence: int,
        target: ConceptReservationState,
        visibility: ConceptReservationVisibility | None = None,
    ) -> ConceptReservationRecord:
        with self._lock:
            record = self._require(reservation_id, tenant_ref)
            owner = _reference(owner_ref, "owner_ref")
            requested_visibility = _strongest_visibility(
                record.visibility, visibility or record.visibility
            )
            if (
                target is not ConceptReservationState.TOMBSTONED
                and _VISIBILITY_RANK[requested_visibility]
                >= _VISIBILITY_RANK[ConceptReservationVisibility.REPOSITORY]
            ):
                target = ConceptReservationState.TOMBSTONED
            # Match native retry semantics: an exact same-owner transition is
            # idempotent when the caller presents the immediately prior fence.
            if (
                record.owner_ref == owner
                and record.state is target
                and record.fence == expected_fence + 1
            ):
                return record
            if record.owner_ref != owner or record.fence != expected_fence:
                raise ConceptReservationFenceConflict(
                    "reservation owner or fence is stale"
                )
            if (
                target is ConceptReservationState.EXPIRED
                and datetime.now(UTC) < record.expires_at
            ):
                raise ConceptReservationConflict(
                    "reservation has not reached its expiry"
                )
            if target not in _transition_map().get(record.state, set()):
                raise ConceptReservationConflict(
                    f"cannot transition {record.state.value} to {target.value}"
                )
            now = max(record.transitioned_at, datetime.now(UTC))
            next_visibility = requested_visibility
            if target is ConceptReservationState.MATERIALIZED:
                next_visibility = _strongest_visibility(
                    next_visibility, ConceptReservationVisibility.FRAGMENT
                )
            elif target is ConceptReservationState.LANDED:
                next_visibility = _strongest_visibility(
                    next_visibility, ConceptReservationVisibility.REPOSITORY
                )
            elif target is ConceptReservationState.TOMBSTONED:
                next_visibility = ConceptReservationVisibility.EXTERNAL
            return self._replace(
                record,
                state=target,
                visibility=next_visibility,
                fence=record.fence + 1,
                transitioned_at=now,
                materialized_at=now
                if target is ConceptReservationState.MATERIALIZED
                else record.materialized_at,
                landed_at=now
                if target is ConceptReservationState.LANDED
                else record.landed_at,
                released_at=now
                if target is ConceptReservationState.RELEASED
                else record.released_at,
                expired_at=now
                if target is ConceptReservationState.EXPIRED
                else record.expired_at,
                tombstoned_at=now
                if target is ConceptReservationState.TOMBSTONED
                else record.tombstoned_at,
            )

    def _replace(
        self, record: ConceptReservationRecord, **changes: Any
    ) -> ConceptReservationRecord:
        next_record = replace(record, **changes)
        self._records[next_record.reservation_id] = next_record
        return next_record


class ConceptReservationService:
    """Lifecycle service combining native authority and local projection."""

    def __init__(
        self, authority: ConceptReservationAuthority, *, repo_root: Any = None
    ) -> None:
        self.authority = authority
        self.repo_root = repo_root

    def reserve(self, request: ConceptReservationRequest) -> ConceptReservationRecord:
        return self.authority.reserve(request)

    def reserve_next_numeric(
        self,
        request: ConceptReservationRequest,
        *,
        concept_prefix: str,
        range_start: int,
        range_end: int,
    ) -> ConceptReservationRecord:
        return reserve_next_numeric(
            self.authority,
            request,
            concept_prefix=concept_prefix,
            range_start=range_start,
            range_end=range_end,
        )

    def materialize(
        self,
        reservation_id: str,
        *,
        tenant_ref: str,
        owner_ref: str,
        expected_fence: int,
    ) -> ConceptReservationRecord:
        current = self.authority.get(reservation_id, tenant_ref=tenant_ref)
        if current.owner_ref != _reference(owner_ref, "owner_ref"):
            raise ConceptReservationUnauthorized(
                "concept reservation belongs to another owner"
            )
        if current.state is ConceptReservationState.MATERIALIZED:
            record = current
        elif current.state is ConceptReservationState.RESERVED:
            record = self.authority.transition(
                reservation_id,
                tenant_ref=tenant_ref,
                owner_ref=owner_ref,
                expected_fence=expected_fence,
                target=ConceptReservationState.MATERIALIZED,
            )
        else:
            raise ConceptReservationConflict(
                f"cannot materialize a {current.state.value} reservation"
            )
        from agent_utilities.governance.concept_allocator import (
            materialize_authoritative_record,
        )

        materialize_authoritative_record(record.to_wire(), repo_root=self.repo_root)
        return record

    def transition(
        self,
        reservation_id: str,
        *,
        tenant_ref: str,
        owner_ref: str,
        expected_fence: int,
        target: ConceptReservationState,
        visibility: ConceptReservationVisibility | None = None,
    ) -> ConceptReservationRecord:
        return self.authority.transition(
            reservation_id,
            tenant_ref=tenant_ref,
            owner_ref=owner_ref,
            expected_fence=expected_fence,
            target=target,
            visibility=visibility,
        )

    def query(
        self,
        *,
        tenant_ref: str,
        namespace: str | None = None,
        state: ConceptReservationState | None = None,
        concept_prefix: str | None = None,
        limit: int = _MAX_LIST_LIMIT,
        cursor: str | None = None,
    ) -> tuple[list[ConceptReservationRecord], str | None]:
        return self.authority.list(
            tenant_ref=tenant_ref,
            namespace=namespace,
            state=state,
            concept_prefix=concept_prefix,
            limit=limit,
            cursor=cursor,
        )


@dataclass(frozen=True, slots=True)
class ProjectionReconciliation:
    """Read-only comparison between authority claims and local code/ledger."""

    matches: tuple[str, ...] = ()
    missing_projection: tuple[str, ...] = ()
    orphan_projection: tuple[str, ...] = ()
    state_mismatch: tuple[str, ...] = ()
    marker_without_claim: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "matches": list(self.matches),
            "missing_projection": list(self.missing_projection),
            "orphan_projection": list(self.orphan_projection),
            "state_mismatch": list(self.state_mismatch),
            "marker_without_claim": list(self.marker_without_claim),
        }


def reconcile_projection(
    authority: ConceptReservationAuthority,
    *,
    tenant_ref: str,
    repo_root: Any,
    max_records: int = 10_000,
) -> ProjectionReconciliation:
    """Compare the native view with local projection and source usage; never write.

    ``max_records`` is a hard safety bound for a corrupted or unexpectedly
    huge authority.  Cursor repetition and a page that exceeds the bound fail
    closed instead of spinning forever or materializing an unbounded response.
    """

    if not 1 <= max_records <= 1_000_000:
        raise ConceptReservationError("max_records is outside the bounded range")

    from agent_utilities.governance import concept_allocator as allocator

    rows: list[ConceptReservationRecord] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        page_limit = min(_MAX_LIST_LIMIT, max_records - len(rows))
        if page_limit <= 0:
            raise AuthorityUnavailable("concept reconciliation exceeded max_records")
        page, cursor = authority.list(
            tenant_ref=tenant_ref, limit=page_limit, cursor=cursor
        )
        rows.extend(page)
        if len(rows) > max_records:
            raise AuthorityUnavailable("concept reconciliation exceeded max_records")
        if cursor is None:
            break
        if cursor in seen_cursors:
            raise AuthorityUnavailable("concept reconciliation cursor did not advance")
        seen_cursors.add(cursor)
    central = {row.concept_id: row for row in rows}
    local_rows = allocator.read_ledger(repo_root)
    local = {str(row["id"]): row for row in local_rows}
    code = set(allocator.scan_code_markers(allocator._default_scan_roots(repo_root)))
    matches: list[str] = []
    missing: list[str] = []
    mismatch: list[str] = []
    for concept_id, row in central.items():
        projection = local.get(concept_id)
        if projection is None:
            missing.append(concept_id)
            continue
        expected = {
            ConceptReservationState.RESERVED: {"reserved", "materialized"},
            ConceptReservationState.MATERIALIZED: {"materialized", "reserved"},
            ConceptReservationState.LANDED: {"landed"},
            ConceptReservationState.TOMBSTONED: {"tombstoned", "landed"},
            ConceptReservationState.RELEASED: {"released", "expired"},
            ConceptReservationState.EXPIRED: {"expired"},
        }[row.state]
        if projection.get("status") not in expected:
            mismatch.append(concept_id)
        elif concept_id in code and row.state not in {
            ConceptReservationState.LANDED,
            ConceptReservationState.TOMBSTONED,
        }:
            # Source visibility has advanced beyond the authority's lifecycle;
            # report it for a fenced transition rather than silently landing a
            # claim during this read-only comparison.
            mismatch.append(concept_id)
        else:
            matches.append(concept_id)
    orphan = sorted(set(local) - set(central))
    marker_without_claim = sorted(code - set(central))
    return ProjectionReconciliation(
        matches=tuple(sorted(matches)),
        missing_projection=tuple(sorted(missing)),
        orphan_projection=tuple(orphan),
        state_mismatch=tuple(sorted(mismatch)),
        marker_without_claim=tuple(marker_without_claim),
    )
