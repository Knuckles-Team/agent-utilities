"""Transport-neutral authority for one served MCP catalog generation.

The multiplexer remains the sole protocol I/O adapter.  This module owns only
the immutable, authorization-partitioned state which must be identical for MCP,
REST, configuration reconciliation, dispatch, reconnect, and readiness:

* one atomic snapshot of tools, resources, resource templates, and prompts;
* monotonic generation and content digest transitions;
* exact refresh and session-resume wire contracts;
* pending list-change high-watermarks; and
* fail-closed stable-dispatch and replica-cohort identity checks.

It deliberately imports neither FastMCP nor a child transport.  Persisted fleet
catalog tables remain a downstream projection, not a second live authority.

CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,256}$"


class _ContractModel(BaseModel):
    """Strict, immutable base for catalog protocol contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CatalogRefreshRequest(_ContractModel):
    """``mcp-catalog-refresh-request/v1`` shared by MCP and REST."""

    request_id: str = Field(pattern=_ID_PATTERN)
    expected_config_revision: str = Field(min_length=1, max_length=128)
    expected_catalog_generation: int = Field(ge=0)
    expected_snapshot_digest: str = Field(pattern=_DIGEST_PATTERN)
    deadline_ms: int = Field(ge=1, le=120_000)


class CatalogRefreshResult(_ContractModel):
    """``mcp-catalog-refresh-result/v1`` shared by MCP and REST."""

    request_id: str = Field(pattern=_ID_PATTERN)
    served_instance_id: str = Field(pattern=_ID_PATTERN)
    release_id: str = Field(min_length=1, max_length=128)
    config_revision: str = Field(min_length=1, max_length=128)
    catalog_generation: int = Field(ge=0)
    snapshot_digest: str = Field(pattern=_DIGEST_PATTERN)
    changed: bool
    pending_list_change_generation: int | None = Field(default=None, ge=1)
    reingestion_state: Literal["reconciled", "unavailable"]
    reconciliation_receipt_digest: str | None = Field(
        default=None, pattern=_DIGEST_PATTERN
    )


CatalogRefreshErrorCode = Literal[
    "authorization-denied",
    "refresh-request-schema-mismatch",
    "catalog-generation-stale",
    "catalog-digest-mismatch",
    "catalog-snapshot-incomplete",
    "refresh-deadline-exceeded",
    "reingestion-unreconciled",
    "replica-generation-divergent",
]


class CatalogRefreshError(_ContractModel):
    """``mcp-catalog-refresh-error/v1`` shared by MCP and REST."""

    request_id: str = Field(pattern=_ID_PATTERN)
    code: CatalogRefreshErrorCode
    retryable: bool
    catalog_generation: int = Field(ge=0)
    snapshot_digest: str = Field(pattern=_DIGEST_PATTERN)
    details_digest: str = Field(pattern=_DIGEST_PATTERN)


class CatalogSessionResumeRequest(_ContractModel):
    """``mcp-catalog-session-resume-request/v1`` shared by MCP and REST."""

    session_id: str = Field(pattern=_ID_PATTERN)
    previous_served_instance_id: str = Field(pattern=_ID_PATTERN)
    release_id: str = Field(min_length=1, max_length=128)
    config_revision: str = Field(min_length=1, max_length=128)
    catalog_generation: int = Field(ge=0)
    snapshot_digest: str = Field(pattern=_DIGEST_PATTERN)
    child_connection_generation: int = Field(ge=0)
    authorization_scope_digest: str = Field(pattern=_DIGEST_PATTERN)
    resume_token_digest: str = Field(pattern=_DIGEST_PATTERN)
    deadline_ms: int = Field(ge=1, le=120_000)


class CatalogSessionResumeResult(_ContractModel):
    """``mcp-catalog-session-resume-result/v1`` shared by MCP and REST."""

    session_id: str = Field(pattern=_ID_PATTERN)
    served_instance_id: str = Field(pattern=_ID_PATTERN)
    release_id: str = Field(min_length=1, max_length=128)
    config_revision: str = Field(min_length=1, max_length=128)
    catalog_generation: int = Field(ge=0)
    snapshot_digest: str = Field(pattern=_DIGEST_PATTERN)
    child_connection_generation: int = Field(ge=0)
    authorization_scope_digest: str = Field(pattern=_DIGEST_PATTERN)
    resume_state: Literal["resumed", "relist-required", "rebound-relist-required"]


CatalogSessionResumeErrorCode = Literal[
    "session-resume-authentication-required",
    "session-resume-authorization-changed",
    "session-resume-release-mismatch",
    "session-resume-config-revision-mismatch",
    "session-resume-catalog-generation-mismatch",
    "session-resume-snapshot-digest-mismatch",
    "child-connection-generation-mismatch",
    "session-resume-token-invalid",
    "replica-generation-divergent",
]


class CatalogSessionResumeError(_ContractModel):
    """``mcp-catalog-session-resume-error/v1`` shared by MCP and REST."""

    session_id: str = Field(pattern=_ID_PATTERN)
    code: CatalogSessionResumeErrorCode
    retryable: bool
    required_action: Literal["reauthenticate", "relist", "reconnect"]
    catalog_generation: int = Field(ge=0)
    snapshot_digest: str = Field(pattern=_DIGEST_PATTERN)


class CatalogIdentity(_ContractModel):
    """Identity carried by every discovery and invocation surface."""

    served_instance_id: str = Field(pattern=_ID_PATTERN)
    release_id: str = Field(min_length=1, max_length=128)
    config_revision: str = Field(min_length=1, max_length=128)
    catalog_generation: int = Field(ge=0)
    snapshot_digest: str = Field(pattern=_DIGEST_PATTERN)
    child_connection_generation: int = Field(ge=0)
    authorization_scope_digest: str = Field(pattern=_DIGEST_PATTERN)


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One canonical protocol descriptor stored as immutable JSON bytes."""

    key: str
    schema_digest: str
    canonical_json: str

    @classmethod
    def from_mapping(cls, key: str, value: Mapping[str, Any]) -> CatalogEntry:
        canonical = _canonical_json(value)
        return cls(
            key=key,
            schema_digest=_digest_text(canonical),
            canonical_json=canonical,
        )

    def as_dict(self) -> dict[str, Any]:
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):  # pragma: no cover - construction invariant
            raise RuntimeError("catalog entry is not an object")
        return value


@dataclass(frozen=True, slots=True)
class ChildCatalogCandidate:
    """One child's complete, bounded four-family protocol candidate."""

    server_name: str
    child_connection_generation: int
    tools: tuple[CatalogEntry, ...]
    resources: tuple[CatalogEntry, ...]
    resource_templates: tuple[CatalogEntry, ...]
    prompts: tuple[CatalogEntry, ...]

    @classmethod
    def build(
        cls,
        *,
        server_name: str,
        child_connection_generation: int,
        tools: Iterable[Mapping[str, Any]],
        resources: Iterable[Mapping[str, Any]],
        resource_templates: Iterable[Mapping[str, Any]],
        prompts: Iterable[Mapping[str, Any]],
    ) -> ChildCatalogCandidate:
        if not server_name or child_connection_generation < 0:
            raise ValueError("catalog child identity is invalid")
        return cls(
            server_name=server_name,
            child_connection_generation=child_connection_generation,
            tools=_entries("name", tools),
            resources=_entries("uri", resources),
            resource_templates=_entries("uriTemplate", resource_templates),
            prompts=_entries("name", prompts),
        )

    def canonical(self) -> dict[str, Any]:
        families: dict[str, list[str]] = {}
        for family_name in ("tools", "resources", "resource_templates", "prompts"):
            entries = getattr(self, family_name)
            families[family_name] = [entry.canonical_json for entry in entries]
        return {
            "server_name": self.server_name,
            "child_connection_generation": self.child_connection_generation,
            **families,
        }


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """One immutable authorization-scoped served catalog snapshot."""

    identity: CatalogIdentity
    children: tuple[ChildCatalogCandidate, ...]

    def tool(
        self, public_name: str
    ) -> tuple[ChildCatalogCandidate, CatalogEntry] | None:
        for child in self.children:
            match = next(
                (entry for entry in child.tools if entry.key == public_name), None
            )
            if match is not None:
                return child, match
        return None


@dataclass(frozen=True, slots=True)
class SessionBinding:
    """Server-minted resume facts for one authenticated client session."""

    identity: CatalogIdentity
    resume_token_digest: str


class CatalogContractError(RuntimeError):
    """Fail-closed catalog contract violation with a closed error code."""

    def __init__(self, code: str, detail: str, *, retryable: bool = False):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.retryable = retryable


class McpCatalogReconciler:
    """The one lower state authority owned by a served multiplexer instance."""

    def __init__(self, *, release_id: str, served_instance_id: str | None = None):
        if not release_id:
            raise ValueError("release_id is required")
        self.release_id = release_id
        self.served_instance_id = served_instance_id or f"graph-os:{uuid.uuid4().hex}"
        self._lock = threading.RLock()
        self._generation = 0
        self._notification_generation = 0
        self._snapshots: dict[str, CatalogSnapshot] = {}
        self._pending: dict[str, int] = {}
        self._sessions: dict[str, SessionBinding] = {}

    def current(
        self, authorization_scope_digest: str, *, config_revision: str
    ) -> CatalogSnapshot:
        """Return the current scope snapshot, creating the immutable empty state."""
        with self._lock:
            current = self._snapshots.get(authorization_scope_digest)
            if current is not None:
                return current
            return self._empty_snapshot(authorization_scope_digest, config_revision)

    def candidate(
        self,
        *,
        authorization_scope_digest: str,
        config_revision: str,
        children: Iterable[ChildCatalogCandidate],
    ) -> CatalogSnapshot:
        """Build, validate, and digest a candidate without publishing it."""
        ordered = tuple(sorted(children, key=lambda child: child.server_name))
        if len({child.server_name for child in ordered}) != len(ordered):
            raise CatalogContractError(
                "catalog-snapshot-incomplete", "duplicate child identity"
            )
        payload = {
            "release_id": self.release_id,
            "config_revision": config_revision,
            "authorization_scope_digest": authorization_scope_digest,
            "children": [child.canonical() for child in ordered],
        }
        digest = _digest_text(_canonical_json(payload))
        with self._lock:
            current = self._snapshots.get(authorization_scope_digest)
            generation = (
                current.identity.catalog_generation
                if current is not None and current.identity.snapshot_digest == digest
                else self._generation + 1
            )
        child_generation = max(
            (child.child_connection_generation for child in ordered), default=0
        )
        identity = CatalogIdentity(
            served_instance_id=self.served_instance_id,
            release_id=self.release_id,
            config_revision=config_revision,
            catalog_generation=generation,
            snapshot_digest=digest,
            child_connection_generation=child_generation,
            authorization_scope_digest=authorization_scope_digest,
        )
        return CatalogSnapshot(identity=identity, children=ordered)

    def publish(
        self, candidate: CatalogSnapshot, *, affected_session_ids: Iterable[str]
    ) -> tuple[CatalogSnapshot, bool]:
        """Atomically publish one validated snapshot and queue its high-watermark."""
        scope = candidate.identity.authorization_scope_digest
        with self._lock:
            previous = self._snapshots.get(scope)
            changed = previous is None or (
                previous.identity.snapshot_digest != candidate.identity.snapshot_digest
            )
            if changed:
                expected = self._generation + 1
                if candidate.identity.catalog_generation != expected:
                    raise CatalogContractError(
                        "catalog-generation-stale", "candidate generation is stale"
                    )
                self._generation = expected
                for session_id in affected_session_ids:
                    self._pending[session_id] = max(
                        expected, self._pending.get(session_id, 0)
                    )
            elif previous is not None:
                candidate = previous
            self._snapshots[scope] = candidate
            return candidate, changed

    def pending_generation(self, session_id: str) -> int | None:
        """Return one session's newest undelivered generation."""
        with self._lock:
            return self._pending.get(session_id)

    def queue_pending(self, session_ids: Iterable[str]) -> int | None:
        """Queue one monotonic high-watermark for explicitly affected sessions."""
        bounded = tuple(dict.fromkeys(session_ids))
        if not bounded:
            return None
        with self._lock:
            self._notification_generation = max(
                self._notification_generation + 1, self._generation
            )
            highwater = self._notification_generation
            for session_id in bounded:
                self._pending[session_id] = max(
                    highwater, self._pending.get(session_id, 0)
                )
            return highwater

    def acknowledge_pending(self, session_id: str, delivered_generation: int) -> None:
        """Acknowledge only the exact high-watermark that was actually sent."""
        with self._lock:
            if self._pending.get(session_id) == delivered_generation:
                self._pending.pop(session_id, None)

    def clear_pending(self) -> None:
        """Drop process-local delivery state during catalog teardown."""
        with self._lock:
            self._pending.clear()

    def has_pending(self, session_id: str) -> bool:
        """Whether one session still owes a list-change notification."""
        return self.pending_generation(session_id) is not None

    def bind_session(
        self,
        session_id: str,
        identity: CatalogIdentity,
        *,
        resume_token_digest: str,
    ) -> None:
        """Bind server-minted resume facts to one authenticated session."""
        if not session_id or not _is_digest(resume_token_digest):
            raise ValueError("session resume binding is invalid")
        with self._lock:
            self._sessions[session_id] = SessionBinding(identity, resume_token_digest)

    def validate_refresh(
        self, request: CatalogRefreshRequest, current: CatalogSnapshot
    ) -> None:
        """Fail closed when optimistic refresh identity is stale."""
        identity = current.identity
        if request.expected_config_revision != identity.config_revision:
            raise CatalogContractError(
                "catalog-generation-stale", "configuration revision changed"
            )
        if request.expected_catalog_generation != identity.catalog_generation:
            raise CatalogContractError(
                "catalog-generation-stale", "catalog generation changed"
            )
        if request.expected_snapshot_digest != identity.snapshot_digest:
            raise CatalogContractError(
                "catalog-digest-mismatch", "catalog digest changed"
            )

    def resolve_tool(
        self,
        *,
        public_name: str,
        expected_generation: int,
        expected_snapshot_digest: str,
        authorization_scope_digest: str,
        config_revision: str,
    ) -> tuple[ChildCatalogCandidate, CatalogEntry]:
        """Resolve one stable-dispatch target from the caller's exact snapshot."""
        snapshot = self.current(
            authorization_scope_digest, config_revision=config_revision
        )
        identity = snapshot.identity
        if identity.catalog_generation != expected_generation:
            raise CatalogContractError(
                "catalog-generation-stale", "stable dispatch generation is stale"
            )
        if identity.snapshot_digest != expected_snapshot_digest:
            raise CatalogContractError(
                "catalog-digest-mismatch", "stable dispatch digest is stale"
            )
        resolved = snapshot.tool(public_name)
        if resolved is None:
            raise CatalogContractError(
                "dispatcher-target-not-visible", "tool is not visible in this scope"
            )
        return resolved

    def resume_session(
        self,
        request: CatalogSessionResumeRequest,
        *,
        authorization_scope_digest: str,
        config_revision: str,
        cohort_homogeneous: bool,
    ) -> CatalogSessionResumeResult:
        """Validate re-authenticated session continuity without replaying a call."""
        if not cohort_homogeneous:
            raise CatalogContractError(
                "replica-generation-divergent", "replica cohort is divergent"
            )
        binding = self._sessions.get(request.session_id)
        if (
            binding is None
            or binding.resume_token_digest != request.resume_token_digest
        ):
            raise CatalogContractError(
                "session-resume-token-invalid", "resume token is invalid"
            )
        current = self.current(
            authorization_scope_digest, config_revision=config_revision
        ).identity
        _validate_resume_identity(request, current)
        if request.authorization_scope_digest != authorization_scope_digest:
            raise CatalogContractError(
                "session-resume-authorization-changed",
                "authorization scope changed",
            )
        same_instance = request.previous_served_instance_id == self.served_instance_id
        child_changed = (
            request.child_connection_generation != current.child_connection_generation
        )
        if child_changed:
            state: Literal["resumed", "relist-required", "rebound-relist-required"] = (
                "relist-required"
            )
        elif same_instance:
            state = "resumed"
        else:
            state = "rebound-relist-required"
        return CatalogSessionResumeResult(
            session_id=request.session_id,
            **current.model_dump(),
            resume_state=state,
        )

    @staticmethod
    def assert_homogeneous(
        local: CatalogIdentity, peers: Iterable[CatalogIdentity]
    ) -> None:
        """Fail readiness/refresh when an equivalent-scope replica diverges."""
        expected = _cohort_key(local)
        for peer in peers:
            if peer.authorization_scope_digest != local.authorization_scope_digest:
                continue
            if _cohort_key(peer) != expected:
                raise CatalogContractError(
                    "replica-generation-divergent", "replica cohort is divergent"
                )

    def _empty_snapshot(
        self, authorization_scope_digest: str, config_revision: str
    ) -> CatalogSnapshot:
        payload = {
            "release_id": self.release_id,
            "config_revision": config_revision,
            "authorization_scope_digest": authorization_scope_digest,
            "children": [],
        }
        identity = CatalogIdentity(
            served_instance_id=self.served_instance_id,
            release_id=self.release_id,
            config_revision=config_revision,
            catalog_generation=0,
            snapshot_digest=_digest_text(_canonical_json(payload)),
            child_connection_generation=0,
            authorization_scope_digest=authorization_scope_digest,
        )
        return CatalogSnapshot(identity=identity, children=())


def refresh_error(
    request_id: str, exc: CatalogContractError, current: CatalogSnapshot
) -> CatalogRefreshError:
    """Encode a closed refresh error without leaking raw exception text."""
    return CatalogRefreshError(
        request_id=request_id,
        code=exc.code,  # type: ignore[arg-type]
        retryable=exc.retryable,
        catalog_generation=current.identity.catalog_generation,
        snapshot_digest=current.identity.snapshot_digest,
        details_digest=_digest_text(exc.detail),
    )


def resume_error(
    session_id: str, exc: CatalogContractError, current: CatalogSnapshot
) -> CatalogSessionResumeError:
    """Encode a closed resume error without exposing authority details."""
    required: Literal["reauthenticate", "relist", "reconnect"] = (
        "reauthenticate" if "auth" in exc.code or "token" in exc.code else "relist"
    )
    return CatalogSessionResumeError(
        session_id=session_id,
        code=exc.code,  # type: ignore[arg-type]
        retryable=exc.retryable,
        required_action=required,
        catalog_generation=current.identity.catalog_generation,
        snapshot_digest=current.identity.snapshot_digest,
    )


def reconciliation_receipt_digest(
    identity: CatalogIdentity, writer_result: Mapping[str, Any]
) -> str:
    """Bind a successful synchronous projection acknowledgement to a snapshot."""
    return _digest_text(
        _canonical_json(
            {
                "identity": identity.model_dump(mode="json"),
                "writer_result": dict(writer_result),
            }
        )
    )


def _entries(
    key_field: str, values: Iterable[Mapping[str, Any]]
) -> tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for value in values:
        key = value.get(key_field)
        if not isinstance(key, str) or not key or key in seen:
            raise CatalogContractError(
                "catalog-snapshot-incomplete", "catalog family key is invalid"
            )
        seen.add(key)
        entries.append(CatalogEntry.from_mapping(key, value))
    return tuple(sorted(entries, key=lambda entry: entry.key))


def _validate_resume_identity(
    request: CatalogSessionResumeRequest, current: CatalogIdentity
) -> None:
    checks = (
        (request.release_id, current.release_id, "session-resume-release-mismatch"),
        (
            request.config_revision,
            current.config_revision,
            "session-resume-config-revision-mismatch",
        ),
        (
            request.catalog_generation,
            current.catalog_generation,
            "session-resume-catalog-generation-mismatch",
        ),
        (
            request.snapshot_digest,
            current.snapshot_digest,
            "session-resume-snapshot-digest-mismatch",
        ),
    )
    for actual, expected, code in checks:
        if actual != expected:
            raise CatalogContractError(code, "session identity changed")


def _cohort_key(identity: CatalogIdentity) -> tuple[Any, ...]:
    return (
        identity.release_id,
        identity.config_revision,
        identity.catalog_generation,
        identity.snapshot_digest,
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CatalogContractError(
            "catalog-snapshot-incomplete", "catalog value is not canonical JSON"
        ) from exc


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "CatalogContractError",
    "CatalogIdentity",
    "CatalogRefreshError",
    "CatalogRefreshRequest",
    "CatalogRefreshResult",
    "CatalogSessionResumeError",
    "CatalogSessionResumeRequest",
    "CatalogSessionResumeResult",
    "CatalogSnapshot",
    "ChildCatalogCandidate",
    "McpCatalogReconciler",
    "reconciliation_receipt_digest",
    "refresh_error",
    "resume_error",
]
