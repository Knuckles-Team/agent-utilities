"""Typed repository and release-control seams for immutable workflows.

The repository stores immutable definitions and CAS-fenced channel pointers.
It intentionally has no claim, lease, retry, result, or completion methods:
execution is delegated to the native epistemic-graph ``WorkItem`` authority.

``InMemoryWorkflowRepository`` is a deterministic reference implementation for
unit tests and local composition.  A durable graph-backed adapter can satisfy
the same ``WorkflowRepository`` protocol without changing this domain core.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .domain import (
    _VERSION_RE,
    ApprovedBinding,
    ChannelName,
    Digest,
    StableId,
    Version,
    WorkflowDefinition,
    WorkflowResolution,
    WorkflowSummary,
    resolution_digest,
)

__all__ = [
    "BindingRegistry",
    "ExactBindingRegistry",
    "InMemoryWorkflowRepository",
    "RejectAllBindingRegistry",
    "ReleasePointer",
    "WorkflowCatalog",
    "WorkflowConflictError",
    "WorkflowDomainError",
    "WorkflowNotFoundError",
    "WorkflowRepository",
]


class WorkflowDomainError(ValueError):
    """Base error for fail-closed workflow-domain operations."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        message = code if not detail else f"{code}: {detail}"
        super().__init__(message)


class WorkflowConflictError(WorkflowDomainError):
    """A CAS or immutable-version precondition failed."""


class WorkflowNotFoundError(WorkflowDomainError):
    """A requested immutable version or release pointer does not exist."""


class ReleasePointer(BaseModel):
    """CAS-fenced channel pointer to one exact immutable version."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    workflow_id: StableId
    channel: ChannelName
    version: Version
    definition_digest: Digest
    generation: int = Field(ge=0)


@runtime_checkable
class BindingRegistry(Protocol):
    """Exact capability approval seam; aliases never resolve implicitly."""

    def is_approved(self, binding: ApprovedBinding) -> bool:
        """Return true only for the exact id/version/digest tuple."""


class RejectAllBindingRegistry:
    """Fail-closed default when no capability authority is configured."""

    def is_approved(self, binding: ApprovedBinding) -> bool:
        del binding
        return False


class ExactBindingRegistry:
    """Small deterministic exact-match registry for composition and fixtures."""

    def __init__(self, bindings: Iterable[ApprovedBinding] = ()) -> None:
        entries: dict[tuple[str, str, str], ApprovedBinding] = {}
        for binding in bindings:
            key = self._key(binding)
            prior = entries.get(key)
            if prior is not None and prior != binding:
                raise WorkflowConflictError("binding_identity_conflict")
            entries[key] = binding
        self._entries = entries

    @staticmethod
    def _key(binding: ApprovedBinding) -> tuple[str, str, str]:
        # A single logical capability/version may never be approved with two
        # different digests.  This catches catalog drift at construction time.
        return (binding.kind, binding.binding_id, binding.version)

    def is_approved(self, binding: ApprovedBinding) -> bool:
        return self._entries.get(self._key(binding)) == binding


@runtime_checkable
class WorkflowRepository(Protocol):
    """Durable storage seam for definitions and CAS release pointers."""

    def put_version(self, definition: WorkflowDefinition) -> None:
        """Store an immutable version, or fail on same-identity drift."""

    def get_version(self, workflow_id: str, version: str) -> WorkflowDefinition | None:
        """Read one exact version; aliases are not accepted by adapters."""

    def read_pointer(self, workflow_id: str, channel: str) -> ReleasePointer | None:
        """Read the current pointer for one channel."""

    def compare_and_set_pointer(
        self,
        *,
        workflow_id: str,
        channel: str,
        expected: ReleasePointer | None,
        version: str,
        definition_digest: str,
    ) -> ReleasePointer:
        """Atomically replace a pointer if its complete prior value matches."""


class InMemoryWorkflowRepository:
    """Thread-safe deterministic reference repository.

    This class is a definition catalog only.  It cannot claim or lease a
    WorkItem and must not be used as an execution queue.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._versions: dict[tuple[str, str], WorkflowDefinition] = {}
        self._pointers: dict[tuple[str, str], ReleasePointer] = {}
        self._history: dict[tuple[str, str], list[ReleasePointer]] = {}

    def put_version(self, definition: WorkflowDefinition) -> None:
        key = (definition.workflow_id, definition.version)
        with self._lock:
            prior = self._versions.get(key)
            if (
                prior is not None
                and prior.definition_digest != definition.definition_digest
            ):
                raise WorkflowConflictError("immutable_version_conflict")
            self._versions[key] = definition

    def get_version(self, workflow_id: str, version: str) -> WorkflowDefinition | None:
        with self._lock:
            return self._versions.get((workflow_id, version))

    def read_pointer(self, workflow_id: str, channel: str) -> ReleasePointer | None:
        with self._lock:
            return self._pointers.get((workflow_id, channel))

    def compare_and_set_pointer(
        self,
        *,
        workflow_id: str,
        channel: str,
        expected: ReleasePointer | None,
        version: str,
        definition_digest: str,
    ) -> ReleasePointer:
        with self._lock:
            current = self._pointers.get((workflow_id, channel))
            if current != expected:
                raise WorkflowConflictError("release_pointer_cas_conflict")

            definition = self._versions.get((workflow_id, version))
            if definition is None:
                raise WorkflowNotFoundError("immutable_version_missing")
            if definition.definition_digest != definition_digest:
                raise WorkflowConflictError("release_digest_mismatch")

            pointer = ReleasePointer(
                workflow_id=workflow_id,
                channel=channel,
                version=version,
                definition_digest=definition_digest,
                generation=0 if current is None else current.generation + 1,
            )
            self._pointers[(workflow_id, channel)] = pointer
            self._history.setdefault((workflow_id, channel), []).append(pointer)
            return pointer

    def release_history(
        self, workflow_id: str, channel: str
    ) -> tuple[ReleasePointer, ...]:
        """Return the immutable pointer history for audit/rollback inspection."""

        with self._lock:
            return tuple(self._history.get((workflow_id, channel), ()))


class WorkflowCatalog:
    """Validate, publish, resolve, and rollback immutable workflow versions."""

    def __init__(
        self,
        repository: WorkflowRepository,
        *,
        bindings: BindingRegistry | None = None,
    ) -> None:
        self.repository = repository
        # Absence of an approval authority must reject rather than accidentally
        # make every definition executable.
        self.bindings = bindings if bindings is not None else RejectAllBindingRegistry()

    def publish(self, definition: WorkflowDefinition) -> str:
        """Validate exact bindings and store one immutable definition."""

        self._validate_bindings(definition)
        self.repository.put_version(definition)
        return definition.definition_digest

    def release(
        self,
        definition: WorkflowDefinition,
        *,
        channel: str,
        expected: ReleasePointer | None = None,
    ) -> ReleasePointer:
        """Publish and CAS a channel to this exact version."""

        channel = _validate_channel(channel)
        self.publish(definition)
        return self.repository.compare_and_set_pointer(
            workflow_id=definition.workflow_id,
            channel=channel,
            expected=expected,
            version=definition.version,
            definition_digest=definition.definition_digest,
        )

    def resolve(self, workflow_id: str, *, channel: str) -> WorkflowResolution:
        """Resolve a channel and re-check all immutable authority evidence."""

        channel = _validate_channel(channel)
        pointer = self.repository.read_pointer(workflow_id, channel)
        if pointer is None:
            raise WorkflowNotFoundError("release_pointer_missing")
        if pointer.workflow_id != workflow_id or pointer.channel != channel:
            raise WorkflowConflictError("release_pointer_identity_mismatch")

        definition = self.repository.get_version(workflow_id, pointer.version)
        if definition is None:
            raise WorkflowNotFoundError("released_immutable_version_missing")
        if definition.definition_digest != pointer.definition_digest:
            raise WorkflowConflictError("released_definition_digest_mismatch")
        self._validate_bindings(definition)

        return WorkflowResolution(
            definition=definition,
            channel=channel,
            pointer_generation=pointer.generation,
            resolution_digest=resolution_digest(
                definition,
                channel=channel,
                pointer_generation=pointer.generation,
            ),
        )

    def rollback(
        self,
        workflow_id: str,
        *,
        channel: str,
        prior_version: str,
        expected: ReleasePointer,
    ) -> ReleasePointer:
        """CAS a channel back to an already-published, lower immutable version."""

        channel = _validate_channel(channel)
        current = self.repository.read_pointer(workflow_id, channel)
        if current != expected:
            raise WorkflowConflictError("rollback_pointer_cas_conflict")
        if not _is_prior_version(prior_version, current.version):
            raise WorkflowConflictError("rollback_target_not_prior")

        target = self.repository.get_version(workflow_id, prior_version)
        if target is None:
            raise WorkflowNotFoundError("rollback_version_missing")
        self._validate_bindings(target)
        return self.repository.compare_and_set_pointer(
            workflow_id=workflow_id,
            channel=channel,
            expected=current,
            version=target.version,
            definition_digest=target.definition_digest,
        )

    def summary(self, workflow_id: str, *, channel: str) -> WorkflowSummary:
        """Return only privacy-safe identity/topology data for a release."""

        resolution = self.resolve(workflow_id, channel=channel)
        return resolution.definition.summary_projection(
            channel=resolution.channel,
            pointer_generation=resolution.pointer_generation,
        )

    def _validate_bindings(self, definition: WorkflowDefinition) -> None:
        all_bindings = [definition.policy_binding]
        all_bindings.extend(step.binding for step in definition.steps)
        all_bindings.extend(
            step.policy_binding
            for step in definition.steps
            if step.policy_binding is not None
        )
        for binding in all_bindings:
            if not self.bindings.is_approved(binding):
                raise WorkflowDomainError("binding_unresolved_or_unapproved")


def _validate_channel(channel: str) -> str:
    if not isinstance(channel, str) or not channel or channel != channel.strip():
        raise WorkflowDomainError("channel_invalid")
    # Let the Pydantic alias enforce the exact lowercase grammar without
    # normalizing a caller's potentially stale alias.
    try:
        pointer = ReleasePointer(
            workflow_id="workflow",
            channel=channel,
            version="0.0.0",
            definition_digest="sha256:" + "0" * 64,
            generation=0,
        )
    except Exception as exc:  # Pydantic's concrete validation details are not API.
        raise WorkflowDomainError("channel_invalid") from exc
    return pointer.channel


def _is_prior_version(candidate: str, current: str) -> bool:
    """Compare concrete semantic versions; aliases and malformed values fail closed."""

    if not _VERSION_RE.fullmatch(candidate) or not _VERSION_RE.fullmatch(current):
        return False

    def key(version: str) -> tuple[int, int, int, int, str]:
        core, _, suffix = version.partition("-")
        core = core.split("+", 1)[0]
        major, minor, patch = (int(part) for part in core.split("."))
        # A release without a prerelease suffix sorts after a prerelease of the
        # same numeric version.  Build metadata does not affect rollback order.
        return major, minor, patch, 1 if not suffix else 0, suffix

    return key(candidate) < key(current)
