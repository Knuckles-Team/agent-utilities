"""Strict references for a resolved run and one native WorkItem admission."""

from __future__ import annotations

import re
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_utilities.control_plane.policy.models import (
    CapabilityBinding,
    PolicyAuthorization,
    Version,
    canonical_digest,
)

__all__ = [
    "AdmissionReceipt",
    "ArtifactRef",
    "InvocationRef",
    "JobRef",
    "NativeAdmissionRequest",
    "NativeWorkItemAdmission",
    "RunAdmissionIdentity",
    "RunRecord",
    "RunRef",
    "RunResolution",
    "TaskRef",
    "ToolBindingRef",
    "TraceRef",
    "deterministic_admission_id",
    "native_work_item_id",
]


_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_./-]{0,127}$")
_REF_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_./-]{0,255}$")
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:/-]{0,255}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

StableId: TypeAlias = Annotated[str, Field(pattern=_ID_RE.pattern, min_length=1)]
OpaqueRef: TypeAlias = Annotated[str, Field(pattern=_REF_RE.pattern, min_length=1)]
StableKey: TypeAlias = Annotated[str, Field(pattern=_KEY_RE.pattern, min_length=1)]
Digest: TypeAlias = Annotated[str, Field(pattern=_DIGEST_RE.pattern)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=False,
        strict=True,
    )


class TraceRef(_FrozenModel):
    trace_id: StableId
    digest: Digest


class ArtifactRef(_FrozenModel):
    artifact_id: StableId
    digest: Digest


class JobRef(_FrozenModel):
    job_id: StableId
    version: Version
    digest: Digest


class TaskRef(_FrozenModel):
    task_id: StableId
    job_id: StableId
    version: Version
    digest: Digest
    depends_on: tuple[StableId, ...] = ()

    @model_validator(mode="after")
    def _dependencies_are_unique(self) -> "TaskRef":
        if self.task_id in self.depends_on:
            raise ValueError("task_self_dependency")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("task_dependency_duplicate")
        return self


class ToolBindingRef(_FrozenModel):
    binding: CapabilityBinding

    @model_validator(mode="after")
    def _tool_only(self) -> "ToolBindingRef":
        if self.binding.kind != "tool":
            raise ValueError("tool_binding_kind_invalid")
        return self

    @property
    def binding_id(self) -> str:
        return self.binding.binding_id

    @property
    def digest(self) -> str:
        return self.binding.digest


class InvocationRef(_FrozenModel):
    invocation_id: StableId
    task_id: StableId
    tool_binding_id: StableId
    tool_binding_digest: Digest
    input_digest: Digest
    trace_ref: TraceRef
    artifact_refs: tuple[ArtifactRef, ...] = ()

    @model_validator(mode="after")
    def _artifacts_are_unique(self) -> "InvocationRef":
        refs = [(artifact.artifact_id, artifact.digest) for artifact in self.artifact_refs]
        if len(refs) != len(set(refs)):
            raise ValueError("invocation_artifact_duplicate")
        return self

    @property
    def invocation_digest(self) -> str:
        return canonical_digest(self)


class RunAdmissionIdentity(_FrozenModel):
    """The stable duplicate-delivery key; request body digest is checked separately."""

    tenant_ref: OpaqueRef
    idempotency_key: StableKey

    @property
    def run_id(self) -> str:
        return deterministic_admission_id(self.tenant_ref, self.idempotency_key)


class RunRef(_FrozenModel):
    run_id: StableId
    resolution_digest: Digest


class RunResolution(_FrozenModel):
    """Exact policy/job/task/tool/trace resolution persisted for one run."""

    tenant_ref: OpaqueRef
    idempotency_key: StableKey
    request_digest: Digest
    authorization: PolicyAuthorization
    job: JobRef
    tasks: tuple[TaskRef, ...] = Field(min_length=1, max_length=256)
    tool_bindings: tuple[ToolBindingRef, ...] = Field(default=(), max_length=256)
    invocations: tuple[InvocationRef, ...] = Field(default=(), max_length=512)
    trace_ref: TraceRef
    artifact_refs: tuple[ArtifactRef, ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def _validate_resolution(self) -> "RunResolution":
        if self.authorization.request_digest != self.request_digest:
            raise ValueError("authorization_request_digest_mismatch")
        if len(self.tasks) > self.authorization.budget.max_tasks:
            raise ValueError("run_task_budget_exceeded")

        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("run_task_duplicate")
        task_map = {task.task_id: task for task in self.tasks}
        for task in self.tasks:
            if task.job_id != self.job.job_id:
                raise ValueError("task_job_reference_mismatch")
            if set(task.depends_on) - set(task_ids):
                raise ValueError("task_dependency_missing")
        depth, fanout = _graph_shape(self.tasks)
        if depth > self.authorization.budget.max_depth:
            raise ValueError("run_depth_budget_exceeded")
        if fanout > self.authorization.budget.max_fanout:
            raise ValueError("run_fanout_budget_exceeded")

        binding_map = {
            binding.binding_id: binding
            for binding in self.tool_bindings
        }
        if len(binding_map) != len(self.tool_bindings):
            raise ValueError("tool_binding_duplicate")
        allowed = {
            (binding.kind, binding.binding_id, binding.version, binding.digest, binding.privilege)
            for binding in self.authorization.bindings
        }
        for tool in self.tool_bindings:
            exact = tool.binding
            if (
                exact.kind,
                exact.binding_id,
                exact.version,
                exact.digest,
                exact.privilege,
            ) not in allowed:
                raise ValueError("tool_binding_not_in_authorization")

        if len(self.invocations) > self.authorization.budget.max_tool_calls:
            raise ValueError("run_tool_budget_exceeded")
        for invocation in self.invocations:
            if invocation.task_id not in task_map:
                raise ValueError("invocation_task_reference_missing")
            binding = binding_map.get(invocation.tool_binding_id)
            if binding is None or binding.digest != invocation.tool_binding_digest:
                raise ValueError("invocation_tool_binding_drift")
        artifact_ids = [(artifact.artifact_id, artifact.digest) for artifact in self.artifact_refs]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("run_artifact_duplicate")
        return self

    @property
    def run_id(self) -> str:
        return deterministic_admission_id(self.tenant_ref, self.idempotency_key)

    @property
    def resolution_digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json", exclude_none=True))

    @property
    def ref(self) -> RunRef:
        return RunRef(run_id=self.run_id, resolution_digest=self.resolution_digest)


class NativeWorkItemAdmission(_FrozenModel):
    """Minimal request to the native WorkItem authority; no claim or lease state."""

    run_id: StableId
    work_item_id: StableId
    tenant_ref: OpaqueRef
    idempotency_key: StableKey
    kind: StableKey
    payload_ref: OpaqueRef
    payload_digest: Digest
    request_digest: Digest
    resolution_digest: Digest
    depends_on: tuple[StableId, ...] = ()

    @model_validator(mode="after")
    def _identity_is_deterministic(self) -> "NativeWorkItemAdmission":
        expected_run = deterministic_admission_id(self.tenant_ref, self.idempotency_key)
        if self.run_id != expected_run:
            raise ValueError("native_admission_run_identity_mismatch")
        if self.work_item_id != native_work_item_id(self.run_id):
            raise ValueError("native_admission_work_item_identity_mismatch")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("native_admission_dependency_duplicate")
        return self


class NativeAdmissionRequest(_FrozenModel):
    """One request that must create exactly one run and one native WorkItem."""

    resolution: RunResolution
    work_item: NativeWorkItemAdmission

    @model_validator(mode="after")
    def _run_and_item_are_one_pair(self) -> "NativeAdmissionRequest":
        resolution = self.resolution
        item = self.work_item
        if item.run_id != resolution.run_id:
            raise ValueError("native_admission_run_mismatch")
        if item.tenant_ref != resolution.tenant_ref:
            raise ValueError("native_admission_tenant_mismatch")
        if item.idempotency_key != resolution.idempotency_key:
            raise ValueError("native_admission_idempotency_mismatch")
        if item.request_digest != resolution.request_digest:
            raise ValueError("native_admission_request_digest_mismatch")
        if item.resolution_digest != resolution.resolution_digest:
            raise ValueError("native_admission_resolution_digest_mismatch")
        return self


class AdmissionReceipt(_FrozenModel):
    run_id: StableId
    work_item_id: StableId
    resolution_digest: Digest
    work_item_digest: Digest
    created: bool


class RunRecord(_FrozenModel):
    """Definition-side record; execution status remains on native WorkItem."""

    resolution: RunResolution
    work_item: NativeWorkItemAdmission
    audit_event_ids: tuple[StableId, ...] = ()


def deterministic_admission_id(tenant_ref: str, idempotency_key: str) -> str:
    """Derive one stable run identity without incorporating request body bytes."""

    if not tenant_ref or tenant_ref != tenant_ref.strip():
        raise ValueError("tenant_ref_invalid")
    if not idempotency_key or idempotency_key != idempotency_key.strip():
        raise ValueError("idempotency_key_invalid")
    digest = canonical_digest(
        {"tenant_ref": tenant_ref, "idempotency_key": idempotency_key}
    )
    return f"run:{digest.removeprefix('sha256:')}"


def native_work_item_id(run_id: str) -> str:
    if not run_id or not run_id.startswith("run:"):
        raise ValueError("run_id_invalid")
    return f"work_item:{canonical_digest(run_id).removeprefix('sha256:')}"


def _graph_shape(tasks: tuple[TaskRef, ...]) -> tuple[int, int]:
    children: dict[str, list[str]] = {task.task_id: [] for task in tasks}
    indegree = {task.task_id: len(task.depends_on) for task in tasks}
    for task in tasks:
        for dependency in task.depends_on:
            if dependency not in children:
                raise ValueError("task_dependency_missing")
            children[dependency].append(task.task_id)
    ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    depth = {task_id: 1 for task_id in indegree}
    visited: list[str] = []
    while ready:
        task_id = ready.pop(0)
        visited.append(task_id)
        for child in sorted(children[task_id]):
            depth[child] = max(depth[child], depth[task_id] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(visited) != len(tasks):
        raise ValueError("task_cycle")
    return max(depth.values(), default=0), max(
        (len(children[task_id]) for task_id in children), default=0
    )
