"""Immutable workflow-domain contracts.

The workflow control plane is intentionally narrower than an execution
runtime.  It owns stable identities, immutable DAG definitions, exact
capability bindings, bounded budgets, and privacy-safe resolution summaries.
It does *not* own claim, lease, retry, result, or completion state; those
mutations belong to the native epistemic-graph ``WorkItem`` protocol.

The models in this module are frozen and reject unknown fields.  Payloads,
secrets, prompts, tool arguments, and results are not part of the contract:
only opaque references, schema digests, and short operator summaries cross
this boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "ABSOLUTE_MAX_COST_MICROS",
    "ABSOLUTE_MAX_DEPTH",
    "ABSOLUTE_MAX_FANOUT",
    "ABSOLUTE_MAX_SECONDS",
    "ABSOLUTE_MAX_STEPS",
    "ABSOLUTE_MAX_TOKENS",
    "ApprovedBinding",
    "ArtifactRef",
    "BindingKind",
    "ChannelName",
    "Digest",
    "StepContract",
    "WorkflowBudget",
    "WorkflowDefinition",
    "WorkflowGraphStats",
    "WorkflowIdentity",
    "WorkflowResolution",
    "WorkflowStep",
    "WorkflowSummary",
    "WorkflowTemplate",
    "WorkflowVersion",
    "canonical_definition",
    "definition_digest",
    "resolution_digest",
]


# These are hard safety ceilings, independent of a caller's requested or
# policy-approved budget.  A deployment can choose smaller ceilings by policy,
# but no workflow definition can raise them through configuration.
ABSOLUTE_MAX_STEPS = 256
ABSOLUTE_MAX_DEPTH = 64
ABSOLUTE_MAX_FANOUT = 32
ABSOLUTE_MAX_TOKENS = 4_000_000
ABSOLUTE_MAX_SECONDS = 86_400
ABSOLUTE_MAX_COST_MICROS = 10_000_000

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_./-]{0,127}$")
_REF_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_./-]{0,255}$")
_VERSION_RE = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CHANNEL_RE = re.compile(r"^[a-z][a-z0-9._-]{0,31}$")
_SENSITIVE_SUMMARY_RE = re.compile(
    r"\b(?:argument|body|credential|password|payload|prompt|result|secret|token)\b",
    re.IGNORECASE,
)

StableId: TypeAlias = Annotated[str, Field(pattern=_ID_RE.pattern, min_length=1)]
OpaqueRef: TypeAlias = Annotated[str, Field(pattern=_REF_RE.pattern, min_length=1)]
Digest: TypeAlias = Annotated[str, Field(pattern=_DIGEST_RE.pattern)]
Version: TypeAlias = Annotated[str, Field(pattern=_VERSION_RE.pattern)]
ChannelName: TypeAlias = Annotated[str, Field(pattern=_CHANNEL_RE.pattern)]
BindingKind: TypeAlias = Literal["agent", "skill", "tool", "policy", "delegation"]


class _FrozenModel(BaseModel):
    """Common strict/frozen base for definition-side data."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=False,
        strict=True,
    )


def _validate_nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}_empty")
    if value != value.strip():
        raise ValueError(f"{field_name}_whitespace")
    return value


def _validate_summary(value: str) -> str:
    value = _validate_nonempty(value, "summary")
    if len(value) > 256:
        raise ValueError("summary_too_long")
    if any(ord(char) < 32 and char not in "\t" for char in value):
        raise ValueError("summary_control_character")
    if _SENSITIVE_SUMMARY_RE.search(value):
        raise ValueError("summary_sensitive_content")
    return value


class ArtifactRef(_FrozenModel):
    """Opaque artifact identity; the artifact bytes never travel with a definition."""

    ref: OpaqueRef
    digest: Digest


class ApprovedBinding(_FrozenModel):
    """Exact, content-addressed capability reference.

    ``version`` is a concrete semantic version and ``digest`` is the exact
    approved artifact digest.  There is intentionally no alias, range, URL,
    inline policy, credential, tool argument, or capability body field.
    """

    kind: BindingKind
    binding_id: StableId
    version: Version
    digest: Digest

    @field_validator("binding_id")
    @classmethod
    def _binding_id_is_not_an_alias(cls, value: str) -> str:
        # The identifier grammar already excludes common alias separators, but
        # this explicit guard keeps the contract obvious if the grammar grows.
        if "@" in value or value.casefold() in {"latest", "current", "default"}:
            raise ValueError("binding_alias_forbidden")
        return value


class StepContract(_FrozenModel):
    """Reference-only input/output contract for one step."""

    contract_id: StableId
    input_schema: ArtifactRef
    output_schema: ArtifactRef
    input_artifacts: tuple[ArtifactRef, ...] = ()
    output_artifacts: tuple[ArtifactRef, ...] = ()

    @model_validator(mode="after")
    def _artifacts_are_unique(self) -> StepContract:
        for field_name in ("input_artifacts", "output_artifacts"):
            refs = getattr(self, field_name)
            identities = [(item.ref, item.digest) for item in refs]
            if len(identities) != len(set(identities)):
                raise ValueError(f"{field_name}_duplicate")
        return self


class WorkflowStep(_FrozenModel):
    """One immutable DAG vertex.

    A step has exactly one approved executable capability binding.  A policy
    override, when present, is also a reference to an approved policy version;
    neither binding contains executable code or an inline policy document.
    """

    step_id: StableId
    binding: ApprovedBinding
    contract: StepContract
    depends_on: tuple[StableId, ...] = ()
    policy_binding: ApprovedBinding | None = None
    summary: str = "step"

    @field_validator("summary")
    @classmethod
    def _summary_is_safe(cls, value: str) -> str:
        return _validate_summary(value)

    @model_validator(mode="after")
    def _dependencies_are_unique_and_not_self(self) -> WorkflowStep:
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("step_dependency_duplicate")
        if self.step_id in self.depends_on:
            raise ValueError("step_self_dependency")
        if self.binding.kind == "policy":
            raise ValueError("step_policy_binding_required_on_policy_field")
        if self.policy_binding is not None and self.policy_binding.kind != "policy":
            raise ValueError("step_policy_binding_kind_invalid")
        return self


class WorkflowBudget(_FrozenModel):
    """Bounded graph and execution budget.

    The definition's ``budget`` is the requested ceiling and ``policy_budget``
    is the immutable ceiling attested by its policy binding.  Validation
    requires every requested dimension to be no greater than the approved
    dimension; no runtime can silently escalate one after resolution.
    """

    max_steps: int = Field(ge=1, le=ABSOLUTE_MAX_STEPS)
    max_depth: int = Field(ge=1, le=ABSOLUTE_MAX_DEPTH)
    max_fanout: int = Field(ge=1, le=ABSOLUTE_MAX_FANOUT)
    max_tool_calls: int = Field(ge=0, le=ABSOLUTE_MAX_STEPS)
    max_tokens: int = Field(ge=1, le=ABSOLUTE_MAX_TOKENS)
    max_seconds: int = Field(ge=1, le=ABSOLUTE_MAX_SECONDS)
    max_cost_micros: int = Field(ge=0, le=ABSOLUTE_MAX_COST_MICROS)


class WorkflowGraphStats(_FrozenModel):
    """Deterministic graph measurements used by bounded admission and summaries."""

    step_count: int = Field(ge=1, le=ABSOLUTE_MAX_STEPS)
    depth: int = Field(ge=1, le=ABSOLUTE_MAX_DEPTH)
    max_fanout: int = Field(ge=0, le=ABSOLUTE_MAX_FANOUT)
    tool_step_count: int = Field(ge=0, le=ABSOLUTE_MAX_STEPS)


class WorkflowIdentity(_FrozenModel):
    """Stable workflow/template/version identity tuple."""

    workflow_id: StableId
    template_id: StableId
    version: Version


class WorkflowTemplate(_FrozenModel):
    """Immutable template metadata, separate from any released version."""

    template_id: StableId
    workflow_id: StableId
    summary: str
    owner_ref: OpaqueRef

    @field_validator("summary")
    @classmethod
    def _summary_is_safe(cls, value: str) -> str:
        return _validate_summary(value)

    @property
    def identity(self) -> tuple[str, str]:
        """Stable ``(workflow_id, template_id)`` identity for this template."""

        return self.workflow_id, self.template_id

    @property
    def template_digest(self) -> str:
        material = {
            "template_id": self.template_id,
            "workflow_id": self.workflow_id,
            "summary": self.summary,
            "owner_ref": self.owner_ref,
        }
        return _sha256_digest(material)


class WorkflowDefinition(_FrozenModel):
    """One immutable, content-addressed workflow version."""

    workflow_id: StableId
    template_id: StableId
    template_digest: Digest
    version: Version
    policy_binding: ApprovedBinding
    budget: WorkflowBudget
    policy_budget: WorkflowBudget
    steps: tuple[WorkflowStep, ...] = Field(min_length=1, max_length=ABSOLUTE_MAX_STEPS)
    summary: str = "immutable workflow definition"

    @field_validator("summary")
    @classmethod
    def _summary_is_safe(cls, value: str) -> str:
        return _validate_summary(value)

    @model_validator(mode="after")
    def _validate_definition(self) -> WorkflowDefinition:
        if self.policy_binding.kind != "policy":
            raise ValueError("workflow_policy_binding_kind_invalid")
        if not _budget_within(self.budget, self.policy_budget):
            raise ValueError("budget_escalation")

        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("workflow_step_duplicate")
        known = set(step_ids)
        for step in self.steps:
            missing = set(step.depends_on) - known
            if missing:
                raise ValueError("workflow_dependency_missing")
            if step.policy_binding is not None and step.policy_binding.kind != "policy":
                raise ValueError("step_policy_binding_kind_invalid")

        stats = _graph_stats(self.steps)
        _enforce_graph_bounds(stats, self.budget)
        return self

    @property
    def identity(self) -> WorkflowIdentity:
        return WorkflowIdentity(
            workflow_id=self.workflow_id,
            template_id=self.template_id,
            version=self.version,
        )

    @property
    def graph_stats(self) -> WorkflowGraphStats:
        return _graph_stats(self.steps)

    @property
    def definition_digest(self) -> str:
        return definition_digest(self)

    def summary_projection(
        self,
        *,
        channel: str | None = None,
        pointer_generation: int | None = None,
    ) -> WorkflowSummary:
        return WorkflowSummary(
            workflow_id=self.workflow_id,
            template_id=self.template_id,
            version=self.version,
            definition_digest=self.definition_digest,
            channel=channel,
            pointer_generation=pointer_generation,
            step_count=self.graph_stats.step_count,
            depth=self.graph_stats.depth,
            max_fanout=self.graph_stats.max_fanout,
            binding_kinds=tuple(
                sorted({step.binding.kind for step in self.steps} | {"policy"})
            ),
        )


# The explicit alias keeps the domain vocabulary useful to callers that refer
# to a released immutable definition as a version.
WorkflowVersion = WorkflowDefinition


class WorkflowSummary(_FrozenModel):
    """Privacy-safe projection; no summaries, bodies, arguments, or results."""

    workflow_id: StableId
    template_id: StableId
    version: Version
    definition_digest: Digest
    channel: ChannelName | None = None
    pointer_generation: int | None = Field(default=None, ge=0)
    step_count: int = Field(ge=1, le=ABSOLUTE_MAX_STEPS)
    depth: int = Field(ge=1, le=ABSOLUTE_MAX_DEPTH)
    max_fanout: int = Field(ge=0, le=ABSOLUTE_MAX_FANOUT)
    binding_kinds: tuple[BindingKind, ...]


class WorkflowResolution(_FrozenModel):
    """Resolved immutable version plus its CAS-fenced channel identity."""

    definition: WorkflowDefinition
    channel: ChannelName
    pointer_generation: int = Field(ge=0)
    resolution_digest: Digest

    @model_validator(mode="after")
    def _digest_is_authoritative(self) -> WorkflowResolution:
        expected = resolution_digest(
            self.definition,
            channel=self.channel,
            pointer_generation=self.pointer_generation,
        )
        if self.resolution_digest != expected:
            raise ValueError("resolution_digest_mismatch")
        return self


def _budget_within(requested: WorkflowBudget, approved: WorkflowBudget) -> bool:
    return all(
        getattr(requested, field_name) <= getattr(approved, field_name)
        for field_name in (
            "max_steps",
            "max_depth",
            "max_fanout",
            "max_tool_calls",
            "max_tokens",
            "max_seconds",
            "max_cost_micros",
        )
    )


def _graph_stats(steps: tuple[WorkflowStep, ...]) -> WorkflowGraphStats:
    """Return deterministic DAG statistics or fail closed on a cycle."""

    step_ids = tuple(step.step_id for step in steps)
    children: dict[str, list[str]] = defaultdict(list)
    indegree = {step_id: 0 for step_id in step_ids}
    for step in steps:
        for dependency in step.depends_on:
            if dependency not in indegree:
                raise ValueError("workflow_dependency_missing")
            children[dependency].append(step.step_id)
            indegree[step.step_id] += 1

    for child_ids in children.values():
        child_ids.sort()

    ready = deque(
        sorted(step_id for step_id, degree in indegree.items() if degree == 0)
    )
    longest = {step_id: 1 for step_id in step_ids}
    visited: list[str] = []
    while ready:
        step_id = ready.popleft()
        visited.append(step_id)
        for child in children.get(step_id, ()):
            longest[child] = max(longest[child], longest[step_id] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                # There are at most ABSOLUTE_MAX_STEPS entries; sorting the
                # small ready set keeps traversal and its digest reproducible.
                ready.append(child)
                ready = deque(sorted(ready))

    if len(visited) != len(step_ids):
        raise ValueError("workflow_cycle")

    return WorkflowGraphStats(
        step_count=len(step_ids),
        depth=max(longest.values()),
        max_fanout=max(
            (len(children.get(step_id, ())) for step_id in step_ids), default=0
        ),
        tool_step_count=sum(1 for step in steps if step.binding.kind == "tool"),
    )


def _enforce_graph_bounds(stats: WorkflowGraphStats, budget: WorkflowBudget) -> None:
    if stats.step_count > budget.max_steps:
        raise ValueError("workflow_step_budget_exceeded")
    if stats.depth > budget.max_depth:
        raise ValueError("workflow_depth_budget_exceeded")
    if stats.max_fanout > budget.max_fanout:
        raise ValueError("workflow_fanout_budget_exceeded")
    if stats.tool_step_count > budget.max_tool_calls:
        raise ValueError("workflow_tool_budget_exceeded")


def _canonical_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, tuple | list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def _sha256_digest(value: object) -> str:
    payload = json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def canonical_definition(definition: WorkflowDefinition) -> dict[str, object]:
    """Return the stable, order-independent representation used for identity."""

    raw = definition.model_dump(mode="json", exclude_none=True)
    # Dependency declaration order has no graph meaning.  Canonicalizing both
    # vertices and incoming edges prevents semantically identical DAGs from
    # receiving different identities merely because a client serialized a list
    # in a different order.
    steps = []
    for step in raw["steps"]:
        step = dict(step)
        step["depends_on"] = sorted(step.get("depends_on", []))
        steps.append(step)
    raw["steps"] = sorted(steps, key=lambda item: str(item["step_id"]))
    return _canonical_value(raw)  # type: ignore[return-value]


def definition_digest(definition: WorkflowDefinition) -> str:
    """Content digest of an immutable definition, including all exact pins."""

    return _sha256_digest(canonical_definition(definition))


def resolution_digest(
    definition: WorkflowDefinition,
    *,
    channel: str,
    pointer_generation: int,
) -> str:
    """Digest of a definition resolved through one CAS-fenced channel."""

    return _sha256_digest(
        {
            "workflow_id": definition.workflow_id,
            "channel": channel,
            "pointer_generation": pointer_generation,
            "version": definition.version,
            "definition_digest": definition.definition_digest,
        }
    )
