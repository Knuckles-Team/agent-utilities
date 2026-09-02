#!/usr/bin/env python3
"""Run privacy-safe semantic and Graph-OS validation for bundled skills.

The harness is deliberately a client of an already deployed Graph-OS endpoint.
It never starts Graph-OS, a model server, or Langfuse.  Cases run sequentially
to keep resource use bounded.  Raw prompts, model output, endpoints, credentials,
trace IDs, and filesystem locations are never written to the report.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import (
    dataclass,
    field,
    is_dataclass,
)
from dataclasses import (
    fields as dataclass_fields,
)
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, TypeGuard

import yaml
from fastmcp.exceptions import ToolError
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, create_model

from agent_utilities.core._env import setting
from agent_utilities.orchestration.run_identity import is_run_id, new_run_id
from agent_utilities.release_catalogs import prebundled_skill_catalog_digest
from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)
from agent_utilities.skills import BUNDLED_SKILLS
from agent_utilities.skills.validation import (
    FORWARD_MATRIX,
    SKILLS_ROOT,
)
from agent_utilities.skills.validation import (
    validate as validate_static_suite,
)

_SAFE_ROUTE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_ERROR = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_MAX_TOOL_PAYLOAD = 64 * 1024
_MAX_TOOL_ITEMS = 4_096
_MAX_TOOL_DEPTH = 24
# Sentinel for "this decoder produced no value", distinct from any JSON value.
_UNDECODED = object()
_ARCHITECTURE_SKILL = "agent-utilities-development"
_ARCHITECTURE_OPERATION_TIMEOUT_SECONDS = 15.0
_ARCHITECTURE_MANIFEST_PATH = "architecture/component-registry.yml"
_ARCHITECTURE_ACTIVE_STATUS = "active"
_ARCHITECTURE_AUTHORITY_STATE = "owner_manifest_authoritative"
_ARCHITECTURE_SOURCE_AUTHORITY = "owner_repository_manifest"
# RF-021 keeps the proposal registry, owner manifests, and generated projection
# as one contract.  These are scenario labels in the synthetic matrix, not a
# second persisted registry schema.
_ARCHITECTURE_WORKFLOW_SCENARIOS = (
    "registry_unavailable",
    "registry_outdated",
    "owner_manifest_identity_disagreement",
    "regeneration_reingestion",
    "finite_exception_metadata",
    "caller_deletion_evidence",
    "concept_discovery_only",
    "plans_cutover",
)
_ARCHITECTURE_LAYOUT_REQUIREMENTS = (
    "layer_boundary_vs_component",
    "parent_layer_no_signature_match",
    "component_owned_roots",
    "worker_lane_shared_file_exception",
)
_ARCHITECTURE_PASS_OUTCOMES = (
    "available",
    "current",
    "matched",
    "not_required",
    "verified",
    "verified",
    "discovery_only",
    "authoritative",
    "implementation_component",
    "verified",
    "verified",
    "verified",
)
# Conceptual phases deliberately map to the existing Graph-OS operation names.
# Local generation, tests, and deletion remain RF-021 evidence obligations; no
# unsupported Graph-OS verb is invented for them.
_ARCHITECTURE_PHASE_OPERATIONS = (
    ("registry_lookup", "graph_query"),
    ("discovery", "graph_search"),
    ("caller_impact", "graph_code"),
)
_TRACE_PAGE_LIMIT = 20
_TRACE_MAX_PAGES = 10
_TRACE_TOOL_ERROR_RETRIES = 2
_TRACE_TOOL_ERROR_RETRY_DELAY_SECONDS = 0.25
_PARENT_INGESTION_POLL_DELAY_SECONDS = 0.25
_DIRECT_MAX_OUTPUT_TOKENS = 1024
_MAX_REPORT_BYTES = 1_000_000
_SKILL_COUNT = len(BUNDLED_SKILLS)
_CASE_COUNT = _SKILL_COUNT * 2
_PASS = "pass"
_FAIL = "fail"
_NA = "not-applicable"
_DIRECT_CASE_LOCK = asyncio.Lock()
_SYNC_CALL_ACTIVE = threading.Lock()
_SYNC_CALL_POISONED = threading.Event()
_AUTHORITY_TRACE_PRECHECK_CAP_SECONDS = 15.0
_AUTHORITY_EXPORT_FLUSH_CAP_SECONDS = 30.0
_AUTHORITY_PARENT_INGESTION_CAP_SECONDS = 15.0
_AUTHORITY_LEASE_SAFETY_SECONDS = 5.0
_AUTHORITY_RENEWAL_TIMEOUT_SECONDS = 30.0
_DIGEST = re.compile(r"^sha256:(?!0{64}$)[a-f0-9]{64}$")
_RELEASE_ID = re.compile(r"^release-[a-z0-9][a-z0-9.-]{2,63}$")
_COMMAND_REFERENCE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_SIGNATURE_ALGORITHMS = frozenset({"ed25519", "ecdsa-p256-sha256", "rsa-pss-sha256"})
_SIGNATURE_VALUE = re.compile(r"^[A-Za-z0-9_-]{43,4096}$")
_KEY_ID = re.compile(r"^key:[a-f0-9]{64}$")
_SIGNER_COMMAND_REFERENCE = "SKILL_VALIDATION_EVIDENCE_SIGNER_COMMAND"
_VERIFIER_COMMAND_REFERENCE = "SKILL_VALIDATION_EVIDENCE_VERIFIER_COMMAND"
_MAX_EXTERNAL_OUTPUT_BYTES = 64 * 1024
_SHELL_EXECUTABLES = frozenset(
    {
        "bash",
        "cmd",
        "cmd.exe",
        "dash",
        "fish",
        "ksh",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "sh",
        "zsh",
    }
)
_CASE_REFERENCE_PATTERNS = {
    "run": re.compile(r"^pref_run_[a-f0-9]{64}$"),
    "trace": re.compile(r"^pref_trace_[a-f0-9]{64}$"),
    "model": re.compile(r"^pref_model_[a-f0-9]{64}$"),
    "skill": re.compile(r"^pref_skill_[a-f0-9]{64}$"),
    "skill_body": re.compile(r"^pref_skill_body_[a-f0-9]{64}$"),
}


class SemanticOutput(BaseModel):
    """Closed response contract used by both execution paths."""

    skill: str = Field(min_length=1, max_length=64)
    mode: Literal["direct", "delegated"]
    selected_routes: list[str] = Field(min_length=1, max_length=16)
    read_only: bool
    privacy_safe: bool
    acceptance_summary: str = Field(
        min_length=1,
        max_length=1_000,
        description=(
            "One short sentence confirming the bounded synthetic validation; "
            "do not reproduce the requested plan or enumerate its steps."
        ),
    )

    model_config = ConfigDict(extra="forbid")


class ArchitectureSharedPath(BaseModel):
    """One finite RF-021 shared-path exception supplied by the owner manifest."""

    path: str = Field(min_length=1, max_length=256)
    kind: Literal["shared_root", "shared_file"]
    owner_component_ids: tuple[str, ...] = Field(min_length=2, max_length=8)
    review_policy: str = Field(min_length=1, max_length=128)
    exception_id: str = Field(pattern=r"^[a-z][a-z0-9.-]{2,127}$")

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArchitectureCandidateIdentity(BaseModel):
    """Exact, digest-bound RF-021 candidate supplied to runtime validation."""

    component_id: str = Field(pattern=r"^[a-z][a-z0-9.-]{2,127}$")
    component_kind: Literal["implementation_component"]
    capability_id: str = Field(pattern=r"^[a-z][a-z0-9.-]{2,127}$")
    parent_component_id: str = Field(pattern=r"^[a-z][a-z0-9.-]{2,127}$")
    parent_layer: str = Field(pattern=r"^[a-z][a-z0-9.-]{1,63}$")
    source_workspace_manifest: Literal["workspace.yml"]
    source_repository_id: str = Field(pattern=r"^[a-z][a-z0-9.-]{2,127}$")
    source_repository_path: str = Field(min_length=1, max_length=256)
    source_manifest_path: Literal["architecture/component-registry.yml"]
    source_revision: str = Field(pattern=r"^[a-f0-9]{40,64}$")
    source_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    authority_signature: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    behavioral_signature: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    dependency_signature: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    identity_policy_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    target_inventory_ref: str = Field(min_length=1, max_length=256)
    owned_source_roots: tuple[str, ...] = Field(min_length=1, max_length=32)
    public_contract_roots: tuple[str, ...] = Field(min_length=1, max_length=32)
    test_roots: tuple[str, ...] = Field(min_length=1, max_length=32)
    generated_roots: tuple[str, ...] = Field(max_length=32)
    shared_paths: tuple[ArchitectureSharedPath, ...] = Field(max_length=16)
    replacement_required: bool
    replaced_component_ids: tuple[str, ...] = Field(max_length=16)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArchitectureRegistryRow(BaseModel):
    """Closed joined ArchitectureComponent/ArchitectureCapability observation."""

    component_id: str
    component_kind: str
    capability_id: str
    implementation_authority_component_id: str
    parent_component_id: str
    parent_layer: str
    source_authority: str
    authority_state: str
    status: str
    source_workspace_manifest: str
    source_repository_id: str
    source_repository_path: str
    source_manifest_path: str
    source_revision: str
    source_digest: str
    authority_signature: str
    behavioral_signature: str
    dependency_signature: str
    identity_policy_digest: str
    target_item_refs: tuple[str, ...]
    owned_source_roots: tuple[str, ...]
    public_contract_roots: tuple[str, ...]
    test_roots: tuple[str, ...]
    generated_roots: tuple[str, ...]
    shared_paths: tuple[ArchitectureSharedPath, ...]
    replaces: tuple[str, ...]
    deletion_proof: str | None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArchitectureDiscoveryRow(BaseModel):
    """Closed advisory discovery row; it can never establish ownership."""

    component_id: str
    capability_id: str
    source_repository_id: str
    source_manifest_path: str
    source_digest: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class DelegationContractError(ValueError):
    """Controlled current-contract diagnostic with no response material."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ValidationChildToolError(RuntimeError):
    """Controlled retryable failure returned by an MCP child tool."""


@dataclass(frozen=True)
class GraphOperationObservation:
    """Bounded metadata retained for one real Graph-OS operation invocation."""

    phase: str
    operation: str
    status: str
    request_digest: str
    response_digest: str
    matched_record_count: int


@dataclass(frozen=True)
class ArchitectureScenarioObservation:
    """One deterministic RF-021 behavioral outcome retained as evidence."""

    scenario: str
    outcome: str


@dataclass(frozen=True)
class ValidationCase:
    """One synthetic case loaded from the checked-in matrix."""

    case_id: str
    skill: str
    mode: Literal["direct", "delegated"]
    model_class: Literal["economy", "standard"]
    task: str = field(repr=False)
    expected_routes: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    read_only: bool
    architecture_candidate: ArchitectureCandidateIdentity | None = None


@dataclass
class CaseResult:
    """Privacy-safe evidence retained for one case."""

    case_id: str
    skill: str
    mode: str
    model_class: str
    model_selection: str = _FAIL
    skill_binding: str = _FAIL
    structural: str = _PASS
    semantic: str = _FAIL
    delegation: str = _NA
    trace: str = _FAIL
    parent_ingestion: str = _FAIL
    trace_linkage: str = "none"
    trace_name: str = ""
    langfuse_match_count: int = 0
    parent_kg_readback_count: int = 0
    selected_routes: tuple[str, ...] = ()
    run_ref: str = ""
    trace_ref: str = ""
    model_ref: str = ""
    skill_ref: str = ""
    skill_body_ref: str = ""
    operation_evidence: tuple[GraphOperationObservation, ...] = ()
    scenario_evidence: tuple[ArchitectureScenarioObservation, ...] = ()
    error_codes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(
            self._required_checks_passed()
            and not self.error_codes
            and self._trace_evidence_exact()
            and self.selected_routes
            and all(_SAFE_ROUTE.fullmatch(route) for route in self.selected_routes)
            and self._all_evidence_exact()
        )

    def _required_checks_passed(self) -> bool:
        """Require every mandatory per-mode check to have recorded a pass."""

        required = [
            self.structural,
            self.semantic,
            self.trace,
            self.parent_ingestion,
            self.model_selection,
            self.skill_binding,
        ]
        if self.mode == "delegated":
            required.append(self.delegation)
        return all(value == _PASS for value in required)

    def _trace_evidence_exact(self) -> bool:
        """Require exactly one run-linked trace and one parent-ingested node."""

        return (
            self.trace_linkage == "run-evidence"
            and self.trace_name == f"graph_run:{self.run_ref}"
            and self.langfuse_match_count == 1
            and self.parent_kg_readback_count == 1
        )

    def _references_valid(self) -> bool:
        """Require every retained opaque reference to match its exact pattern."""

        return all(
            pattern.fullmatch(value) is not None
            for pattern, value in (
                (_CASE_REFERENCE_PATTERNS["run"], self.run_ref),
                (_CASE_REFERENCE_PATTERNS["trace"], self.trace_ref),
                (_CASE_REFERENCE_PATTERNS["model"], self.model_ref),
                (_CASE_REFERENCE_PATTERNS["skill"], self.skill_ref),
                (_CASE_REFERENCE_PATTERNS["skill_body"], self.skill_body_ref),
            )
        )

    def _all_evidence_exact(self) -> bool:
        """Require opaque runtime references and the skill-specific evidence."""

        return self._references_valid() and self._architecture_evidence_exact()

    def _architecture_evidence_exact(self) -> bool:
        """Require the exact joined/discovery/caller evidence for dev-skill cases."""

        if self.skill != _ARCHITECTURE_SKILL:
            return not self.operation_evidence and not self.scenario_evidence
        return (
            self._architecture_operations_exact()
            and self._architecture_scenarios_exact()
        )

    def _architecture_operations_exact(self) -> bool:
        """Require the three candidate-bound tool observations in contract order."""

        expected = tuple(
            (phase, operation, status)
            for (phase, operation), status in zip(
                _ARCHITECTURE_PHASE_OPERATIONS,
                ("verified", "advisory", "grounded"),
                strict=True,
            )
        )
        observed = tuple(
            (item.phase, item.operation, item.status)
            for item in self.operation_evidence
        )
        return (
            observed == expected
            and tuple(item.matched_record_count for item in self.operation_evidence[:2])
            == (1, 1)
            and 1 <= self.operation_evidence[2].matched_record_count <= 32
            and all(
                _DIGEST.fullmatch(item.request_digest) is not None
                and _DIGEST.fullmatch(item.response_digest) is not None
                for item in self.operation_evidence
            )
        )

    def _architecture_scenarios_exact(self) -> bool:
        """Require every structured scenario and no failing outcome."""

        expected = tuple(
            ArchitectureScenarioObservation(scenario, outcome)
            for scenario, outcome in zip(
                architecture_workflow_scenarios(),
                _ARCHITECTURE_PASS_OUTCOMES,
                strict=True,
            )
        )
        return self.scenario_evidence == expected

    def add_error(self, code: str) -> None:
        normalized = re.sub(r"[^a-z0-9_]+", "_", code.casefold()).strip("_")
        if not _SAFE_ERROR.fullmatch(normalized):
            normalized = "validation_error"
        if normalized not in self.error_codes:
            self.error_codes.append(normalized)


@dataclass(frozen=True)
class TraceRecord:
    """Metadata-only evidence retained from one exact-name trace lookup."""

    name: str
    evidence: dict[str, str]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_digest(value: str, field: str) -> str:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field}_invalid")
    return value


def _architecture_candidate_ref(candidate: ArchitectureCandidateIdentity) -> str:
    """Return a content-bound reference without persisting candidate identities."""

    digest = hashlib.sha256(
        _canonical_bytes(candidate.model_dump(mode="json"))
    ).hexdigest()
    return f"pref_architecture_candidate_{digest}"


def _canonical_owner_path(value: str) -> bool:
    """Return whether an owner root is finite, relative, and non-patterned."""

    path = Path(value)
    return bool(
        value
        and len(value) <= 256
        and not path.is_absolute()
        and value == path.as_posix()
        and all(part not in {"", ".", ".."} for part in path.parts)
        and not any(marker in value for marker in ("*", "?", "[", "]", "\x00"))
    )


def _owner_paths_overlap(left: str, right: str) -> bool:
    """Return whether two canonical owner paths contain one another."""

    return bool(
        left == right
        or left.startswith(f"{right.rstrip('/')}/")
        or right.startswith(f"{left.rstrip('/')}/")
    )


def _validate_architecture_candidate(
    candidate: ArchitectureCandidateIdentity,
) -> ArchitectureCandidateIdentity:
    """Fail closed on a broad, overlapping, or internally inconsistent candidate."""

    _validate_architecture_candidate_digests(candidate)
    roots = _validate_architecture_candidate_roots(candidate)
    _validate_architecture_candidate_identity(candidate)
    _validate_architecture_candidate_shared_paths(candidate, roots)
    return candidate


def _validate_architecture_candidate_digests(
    candidate: ArchitectureCandidateIdentity,
) -> None:
    """Require every externally supplied content identity to be non-sentinel."""

    digest_fields = (
        "source_digest",
        "authority_signature",
        "behavioral_signature",
        "dependency_signature",
        "identity_policy_digest",
    )
    invalid = next(
        (
            field_name
            for field_name in digest_fields
            if _DIGEST.fullmatch(str(getattr(candidate, field_name))) is None
        ),
        None,
    )
    if invalid is not None:
        raise ValueError(f"architecture_candidate_{invalid}_invalid")


def _validate_architecture_candidate_roots(
    candidate: ArchitectureCandidateIdentity,
) -> tuple[str, ...]:
    """Require unique finite roots and isolate generated from handwritten paths."""

    root_groups = (
        candidate.owned_source_roots,
        candidate.public_contract_roots,
        candidate.test_roots,
        candidate.generated_roots,
    )
    _validate_architecture_root_groups(root_groups)
    _validate_generated_root_isolation(root_groups[:3], candidate.generated_roots)
    flattened = [root for group in root_groups for root in group]
    return tuple(flattened)


def _validate_architecture_root_groups(
    root_groups: tuple[tuple[str, ...], ...],
) -> None:
    """Require every owner root to be finite and unique within its role."""

    if any(not _canonical_owner_path(root) for group in root_groups for root in group):
        raise ValueError("architecture_candidate_root_invalid")
    if any(len(group) != len(set(group)) for group in root_groups):
        raise ValueError("architecture_candidate_root_duplicate")


def _validate_generated_root_isolation(
    handwritten_groups: tuple[tuple[str, ...], ...], generated_roots: tuple[str, ...]
) -> None:
    """Keep generated ownership disjoint from handwritten owner roots."""

    if any(
        _owner_paths_overlap(handwritten_root, generated_root)
        for group in handwritten_groups
        for handwritten_root in group
        for generated_root in generated_roots
    ):
        raise ValueError("architecture_candidate_generated_root_overlap")


def _validate_architecture_candidate_identity(
    candidate: ArchitectureCandidateIdentity,
) -> None:
    """Require canonical parent, manifest, and replacement identity semantics."""

    if candidate.parent_component_id == candidate.component_id:
        raise ValueError("architecture_candidate_parent_invalid")
    if candidate.source_manifest_path != _ARCHITECTURE_MANIFEST_PATH:
        raise ValueError("architecture_candidate_manifest_path_invalid")
    if candidate.replacement_required != bool(candidate.replaced_component_ids):
        raise ValueError("architecture_candidate_replacement_contract_invalid")


def _validate_architecture_candidate_shared_paths(
    candidate: ArchitectureCandidateIdentity, exclusive_roots: tuple[str, ...]
) -> None:
    """Require complete co-ownership and disjoint shared-path exceptions."""

    for shared in candidate.shared_paths:
        _validate_architecture_shared_path(candidate, shared, exclusive_roots)


def _validate_architecture_shared_path(
    candidate: ArchitectureCandidateIdentity,
    shared: ArchitectureSharedPath,
    exclusive_roots: tuple[str, ...],
) -> None:
    """Validate one finite shared-root or shared-file exception."""

    if not _canonical_owner_path(shared.path):
        raise ValueError("architecture_candidate_shared_path_invalid")
    if candidate.component_id not in shared.owner_component_ids:
        raise ValueError("architecture_candidate_shared_owner_missing")
    if len(shared.owner_component_ids) != len(set(shared.owner_component_ids)):
        raise ValueError("architecture_candidate_shared_owner_duplicate")
    if any(_owner_paths_overlap(shared.path, root) for root in exclusive_roots):
        raise ValueError("architecture_candidate_shared_exclusive_overlap")


def _case_contract(case: ValidationCase) -> dict[str, Any]:
    """Return the content-free canonical contract bound into release evidence."""

    contract = {
        "id": case.case_id,
        "skill": case.skill,
        "mode": case.mode,
        "modelClass": case.model_class,
        "taskDigest": _digest_bytes(case.task.encode("utf-8")),
        "expectedRoutes": list(case.expected_routes),
        "allowedTools": list(case.allowed_tools),
        "readOnly": case.read_only,
    }
    contract.update(_architecture_contract_binding(case))
    return contract


def _architecture_contract_binding(case: ValidationCase) -> dict[str, str]:
    """Bind development cases to a candidate without changing other contracts."""

    if case.architecture_candidate is None:
        return {}
    return {
        "architectureCandidateRef": _architecture_candidate_ref(
            case.architecture_candidate
        )
    }


def _test_catalog_evidence(cases: list[ValidationCase]) -> dict[str, Any]:
    matrix = yaml.safe_load(FORWARD_MATRIX.read_text(encoding="utf-8"))
    contracts = [_case_contract(case) for case in cases]
    if (
        len(contracts) != _CASE_COUNT
        or len({item["id"] for item in contracts}) != _CASE_COUNT
    ):
        raise RuntimeError("test_catalog_not_exact")
    case_digests = {
        item["id"]: _digest_bytes(_canonical_bytes(item)) for item in contracts
    }
    return {
        "testCatalogDigest": _digest_bytes(_canonical_bytes(matrix)),
        "caseCatalogDigest": _digest_bytes(
            _canonical_bytes(
                [
                    {"caseId": case_id, "caseDigest": case_digests[case_id]}
                    for case_id in sorted(case_digests)
                ]
            )
        ),
        "caseDigests": case_digests,
    }


def load_matrix() -> tuple[dict[str, int | bool], list[ValidationCase]]:
    """Load the already statically validated version-2 matrix."""

    raw = yaml.safe_load(FORWARD_MATRIX.read_text(encoding="utf-8")) or {}
    defaults = dict(raw["runtime_defaults"])
    cases = [
        ValidationCase(
            case_id=str(item["id"]),
            skill=str(item["skill"]),
            mode=str(item["mode"]),  # type: ignore[arg-type]
            model_class=str(item["model_class"]),  # type: ignore[arg-type]
            task=str(item["task"]),
            expected_routes=tuple(str(route) for route in item["expected_routes"]),
            allowed_tools=tuple(str(tool) for tool in item["allowed_tools"]),
            read_only=bool(item["read_only"]),
            architecture_candidate=_architecture_candidate_from_matrix(item),
        )
        for item in raw["cases"]
    ]
    return defaults, cases


def _architecture_candidate_from_matrix(
    item: dict[str, Any],
) -> ArchitectureCandidateIdentity | None:
    """Parse one optional candidate through the closed runtime model."""

    if "architecture_candidate" not in item:
        return None
    candidate = ArchitectureCandidateIdentity.model_validate(
        item["architecture_candidate"]
    )
    return _validate_architecture_candidate(candidate)


def _skill_body(skill: str) -> str:
    text = (SKILLS_ROOT / skill / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n.*?\n---\n(.*)$", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _skill_instruction_digest(skill: str) -> str:
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        runnable_skill_digest,
    )

    return runnable_skill_digest(_skill_runtime_body(skill))


def _skill_runtime_body(skill: str) -> str:
    body, _privacy = PersistencePrivacyGuard().sanitize_text(_skill_body(skill))
    return body


class _SkillValidationEvidenceSource:
    """Bounded authoritative evidence for an isolated direct validation case.

    Direct cases execute outside the Graph-OS process, but authenticated model
    calls still have to cross the mandatory ContextCompiler boundary.  Supplying
    the checked-in, privacy-sanitized skill body as the only candidate preserves
    that production invariant without opening a second graph engine or granting
    the validator ambient access to runtime data.
    """

    def __init__(self, skill: str) -> None:
        self._skill = skill
        self._body = _skill_runtime_body(skill)
        self.node_id = persistence_reference(
            "skill", skill, namespace="skill-validation-evidence"
        )

    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int = 8,
        as_of: str | None = None,
        session: Any | None = None,
    ) -> list[dict[str, Any]]:
        del query, as_of, session
        if top_k < 1:
            return []
        return [
            {
                "id": self.node_id,
                "kind": "skill_instruction",
                "content": self._body,
                "score": 1.0,
                "confidence": 1.0,
                "source_refs": [f"skill://{self._skill}"],
            }
        ]

    def retrieve_epistemic_view(self, query: str, *, top_k: int = 8) -> dict[str, Any]:
        del query, top_k
        return {}


class _ReadOnlyValidationMarkingStore:
    """Empty mandatory-marking authority for one synthetic evidence source."""

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        del params
        normalized = " ".join(str(query or "").split()).casefold()
        if not normalized.startswith("match ") or " return " not in normalized:
            raise PermissionError("skill_validation_marking_store_is_read_only")
        return []


@contextmanager
def _direct_evidence_authority(
    skill: str,
) -> Iterator[_SkillValidationEvidenceSource]:
    """Scope the explicit compiler, ACL, and marking authority for one case."""

    from agent_utilities.core.contextual_model import use_context_compiler_engine
    from agent_utilities.knowledge_graph.core.company_brain_runtime import (
        get_company_brain,
    )
    from agent_utilities.knowledge_graph.ontology.permissioning import (
        use_marking_authority,
    )
    from agent_utilities.models.company_brain import (
        ActorType,
        DataClassification,
        NodeACL,
    )

    source = _SkillValidationEvidenceSource(skill)
    permissions = get_company_brain().permissions
    acl = NodeACL(
        node_id=source.node_id,
        classification=DataClassification.INTERNAL,
        read_roles=["kg:admin"],
        data_owner="skill-validation-authority",
        data_owner_type=ActorType.SYSTEM,
    )
    with (
        use_marking_authority(_ReadOnlyValidationMarkingStore()),
        permissions.use_acl(acl),
        use_context_compiler_engine(source),
    ):
        yield source


def _contract_instruction(case: ValidationCase) -> str:
    routes = ", ".join(case.expected_routes)
    return (
        "This is a synthetic, read-only validation. Do not mutate state, create "
        "schedules, contact external systems, reveal configuration, or reproduce "
        "the skill text. Apply the skill to the synthetic task internally; the "
        "JSON validation envelope is the only response artifact. Return "
        "only one JSON object with exactly these keys: skill, mode, "
        "selected_routes, read_only, privacy_safe, acceptance_summary. Set mode to "
        f"{case.mode!r}. Set skill to {case.skill!r}. The statically certified "
        "route contract for this case is "
        f"[{routes}]; copy each of those operation slugs exactly once into "
        "selected_routes and add no other route. Set read_only and privacy_safe "
        "to true. Keep acceptance_summary to one plain sentence of at most 240 "
        "characters; it confirms the validation and does not reproduce the plan. "
        "Do not include paths, endpoints, identities, "
        "credentials, source records, or trace identifiers."
    )


def _direct_execution_prompt(case: ValidationCase) -> str:
    """Place the closed response contract after the synthetic direct task.

    The skill body and contract remain system instructions.  Repeating the
    contract after the task also keeps the final user-level instruction aligned
    with the prompted-output schema, preventing plan-shaped task wording from
    displacing the required JSON response on smaller local models.
    """

    return f"{case.task}\n\n{_contract_instruction(case)}"


def _direct_semantic_output_type(case: ValidationCase) -> Any:
    """Return the provider-neutral, case-exact JSON contract for direct validation.

    The direct system instruction requires a bare JSON object.  PydanticAI's
    default model output protocol is a tool call, which conflicts with that
    instruction and is not uniformly implemented by local OpenAI-compatible
    runtimes.  Prompted output makes the wire contract match the instruction
    while retaining Pydantic validation and bounded output retries.  The
    per-case route enum, cardinality, and set validator move the closed route
    contract into model-output validation so an otherwise well-formed response
    with extra or missing routes is retried instead of failing only after the
    model run has completed.
    """

    from pydantic_ai import PromptedOutput

    expected_routes = tuple(case.expected_routes)
    expected_route_set = frozenset(expected_routes)

    def validate_exact_routes(routes: list[str]) -> list[str]:
        if len(routes) != len(expected_routes) or set(routes) != expected_route_set:
            raise ValueError("selected_routes_must_match_case_contract")
        return routes

    route_literal = Literal.__getitem__(expected_routes)
    selected_routes_type = Annotated[
        # `route_literal` is a Literal type built from a runtime tuple (the
        # per-case route set), so it can never be a statically-recognized
        # type alias -- mypy's "variable vs type alias" rule (valid-type)
        # rejects any name used here that wasn't defined via a literal
        # `Literal[...]`/`Union[...]` expression, regardless of annotation
        # or cast. This is unavoidable runtime type construction, not a bug.
        list[route_literal],  # type: ignore[valid-type]
        Field(min_length=len(expected_routes), max_length=len(expected_routes)),
        AfterValidator(validate_exact_routes),
    ]
    case_output = create_model(
        "DirectSemanticOutput",
        __base__=SemanticOutput,
        selected_routes=(selected_routes_type, ...),
    )

    return PromptedOutput(
        case_output,
        name="bundled skill validation",
        description="Return the closed, privacy-safe synthetic validation result.",
        template=(
            "Return exactly one JSON object that validates against this JSON Schema. "
            "Do not wrap it in Markdown or add text before or after it.\n{schema}"
        ),
    )


def validate_semantic_output(case: ValidationCase, output: SemanticOutput) -> list[str]:
    """Return controlled error codes without retaining raw model output."""

    errors: list[str] = []
    if output.skill != case.skill:
        errors.append("semantic_skill_mismatch")
    if output.mode != case.mode:
        errors.append("semantic_mode_mismatch")
    if not output.read_only or not case.read_only:
        errors.append("semantic_not_read_only")
    if not output.privacy_safe:
        errors.append("semantic_privacy_not_acknowledged")
    errors.extend(_semantic_route_errors(case, output.selected_routes))
    _clean, privacy = PersistencePrivacyGuard().sanitize(output.model_dump())
    if privacy.changed:
        errors.append("semantic_output_privacy_violation")
    return errors


def _semantic_route_errors(case: ValidationCase, routes: list[str]) -> list[str]:
    """Return the route-contract error codes in their declared report order."""

    errors: list[str] = []
    if len(routes) != len(set(routes)) or any(
        not _SAFE_ROUTE.fullmatch(route) for route in routes
    ):
        errors.append("semantic_routes_invalid")
    expected_routes = set(case.expected_routes)
    selected_routes = set(routes)
    if not expected_routes.issubset(selected_routes):
        errors.append("semantic_routes_incomplete")
    if selected_routes - expected_routes:
        errors.append("semantic_routes_unexpected")
    return errors


def _parse_json_text(value: str) -> Any:
    if len(value) > _MAX_TOOL_PAYLOAD:
        raise ValueError("payload_too_large")
    text = value.strip()
    if len(text.encode("utf-8")) > _MAX_TOOL_PAYLOAD:
        raise ValueError("payload_too_large")
    parsed = json.loads(text)
    _validate_tool_payload_bounds(parsed)
    return parsed


class _PayloadScan:
    """One bounded, cycle-safe traversal budget for an MCP payload tree.

    The check order per node is load-bearing and matches the original inline
    traversal exactly: depth, then item count, then the per-type charge (which
    for a container is cycle, then width, then expansion), then the remaining
    byte budget.
    """

    __slots__ = ("items", "remaining", "seen")

    def __init__(self) -> None:
        self.remaining = _MAX_TOOL_PAYLOAD
        self.items = 0
        self.seen: set[int] = set()

    def visit(self, current: Any, depth: int, stack: list[tuple[Any, int]]) -> None:
        """Charge one popped node against the budget and queue its children."""

        if depth > _MAX_TOOL_DEPTH:
            raise ValueError("payload_too_deep")
        self.items += 1
        if self.items > _MAX_TOOL_ITEMS:
            raise ValueError("payload_too_many_items")
        self._charge(current, depth, stack)
        if self.remaining < 0:
            raise ValueError("payload_too_large")

    def _charge(self, current: Any, depth: int, stack: list[tuple[Any, int]]) -> None:
        if current is None or isinstance(current, bool | int | float):
            self.remaining -= 16
        elif isinstance(current, str):
            self._charge_text(current)
        elif isinstance(current, bytes):
            self.remaining -= len(current)
        elif isinstance(current, dict):
            self._expand_mapping(current, depth, stack)
        elif isinstance(current, list | tuple):
            self._expand_sequence(current, depth, stack)
        elif _is_fastmcp_structured_dataclass(current):
            self._expand_structured(current, depth, stack)
        else:
            raise TypeError("payload_type_invalid")

    def _charge_text(self, current: str) -> None:
        if len(current) > self.remaining:
            raise ValueError("payload_too_large")
        self.remaining -= len(current.encode("utf-8"))

    def _enter_container(self, current: Any, width: int) -> None:
        """Reject a cycle, then an over-wide container, before expanding it."""

        identity = id(current)
        if identity in self.seen:
            raise ValueError("payload_cycle")
        self.seen.add(identity)
        if width > _MAX_TOOL_ITEMS - self.items:
            raise ValueError("payload_too_many_items")

    def _expand_mapping(
        self, current: dict[Any, Any], depth: int, stack: list[tuple[Any, int]]
    ) -> None:
        self._enter_container(current, len(current))
        for key, item in current.items():
            if not isinstance(key, str):
                raise TypeError("payload_key_invalid")
            stack.append((item, depth + 1))
            stack.append((key, depth + 1))

    def _expand_sequence(
        self,
        current: list[Any] | tuple[Any, ...],
        depth: int,
        stack: list[tuple[Any, int]],
    ) -> None:
        self._enter_container(current, len(current))
        stack.extend((item, depth + 1) for item in current)

    def _expand_structured(
        self, current: Any, depth: int, stack: list[tuple[Any, int]]
    ) -> None:
        members = dataclass_fields(current)
        self._enter_container(current, len(members))
        for member in members:
            stack.append((getattr(current, member.name), depth + 1))
            stack.append((member.name, depth + 1))


def _validate_tool_payload_bounds(value: Any) -> None:
    """Reject oversized, cyclic, deep, or non-data MCP payloads before use."""

    scan = _PayloadScan()
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        scan.visit(current, depth, stack)


def _is_fastmcp_structured_dataclass(value: Any) -> bool:
    """Recognize only FastMCP's validated JSON-schema result containers."""

    return bool(
        not isinstance(value, type)
        and is_dataclass(value)
        and type(value).__module__ == "fastmcp.utilities.json_schema_type"
    )


def _normalize_fastmcp_structured_data(value: Any) -> Any:
    """Convert an already bounded FastMCP result tree into plain JSON data."""

    if _is_fastmcp_structured_dataclass(value):
        return {
            member.name: _normalize_fastmcp_structured_data(getattr(value, member.name))
            for member in dataclass_fields(value)
        }
    if isinstance(value, dict):
        return {
            key: _normalize_fastmcp_structured_data(item) for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_normalize_fastmcp_structured_data(item) for item in value]
    return value


def _decode_structured_value(value: Any) -> Any:
    """Decode one non-empty ``data``/``structured_content`` attribute."""

    if isinstance(value, str):
        try:
            return _parse_json_text(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    _validate_tool_payload_bounds(value)
    return _normalize_fastmcp_structured_data(value)


def _decode_structured_attributes(result: Any) -> Any:
    """Decode the first populated structured attribute, or ``_UNDECODED``."""

    for attr in ("data", "structured_content"):
        value = getattr(result, attr, None)
        if value not in (None, {}):
            return _decode_structured_value(value)
    return _UNDECODED


def _bounded_content_texts(content: list[Any]) -> list[str]:
    """Collect the bounded text blocks of an MCP content list."""

    texts: list[str] = []
    characters = 0
    for item in content:
        text = str(getattr(item, "text", ""))
        characters += len(text) + (1 if text and texts else 0)
        if characters > _MAX_TOOL_PAYLOAD:
            raise ValueError("payload_too_large")
        if text:
            texts.append(text)
    return texts


def _decode_content_list(content: list[Any]) -> Any:
    """Decode an MCP content list, or ``_UNDECODED`` when it carries no text."""

    if len(content) > _MAX_TOOL_ITEMS:
        raise ValueError("payload_too_many_items")
    joined = "\n".join(_bounded_content_texts(content))
    if not joined:
        return _UNDECODED
    try:
        return _parse_json_text(joined)
    except json.JSONDecodeError:
        return joined


def _decode_text_result(result: str) -> Any:
    """Decode a bare string result as JSON, or as bounded text."""

    try:
        return _parse_json_text(result)
    except json.JSONDecodeError:
        if len(result) > _MAX_TOOL_PAYLOAD:
            raise ValueError("payload_too_large") from None
        return result


def _decode_tool_result(result: Any) -> Any:
    decoded = _decode_structured_attributes(result)
    if decoded is not _UNDECODED:
        return decoded
    content = getattr(result, "content", None)
    if isinstance(content, list):
        decoded = _decode_content_list(content)
        if decoded is not _UNDECODED:
            return decoded
    if isinstance(result, str):
        return _decode_text_result(result)
    _validate_tool_payload_bounds(result)
    return result


def _extract_delegation_envelope(value: Any) -> tuple[Any, str]:
    """Validate the current outer ``graph_orchestrate`` contract exactly."""

    # Depending on the negotiated MCP result schema, a string-returning tool's
    # structured payload may itself be the JSON string rather than the
    # ``{"result": ...}`` object below.
    if isinstance(value, str):
        value = _parse_json_text(value)
    # FastMCP exposes a tool annotated as returning ``str`` through the current
    # structured-result envelope.  Unwrap that wire-level representation once;
    # the contained GraphOS object is still validated against the sole strict
    # delegation contract below.
    if (
        isinstance(value, dict)
        and set(value) == {"result"}
        and isinstance(value["result"], str)
    ):
        value = _parse_json_text(value["result"])
    if not isinstance(value, dict):
        raise DelegationContractError("delegation_response_not_object")
    allowed = {"output", "run_id", "mermaid"}
    if not {"output", "run_id"}.issubset(value) or set(value) - allowed:
        raise DelegationContractError("delegation_response_schema_invalid")
    run_id = str(value["run_id"] or "")
    if not is_run_id(run_id):
        raise DelegationContractError("delegation_run_id_invalid")
    return value["output"], run_id


def _semantic_from_delegation_output(output: Any) -> SemanticOutput:
    if isinstance(output, str):
        try:
            output = _parse_json_text(output)
        except json.JSONDecodeError as exc:
            raise DelegationContractError("delegation_output_not_json") from exc
    return SemanticOutput.model_validate(output)


def _extract_semantic_payload(value: Any) -> tuple[SemanticOutput, str]:
    """Validate the outer envelope and closed semantic response contract."""

    output, run_id = _extract_delegation_envelope(value)
    return _semantic_from_delegation_output(output), run_id


def _usage_counts(run_result: Any) -> dict[str, int]:
    usage_fn = getattr(run_result, "usage", None)
    usage = usage_fn() if callable(usage_fn) else usage_fn
    if usage is None:
        return {}

    def count(*names: str) -> int:
        for name in names:
            raw = getattr(usage, name, None)
            if raw is not None:
                try:
                    return max(0, int(raw))
                except (TypeError, ValueError):
                    return 0
        return 0

    prompt = count("input_tokens", "request_tokens", "prompt_tokens")
    response = count("output_tokens", "response_tokens", "completion_tokens")
    total = count("total_tokens") or prompt + response
    return {"prompt": prompt, "response": response, "total": total}


async def _call_tool(
    client: Any, name: str, arguments: dict[str, Any], timeout: float
) -> Any:
    result = await asyncio.wait_for(
        client.call_tool(name, arguments), timeout=max(1.0, timeout)
    )
    # MCP's wire/model field is ``isError``. Some client adapters expose the
    # snake-case convenience alias; honor both so a child failure can never be
    # decoded as a successful (usually empty) payload.
    if bool(getattr(result, "isError", False)) or bool(
        getattr(result, "is_error", False)
    ):
        raise ValidationChildToolError("mcp_tool_error")
    return _decode_tool_result(result)


def architecture_workflow_scenarios() -> tuple[str, ...]:
    """Return the deterministic RF-021 scenario labels in matrix order."""

    return _ARCHITECTURE_WORKFLOW_SCENARIOS + _ARCHITECTURE_LAYOUT_REQUIREMENTS


def _architecture_operation_specs(
    candidate: ArchitectureCandidateIdentity,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    """Build candidate-bound requests for the three real Graph-OS operations."""

    query = (
        "MATCH (component:ArchitectureComponent {component_id: $component_id})"
        "-[:IMPLEMENTS]->"
        "(capability:ArchitectureCapability {capability_id: $capability_id}) "
        "WHERE component.source_repository_id = $source_repository_id "
        "RETURN component.component_id AS component_id, "
        "component.component_kind AS component_kind, "
        "capability.capability_id AS capability_id, "
        "capability.implementation_authority_component_id "
        "AS implementation_authority_component_id, "
        "component.parent_component_id AS parent_component_id, "
        "component.parent_layer AS parent_layer, "
        "component.source_authority AS source_authority, "
        "component.authority_state AS authority_state, "
        "component.status AS status, "
        "component.source_workspace_manifest AS source_workspace_manifest, "
        "component.source_repository_id AS source_repository_id, "
        "component.source_repository_path AS source_repository_path, "
        "component.source_manifest_path AS source_manifest_path, "
        "component.source_revision AS source_revision, "
        "component.source_digest AS source_digest, "
        "component.authority_signature AS authority_signature, "
        "component.behavioral_signature AS behavioral_signature, "
        "component.dependency_signature AS dependency_signature, "
        "component.identity_policy_digest AS identity_policy_digest, "
        "component.target_item_refs AS target_item_refs, "
        "component.owned_source_roots AS owned_source_roots, "
        "component.public_contract_roots AS public_contract_roots, "
        "component.test_roots AS test_roots, "
        "component.generated_roots AS generated_roots, "
        "component.shared_paths AS shared_paths, "
        "component.replaces AS replaces, "
        "component.deletion_proof AS deletion_proof LIMIT 2"
    )
    params = json.dumps(
        {
            "capability_id": candidate.capability_id,
            "component_id": candidate.component_id,
            "source_repository_id": candidate.source_repository_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    exact_identity = f"{candidate.component_id} {candidate.capability_id}"
    caller_query = (
        f"live callers of {candidate.capability_id} implemented by "
        f"{candidate.component_id}"
    )
    return (
        (
            "registry_lookup",
            "graph_query",
            {"cypher": query, "params": params, "scope": "local"},
        ),
        (
            "discovery",
            "graph_search",
            {"query": exact_identity, "mode": "hybrid", "top_k": 8},
        ),
        (
            "caller_impact",
            "graph_code",
            {
                "action": "code_context",
                "query": caller_query,
                "target": "usage",
                "top_k": 8,
            },
        ),
    )


def _architecture_payload_rows(payload: Any, key: str, limit: int) -> list[Any]:
    """Read one exact bounded row collection and reject text-shaped success."""

    if not isinstance(payload, dict) or key not in payload:
        raise ValueError("architecture_response_schema_invalid")
    rows = payload[key]
    if not isinstance(rows, list) or len(rows) > limit:
        raise ValueError("architecture_response_rows_invalid")
    return rows


def _architecture_registry_row(
    candidate: ArchitectureCandidateIdentity, payload: Any
) -> ArchitectureRegistryRow:
    """Return the unique authoritative joined row for the exact candidate."""

    rows = _architecture_payload_rows(payload, "rows", 2)
    if not rows:
        raise ValueError("architecture_registry_unavailable")
    if len(rows) != 1:
        raise ValueError("architecture_registry_duplicate_authority")
    try:
        row = ArchitectureRegistryRow.model_validate(rows[0])
    except Exception as exc:
        raise ValueError("architecture_registry_row_invalid") from exc
    _verify_architecture_registry_identity(candidate, row)
    _verify_architecture_registry_ownership(candidate, row)
    _verify_architecture_registry_replacement(candidate, row)
    return row


def _architecture_registry_expected_identity(
    candidate: ArchitectureCandidateIdentity,
) -> dict[str, str]:
    """Return the exact owner-manifest and signature identity to compare."""

    return {
        "component_id": candidate.component_id,
        "component_kind": candidate.component_kind,
        "capability_id": candidate.capability_id,
        "implementation_authority_component_id": candidate.component_id,
        "parent_component_id": candidate.parent_component_id,
        "parent_layer": candidate.parent_layer,
        "source_authority": _ARCHITECTURE_SOURCE_AUTHORITY,
        "authority_state": _ARCHITECTURE_AUTHORITY_STATE,
        "status": _ARCHITECTURE_ACTIVE_STATUS,
        "source_workspace_manifest": candidate.source_workspace_manifest,
        "source_repository_id": candidate.source_repository_id,
        "source_repository_path": candidate.source_repository_path,
        "source_manifest_path": _ARCHITECTURE_MANIFEST_PATH,
        "source_revision": candidate.source_revision,
        "source_digest": candidate.source_digest,
        "authority_signature": candidate.authority_signature,
        "behavioral_signature": candidate.behavioral_signature,
        "dependency_signature": candidate.dependency_signature,
        "identity_policy_digest": candidate.identity_policy_digest,
    }


def _verify_architecture_registry_identity(
    candidate: ArchitectureCandidateIdentity, row: ArchitectureRegistryRow
) -> None:
    """Reject stale/proposal state separately from source identity disagreement."""

    expected = _architecture_registry_expected_identity(candidate)
    observed = row.model_dump(mode="json")
    disagreements = {
        field_name
        for field_name, value in expected.items()
        if observed[field_name] != value
    }
    if not disagreements:
        return
    stale_fields = {
        "status",
        "source_revision",
        "source_authority",
        "authority_state",
    }
    if disagreements.intersection(stale_fields):
        raise ValueError("architecture_registry_outdated")
    raise ValueError("architecture_owner_manifest_identity_disagreement")


def _verify_architecture_registry_ownership(
    candidate: ArchitectureCandidateIdentity, row: ArchitectureRegistryRow
) -> None:
    """Verify the sole target-inventory link and every finite owner root."""

    if row.target_item_refs != (candidate.target_inventory_ref,):
        raise ValueError("architecture_target_inventory_link_invalid")
    expected = (
        candidate.owned_source_roots,
        candidate.public_contract_roots,
        candidate.test_roots,
        candidate.generated_roots,
        candidate.shared_paths,
    )
    observed = (
        row.owned_source_roots,
        row.public_contract_roots,
        row.test_roots,
        row.generated_roots,
        row.shared_paths,
    )
    if observed != expected:
        raise ValueError("architecture_owner_roots_invalid")


def _verify_architecture_registry_replacement(
    candidate: ArchitectureCandidateIdentity, row: ArchitectureRegistryRow
) -> None:
    """Require exact replacement identities and deletion proof when applicable."""

    if row.replaces != candidate.replaced_component_ids:
        raise ValueError("architecture_replacement_identity_invalid")
    if candidate.replacement_required:
        if row.deletion_proof is None or _DIGEST.fullmatch(row.deletion_proof) is None:
            raise ValueError("architecture_deletion_proof_missing")
        return
    if row.deletion_proof is not None:
        raise ValueError("architecture_unrelated_deletion_proof")


def _architecture_discovery_rows(
    candidate: ArchitectureCandidateIdentity, payload: Any
) -> tuple[ArchitectureDiscoveryRow, ...]:
    """Validate exact advisory discovery rows without treating them as ownership."""

    rows = _architecture_payload_rows(payload, "results", 8)
    if not rows:
        raise ValueError("architecture_discovery_unavailable")
    try:
        parsed = tuple(ArchitectureDiscoveryRow.model_validate(row) for row in rows)
    except Exception as exc:
        raise ValueError("architecture_discovery_row_invalid") from exc
    if len(parsed) != 1:
        raise ValueError("architecture_discovery_unrelated_or_duplicate")
    row = parsed[0]
    if (
        row.component_id != candidate.component_id
        or row.capability_id != candidate.capability_id
        or row.source_repository_id != candidate.source_repository_id
        or row.source_manifest_path != _ARCHITECTURE_MANIFEST_PATH
        or row.source_digest != candidate.source_digest
    ):
        raise ValueError("architecture_discovery_unrelated_or_duplicate")
    return parsed


def _path_belongs_to_candidate(
    candidate: ArchitectureCandidateIdentity, value: str
) -> bool:
    roots = (
        *candidate.owned_source_roots,
        *candidate.public_contract_roots,
        *candidate.test_roots,
    )
    return _canonical_owner_path(value) and any(
        value == root or value.startswith(f"{root.rstrip('/')}/") for root in roots
    )


def _architecture_caller_count(
    candidate: ArchitectureCandidateIdentity, payload: Any
) -> int:
    """Require grounded, owner-scoped caller rows from the typed code-context bundle."""

    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError("architecture_caller_evidence_unavailable")
    spans = payload.get("evidence_spans")
    trace = payload.get("reasoning_trace")
    if not isinstance(spans, list) or not 1 <= len(spans) <= 32:
        raise ValueError("architecture_caller_evidence_invalid")
    if not isinstance(trace, list) or len(trace) > 64:
        raise ValueError("architecture_caller_evidence_invalid")
    cited = _architecture_citations(candidate, spans)
    callers = _architecture_callers(trace)
    grounded = sum(
        _architecture_caller_grounded(caller, cited) for caller in callers[:32]
    )
    if grounded < 1:
        raise ValueError("architecture_live_caller_missing")
    return grounded


def _architecture_citations(
    candidate: ArchitectureCandidateIdentity, spans: list[Any]
) -> set[tuple[str, int]]:
    """Return only bounded owner-scoped file/line citations."""

    return {
        (str(span.get("file") or ""), int(span.get("line") or 0))
        for span in spans
        if isinstance(span, dict)
        and isinstance(span.get("line"), int)
        and _path_belongs_to_candidate(candidate, str(span.get("file") or ""))
    }


def _architecture_callers(trace: list[Any]) -> list[Any]:
    """Extract caller rows only from the typed code-context sections step."""

    callers: list[Any] = []
    for step in trace:
        if not isinstance(step, dict) or step.get("step") != "sections":
            continue
        sections = step.get("sections")
        if isinstance(sections, dict) and isinstance(sections.get("callers"), list):
            callers.extend(sections["callers"])
    return callers


def _architecture_caller_grounded(caller: Any, citations: set[tuple[str, int]]) -> bool:
    """Return whether one caller has an identical retained citation."""

    if not isinstance(caller, dict) or not isinstance(caller.get("line"), int):
        return False
    citation = (str(caller.get("file") or ""), int(caller["line"]))
    return citation in citations


def _architecture_error_code(exc: Exception) -> str:
    code = str(exc)
    return code if _SAFE_ERROR.fullmatch(code) else "architecture_response_invalid"


def _architecture_scenario_observations(
    _candidate: ArchitectureCandidateIdentity, errors: set[str]
) -> tuple[ArchitectureScenarioObservation, ...]:
    """Project actual gate results into the deterministic RF-021 scenario schema."""

    registry_missing = "architecture_registry_unavailable" in errors
    registry_stale = bool(
        errors
        & {
            "architecture_registry_outdated",
            "architecture_registry_duplicate_authority",
        }
    )
    identity_mismatch = "architecture_owner_manifest_identity_disagreement" in errors
    refresh_required = bool(errors)
    rejected = bool(errors)
    outcomes = {
        "registry_unavailable": _scenario_outcome(
            registry_missing, "fail_closed", "available"
        ),
        "registry_outdated": _scenario_outcome(registry_stale, "rejected", "current"),
        "owner_manifest_identity_disagreement": _scenario_outcome(
            identity_mismatch, "rejected", "matched"
        ),
        "regeneration_reingestion": _scenario_outcome(
            refresh_required, "rf021_handoff", "not_required"
        ),
        "finite_exception_metadata": _scenario_outcome(
            rejected, "rejected", "verified"
        ),
        "caller_deletion_evidence": _scenario_outcome(rejected, "rejected", "verified"),
        "concept_discovery_only": "discovery_only",
        "plans_cutover": _scenario_outcome(rejected, "rejected", "authoritative"),
        "layer_boundary_vs_component": _scenario_outcome(
            rejected, "rejected", "implementation_component"
        ),
        "parent_layer_no_signature_match": _scenario_outcome(
            rejected, "rejected", "verified"
        ),
        "component_owned_roots": _scenario_outcome(rejected, "rejected", "verified"),
        "worker_lane_shared_file_exception": _scenario_outcome(
            rejected, "rejected", "verified"
        ),
    }
    return tuple(
        ArchitectureScenarioObservation(scenario, outcomes[scenario])
        for scenario in architecture_workflow_scenarios()
    )


def _scenario_outcome(condition: bool, when_true: str, when_false: str) -> str:
    """Select one controlled scenario result without embedding scenario logic."""

    return when_true if condition else when_false


async def _capture_architecture_operations(
    case: ValidationCase,
    result: CaseResult,
    *,
    client: Any,
    timeout: float,
) -> tuple[GraphOperationObservation, ...]:
    """Invoke and capture every RF-021 Graph-OS operation for the dev skill.

    Raw response bodies never reach ``CaseResult`` or a persisted report. Exact
    request/response digests, typed status, and bounded match counts are retained.
    All three calls are attempted so an earlier failure cannot hide another one.
    """

    if case.skill != _ARCHITECTURE_SKILL:
        return ()

    candidate = case.architecture_candidate
    if candidate is None:
        result.add_error("architecture_candidate_identity_missing")
        return ()

    if not _architecture_routes_valid(case):
        result.add_error("architecture_operation_contract_invalid")
        result.operation_evidence = ()
        return ()

    specs = _architecture_operation_specs(candidate)
    budget = min(_ARCHITECTURE_OPERATION_TIMEOUT_SECONDS, max(1.0, timeout))
    deadline = time.monotonic() + budget
    observations, payloads = await _invoke_architecture_operations(
        client, specs, deadline
    )
    counts, validation_errors = _validate_architecture_payloads(candidate, payloads)
    captured = _finalize_architecture_observations(observations, counts)

    result.operation_evidence = captured
    result.scenario_evidence = _architecture_scenario_observations(
        candidate, validation_errors
    )
    for code in sorted(validation_errors):
        result.add_error(code)
    if validation_errors:
        result.add_error("architecture_registry_regenerate_reingest_required")
    return captured


def _architecture_routes_valid(case: ValidationCase) -> bool:
    """Require every real operation to be part of the case route contract."""

    required = {operation for _phase, operation in _ARCHITECTURE_PHASE_OPERATIONS}
    return required.issubset(case.expected_routes)


async def _invoke_architecture_operations(
    client: Any,
    specs: tuple[tuple[str, str, dict[str, Any]], ...],
    deadline: float,
) -> tuple[list[GraphOperationObservation], dict[str, Any]]:
    """Invoke all candidate-bound operations and retain bounded observations."""

    observations: list[GraphOperationObservation] = []
    payloads: dict[str, Any] = {}
    for phase, operation, arguments in specs:
        observation, payload = await _invoke_architecture_operation(
            client, phase, operation, arguments, deadline
        )
        observations.append(observation)
        if payload is not None:
            payloads[phase] = payload
    return observations, payloads


async def _invoke_architecture_operation(
    client: Any,
    phase: str,
    operation: str,
    arguments: dict[str, Any],
    deadline: float,
) -> tuple[GraphOperationObservation, Any | None]:
    """Invoke one operation within the shared deadline and digest its response."""

    request_digest = _digest_bytes(_canonical_bytes(arguments))
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return _architecture_failed_observation(
            phase, operation, request_digest, "timeout"
        ), None
    try:
        payload = await _call_tool(client, operation, arguments, remaining)
    except TimeoutError:
        return _architecture_failed_observation(
            phase, operation, request_digest, "timeout"
        ), None
    except Exception:  # noqa: BLE001 - retain only typed status evidence
        return _architecture_failed_observation(
            phase, operation, request_digest, "error"
        ), None
    return (
        GraphOperationObservation(
            phase,
            operation,
            "observed",
            request_digest,
            _digest_bytes(_canonical_bytes(payload)),
            0,
        ),
        payload,
    )


def _architecture_failed_observation(
    phase: str, operation: str, request_digest: str, status: str
) -> GraphOperationObservation:
    """Build one content-free timeout or tool-error observation."""

    return GraphOperationObservation(
        phase,
        operation,
        status,
        request_digest,
        _digest_bytes(_canonical_bytes({"status": status})),
        0,
    )


def _architecture_registry_count(
    candidate: ArchitectureCandidateIdentity, payload: Any
) -> int:
    _architecture_registry_row(candidate, payload)
    return 1


def _architecture_discovery_count(
    candidate: ArchitectureCandidateIdentity, payload: Any
) -> int:
    """Return the number of exact advisory discovery rows."""

    return len(_architecture_discovery_rows(candidate, payload))


def _validate_architecture_payloads(
    candidate: ArchitectureCandidateIdentity, payloads: dict[str, Any]
) -> tuple[dict[str, int], set[str]]:
    """Run the three typed validators and return controlled errors only."""

    validators = {
        "registry_lookup": _architecture_registry_count,
        "discovery": _architecture_discovery_count,
        "caller_impact": _architecture_caller_count,
    }
    counts: dict[str, int] = {}
    errors: set[str] = set()
    for phase, validator in validators.items():
        if phase not in payloads:
            errors.add("architecture_operation_unavailable")
            continue
        try:
            counts[phase] = validator(candidate, payloads[phase])
        except Exception as exc:  # noqa: BLE001 - convert to controlled code
            errors.add(_architecture_error_code(exc))
    return counts, errors


def _architecture_verified_status(phase: str) -> str:
    """Return the evidence role for one validated operation phase."""

    return {
        "registry_lookup": "verified",
        "discovery": "advisory",
        "caller_impact": "grounded",
    }[phase]


def _finalize_architecture_observations(
    observations: list[GraphOperationObservation], counts: dict[str, int]
) -> tuple[GraphOperationObservation, ...]:
    """Attach typed validation roles and record counts to tool evidence."""

    return tuple(
        GraphOperationObservation(
            observation.phase,
            observation.operation,
            _architecture_verified_status(observation.phase)
            if observation.phase in counts
            else observation.status,
            observation.request_digest,
            observation.response_digest,
            counts.get(observation.phase, 0),
        )
        for observation in observations
    )


async def _verified_validation_session(
    headers: dict[str, str], *, minimum_ttl_seconds: int
) -> Any:
    """Validate one MCP bearer and mint sufficiently current graph authority."""

    if minimum_ttl_seconds < 0:
        raise ValueError("minimum_ttl_seconds_must_be_non_negative")

    authorization = str(
        headers.get("Authorization") or headers.get("authorization") or ""
    )
    scheme, separator, token = authorization.partition(" ")
    if scheme.casefold() != "bearer" or not separator or not token.strip():
        raise RuntimeError("direct_identity_unavailable")
    from agent_utilities.security.request_identity import (
        actor_from_bearer_token,
        mint_graph_session,
    )

    actor = await actor_from_bearer_token(token.strip())
    session = mint_graph_session(actor)
    session.engine_verified_context()
    session.ensure_authority_current(minimum_ttl_seconds=minimum_ttl_seconds)
    return session


def minimum_campaign_authority_ttl_seconds(
    *,
    case_timeout: float,
    trace_timeout: float,
    shutdown_grace: float,
) -> int:
    """Return the lease needed for the campaign's longest bounded case.

    The lease spans the trace precheck, model call, exporter flush, exact-trace
    wait, parent-ingestion read-back, controlled shutdown, and a small expiry
    boundary margin. Keeping the calculation here makes deployment validation
    and runtime renewal share one definition instead of independent TTL floors.
    """

    windows = (case_timeout, trace_timeout, shutdown_grace)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        for value in windows
    ):
        raise ValueError("campaign_authority_window_invalid")
    if case_timeout <= 0 or trace_timeout <= 0 or shutdown_grace < 0:
        raise ValueError("campaign_authority_window_invalid")
    bounded_seconds = (
        min(_AUTHORITY_TRACE_PRECHECK_CAP_SECONDS, trace_timeout)
        + case_timeout
        + min(_AUTHORITY_EXPORT_FLUSH_CAP_SECONDS, case_timeout)
        + trace_timeout
        + min(_AUTHORITY_PARENT_INGESTION_CAP_SECONDS, trace_timeout)
        + shutdown_grace
        + _AUTHORITY_LEASE_SAFETY_SECONDS
    )
    return math.ceil(bounded_seconds)


def _direct_case_minimum_authority_ttl(
    *, case_timeout: float, trace_timeout: float
) -> int:
    """Return the lease required to cover one bounded direct validation case."""

    return minimum_campaign_authority_ttl_seconds(
        case_timeout=case_timeout,
        trace_timeout=trace_timeout,
        shutdown_grace=0.0,
    )


async def _renew_direct_validation_session(
    *, expected_authority: dict[str, Any], minimum_ttl_seconds: int
) -> Any:
    """Mint current direct-case authority without changing its verified grant."""

    from agent_utilities.knowledge_graph.core.session import SessionExpiredError
    from agent_utilities.mcp.client_credentials import child_auth_header, get_provider

    async def mint_from_current_bearer() -> Any:
        headers = await _bounded_sync_call(
            lambda: child_auth_header({}),
            _AUTHORITY_RENEWAL_TIMEOUT_SECONDS,
        )
        return await _verified_validation_session(
            headers, minimum_ttl_seconds=minimum_ttl_seconds
        )

    try:
        session = await mint_from_current_bearer()
    except SessionExpiredError:
        # The provider normally refreshes within its expiry skew. A direct case
        # may require a longer lease than that skew, so proactively rotate the
        # bearer once and re-verify it instead of starting work that can expire.
        provider = get_provider()
        if provider is None:
            raise RuntimeError("direct_identity_renewal_unavailable") from None
        await _bounded_sync_call(
            lambda: provider.get_token(force=True),
            _AUTHORITY_RENEWAL_TIMEOUT_SECONDS,
        )
        session = await mint_from_current_bearer()

    if session.engine_verified_context() != expected_authority:
        raise RuntimeError("direct_identity_authority_changed")
    return session


async def _renew_delegated_validation_session(
    *, expected_authority: dict[str, Any], minimum_ttl_seconds: int
) -> Any:
    """Force a fresh bearer before one bounded delegated validation case."""

    from agent_utilities.mcp.client_credentials import child_auth_header, get_provider

    provider = get_provider()
    if provider is None:
        raise RuntimeError("delegated_identity_renewal_unavailable")
    await _bounded_sync_call(
        lambda: provider.get_token(force=True),
        _AUTHORITY_RENEWAL_TIMEOUT_SECONDS,
    )
    headers = await _bounded_sync_call(
        lambda: child_auth_header({}),
        _AUTHORITY_RENEWAL_TIMEOUT_SECONDS,
    )
    session = await _verified_validation_session(
        headers, minimum_ttl_seconds=minimum_ttl_seconds
    )
    if session.engine_verified_context() != expected_authority:
        raise RuntimeError("delegated_identity_authority_changed")
    return session


async def _ensure_tool(client: Any, tool: str, timeout: float) -> None:
    names = await _list_tool_names(client, timeout)
    if tool in names:
        return
    if "load_tools" not in names:
        raise RuntimeError("tool_loader_unavailable")
    await _call_tool(client, "load_tools", {"tools": [tool]}, timeout)
    names = await _list_tool_names(client, timeout)
    if tool not in names:
        raise RuntimeError("required_tool_unavailable")


async def _list_tool_names(client: Any, timeout: float) -> set[str]:
    """List a bounded MCP tool surface under the caller's wall-clock budget."""

    entries = await asyncio.wait_for(client.list_tools(), timeout=max(1.0, timeout))
    if not isinstance(entries, list) or len(entries) > _MAX_TOOL_ITEMS:
        raise RuntimeError("tool_catalog_invalid")
    names = {str(getattr(entry, "name", "") or "") for entry in entries}
    if "" in names or any(len(name) > 256 for name in names):
        raise RuntimeError("tool_catalog_invalid")
    return names


def _langfuse_tool_candidates(catalog: Any) -> list[str]:
    """Extract the safely named prefixed `langfuse_observability` entries."""

    candidates = []
    if isinstance(catalog, dict):
        for entry in catalog.get("tools") or []:
            if (
                isinstance(entry, dict)
                and entry.get("tool") == "langfuse_observability"
            ):
                candidates.append(str(entry.get("prefixed_name") or ""))
    return [name for name in candidates if _SAFE_ROUTE.fullmatch(name)]


async def _load_langfuse_tool(client: Any, timeout: float) -> str:
    """Discover and load Langfuse through Graph-OS, never a direct endpoint."""

    names = await _list_tool_names(client, timeout)
    if "list_catalog" not in names or "load_tools" not in names:
        raise RuntimeError("fleet_catalog_unavailable")
    catalog = await _call_tool(
        client,
        "list_catalog",
        {"server": "langfuse-mcp", "include_tools": True},
        timeout,
    )
    candidates = _langfuse_tool_candidates(catalog)
    if len(candidates) != 1:
        raise RuntimeError("langfuse_tool_discovery_failed")
    if candidates[0] not in names:
        await _call_tool(client, "load_tools", {"tools": candidates}, timeout)
        names = await _list_tool_names(client, timeout)
        if candidates[0] not in names:
            raise RuntimeError("langfuse_tool_load_failed")
    return candidates[0]


async def _await_sync_worker(
    completed: threading.Event, poisoned: threading.Event, timeout: float
) -> None:
    """Await the single SDK worker, poisoning the slot if it is abandoned."""

    try:
        deadline = time.monotonic() + max(1.0, timeout)
        while not completed.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if completed.is_set():
                    break
                poisoned.set()
                raise TimeoutError("blocking_sdk_timeout")
            await asyncio.sleep(min(0.05, remaining))
    except BaseException:
        if not completed.is_set():
            poisoned.set()
        raise


async def _bounded_sync_call(function: Any, timeout: float) -> Any:
    """Run at most one blocking SDK call and fail closed after abandonment.

    CPython cannot safely terminate a thread blocked inside a third-party SDK.
    A timeout or caller cancellation therefore poisons this certification
    process: the daemon may finish during shutdown, but no second SDK worker is
    admitted and the validation run aborts instead of accumulating work beyond
    its budget.
    """

    active_guard = _SYNC_CALL_ACTIVE
    poisoned = _SYNC_CALL_POISONED
    if poisoned.is_set():
        raise RuntimeError("blocking_sdk_worker_abandoned")
    if not active_guard.acquire(blocking=False):
        raise RuntimeError("blocking_sdk_worker_active")
    if poisoned.is_set():
        active_guard.release()
        raise RuntimeError("blocking_sdk_worker_abandoned")

    outcome: list[tuple[bool, Any]] = []
    completed = threading.Event()
    caller_context = copy_context()

    def invoke() -> None:
        try:
            outcome.append((True, caller_context.run(function)))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller task
            outcome.append((False, exc))
        finally:
            active_guard.release()
            completed.set()

    try:
        threading.Thread(
            target=invoke,
            daemon=True,
            name="skill-validation-sdk",
        ).start()
    except BaseException:
        active_guard.release()
        raise
    await _await_sync_worker(completed, poisoned, timeout)
    succeeded, value = outcome[0]
    if succeeded:
        return value
    raise value


async def _verify_langfuse_posture(
    client: Any, langfuse_tool: str, timeout: float
) -> None:
    """Prove the mounted child is enforcing metadata-only retention."""

    posture = await _call_tool(
        client,
        langfuse_tool,
        {"action": "runtime_posture"},
        timeout,
    )
    if posture != {
        "content_capture_enabled": False,
        "metadata_only": True,
    }:
        raise RuntimeError("langfuse_content_posture_invalid")


def _expected_trace_name(run_id: str, tenant_id: str) -> str:
    """Derive the exact opaque trace name emitted for a runtime run."""

    from agent_utilities.usage.privacy import normalize_run_id

    name = f"graph_run:{normalize_run_id(run_id, tenant_id=tenant_id)}"
    if not re.fullmatch(r"graph_run:pref_run_[a-f0-9]{64}", name):
        raise RuntimeError("trace_expected_name_invalid")
    return name


def _trace_row_evidence(row: dict[str, Any]) -> dict[str, str]:
    """Extract only the closed opaque evidence contract from trace metadata."""

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    patterns = {
        "run_ref": re.compile(r"pref_run_[a-f0-9]{64}"),
        "model_ref": re.compile(r"pref_model_[a-f0-9]{64}"),
        "skill_ref": re.compile(r"pref_skill_[a-f0-9]{64}"),
        "skill_body_ref": re.compile(r"pref_skill_body_[a-f0-9]{64}"),
        "model_class": re.compile(r"(?:economy|standard)"),
    }
    evidence: dict[str, str] = {}
    for key, pattern in patterns.items():
        if key not in metadata:
            continue
        value = metadata[key]
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise RuntimeError("trace_evidence_invalid")
        evidence[key] = value
    return evidence


def _trace_list_arguments(
    page: int, from_timestamp: str | None, expected_name: str
) -> dict[str, Any]:
    """Build one bounded, exact-name `trace_list` request for a single page."""

    args: dict[str, Any] = {
        "action": "trace_list",
        "page": page,
        # Cases run sequentially and each window expects one run-linked trace.
        # Small pages stay below GraphOS's delegated-value boundary even when
        # the shared project contains content-heavy automatic telemetry.
        "limit": _TRACE_PAGE_LIMIT,
        "order_by": "timestamp.desc",
        "fields": "core,basic,metadata",
    }
    if from_timestamp:
        args["from_timestamp"] = from_timestamp
    # Filter at the provider boundary so unrelated shared-project traffic
    # cannot consume the bounded page window or expand metadata exposure.
    args["name"] = expected_name
    return args


def _collect_trace_rows(
    snapshot: dict[str, TraceRecord], rows: list[Any], expected_name: str
) -> None:
    """Retain only exact-name rows and their closed evidence metadata."""

    for row in rows:
        if not isinstance(row, dict):
            continue
        trace_id = str(row.get("id") or "")
        name = str(row.get("name") or "")
        if trace_id and len(trace_id) <= 256 and name == expected_name:
            snapshot[trace_id] = TraceRecord(
                name=name,
                evidence=_trace_row_evidence(row),
            )


async def _trace_snapshot(
    client: Any,
    langfuse_tool: str,
    timeout: float,
    *,
    from_timestamp: str | None = None,
    expected_name: str,
) -> dict[str, TraceRecord]:
    if not re.fullmatch(r"graph_run:pref_run_[a-f0-9]{64}", expected_name):
        raise RuntimeError("trace_expected_name_invalid")
    snapshot: dict[str, TraceRecord] = {}
    deadline = time.monotonic() + max(1.0, timeout)
    max_pages = _TRACE_MAX_PAGES if from_timestamp else 1
    for page in range(1, max_pages + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("trace_snapshot_timeout")
        args = _trace_list_arguments(page, from_timestamp, expected_name)
        payload = await _call_tool(client, langfuse_tool, args, min(remaining, timeout))
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("trace_snapshot_invalid")
        _collect_trace_rows(snapshot, rows, expected_name)
        if len(rows) < _TRACE_PAGE_LIMIT:
            return snapshot
    if from_timestamp:
        raise RuntimeError("trace_snapshot_boundary_exceeded")
    return snapshot


async def _next_transient_retry(attempts: int, deadline: float) -> int | None:
    """Charge one bounded retry for a typed child-tool failure.

    Returns the new attempt count after sleeping the backoff, or ``None`` when
    the retry budget or the caller's deadline is exhausted and the typed
    failure must propagate as a certification gate.
    """

    if attempts >= _TRACE_TOOL_ERROR_RETRIES:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    await asyncio.sleep(min(_TRACE_TOOL_ERROR_RETRY_DELAY_SECONDS, remaining))
    return attempts + 1


def _matched_expected_trace(
    current: dict[str, TraceRecord], expected_evidence: dict[str, str]
) -> str | None:
    """Return the single exact-evidence trace id, or None while none exists."""

    matching = sorted(current)
    if len(matching) == 1:
        record = current[matching[0]]
        if any(
            record.evidence.get(key) != value
            for key, value in expected_evidence.items()
        ):
            raise RuntimeError("trace_evidence_mismatch")
        return matching[0]
    if len(matching) > 1:
        raise RuntimeError("trace_run_identifier_ambiguous")
    return None


async def _wait_for_expected_trace(
    client: Any,
    langfuse_tool: str,
    started_at: str,
    expected_name: str,
    expected_evidence: dict[str, str],
    timeout: float,
) -> tuple[str, str]:
    """Require one exact run trace with the case's controlled evidence metadata."""

    if not expected_evidence or expected_evidence.get("run_ref") != (
        expected_name.removeprefix("graph_run:")
    ):
        raise RuntimeError("trace_expected_evidence_invalid")

    deadline = time.monotonic() + timeout
    transient_tool_errors = 0
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            current = await _trace_snapshot(
                client,
                langfuse_tool,
                min(15.0, remaining),
                from_timestamp=started_at,
                expected_name=expected_name,
            )
        except (ToolError, ValidationChildToolError):
            # An exact-name trace read is idempotent, and GraphOS may fail the
            # outer call after a successful provider read when its mandatory
            # parent ChangeEnvelope races another graph writer. Retry only the
            # typed child-tool failure, keep the attempt count bounded, and let
            # persistent provider/ingestion failures remain certification gates.
            attempts = await _next_transient_retry(transient_tool_errors, deadline)
            if attempts is None:
                raise
            transient_tool_errors = attempts
            continue
        matched = _matched_expected_trace(current, expected_evidence)
        if matched is not None:
            return matched, "run-evidence"
        await asyncio.sleep(1.0)
    raise TimeoutError("trace_not_observed")


def _require_parent_ingestion_inputs(expected_name: str, timeout: float) -> None:
    """Require an exact opaque trace name and a finite, positive time budget."""

    if not re.fullmatch(r"graph_run:pref_run_[a-f0-9]{64}", expected_name):
        raise RuntimeError("trace_expected_name_invalid")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("trace_parent_ingestion_timeout_invalid")


def _parent_ingestion_query(expected_name: str) -> dict[str, Any]:
    """Build the bounded, identity-retaining parent-ingestion readback query."""

    return {
        "cypher": (
            "MATCH (n:Trace) WHERE n.name = $name "
            "RETURN n.id AS id, n.name AS name LIMIT 2"
        ),
        "params": json.dumps({"name": expected_name}, separators=(",", ":")),
        "scope": "local",
    }


async def _verify_parent_ingested_trace(
    client: Any,
    expected_name: str,
    timeout: float,
) -> int:
    """Require exactly one parent-mediated KG node for an exact opaque trace."""

    _require_parent_ingestion_inputs(expected_name, timeout)
    arguments = _parent_ingestion_query(expected_name)
    deadline = time.monotonic() + timeout
    transient_tool_errors = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("trace_parent_ingestion_not_observed")
        try:
            payload = await _call_tool(
                client,
                "graph_query",
                arguments,
                min(15.0, remaining),
            )
        except (ToolError, ValidationChildToolError):
            attempts = await _next_transient_retry(transient_tool_errors, deadline)
            if attempts is None:
                raise
            transient_tool_errors = attempts
            continue
        count = _parent_ingested_trace_count(payload, expected_name=expected_name)
        if count == 1:
            return count
        if count != 0:
            raise RuntimeError("trace_parent_ingestion_mismatch")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("trace_parent_ingestion_not_observed")
        await asyncio.sleep(min(_PARENT_INGESTION_POLL_DELAY_SECONDS, remaining))


def _graph_query_trace(payload: Any) -> dict[str, Any] | None:
    """Return the single closed `graph_query` step of an EvidenceBundle trace."""

    if not isinstance(payload, dict):
        return None
    reasoning_trace = payload.get("reasoning_trace")
    if not isinstance(reasoning_trace, list) or any(
        not isinstance(item, dict) for item in reasoning_trace
    ):
        return None
    query_traces = [
        item for item in reasoning_trace if item.get("step") == "graph_query"
    ]
    if len(query_traces) != 1:
        return None
    trace = query_traces[0]
    return trace if set(trace) == {"step", "payload"} else None


def _graph_query_rows(payload: Any) -> list[Any] | None:
    """Return the bounded row projection of GraphQuery's EvidenceBundle trace."""

    trace = _graph_query_trace(payload)
    if trace is None:
        return None
    aggregate = trace.get("payload")
    if not isinstance(aggregate, dict) or set(aggregate) != {"rows"}:
        return None
    rows = aggregate.get("rows")
    if not isinstance(rows, list) or len(rows) > 2:
        return None
    return rows


def _governed_trace_row(row: Any, expected_name: str) -> bool:
    """Accept only a two-field row whose node identity and name are governed."""

    if not isinstance(row, dict) or set(row) != {"id", "name"}:
        return False
    node_id = row.get("id")
    if (
        not isinstance(node_id, str)
        or re.fullmatch(r"langfuse:trace:[a-f0-9]{32}", node_id) is None
    ):
        return False
    return row.get("name") == expected_name


def _parent_ingested_trace_count(payload: Any, *, expected_name: str) -> int | None:
    """Count only governed trace-id rows in GraphQuery's EvidenceBundle trace.

    Public graph reads must retain node identity so tenant, ACL, visibility, and
    audit enforcement can govern every returned row.  The query is bounded at
    two rows: one is the required materialization, zero is missing, and two
    proves an ambiguous duplicate.  Accept only that closed projection; an
    aggregate without node identity, a similarly named claim, or a widened row
    is not proof of parent-mediated ingestion.
    """

    rows = _graph_query_rows(payload)
    if rows is None:
        return None
    if not all(_governed_trace_row(row, expected_name) for row in rows):
        return None
    return len(rows)


def _opaque_ref(kind: str, value: str) -> str:
    return persistence_reference(kind, value, namespace="skill-validation")


def _expected_delegated_model_ref(model_class: str) -> str:
    """Resolve the exact configured model identity for a delegated model class."""

    from agent_utilities.orchestration.agent_runner import (
        _configured_model_for_class,
    )

    selected = _configured_model_for_class(model_class)
    return persistence_reference("model", selected.id, namespace="orchestration-run")


def _validate_delegated_runtime_evidence(
    case: ValidationCase, status: dict[str, Any]
) -> tuple[list[str], str, str, str]:
    """Validate actual trace metadata, never the fixture's requested label alone."""
    errors: list[str] = []
    model_ref = str(status.get("model_ref") or "")
    skill_ref = str(status.get("skill_ref") or "")
    digest = str(status.get("skill_instruction_digest") or "")
    expected_skill_ref = persistence_reference(
        "skill", case.skill, namespace="execution-trace"
    )
    expected_model_ref = _expected_delegated_model_ref(case.model_class)
    if str(status.get("model_class") or "") != case.model_class:
        errors.append("model_class_mismatch")
    if not model_ref:
        errors.append("model_reference_missing")
    elif model_ref != expected_model_ref:
        errors.append("model_reference_mismatch")
    if skill_ref != expected_skill_ref:
        errors.append("skill_reference_mismatch")
    if digest != _skill_instruction_digest(case.skill):
        errors.append("skill_instruction_digest_mismatch")
    return errors, model_ref, skill_ref, digest


def _delegation_terminal_error_code(status: dict[str, Any]) -> str | None:
    """Classify terminal failure metadata without retaining its raw text."""

    state = str(status.get("status") or "").strip().casefold()
    if state == "completed":
        return None
    error_text = str(status.get("error") or "")
    known_types = (
        "ContextCompilationError",
        "PermissionError",
        "SessionRequiredError",
        "ScopeError",
        "TransportSecurityError",
        "ValidationError",
        "TimeoutError",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "ImportError",
        "ConnectionError",
        "HTTPStatusError",
        "ModelHTTPError",
        "UnexpectedModelBehavior",
        "ToolError",
    )
    failure_type = next((name for name in known_types if name in error_text), "")
    if failure_type:
        return f"delegation_terminal_{failure_type.casefold()}"
    normalized_state = re.sub(r"[^a-z0-9_]+", "_", state).strip("_")
    return f"delegation_terminal_{normalized_state or 'failure'}"


def _validation_reasoning_effort(model_class: str, *, delegated: bool) -> str | None:
    """Return the provider-neutral reasoning override for validation.

    Economy validation must not send the OpenAI-compatible ``"none"``
    extension: it is not part of the portable effort vocabulary and some
    otherwise compatible runtimes reject it.  Direct model construction uses
    ``None`` to omit the field; the string-only MCP delegation surface uses an
    empty value, which ``graph_orchestrate`` converts to the same omission.
    Standard delegated cases retain their bounded ``low`` effort.
    """

    if model_class == "economy":
        return "" if delegated else None
    return "low" if delegated else None


async def _attach_trace_evidence(
    result: CaseResult,
    *,
    client: Any,
    langfuse_tool: str,
    started_at: str,
    expected_trace_name: str,
    expected_trace_evidence: dict[str, str],
    trace_timeout: float,
) -> None:
    """Record the exact trace and its parent-ingested node, or a typed error."""

    try:
        trace_id, linkage = await _wait_for_expected_trace(
            client,
            langfuse_tool,
            started_at,
            expected_trace_name,
            expected_trace_evidence,
            trace_timeout,
        )
        result.trace = _PASS
        result.trace_linkage = linkage
        result.trace_name = expected_trace_name
        result.langfuse_match_count = 1
        result.trace_ref = _opaque_ref("trace", trace_id)
        result.parent_kg_readback_count = await _verify_parent_ingested_trace(
            client, expected_trace_name, min(15.0, trace_timeout)
        )
        result.parent_ingestion = _PASS
    except Exception as exc:  # noqa: BLE001 - report only the exception class
        result.add_error(f"trace_or_ingestion_{type(exc).__name__}")


def _direct_case_model(case: ValidationCase, result: CaseResult) -> tuple[Any, str]:
    """Bind the configured model and skill identity onto the case result."""

    from agent_utilities.core.model_factory import create_model
    from agent_utilities.orchestration.agent_runner import (
        _configured_model_for_class,
    )

    selected_model = _configured_model_for_class(case.model_class)
    model = create_model(
        model_id=selected_model.id,
        reasoning_effort=_validation_reasoning_effort(
            case.model_class, delegated=False
        ),
    )
    model_name = str(getattr(model, "model_name", "") or "")
    if not model_name:
        raise RuntimeError("runtime_model_identity_unavailable")
    result.model_ref = _opaque_ref("model", model_name)
    expected_model_ref = _opaque_ref("model", selected_model.id)
    if result.model_ref != expected_model_ref:
        result.add_error("direct_model_selection_mismatch")
    else:
        result.model_selection = _PASS
    instruction_digest = _skill_instruction_digest(case.skill)
    result.skill_ref = persistence_reference(
        "skill", case.skill, namespace="execution-trace"
    )
    result.skill_body_ref = _opaque_ref("skill_body", instruction_digest)
    result.skill_binding = _PASS
    return model, model_name


def _direct_model_settings(
    *, system_prompt: str, model_identity: str, case_timeout: float
) -> Any:
    """Build bounded direct-run settings, folding the provider prompt-cache hint."""

    from pydantic_ai import ModelSettings

    direct_model_settings: Any = ModelSettings(
        # The closed JSON contract is intentionally small. A bounded
        # generation keeps CPU-only local-model validation practical.
        max_tokens=_DIRECT_MAX_OUTPUT_TOKENS,
        temperature=0.0,
        timeout=case_timeout,
    )
    try:
        # D-54c-4 — this call bypasses attach_profile_resolver (it invokes
        # agent.run() directly with an explicit model_settings), so fold the
        # provider-native prompt-cache directive here too (CONCEPT:AU-ORCH.optimization.provider-prompt-cache).
        from agent_utilities.caching.prompt_cache import fold_prompt_cache_hint

        return fold_prompt_cache_hint(
            direct_model_settings,
            system_prompt=system_prompt,
            model_identity=model_identity,
        )
    except Exception:  # noqa: BLE001 - prompt-cache hint is best-effort
        return direct_model_settings


def _record_direct_semantic(case: ValidationCase, result: CaseResult, run: Any) -> None:
    """Validate the closed semantic contract of a direct run into the result."""

    semantic = SemanticOutput.model_validate(run.output)
    semantic_errors = validate_semantic_output(case, semantic)
    result.selected_routes = tuple(sorted(semantic.selected_routes))
    for error in semantic_errors:
        result.add_error(error)
    result.semantic = _PASS if not semantic_errors else _FAIL


async def _export_direct_trace(
    result: CaseResult,
    *,
    run: Any,
    validation_run_id: str,
    model_name: str,
    trace_evidence: dict[str, str],
    case_timeout: float,
) -> None:
    """Emit the exact run trace through the single bounded blocking-SDK slot."""

    from agent_utilities.observability.langfuse_exporter import get_langfuse_exporter

    exporter = get_langfuse_exporter()
    if exporter is None:
        result.add_error("trace_exporter_unavailable")
        return

    def emit_trace() -> bool | None:
        if not exporter.enabled:
            return None
        emitted = exporter.export_graph_run(
            run_id=validation_run_id,
            query="",
            status=("success" if result.semantic == _PASS else "validation_failed"),
            token_usage=_usage_counts(run),
            model=model_name,
            metadata={"validation_kind": "bundled_skill_direct"},
            evidence={
                key: value for key, value in trace_evidence.items() if key != "run_ref"
            },
        )
        exporter.flush()
        return emitted

    emitted = await _bounded_sync_call(emit_trace, min(30.0, case_timeout))
    if emitted is None:
        result.add_error("trace_exporter_unavailable")
    elif not emitted:
        result.add_error("trace_export_failed")


async def _execute_direct_case(
    case: ValidationCase,
    result: CaseResult,
    *,
    validation_run_id: str,
    expected_trace_name: str,
    case_timeout: float,
) -> dict[str, str]:
    """Run one direct in-process case and return its expected trace evidence."""

    expected_trace_evidence: dict[str, str] = {}
    async with _DIRECT_CASE_LOCK:
        try:
            from agent_utilities.core.contextual_model import create_context_agent

            with _direct_evidence_authority(case.skill):
                model, model_name = _direct_case_model(case, result)
                expected_trace_evidence = {
                    "run_ref": expected_trace_name.removeprefix("graph_run:"),
                    "model_ref": result.model_ref,
                    "model_class": case.model_class,
                    "skill_ref": result.skill_ref,
                    "skill_body_ref": result.skill_body_ref,
                }
                direct_system_prompt = (
                    f"{_skill_runtime_body(case.skill)}\n\n"
                    f"{_contract_instruction(case)}"
                )
                agent = create_context_agent(
                    model=model,
                    output_type=_direct_semantic_output_type(case),
                    system_prompt=direct_system_prompt,
                    model_settings=_direct_model_settings(
                        system_prompt=direct_system_prompt,
                        model_identity=model_name,
                        case_timeout=case_timeout,
                    ),
                    retries=2,
                )
                run = await asyncio.wait_for(
                    agent.run(_direct_execution_prompt(case)), timeout=case_timeout
                )
                _record_direct_semantic(case, result, run)
                await _export_direct_trace(
                    result,
                    run=run,
                    validation_run_id=validation_run_id,
                    model_name=model_name,
                    trace_evidence=expected_trace_evidence,
                    case_timeout=case_timeout,
                )
                result.run_ref = expected_trace_name.removeprefix("graph_run:")
        except Exception as exc:  # noqa: BLE001 - report only the exception class
            result.add_error(f"direct_{type(exc).__name__}")
    return expected_trace_evidence


async def _run_direct_case(
    case: ValidationCase,
    *,
    client: Any,
    langfuse_tool: str,
    tenant_id: str,
    case_timeout: float,
    trace_timeout: float,
) -> CaseResult:
    result = CaseResult(
        case_id=case.case_id,
        skill=case.skill,
        mode=case.mode,
        model_class=case.model_class,
    )
    validation_run_id = new_run_id()
    expected_trace_name = _expected_trace_name(validation_run_id, tenant_id)
    try:
        await _capture_architecture_operations(
            case,
            result,
            client=client,
            timeout=case_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - retain only controlled diagnostics
        result.add_error(f"architecture_probe_{type(exc).__name__}")
        return result
    if result.error_codes:
        return result
    try:
        existing = await _trace_snapshot(
            client,
            langfuse_tool,
            min(15.0, trace_timeout),
            expected_name=expected_trace_name,
        )
    except Exception as exc:  # noqa: BLE001 - controlled type-only evidence
        result.add_error(f"trace_precheck_{type(exc).__name__}")
        return result
    if existing:
        result.add_error("trace_run_identifier_preexisting")
        return result
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    expected_trace_evidence = await _execute_direct_case(
        case,
        result,
        validation_run_id=validation_run_id,
        expected_trace_name=expected_trace_name,
        case_timeout=case_timeout,
    )
    if _SYNC_CALL_POISONED.is_set():
        return result
    if not expected_trace_evidence:
        result.add_error("trace_expected_evidence_unavailable")
        return result
    await _attach_trace_evidence(
        result,
        client=client,
        langfuse_tool=langfuse_tool,
        started_at=started_at,
        expected_trace_name=expected_trace_name,
        expected_trace_evidence=expected_trace_evidence,
        trace_timeout=trace_timeout,
    )
    return result


def _delegation_request(
    case: ValidationCase, *, max_steps: int, token_budget: int
) -> dict[str, Any]:
    """Build the bounded `graph_orchestrate` request for one delegated case."""

    return {
        "agent_name": case.skill,
        "task": f"{case.task}\n\n{_contract_instruction(case)}",
        "max_steps": max_steps,
        "budget_tokens": token_budget,
        "allowed_tools": ",".join(case.allowed_tools),
        "reasoning_effort": _validation_reasoning_effort(
            case.model_class, delegated=True
        ),
        "model_class": case.model_class,
        "response_format": "json",
    }


def _record_delegated_semantic(
    case: ValidationCase, result: CaseResult, output: Any
) -> None:
    """Validate the delegated semantic contract, retaining only error codes."""

    try:
        semantic = _semantic_from_delegation_output(output)
        semantic_errors = validate_semantic_output(case, semantic)
        result.selected_routes = tuple(sorted(semantic.selected_routes))
        for error in semantic_errors:
            result.add_error(error)
        result.semantic = _PASS if not semantic_errors else _FAIL
    except Exception as exc:  # noqa: BLE001 - controlled semantic evidence only
        if isinstance(exc, DelegationContractError):
            result.add_error(exc.code)
        else:
            result.add_error(f"delegated_semantic_{type(exc).__name__}")


def _delegated_check_status(evidence_errors: list[str], prefix: str) -> str:
    """Pass a delegated check only when no evidence error carries its prefix."""

    return (
        _PASS
        if not any(error.startswith(prefix) for error in evidence_errors)
        else _FAIL
    )


def _record_delegated_evidence(
    case: ValidationCase, result: CaseResult, status: dict[str, Any]
) -> str:
    """Record the delegated run's controlled evidence; return its skill digest."""

    terminal_error = _delegation_terminal_error_code(status)
    if terminal_error:
        result.add_error(terminal_error)
    (
        evidence_errors,
        model_ref,
        skill_ref,
        digest,
    ) = _validate_delegated_runtime_evidence(case, status)
    for error in evidence_errors:
        result.add_error(error)
    result.model_ref = model_ref
    result.skill_ref = skill_ref
    result.skill_body_ref = _opaque_ref("skill_body", digest) if digest else ""
    result.model_selection = _delegated_check_status(evidence_errors, "model_")
    result.skill_binding = _delegated_check_status(evidence_errors, "skill_")
    result.delegation = (
        _PASS if not evidence_errors and terminal_error is None else _FAIL
    )
    return digest


def _delegated_trace_evidence(
    case: ValidationCase, result: CaseResult, expected_trace_name: str, digest: str
) -> dict[str, str]:
    """Return the exact trace evidence, or empty when a reference is missing."""

    if not (result.model_ref and result.skill_ref and digest):
        return {}
    return {
        "run_ref": expected_trace_name.removeprefix("graph_run:"),
        "model_ref": result.model_ref,
        "model_class": case.model_class,
        "skill_ref": result.skill_ref,
        "skill_body_ref": _opaque_ref("skill_body", digest),
    }


async def _record_delegated_run(
    case: ValidationCase,
    result: CaseResult,
    *,
    client: Any,
    run_id: str,
    expected_trace_name: str,
    case_timeout: float,
) -> dict[str, str]:
    """Await the delegated run and record its controlled completion evidence."""

    if not run_id:
        result.add_error("delegation_run_handle_missing")
        return {}
    result.run_ref = expected_trace_name.removeprefix("graph_run:")
    status = await _wait_for_run_completion(client, run_id, min(case_timeout, 30.0))
    digest = _record_delegated_evidence(case, result, status)
    return _delegated_trace_evidence(case, result, expected_trace_name, digest)


async def _run_delegated_case(
    case: ValidationCase,
    *,
    client: Any,
    langfuse_tool: str,
    tenant_id: str,
    max_steps: int,
    token_budget: int,
    case_timeout: float,
    trace_timeout: float,
) -> CaseResult:
    result = CaseResult(
        case_id=case.case_id,
        skill=case.skill,
        mode=case.mode,
        model_class=case.model_class,
        delegation=_FAIL,
    )
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    run_id = ""
    expected_trace_name = ""
    expected_trace_evidence: dict[str, str] = {}
    try:
        await _capture_architecture_operations(
            case,
            result,
            client=client,
            timeout=case_timeout,
        )
        if result.error_codes:
            return result
        response = await _call_tool(
            client,
            "graph_orchestrate",
            _delegation_request(case, max_steps=max_steps, token_budget=token_budget),
            case_timeout,
        )
        output, run_id = _extract_delegation_envelope(response)
        expected_trace_name = _expected_trace_name(run_id, tenant_id)
        _record_delegated_semantic(case, result, output)
        expected_trace_evidence = await _record_delegated_run(
            case,
            result,
            client=client,
            run_id=run_id,
            expected_trace_name=expected_trace_name,
            case_timeout=case_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - retain only controlled diagnostics
        if isinstance(exc, DelegationContractError):
            result.add_error(exc.code)
        else:
            result.add_error(f"delegated_{type(exc).__name__}")

    if not run_id or not expected_trace_name:
        result.add_error("trace_run_identifier_unavailable")
    elif not expected_trace_evidence:
        result.add_error("trace_expected_evidence_unavailable")
    else:
        await _attach_trace_evidence(
            result,
            client=client,
            langfuse_tool=langfuse_tool,
            started_at=started_at,
            expected_trace_name=expected_trace_name,
            expected_trace_evidence=expected_trace_evidence,
            trace_timeout=trace_timeout,
        )
    return result


async def _wait_for_run_completion(
    client: Any, run_id: str, timeout: float
) -> dict[str, Any]:
    """Poll the focused job surface until the delegated run is terminal."""

    if not is_run_id(run_id):
        raise DelegationContractError("delegation_run_id_invalid")

    deadline = time.monotonic() + max(1.0, timeout)
    failed_states = {
        "cancelled",
        "canceled",
        "dead_letter",
        "denied",
        "error",
        "failed",
        "rejected",
    }
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("delegation_status_timeout")
        status = await _call_tool(
            client,
            "graph_jobs",
            {"action": "status", "job_id": run_id},
            min(remaining, 15.0),
        )
        if not isinstance(status, dict):
            raise RuntimeError("delegation_status_not_object")
        state = str(status.get("status") or "").strip().casefold()
        if state == "completed" or state in failed_states or state == "degraded":
            return status
        await asyncio.sleep(min(0.5, max(0.0, remaining)))


def _report_payload(content: str) -> bytes:
    """Encode one already-controlled report after a final privacy gate."""

    _clean, privacy = PersistencePrivacyGuard().sanitize_text(content)
    if privacy.changed:
        raise RuntimeError("report_privacy_gate_failed")
    payload = content.encode("utf-8")
    if not 1 <= len(payload) <= _MAX_REPORT_BYTES:
        raise RuntimeError("report_size_invalid")
    return payload


def _raise_report_directory_error(
    part: str, directory_fd: int, exc: OSError
) -> NoReturn:
    """Classify a failed component open without ever following the component."""

    try:
        metadata = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        raise RuntimeError("report_directory_invalid") from None
    code = (
        "report_directory_symlink"
        if stat.S_ISLNK(metadata.st_mode)
        else "report_directory_invalid"
    )
    raise RuntimeError(code) from exc


def _create_report_component(directory_fd: int, part: str, flags: int) -> int:
    """Create one missing component 0700 and reopen it no-follow."""

    try:
        os.mkdir(part, mode=0o700, dir_fd=directory_fd)
        _fsync_report_directory(directory_fd)
    except FileExistsError:
        pass
    try:
        return os.open(part, flags, dir_fd=directory_fd)
    except OSError as exc:
        _raise_report_directory_error(part, directory_fd, exc)


def _open_report_component(directory_fd: int, part: str, flags: int) -> int:
    """Open one path component no-follow, creating it when it is absent."""

    if part in {"", ".", ".."}:
        raise RuntimeError("report_directory_invalid")
    try:
        return os.open(part, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return _create_report_component(directory_fd, part, flags)
    except OSError as exc:
        _raise_report_directory_error(part, directory_fd, exc)


def _open_report_directory(path: Path) -> int:
    """Open or create a POSIX directory by traversing every component no-follow."""

    absolute = path.absolute()
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current_fd = os.open(absolute.anchor, directory_flags)
    try:
        for part in absolute.parts[1:]:
            next_fd = _open_report_component(current_fd, part, directory_flags)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _check_report_destination(directory_fd: int, filename: str) -> None:
    """Reject a symlink or non-regular destination without following it."""

    try:
        metadata = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError("report_destination_symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("report_destination_invalid")


def _fsync_report_directory(directory_fd: int) -> None:
    """Persist a directory update when the host filesystem supports it."""

    try:
        os.fsync(directory_fd)
    except OSError as exc:
        unsupported = {
            errno.EBADF,
            errno.EINVAL,
            getattr(errno, "ENOTSUP", -1),
            getattr(errno, "EOPNOTSUPP", -1),
        }
        if exc.errno not in unsupported:
            raise


def _publish_report_posix(destination: Path, payload: bytes) -> None:
    """Publish through a no-follow directory descriptor on POSIX."""

    directory_fd = _open_report_directory(destination.parent)
    temporary_name = ""
    try:
        _check_report_destination(directory_fd, destination.name)
        create_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for _attempt in range(8):
            candidate = f".{destination.name}.{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    create_flags,
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        else:
            raise RuntimeError("report_temporary_unavailable")
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _check_report_destination(directory_fd, destination.name)
        os.replace(
            temporary_name,
            destination.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = ""
        _fsync_report_directory(directory_fd)
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def publish_report(destination: Path, content: str) -> None:
    """Publish a bounded report only where descriptor-safe privacy is available."""

    if destination.name in {"", ".", ".."}:
        raise RuntimeError("report_destination_invalid")
    if os.name != "posix":
        raise RuntimeError("report_platform_unsupported")
    payload = _report_payload(content)
    _publish_report_posix(destination, payload)


def _validate_result_set(results: list[CaseResult], *, mode: str) -> None:
    """Require the exact selected catalog and unique nonempty evidence refs."""

    _defaults, catalog = load_matrix()
    expected = {
        case.case_id: (case.skill, case.mode, case.model_class)
        for case in catalog
        if mode in {"all", case.mode}
    }
    actual_ids = [result.case_id for result in results]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected):
        raise RuntimeError("runtime_case_set_invalid")
    if any(
        (result.skill, result.mode, result.model_class) != expected[result.case_id]
        for result in results
    ):
        raise RuntimeError("runtime_case_contract_invalid")
    _require_unique_evidence_references(results)


def _require_unique_evidence_references(results: list[CaseResult]) -> None:
    """Reject any two cases claiming the same run or trace reference."""

    for attribute in ("run_ref", "trace_ref"):
        references = [
            str(getattr(result, attribute))
            for result in results
            if getattr(result, attribute)
        ]
        if len(references) != len(set(references)):
            raise RuntimeError("runtime_evidence_reference_collision")


# Column order of the per-skill table; changing either tuple changes the report.
_DIRECT_REPORT_CHECKS = (
    "structural",
    "model_selection",
    "skill_binding",
    "semantic",
    "trace",
    "parent_ingestion",
)
_DELEGATED_REPORT_CHECKS = (
    "structural",
    "model_selection",
    "skill_binding",
    "semantic",
    "delegation",
    "trace",
    "parent_ingestion",
)


def _report_header_lines(generated_at: str) -> list[str]:
    """Return the report preamble and the per-skill table header."""

    return [
        "# Agent Utilities consolidated skill validation matrix",
        "",
        f"Generated: {generated_at}",
        "",
        "Validation used synthetic, read-only cases, sequential execution, "
        "metadata-only observability, and neutral `skill://` references. Raw model "
        "output, prompts, endpoints, credentials, identities, trace identifiers, and "
        "filesystem locations are intentionally absent.",
        "",
        "## Per-skill result",
        "",
        "| Skill | Direct static | Direct model selection | Direct skill binding | Direct semantic | Direct trace | Direct KG ingest | Delegated static | Delegated model selection | Delegated skill binding | Delegated semantic | Graph-OS delegation | Delegated trace | Delegated KG ingest | Paired result |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]


def _report_check_cells(
    result: CaseResult | None, checks: tuple[str, ...]
) -> list[str]:
    """Render one mode's check columns, or `not-run` when the case is absent."""

    if result is None:
        return ["not-run"] * len(checks)
    return [str(getattr(result, name)) for name in checks]


def _skill_matrix_row(skill: str, pair: dict[str, CaseResult]) -> str:
    """Render one skill's direct/delegated row of the per-skill table."""

    direct = pair.get("direct")
    delegated = pair.get("delegated")
    pair_passed = bool(direct and delegated and direct.passed and delegated.passed)
    cells = [
        f"`{skill}`",
        *_report_check_cells(direct, _DIRECT_REPORT_CHECKS),
        *_report_check_cells(delegated, _DELEGATED_REPORT_CHECKS),
        _PASS if pair_passed else _FAIL,
    ]
    return "| " + " | ".join(cells) + " |"


def _evidence_report_row(result: CaseResult) -> str:
    """Render one case's row of the privacy-safe evidence table."""

    routes = ", ".join(f"`{route}`" for route in result.selected_routes) or "none"
    errors = ", ".join(f"`{code}`" for code in result.error_codes) or "none"
    return (
        f"| `{result.case_id}` | {routes} | `{result.model_ref or 'none'}` | "
        f"`{result.skill_ref or 'none'}` | `{result.skill_body_ref or 'none'}` | "
        f"`{result.run_ref or 'none'}` | `{result.trace_ref or 'none'}` | "
        f"{result.trace_linkage} | {errors} |"
    )


def _report_aggregate_lines(
    results: list[CaseResult], by_skill: dict[str, dict[str, CaseResult]]
) -> list[str]:
    """Render the aggregate section and its linkage/ingestion method notes."""

    passed = sum(result.passed for result in results)
    fully_passed = sum(
        all(item.passed for item in pair.values()) and len(pair) == 2
        for pair in by_skill.values()
    )
    return [
        "",
        "## Aggregate",
        "",
        f"- Cases passed: {passed}/{len(results)}",
        f"- Skills fully passed: {fully_passed}/{len(by_skill)}",
        "- Trace linkage method: one exact-name `graph_run` trace whose metadata binds the case run, configured model, model class, skill, and skill body, queried through the Langfuse MCP tool mounted by Graph-OS.",
        "- Parent-ingestion proof: each exact trace resolves to exactly one `Trace` node written by Graph-OS parent mediation under verified `kg:write` authority.",
        "",
    ]


def render_report(results: list[CaseResult], *, generated_at: str) -> str:
    """Render only controlled fields and opaque references."""

    by_skill: dict[str, dict[str, CaseResult]] = {}
    for result in results:
        by_skill.setdefault(result.skill, {})[result.mode] = result
    lines = _report_header_lines(generated_at)
    lines.extend(
        _skill_matrix_row(skill, by_skill[skill]) for skill in sorted(by_skill)
    )
    lines.extend(
        [
            "",
            "## Privacy-safe evidence",
            "",
            "| Case | Routes selected | Model reference | Skill reference | Skill body reference | Run reference | Trace reference | Linkage | Errors |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    lines.extend(
        _evidence_report_row(result)
        for result in sorted(results, key=lambda item: item.case_id)
    )
    lines.extend(_report_aggregate_lines(results, by_skill))
    rendered = "\n".join(lines)
    _clean, privacy = PersistencePrivacyGuard().sanitize_text(rendered)
    if privacy.changed:
        raise RuntimeError("report_privacy_gate_failed")
    return rendered


def _valid_command_word(item: object) -> bool:
    """Accept only a bounded, NUL-free, non-empty argv word."""

    return isinstance(item, str) and 0 < len(item) <= 4_096 and "\x00" not in item


def _valid_command_argv(argv: object) -> TypeGuard[list[str]]:
    """Accept only a bounded list of valid argv words."""

    return (
        isinstance(argv, list)
        and 1 <= len(argv) <= 32
        and all(_valid_command_word(item) for item in argv)
    )


def _external_executable_unsafe(
    executable: Path,
    original: os.stat_result,
    canonical: Path,
    metadata: os.stat_result,
) -> bool:
    """Reject a relative, symlinked, swapped, shell, or non-executable target."""

    return (
        not executable.is_absolute()
        or stat.S_ISLNK(original.st_mode)
        or not stat.S_ISREG(original.st_mode)
        or (original.st_dev, original.st_ino) != (metadata.st_dev, metadata.st_ino)
        or canonical.name.casefold() in _SHELL_EXECUTABLES
        or canonical.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or not os.access(canonical, os.X_OK)
    )


def _validate_external_command_argv(argv: object) -> list[str]:
    """Resolve one bounded, non-shell external command without executing it."""

    if not _valid_command_argv(argv):
        raise RuntimeError("evidence_command_reference_invalid")
    executable = Path(argv[0])
    try:
        original = executable.lstat()
        canonical = executable.resolve(strict=True)
        metadata = canonical.lstat()
    except OSError as exc:
        raise RuntimeError("evidence_command_reference_invalid") from exc
    if _external_executable_unsafe(executable, original, canonical, metadata):
        raise RuntimeError("evidence_command_reference_invalid")
    return [str(canonical), *argv[1:]]


def _external_command(reference: str) -> list[str]:
    if _COMMAND_REFERENCE.fullmatch(reference) is None:
        raise RuntimeError("evidence_command_reference_invalid")
    raw = str(setting(reference, "") or "")
    if not raw:
        raise RuntimeError("evidence_command_reference_unresolved")
    try:
        argv = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("evidence_command_reference_invalid") from exc
    return _validate_external_command_argv(argv)


def _external_json(reference: str, payload: bytes) -> dict[str, Any]:
    completed = subprocess.run(
        _external_command(reference),
        input=payload,
        capture_output=True,
        check=False,
        timeout=120,
        close_fds=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("external_evidence_command_failed")
    if len(completed.stdout) > _MAX_EXTERNAL_OUTPUT_BYTES:
        raise RuntimeError("external_evidence_output_too_large")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("external_evidence_output_invalid") from exc
    if not isinstance(response, dict):
        raise RuntimeError("external_evidence_output_invalid")
    return response


def _signature_from_response(
    response: dict[str, Any], *, subject_digest: str
) -> dict[str, str]:
    signature = {
        "algorithm": str(response.get("algorithm") or ""),
        "keyId": str(response.get("keyId") or ""),
        "signature": str(response.get("signature") or ""),
        "subjectDigest": str(response.get("subjectDigest") or ""),
    }
    if (
        set(response) != set(signature)
        or signature["algorithm"] not in _SIGNATURE_ALGORITHMS
        or _KEY_ID.fullmatch(signature["keyId"]) is None
        or _SIGNATURE_VALUE.fullmatch(signature["signature"]) is None
        or signature["subjectDigest"] != subject_digest
    ):
        raise RuntimeError("evidence_signature_invalid")
    return signature


def sign_and_verify_evidence(
    unsigned: dict[str, Any], *, signer_reference: str, verifier_reference: str
) -> dict[str, Any]:
    """Sign canonical evidence externally and require the independent verifier."""

    if "signature" in unsigned:
        raise RuntimeError("evidence_unsigned_contract_invalid")
    subject_digest = _digest_bytes(_canonical_bytes(unsigned))
    signature = _signature_from_response(
        _external_json(signer_reference, _canonical_bytes(unsigned)),
        subject_digest=subject_digest,
    )
    signed = {**unsigned, "signature": signature}
    verification = _external_json(verifier_reference, _canonical_bytes(signed))
    if verification != {
        "verified": True,
        "subjectDigest": subject_digest,
        "keyId": signature["keyId"],
    }:
        raise RuntimeError("evidence_verification_failed")
    return signed


def verify_signed_evidence(
    signed: dict[str, Any], *, verifier_reference: str
) -> dict[str, Any]:
    """Independently verify one closed evidence document.

    The verifier receives the canonical signed document over stdin and must
    return the exact bounded acknowledgement used by the producer.  This
    function never trusts a producer-side verification result and never emits
    signer, command, path, endpoint, or identity material.
    """

    if not isinstance(signed, dict) or "signature" not in signed:
        raise RuntimeError("evidence_signed_contract_invalid")
    signature_value = signed.get("signature")
    if not isinstance(signature_value, dict):
        raise RuntimeError("evidence_signature_invalid")
    unsigned = {key: value for key, value in signed.items() if key != "signature"}
    subject_digest = _digest_bytes(_canonical_bytes(unsigned))
    signature = _signature_from_response(signature_value, subject_digest=subject_digest)
    verification = _external_json(verifier_reference, _canonical_bytes(signed))
    expected = {
        "verified": True,
        "subjectDigest": subject_digest,
        "keyId": signature["keyId"],
    }
    if verification != expected:
        raise RuntimeError("evidence_verification_failed")
    return unsigned


def _controlled_ref(value: str) -> str | None:
    """Retain an opaque reference only when it matches the exact ref pattern."""

    return value if re.fullmatch(r"pref_[a-z_]+_[a-f0-9]{64}", value or "") else None


def _controlled_trace_name(value: str) -> str | None:
    """Retain a trace name only when it matches the exact opaque run pattern."""

    return (
        value if re.fullmatch(r"graph_run:pref_run_[a-f0-9]{64}", value or "") else None
    )


def _require_exact_case_set(
    results: list[CaseResult],
    result_by_id: dict[str, CaseResult],
    cases: list[ValidationCase],
) -> None:
    """Require exactly one result per catalog case, with no duplicate ids."""

    if (
        len(results) != _CASE_COUNT
        or len(result_by_id) != _CASE_COUNT
        or set(result_by_id) != {case.case_id for case in cases}
    ):
        raise RuntimeError("runtime_case_set_not_exact")


def _evidence_case_entry(
    case: ValidationCase, result: CaseResult, case_digest: str
) -> dict[str, Any]:
    """Build the closed, content-free evidence subject for one case."""

    return {
        "caseId": case.case_id,
        "caseDigest": case_digest,
        "skill": case.skill,
        "mode": case.mode,
        "modelClass": result.model_class,
        "status": _PASS if result.passed else _FAIL,
        "checks": {
            "structural": result.structural,
            "modelSelection": result.model_selection,
            "skillBinding": result.skill_binding,
            "semantic": result.semantic,
            "delegation": result.delegation,
            "trace": result.trace,
            "parentKnowledgeGraph": result.parent_ingestion,
        },
        "skillRef": _controlled_ref(result.skill_ref),
        "skillBodyRef": _controlled_ref(result.skill_body_ref),
        "runRef": _controlled_ref(result.run_ref),
        "traceRef": _controlled_ref(result.trace_ref),
        "langfuse": {
            "lookupMethod": "exact-name",
            "metadataOnly": True,
            "traceName": _controlled_trace_name(result.trace_name),
            "matchCount": result.langfuse_match_count,
            "linkage": result.trace_linkage,
        },
        "parentKnowledgeGraph": {
            "readbackMethod": "exact-trace-name",
            "matchCount": result.parent_kg_readback_count,
        },
        "architecture": _architecture_evidence_entry(case, result),
        "errorCodes": sorted(result.error_codes),
    }


def _architecture_evidence_entry(
    case: ValidationCase, result: CaseResult
) -> dict[str, Any]:
    """Build the exact content-free architecture observation block."""

    candidate_ref = (
        _architecture_candidate_ref(case.architecture_candidate)
        if case.architecture_candidate is not None
        else ""
    )
    return {
        "candidateRef": _controlled_ref(candidate_ref),
        "operations": [
            {
                "phase": item.phase,
                "operation": item.operation,
                "status": item.status,
                "requestDigest": item.request_digest,
                "responseDigest": item.response_digest,
                "matchedRecordCount": item.matched_record_count,
            }
            for item in result.operation_evidence
        ],
        "scenarios": [
            {"scenario": item.scenario, "outcome": item.outcome}
            for item in result.scenario_evidence
        ],
    }


def _fully_passed_skill_count(results: list[CaseResult]) -> int:
    """Count skills whose direct and delegated cases are both present and passed."""

    skills = {result.skill for result in results}
    return sum(
        len(items) == 2 and all(item.passed for item in items)
        for skill in skills
        for items in [[item for item in results if item.skill == skill]]
    )


def _evidence_result_block(passed: int, fully_passed: int) -> dict[str, Any]:
    """Build the aggregate result block of the evidence subject."""

    exact = passed == _CASE_COUNT and fully_passed == _SKILL_COUNT
    return {
        "status": _PASS if exact else _FAIL,
        "passedCases": passed,
        "totalCases": _CASE_COUNT,
        "fullyPassedSkills": fully_passed,
        "totalSkills": _SKILL_COUNT,
    }


def build_evidence(
    results: list[CaseResult],
    *,
    generated_at: str,
    release_id: str,
    release_specification_digest: str,
    promotion_evidence_digest: str,
    graph_os_digest: str,
    engine_digest: str,
    runtime_config_digest: str,
    runtime_profile_digest: str,
    model_registry_digest: str,
) -> dict[str, Any]:
    """Build the closed, content-free exact-release skill evidence subject."""

    if _RELEASE_ID.fullmatch(release_id) is None:
        raise ValueError("release_id_invalid")
    _require_digest(release_specification_digest, "release_specification_digest")
    _require_digest(promotion_evidence_digest, "promotion_evidence_digest")
    _require_digest(graph_os_digest, "graph_os_digest")
    _require_digest(engine_digest, "engine_digest")
    _require_digest(runtime_config_digest, "runtime_config_digest")
    _require_digest(runtime_profile_digest, "runtime_profile_digest")
    _require_digest(model_registry_digest, "model_registry_digest")
    _defaults, cases = load_matrix()
    catalog = _test_catalog_evidence(cases)
    result_by_id = {result.case_id: result for result in results}
    _require_exact_case_set(results, result_by_id, cases)
    evidence_cases = [
        _evidence_case_entry(
            case,
            result_by_id[case.case_id],
            catalog["caseDigests"][case.case_id],
        )
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    passed = sum(result.passed for result in results)
    fully_passed = _fully_passed_skill_count(results)
    evidence = {
        "apiVersion": "graphos.io/v2",
        "kind": "PrebundledSkillValidationEvidence",
        "evidenceVersion": 2,
        "generatedAt": generated_at,
        "release": {
            "id": release_id,
            "specificationDigest": release_specification_digest,
            "promotionEvidenceDigest": promotion_evidence_digest,
            "graphOsDigest": graph_os_digest,
            "engineDigest": engine_digest,
        },
        "runtime": {
            "configurationDigest": runtime_config_digest,
            "profileDigest": runtime_profile_digest,
            "modelRegistryDigest": model_registry_digest,
            "sequential": True,
            "metadataOnlyObservability": True,
        },
        "catalog": {
            "skillCount": _SKILL_COUNT,
            "skillCatalogDigest": prebundled_skill_catalog_digest(SKILLS_ROOT),
            "testCaseCount": _CASE_COUNT,
            "testCatalogDigest": catalog["testCatalogDigest"],
            "caseCatalogDigest": catalog["caseCatalogDigest"],
        },
        "cases": evidence_cases,
        "result": _evidence_result_block(passed, fully_passed),
        "privacy": {
            "containsPrompts": False,
            "containsModelOutput": False,
            "containsEndpoints": False,
            "containsCredentials": False,
            "containsIdentities": False,
            "containsFilesystemLocations": False,
            "containsRawTraceIdentifiers": False,
        },
    }
    _clean, privacy = PersistencePrivacyGuard().sanitize(evidence)
    if privacy.changed:
        raise RuntimeError("evidence_privacy_gate_failed")
    return evidence


def render_evidence(evidence: dict[str, Any]) -> str:
    rendered = json.dumps(evidence, sort_keys=True, indent=2) + "\n"
    _report_payload(rendered)
    return rendered


def _validated_graph_os_url(args: argparse.Namespace) -> str:
    """Require a configured Graph-OS URL and a metadata-only, ingesting runtime."""

    from agent_utilities.core.config import config, setting

    graph_os_url = str(args.graph_os_url or config.mcp_url or "").strip()
    if not graph_os_url:
        raise RuntimeError("graph_os_url_unconfigured")
    capture_content = str(setting("LANGFUSE_CAPTURE_CONTENT", "false") or "false")
    if capture_content.strip().casefold() in {"1", "true", "yes", "on"}:
        raise RuntimeError("langfuse_content_capture_must_be_disabled")
    if not config.langfuse_kg_auto_ingest:
        raise RuntimeError("langfuse_parent_ingestion_required")
    return graph_os_url


async def _prepare_validation_tools(
    client: Any, cases: list[ValidationCase], tenant_id: str
) -> str:
    """Load the exact tool surface and prove no probe trace already exists."""

    await _ensure_tool(client, "graph_orchestrate", 30.0)
    await _ensure_tool(client, "graph_query", 30.0)
    if any(case.skill == _ARCHITECTURE_SKILL for case in cases):
        await _ensure_tool(client, "graph_search", 30.0)
        await _ensure_tool(client, "graph_code", 30.0)
    if any(case.mode == "delegated" for case in cases):
        await _ensure_tool(client, "graph_jobs", 30.0)
    langfuse_tool = await _load_langfuse_tool(client, 30.0)
    await _verify_langfuse_posture(client, langfuse_tool, 30.0)
    probe_name = _expected_trace_name(new_run_id(), tenant_id)
    if await _trace_snapshot(
        client,
        langfuse_tool,
        30.0,
        expected_name=probe_name,
    ):
        raise RuntimeError("trace_probe_collision")
    return langfuse_tool


async def _run_validation_case(
    case: ValidationCase,
    *,
    client: Any,
    langfuse_tool: str,
    tenant_id: str,
    defaults: dict[str, int | bool],
    args: argparse.Namespace,
    expected_authority: Any,
) -> CaseResult:
    """Renew the per-mode authority and run one case on its execution path."""

    trace_timeout = float(defaults["trace_timeout_seconds"])
    minimum_ttl_seconds = _direct_case_minimum_authority_ttl(
        case_timeout=args.case_timeout, trace_timeout=trace_timeout
    )
    if case.mode == "direct":
        from agent_utilities.knowledge_graph.core.session import use_session
        from agent_utilities.security.brain_context import use_actor

        validation_session = await _renew_direct_validation_session(
            expected_authority=expected_authority,
            minimum_ttl_seconds=minimum_ttl_seconds,
        )
        with (
            use_actor(validation_session.actor),
            use_session(validation_session),
        ):
            return await _run_direct_case(
                case,
                client=client,
                langfuse_tool=langfuse_tool,
                tenant_id=tenant_id,
                case_timeout=args.case_timeout,
                trace_timeout=trace_timeout,
            )
    await _renew_delegated_validation_session(
        expected_authority=expected_authority,
        minimum_ttl_seconds=minimum_ttl_seconds,
    )
    return await _run_delegated_case(
        case,
        client=client,
        langfuse_tool=langfuse_tool,
        tenant_id=tenant_id,
        max_steps=int(defaults["max_steps"]),
        token_budget=int(defaults["token_budget"]),
        case_timeout=args.case_timeout,
        trace_timeout=trace_timeout,
    )


async def run(args: argparse.Namespace) -> list[CaseResult]:
    defaults, all_cases = load_matrix()
    cases = [case for case in all_cases if args.mode in {"all", case.mode}]

    graph_os_url = _validated_graph_os_url(args)

    from agent_utilities.mcp.client_credentials import child_auth, child_auth_header
    from agent_utilities.mcp.toolset_factory import build_http_toolset

    headers = child_auth_header({})
    identity_session = await _verified_validation_session(
        headers, minimum_ttl_seconds=1
    )
    tenant_id = str(identity_session.tenant)
    expected_authority = identity_session.engine_verified_context()
    toolset = build_http_toolset(
        graph_os_url,
        auth=child_auth({}),
        timeout=args.case_timeout,
        toolset_id="skill-validation",
    )
    results: list[CaseResult] = []
    async with toolset.client as client:
        langfuse_tool = await _prepare_validation_tools(client, cases, tenant_id)
        for case in cases:
            item = await _run_validation_case(
                case,
                client=client,
                langfuse_tool=langfuse_tool,
                tenant_id=tenant_id,
                defaults=defaults,
                args=args,
                expected_authority=expected_authority,
            )
            results.append(item)
            if _SYNC_CALL_POISONED.is_set():
                raise RuntimeError("blocking_sdk_worker_abandoned")
    return results


def _build_argument_parser() -> argparse.ArgumentParser:
    """Declare the full command-line surface of the validation harness."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "delegated", "all"), default="all")
    parser.add_argument(
        "--graph-os-url",
        default="",
        help="Existing Graph-OS streamable-HTTP URL; defaults to AgentConfig MCP_URL.",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=120.0,
        help="Per-case wall-clock limit in seconds (1-600).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional Markdown output destination; its location is never recorded.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=None,
        help="Strict signed JSON evidence destination used by --mode all.",
    )
    parser.add_argument("--release-id", default="")
    parser.add_argument("--release-specification-digest", default="")
    parser.add_argument("--promotion-evidence-digest", default="")
    parser.add_argument("--graph-os-digest", default="")
    parser.add_argument("--engine-digest", default="")
    parser.add_argument("--runtime-config-digest", default="")
    parser.add_argument("--runtime-profile-digest", default="")
    parser.add_argument("--model-registry-digest", default="")
    parser.add_argument(
        "--signer-command-ref",
        default=_SIGNER_COMMAND_REFERENCE,
        help="Environment variable containing the external signer JSON argv.",
    )
    parser.add_argument(
        "--verifier-command-ref",
        default=_VERIFIER_COMMAND_REFERENCE,
        help="Environment variable containing the external verifier JSON argv.",
    )
    return parser


def _release_argument_values(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the exact-release argument values in their declared order."""

    return (
        args.release_id,
        args.release_specification_digest,
        args.promotion_evidence_digest,
        args.graph_os_digest,
        args.engine_digest,
        args.runtime_config_digest,
        args.runtime_profile_digest,
        args.model_registry_digest,
    )


def _validate_release_destinations(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Require both publication destinations, colocated and correctly suffixed."""

    if (
        args.report is None
        or args.evidence is None
        or not all(_release_argument_values(args))
    ):
        parser.error(
            "--mode all requires --report, --evidence, --release-id, "
            "--release-specification-digest, --promotion-evidence-digest, "
            "--graph-os-digest, --engine-digest, --runtime-config-digest, "
            "--runtime-profile-digest, and --model-registry-digest"
        )
    if args.report.parent.absolute() != args.evidence.parent.absolute():
        parser.error("--report and --evidence must be published alongside")
    if args.report.suffix.casefold() != ".md" or args.evidence.suffix != ".json":
        parser.error("--report must be Markdown and --evidence must be JSON")


def _validate_release_references(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Require an exact release id, real digests, and signer/verifier refs."""

    if _RELEASE_ID.fullmatch(args.release_id) is None:
        parser.error("--release-id is invalid")
    for option, value in (
        ("--release-specification-digest", args.release_specification_digest),
        ("--promotion-evidence-digest", args.promotion_evidence_digest),
        ("--graph-os-digest", args.graph_os_digest),
        ("--engine-digest", args.engine_digest),
        ("--runtime-config-digest", args.runtime_config_digest),
        ("--runtime-profile-digest", args.runtime_profile_digest),
        ("--model-registry-digest", args.model_registry_digest),
    ):
        if _DIGEST.fullmatch(value) is None:
            parser.error(f"{option} must be a non-sentinel sha256 digest")
    for option, value in (
        ("--signer-command-ref", args.signer_command_ref),
        ("--verifier-command-ref", args.verifier_command_ref),
    ):
        if _COMMAND_REFERENCE.fullmatch(value) is None:
            parser.error(f"{option} must be an environment reference")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if not 1.0 <= args.case_timeout <= 600.0:
        parser.error("--case-timeout must be between 1 and 600 seconds")
    if args.mode == "all":
        _validate_release_destinations(parser, args)
        _validate_release_references(parser, args)
    elif args.evidence is not None or any(_release_argument_values(args)):
        parser.error("exact release evidence is emitted only by --mode all")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    static_errors = validate_static_suite()
    if static_errors:
        print(f"Static skill validation failed with {len(static_errors)} issue(s).")
        return 2
    try:
        results = asyncio.run(run(args))
        _validate_result_set(results, mode=args.mode)
        generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        report = render_report(results, generated_at=generated_at)
        if args.mode == "all":
            unsigned = build_evidence(
                results,
                generated_at=generated_at,
                release_id=args.release_id,
                release_specification_digest=args.release_specification_digest,
                promotion_evidence_digest=args.promotion_evidence_digest,
                graph_os_digest=args.graph_os_digest,
                engine_digest=args.engine_digest,
                runtime_config_digest=args.runtime_config_digest,
                runtime_profile_digest=args.runtime_profile_digest,
                model_registry_digest=args.model_registry_digest,
            )
            evidence = sign_and_verify_evidence(
                unsigned,
                signer_reference=args.signer_command_ref,
                verifier_reference=args.verifier_command_ref,
            )
            publish_report(args.evidence, render_evidence(evidence))
            publish_report(args.report, report)
        elif args.report is not None:
            publish_report(args.report, report)
        else:
            print(report)
    except Exception as exc:  # noqa: BLE001 - never print environment-bearing messages
        print(f"Runtime skill validation failed ({type(exc).__name__}).")
        return 2
    passed = sum(result.passed for result in results)
    print(f"Runtime skill validation: {passed}/{len(results)} cases passed.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
