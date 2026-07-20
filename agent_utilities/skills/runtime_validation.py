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
from typing import Annotated, Any, Literal

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
_TRACE_PAGE_LIMIT = 20
_TRACE_MAX_PAGES = 10
_TRACE_TOOL_ERROR_RETRIES = 2
_TRACE_TOOL_ERROR_RETRY_DELAY_SECONDS = 0.25
_PARENT_INGESTION_POLL_DELAY_SECONDS = 0.25
_DIRECT_MAX_OUTPUT_TOKENS = 1024
_MAX_REPORT_BYTES = 1_000_000
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


class DelegationContractError(ValueError):
    """Controlled current-contract diagnostic with no response material."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ValidationChildToolError(RuntimeError):
    """Controlled retryable failure returned by an MCP child tool."""


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
    error_codes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
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
        references_valid = all(
            pattern.fullmatch(value) is not None
            for pattern, value in (
                (_CASE_REFERENCE_PATTERNS["run"], self.run_ref),
                (_CASE_REFERENCE_PATTERNS["trace"], self.trace_ref),
                (_CASE_REFERENCE_PATTERNS["model"], self.model_ref),
                (_CASE_REFERENCE_PATTERNS["skill"], self.skill_ref),
                (_CASE_REFERENCE_PATTERNS["skill_body"], self.skill_body_ref),
            )
        )
        return bool(
            all(value == _PASS for value in required)
            and not self.error_codes
            and self.trace_linkage == "run-evidence"
            and self.trace_name == f"graph_run:{self.run_ref}"
            and self.langfuse_match_count == 1
            and self.parent_kg_readback_count == 1
            and self.selected_routes
            and all(_SAFE_ROUTE.fullmatch(route) for route in self.selected_routes)
            and references_valid
        )

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


def _case_contract(case: ValidationCase) -> dict[str, Any]:
    """Return the content-free canonical contract bound into release evidence."""

    return {
        "id": case.case_id,
        "skill": case.skill,
        "mode": case.mode,
        "modelClass": case.model_class,
        "taskDigest": _digest_bytes(case.task.encode("utf-8")),
        "expectedRoutes": list(case.expected_routes),
        "allowedTools": list(case.allowed_tools),
        "readOnly": case.read_only,
    }


def _test_catalog_evidence(cases: list[ValidationCase]) -> dict[str, Any]:
    matrix = yaml.safe_load(FORWARD_MATRIX.read_text(encoding="utf-8"))
    contracts = [_case_contract(case) for case in cases]
    if len(contracts) != 20 or len({item["id"] for item in contracts}) != 20:
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
        )
        for item in raw["cases"]
    ]
    return defaults, cases


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
        self, query: str, *, top_k: int = 8, as_of: str | None = None
    ) -> list[dict[str, Any]]:
        del query, as_of
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
        list[route_literal],
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
    routes = output.selected_routes
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
    _clean, privacy = PersistencePrivacyGuard().sanitize(output.model_dump())
    if privacy.changed:
        errors.append("semantic_output_privacy_violation")
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


def _validate_tool_payload_bounds(value: Any) -> None:
    """Reject oversized, cyclic, deep, or non-data MCP payloads before use."""

    remaining = _MAX_TOOL_PAYLOAD
    items = 0
    seen: set[int] = set()
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_TOOL_DEPTH:
            raise ValueError("payload_too_deep")
        items += 1
        if items > _MAX_TOOL_ITEMS:
            raise ValueError("payload_too_many_items")
        if current is None or isinstance(current, bool | int | float):
            remaining -= 16
        elif isinstance(current, str):
            if len(current) > remaining:
                raise ValueError("payload_too_large")
            remaining -= len(current.encode("utf-8"))
        elif isinstance(current, bytes):
            remaining -= len(current)
        elif isinstance(current, dict):
            identity = id(current)
            if identity in seen:
                raise ValueError("payload_cycle")
            seen.add(identity)
            if len(current) > _MAX_TOOL_ITEMS - items:
                raise ValueError("payload_too_many_items")
            for key, item in current.items():
                if not isinstance(key, str):
                    raise TypeError("payload_key_invalid")
                stack.append((item, depth + 1))
                stack.append((key, depth + 1))
        elif isinstance(current, list | tuple):
            identity = id(current)
            if identity in seen:
                raise ValueError("payload_cycle")
            seen.add(identity)
            if len(current) > _MAX_TOOL_ITEMS - items:
                raise ValueError("payload_too_many_items")
            stack.extend((item, depth + 1) for item in current)
        elif _is_fastmcp_structured_dataclass(current):
            identity = id(current)
            if identity in seen:
                raise ValueError("payload_cycle")
            seen.add(identity)
            members = dataclass_fields(current)
            if len(members) > _MAX_TOOL_ITEMS - items:
                raise ValueError("payload_too_many_items")
            for member in members:
                stack.append((getattr(current, member.name), depth + 1))
                stack.append((member.name, depth + 1))
        else:
            raise TypeError("payload_type_invalid")
        if remaining < 0:
            raise ValueError("payload_too_large")


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


def _decode_tool_result(result: Any) -> Any:
    for attr in ("data", "structured_content"):
        value = getattr(result, attr, None)
        if value not in (None, {}):
            if isinstance(value, str):
                try:
                    return _parse_json_text(value)
                except json.JSONDecodeError:
                    return value
            if isinstance(value, BaseModel):
                value = value.model_dump(mode="json")
            _validate_tool_payload_bounds(value)
            return _normalize_fastmcp_structured_data(value)
    content = getattr(result, "content", None)
    if isinstance(content, list):
        if len(content) > _MAX_TOOL_ITEMS:
            raise ValueError("payload_too_many_items")
        texts: list[str] = []
        characters = 0
        for item in content:
            text = str(getattr(item, "text", ""))
            characters += len(text) + (1 if text and texts else 0)
            if characters > _MAX_TOOL_PAYLOAD:
                raise ValueError("payload_too_large")
            if text:
                texts.append(text)
        joined = "\n".join(texts)
        if joined:
            try:
                return _parse_json_text(joined)
            except json.JSONDecodeError:
                return joined
    if isinstance(result, str):
        try:
            return _parse_json_text(result)
        except json.JSONDecodeError:
            if len(result) > _MAX_TOOL_PAYLOAD:
                raise ValueError("payload_too_large") from None
            return result
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
    candidates = []
    if isinstance(catalog, dict):
        for entry in catalog.get("tools") or []:
            if (
                isinstance(entry, dict)
                and entry.get("tool") == "langfuse_observability"
            ):
                candidates.append(str(entry.get("prefixed_name") or ""))
    candidates = [name for name in candidates if _SAFE_ROUTE.fullmatch(name)]
    if len(candidates) != 1:
        raise RuntimeError("langfuse_tool_discovery_failed")
    if candidates[0] not in names:
        await _call_tool(client, "load_tools", {"tools": candidates}, timeout)
        names = await _list_tool_names(client, timeout)
        if candidates[0] not in names:
            raise RuntimeError("langfuse_tool_load_failed")
    return candidates[0]


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
        payload = await _call_tool(client, langfuse_tool, args, min(remaining, timeout))
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("trace_snapshot_invalid")
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
        if len(rows) < _TRACE_PAGE_LIMIT:
            return snapshot
    if from_timestamp:
        raise RuntimeError("trace_snapshot_boundary_exceeded")
    return snapshot


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
            if transient_tool_errors >= _TRACE_TOOL_ERROR_RETRIES:
                raise
            transient_tool_errors += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            await asyncio.sleep(min(_TRACE_TOOL_ERROR_RETRY_DELAY_SECONDS, remaining))
            continue
        matching = sorted(current)
        if len(matching) == 1:
            record = current[matching[0]]
            if any(
                record.evidence.get(key) != value
                for key, value in expected_evidence.items()
            ):
                raise RuntimeError("trace_evidence_mismatch")
            return matching[0], "run-evidence"
        if len(matching) > 1:
            raise RuntimeError("trace_run_identifier_ambiguous")
        await asyncio.sleep(1.0)
    raise TimeoutError("trace_not_observed")


async def _verify_parent_ingested_trace(
    client: Any,
    expected_name: str,
    timeout: float,
) -> int:
    """Require exactly one parent-mediated KG node for an exact opaque trace."""

    if not re.fullmatch(r"graph_run:pref_run_[a-f0-9]{64}", expected_name):
        raise RuntimeError("trace_expected_name_invalid")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("trace_parent_ingestion_timeout_invalid")
    arguments = {
        "cypher": (
            "MATCH (n:Trace) WHERE n.name = $name "
            "RETURN n.id AS id, n.name AS name LIMIT 2"
        ),
        "params": json.dumps({"name": expected_name}, separators=(",", ":")),
        "scope": "local",
    }
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
            if transient_tool_errors >= _TRACE_TOOL_ERROR_RETRIES:
                raise
            transient_tool_errors += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            await asyncio.sleep(min(_TRACE_TOOL_ERROR_RETRY_DELAY_SECONDS, remaining))
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


def _parent_ingested_trace_count(payload: Any, *, expected_name: str) -> int | None:
    """Count only governed trace-id rows in GraphQuery's EvidenceBundle trace.

    Public graph reads must retain node identity so tenant, ACL, visibility, and
    audit enforcement can govern every returned row.  The query is bounded at
    two rows: one is the required materialization, zero is missing, and two
    proves an ambiguous duplicate.  Accept only that closed projection; an
    aggregate without node identity, a similarly named claim, or a widened row
    is not proof of parent-mediated ingestion.
    """

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
    if set(trace) != {"step", "payload"}:
        return None
    aggregate = trace.get("payload")
    if not isinstance(aggregate, dict) or set(aggregate) != {"rows"}:
        return None
    rows = aggregate.get("rows")
    if not isinstance(rows, list) or len(rows) > 2:
        return None
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "name"}:
            return None
        node_id = row.get("id")
        if (
            not isinstance(node_id, str)
            or re.fullmatch(r"langfuse:trace:[a-f0-9]{32}", node_id) is None
        ):
            return None
        if row.get("name") != expected_name:
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
    expected_trace_evidence: dict[str, str] = {}
    async with _DIRECT_CASE_LOCK:
        try:
            from pydantic_ai import ModelSettings

            from agent_utilities.core.contextual_model import create_context_agent
            from agent_utilities.core.model_factory import create_model
            from agent_utilities.orchestration.agent_runner import (
                _configured_model_for_class,
            )

            with _direct_evidence_authority(case.skill):
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
                expected_trace_evidence = {
                    "run_ref": expected_trace_name.removeprefix("graph_run:"),
                    "model_ref": result.model_ref,
                    "model_class": case.model_class,
                    "skill_ref": result.skill_ref,
                    "skill_body_ref": result.skill_body_ref,
                }
                agent = create_context_agent(
                    model=model,
                    output_type=_direct_semantic_output_type(case),
                    system_prompt=(
                        f"{_skill_runtime_body(case.skill)}\n\n"
                        f"{_contract_instruction(case)}"
                    ),
                    model_settings=ModelSettings(
                        # The closed JSON contract is intentionally small. A bounded
                        # generation keeps CPU-only local-model validation practical.
                        max_tokens=_DIRECT_MAX_OUTPUT_TOKENS,
                        temperature=0.0,
                        timeout=case_timeout,
                    ),
                    retries=2,
                )
                run = await asyncio.wait_for(
                    agent.run(_direct_execution_prompt(case)), timeout=case_timeout
                )
                semantic = SemanticOutput.model_validate(run.output)
                semantic_errors = validate_semantic_output(case, semantic)
                result.selected_routes = tuple(sorted(semantic.selected_routes))
                for error in semantic_errors:
                    result.add_error(error)
                result.semantic = _PASS if not semantic_errors else _FAIL

                from agent_utilities.observability.langfuse_exporter import (
                    get_langfuse_exporter,
                )

                exporter = get_langfuse_exporter()
                if exporter is None:
                    result.add_error("trace_exporter_unavailable")
                else:

                    def emit_trace() -> bool | None:
                        if not exporter.enabled:
                            return None
                        emitted = exporter.export_graph_run(
                            run_id=validation_run_id,
                            query="",
                            status=(
                                "success"
                                if result.semantic == _PASS
                                else "validation_failed"
                            ),
                            token_usage=_usage_counts(run),
                            model=model_name,
                            metadata={"validation_kind": "bundled_skill_direct"},
                            evidence={
                                key: value
                                for key, value in expected_trace_evidence.items()
                                if key != "run_ref"
                            },
                        )
                        exporter.flush()
                        return emitted

                    emitted = await _bounded_sync_call(
                        emit_trace, min(30.0, case_timeout)
                    )
                    if emitted is None:
                        result.add_error("trace_exporter_unavailable")
                    elif not emitted:
                        result.add_error("trace_export_failed")
                result.run_ref = expected_trace_name.removeprefix("graph_run:")
        except Exception as exc:  # noqa: BLE001 - report only the exception class
            result.add_error(f"direct_{type(exc).__name__}")

    if _SYNC_CALL_POISONED.is_set():
        return result
    if not expected_trace_evidence:
        result.add_error("trace_expected_evidence_unavailable")
    else:
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
    return result


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
        response = await _call_tool(
            client,
            "graph_orchestrate",
            {
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
            },
            case_timeout,
        )
        output, run_id = _extract_delegation_envelope(response)
        expected_trace_name = _expected_trace_name(run_id, tenant_id)
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
        if not run_id:
            result.add_error("delegation_run_handle_missing")
        else:
            result.run_ref = expected_trace_name.removeprefix("graph_run:")
            status = await _wait_for_run_completion(
                client, run_id, min(case_timeout, 30.0)
            )
            terminal_error = _delegation_terminal_error_code(status)
            if terminal_error:
                result.add_error(terminal_error)
            evidence_errors, model_ref, skill_ref, digest = (
                _validate_delegated_runtime_evidence(case, status)
            )
            for error in evidence_errors:
                result.add_error(error)
            result.model_ref = model_ref
            result.skill_ref = skill_ref
            result.skill_body_ref = _opaque_ref("skill_body", digest) if digest else ""
            result.model_selection = (
                _PASS
                if not any(error.startswith("model_") for error in evidence_errors)
                else _FAIL
            )
            result.skill_binding = (
                _PASS
                if not any(error.startswith("skill_") for error in evidence_errors)
                else _FAIL
            )
            result.delegation = (
                _PASS if not evidence_errors and terminal_error is None else _FAIL
            )
            if result.model_ref and result.skill_ref and digest:
                expected_trace_evidence = {
                    "run_ref": expected_trace_name.removeprefix("graph_run:"),
                    "model_ref": result.model_ref,
                    "model_class": case.model_class,
                    "skill_ref": result.skill_ref,
                    "skill_body_ref": _opaque_ref("skill_body", digest),
                }
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
            if part in {"", ".", ".."}:
                raise RuntimeError("report_directory_invalid")
            try:
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current_fd)
                    _fsync_report_directory(current_fd)
                except FileExistsError:
                    pass
                try:
                    next_fd = os.open(part, directory_flags, dir_fd=current_fd)
                except OSError as exc:
                    try:
                        metadata = os.stat(
                            part,
                            dir_fd=current_fd,
                            follow_symlinks=False,
                        )
                    except OSError:
                        raise RuntimeError("report_directory_invalid") from None
                    code = (
                        "report_directory_symlink"
                        if stat.S_ISLNK(metadata.st_mode)
                        else "report_directory_invalid"
                    )
                    raise RuntimeError(code) from exc
            except OSError as exc:
                try:
                    metadata = os.stat(
                        part,
                        dir_fd=current_fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    raise RuntimeError("report_directory_invalid") from None
                code = (
                    "report_directory_symlink"
                    if stat.S_ISLNK(metadata.st_mode)
                    else "report_directory_invalid"
                )
                raise RuntimeError(code) from exc
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
    for attribute in ("run_ref", "trace_ref"):
        references = [
            str(getattr(result, attribute))
            for result in results
            if getattr(result, attribute)
        ]
        if len(references) != len(set(references)):
            raise RuntimeError("runtime_evidence_reference_collision")


def render_report(results: list[CaseResult], *, generated_at: str) -> str:
    """Render only controlled fields and opaque references."""

    by_skill: dict[str, dict[str, CaseResult]] = {}
    for result in results:
        by_skill.setdefault(result.skill, {})[result.mode] = result
    lines = [
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
    for skill in sorted(by_skill):
        direct = by_skill[skill].get("direct")
        delegated = by_skill[skill].get("delegated")
        pair_passed = bool(direct and delegated and direct.passed and delegated.passed)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{skill}`",
                    direct.structural if direct else "not-run",
                    direct.model_selection if direct else "not-run",
                    direct.skill_binding if direct else "not-run",
                    direct.semantic if direct else "not-run",
                    direct.trace if direct else "not-run",
                    direct.parent_ingestion if direct else "not-run",
                    delegated.structural if delegated else "not-run",
                    delegated.model_selection if delegated else "not-run",
                    delegated.skill_binding if delegated else "not-run",
                    delegated.semantic if delegated else "not-run",
                    delegated.delegation if delegated else "not-run",
                    delegated.trace if delegated else "not-run",
                    delegated.parent_ingestion if delegated else "not-run",
                    _PASS if pair_passed else _FAIL,
                ]
            )
            + " |"
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
    for result in sorted(results, key=lambda item: item.case_id):
        routes = ", ".join(f"`{route}`" for route in result.selected_routes) or "none"
        errors = ", ".join(f"`{code}`" for code in result.error_codes) or "none"
        lines.append(
            f"| `{result.case_id}` | {routes} | `{result.model_ref or 'none'}` | "
            f"`{result.skill_ref or 'none'}` | `{result.skill_body_ref or 'none'}` | "
            f"`{result.run_ref or 'none'}` | `{result.trace_ref or 'none'}` | "
            f"{result.trace_linkage} | {errors} |"
        )
    passed = sum(result.passed for result in results)
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Cases passed: {passed}/{len(results)}",
            f"- Skills fully passed: {sum(all(item.passed for item in pair.values()) and len(pair) == 2 for pair in by_skill.values())}/{len(by_skill)}",
            "- Trace linkage method: one exact-name `graph_run` trace whose metadata binds the case run, configured model, model class, skill, and skill body, queried through the Langfuse MCP tool mounted by Graph-OS.",
            "- Parent-ingestion proof: each exact trace resolves to exactly one `Trace` node written by Graph-OS parent mediation under verified `kg:write` authority.",
            "",
        ]
    )
    rendered = "\n".join(lines)
    _clean, privacy = PersistencePrivacyGuard().sanitize_text(rendered)
    if privacy.changed:
        raise RuntimeError("report_privacy_gate_failed")
    return rendered


def _validate_external_command_argv(argv: object) -> list[str]:
    """Resolve one bounded, non-shell external command without executing it."""

    if (
        not isinstance(argv, list)
        or not 1 <= len(argv) <= 32
        or not all(
            isinstance(item, str) and 0 < len(item) <= 4_096 and "\x00" not in item
            for item in argv
        )
    ):
        raise RuntimeError("evidence_command_reference_invalid")
    executable = Path(argv[0])
    try:
        original = executable.lstat()
        canonical = executable.resolve(strict=True)
        metadata = canonical.lstat()
    except OSError as exc:
        raise RuntimeError("evidence_command_reference_invalid") from exc
    if (
        not executable.is_absolute()
        or stat.S_ISLNK(original.st_mode)
        or not stat.S_ISREG(original.st_mode)
        or (original.st_dev, original.st_ino) != (metadata.st_dev, metadata.st_ino)
        or canonical.name.casefold() in _SHELL_EXECUTABLES
        or canonical.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or not os.access(canonical, os.X_OK)
    ):
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
    expected_ids = {case.case_id for case in cases}
    if (
        len(results) != 20
        or len(result_by_id) != 20
        or set(result_by_id) != expected_ids
    ):
        raise RuntimeError("runtime_case_set_not_exact")

    evidence_cases: list[dict[str, Any]] = []

    def controlled_ref(value: str) -> str | None:
        return (
            value if re.fullmatch(r"pref_[a-z_]+_[a-f0-9]{64}", value or "") else None
        )

    def controlled_trace_name(value: str) -> str | None:
        return (
            value
            if re.fullmatch(r"graph_run:pref_run_[a-f0-9]{64}", value or "")
            else None
        )

    for case in sorted(cases, key=lambda item: item.case_id):
        result = result_by_id[case.case_id]
        evidence_cases.append(
            {
                "caseId": case.case_id,
                "caseDigest": catalog["caseDigests"][case.case_id],
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
                "skillRef": controlled_ref(result.skill_ref),
                "skillBodyRef": controlled_ref(result.skill_body_ref),
                "runRef": controlled_ref(result.run_ref),
                "traceRef": controlled_ref(result.trace_ref),
                "langfuse": {
                    "lookupMethod": "exact-name",
                    "metadataOnly": True,
                    "traceName": controlled_trace_name(result.trace_name),
                    "matchCount": result.langfuse_match_count,
                    "linkage": result.trace_linkage,
                },
                "parentKnowledgeGraph": {
                    "readbackMethod": "exact-trace-name",
                    "matchCount": result.parent_kg_readback_count,
                },
                "errorCodes": sorted(result.error_codes),
            }
        )

    passed = sum(result.passed for result in results)
    skills = {result.skill for result in results}
    fully_passed = sum(
        len(items) == 2 and all(item.passed for item in items)
        for skill in skills
        for items in [[item for item in results if item.skill == skill]]
    )
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
            "skillCount": 10,
            "skillCatalogDigest": prebundled_skill_catalog_digest(SKILLS_ROOT),
            "testCaseCount": 20,
            "testCatalogDigest": catalog["testCatalogDigest"],
            "caseCatalogDigest": catalog["caseCatalogDigest"],
        },
        "cases": evidence_cases,
        "result": {
            "status": _PASS if passed == 20 and fully_passed == 10 else _FAIL,
            "passedCases": passed,
            "totalCases": 20,
            "fullyPassedSkills": fully_passed,
            "totalSkills": 10,
        },
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


async def run(args: argparse.Namespace) -> list[CaseResult]:
    defaults, all_cases = load_matrix()
    cases = [case for case in all_cases if args.mode in {"all", case.mode}]

    from agent_utilities.core.config import config, setting
    from agent_utilities.mcp.client_credentials import child_auth, child_auth_header
    from agent_utilities.mcp.toolset_factory import build_http_toolset

    graph_os_url = str(args.graph_os_url or config.mcp_url or "").strip()
    if not graph_os_url:
        raise RuntimeError("graph_os_url_unconfigured")
    capture_content = str(setting("LANGFUSE_CAPTURE_CONTENT", "false") or "false")
    if capture_content.strip().casefold() in {"1", "true", "yes", "on"}:
        raise RuntimeError("langfuse_content_capture_must_be_disabled")
    if not config.langfuse_kg_auto_ingest:
        raise RuntimeError("langfuse_parent_ingestion_required")

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
        await _ensure_tool(client, "graph_orchestrate", 30.0)
        await _ensure_tool(client, "graph_query", 30.0)
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
        for case in cases:
            if case.mode == "direct":
                from agent_utilities.knowledge_graph.core.session import use_session
                from agent_utilities.security.brain_context import use_actor

                validation_session = await _renew_direct_validation_session(
                    expected_authority=expected_authority,
                    minimum_ttl_seconds=_direct_case_minimum_authority_ttl(
                        case_timeout=args.case_timeout,
                        trace_timeout=float(defaults["trace_timeout_seconds"]),
                    ),
                )
                with (
                    use_actor(validation_session.actor),
                    use_session(validation_session),
                ):
                    item = await _run_direct_case(
                        case,
                        client=client,
                        langfuse_tool=langfuse_tool,
                        tenant_id=tenant_id,
                        case_timeout=args.case_timeout,
                        trace_timeout=float(defaults["trace_timeout_seconds"]),
                    )
            else:
                await _renew_delegated_validation_session(
                    expected_authority=expected_authority,
                    minimum_ttl_seconds=_direct_case_minimum_authority_ttl(
                        case_timeout=args.case_timeout,
                        trace_timeout=float(defaults["trace_timeout_seconds"]),
                    ),
                )
                item = await _run_delegated_case(
                    case,
                    client=client,
                    langfuse_tool=langfuse_tool,
                    tenant_id=tenant_id,
                    max_steps=int(defaults["max_steps"]),
                    token_budget=int(defaults["token_budget"]),
                    case_timeout=args.case_timeout,
                    trace_timeout=float(defaults["trace_timeout_seconds"]),
                )
            results.append(item)
            if _SYNC_CALL_POISONED.is_set():
                raise RuntimeError("blocking_sdk_worker_abandoned")
    return results


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
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
    args = parser.parse_args(argv)
    if not 1.0 <= args.case_timeout <= 600.0:
        parser.error("--case-timeout must be between 1 and 600 seconds")
    release_values = (
        args.release_id,
        args.release_specification_digest,
        args.promotion_evidence_digest,
        args.graph_os_digest,
        args.engine_digest,
        args.runtime_config_digest,
        args.runtime_profile_digest,
        args.model_registry_digest,
    )
    if args.mode == "all":
        if args.report is None or args.evidence is None or not all(release_values):
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
    elif args.evidence is not None or any(release_values):
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
