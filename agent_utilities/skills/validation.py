#!/usr/bin/env python3
"""Validate the consolidated agent-utilities pre-bundled skill suite."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from agent_utilities.mcp.skill_coverage import parse_graph_os_sidecar
from agent_utilities.skills import BUNDLED_SKILLS

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = Path(__file__).resolve().parent
FORWARD_MATRIX = SKILLS_ROOT / "runtime_validation.yaml"
EXPERT_PROMPT = PACKAGE_ROOT / "prompts" / "agent-utilities-expert.json"

EXPECTED_SKILLS = frozenset(BUNDLED_SKILLS)

_REQUIRED_WORKFLOW_TERMS: dict[str, frozenset[str]] = {
    "agent-utilities-deployment": frozenset(
        {"migration", "persisted-format", "upgrade"}
    ),
    "graph-engine-and-modalities": frozenset(
        {
            "sql",
            "sparql",
            "reasoning",
            "consensus",
            "tenancy",
            "rbac",
            "administration",
        }
    ),
    "graph-runtime-and-governance": frozenset({"troubleshoot"}),
}
_REQUIRED_WORKFLOW_ROUTES: dict[str, frozenset[str]] = {
    "graph-engine-and-modalities": frozenset(
        {
            "engine_admin",
            "engine_consensus",
            "engine_query",
            "engine_rbac",
            "engine_rdf",
            "engine_reasoning",
            "engine_tenants",
        }
    )
}

_PRIVATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "absolute filesystem path",
        re.compile(
            r"(?:^|[\s`'\"])/(?:home|Users|mnt|root|tmp|var|srv|opt|etc|workspace)/"
        ),
    ),
    ("Windows filesystem path", re.compile(r"\b[A-Za-z]:\\\\")),
    ("UNC filesystem path", re.compile(r"\\\\[^\s\\]+\\[^\s\\]+")),
    ("home-relative filesystem path", re.compile(r"~[/\\\\]")),
    ("literal IPv4 address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    (
        # Excludes the IANA-reserved documentation domains (RFC 2606/6761:
        # example.com/.net/.org/.test), the RDF ecosystem's "ex" placeholder
        # namespace convention, and an elided "…" placeholder segment (e.g.
        # UQL/SPARQL syntax references illustrating <http://ex/Class> or
        # <http://…/Class> IRIs) — none are a resolvable, leakable endpoint.
        "network endpoint",
        re.compile(
            r"\b(?:https?|wss?|tcp|udp|ssh|unix)://"
            r"(?!ex/|example\.(?:com|net|org|test)\b|…)",
            re.IGNORECASE,
        ),
    ),
    (
        "private endpoint suffix",
        re.compile(r"\.(?:arpa|internal|local|corp|lan)\b", re.IGNORECASE),
    ),
    (
        "email address",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    ),
    ("embedded secret reference", re.compile(r"\b(?:vault|secret)://", re.IGNORECASE)),
    ("private key material", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
)

_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DIGEST = re.compile(r"^sha256:(?!0{64}$)[a-f0-9]{64}$")
_ARCHITECTURE_SOURCE_REVISION = re.compile(r"^[a-f0-9]{40,64}$")
_ARCHITECTURE_COMPONENT_ID = re.compile(r"^[a-z][a-z0-9.-]{2,127}$")
_ARCHITECTURE_TARGET_REF = re.compile(r"^tgt:[a-z][a-z0-9-]*:v[0-9]+:[a-f0-9]{64}$")
_ARCHITECTURE_SOURCE_WORKSPACE_MANIFEST = "workspace.yml"
_ARCHITECTURE_SOURCE_REPOSITORY_ID = "agent-utilities"
_ARCHITECTURE_SOURCE_REPOSITORY_PATH = "agent-packages/agent-utilities"
_ARCHITECTURE_MANIFEST_PATH = "architecture/component-registry.yml"
_ARCHITECTURE_SOURCE_AUTHORITY = "owner_repository_manifest"
_ARCHITECTURE_AUTHORITY_STATE = "owner_manifest_authoritative"
_ARCHITECTURE_ACTIVE_STATUS = "active"
_ARCHITECTURE_MANIFEST_SCHEMA = "au-architecture-component-owner-manifest/v1"
_ARCHITECTURE_MANIFEST_GENERATOR = (
    "agent_utilities.skills.validation:architecture_candidate_from_owner_manifest"
)
_ARCHITECTURE_MANIFEST_CANONICALIZATION = "json-sort-keys-utf8"
_ARCHITECTURE_OWNER_MANIFEST = (
    PACKAGE_ROOT.parent / "architecture" / "component-registry.yml"
)
_ARCHITECTURE_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "owner_repository",
        "manifest_version",
        "generator",
        "canonical_source",
        "components",
        "integrity",
    }
)
_ARCHITECTURE_CANONICAL_SOURCE_FIELDS = frozenset(
    {
        "workspace_manifest",
        "repository_id",
        "repository_path",
        "manifest_path",
        "source_revision",
        "source_authority",
        "authority_state",
    }
)
_ARCHITECTURE_INTEGRITY_FIELDS = frozenset({"canonicalization", "source_digest"})
_ARCHITECTURE_OWNER_COMPONENT_FIELDS = frozenset(
    {
        "component_id",
        "component_kind",
        "capability_id",
        "parent_component_id",
        "parent_layer",
        "source_workspace_manifest",
        "source_repository_id",
        "source_repository_path",
        "source_manifest_path",
        "source_revision",
        "authority_signature",
        "behavioral_signature",
        "dependency_signature",
        "identity_policy_digest",
        "target_inventory_ref",
        "owned_source_roots",
        "public_contract_roots",
        "test_roots",
        "generated_roots",
        "shared_paths",
        "replacement_required",
        "replaced_component_ids",
        "status",
    }
)
_ARCHITECTURE_CANDIDATE_FIELDS = frozenset(
    {
        "component_id",
        "component_kind",
        "capability_id",
        "parent_component_id",
        "parent_layer",
        "source_workspace_manifest",
        "source_repository_id",
        "source_repository_path",
        "source_manifest_path",
        "source_revision",
        "source_digest",
        "authority_signature",
        "behavioral_signature",
        "dependency_signature",
        "identity_policy_digest",
        "target_inventory_ref",
        "owned_source_roots",
        "public_contract_roots",
        "test_roots",
        "generated_roots",
        "shared_paths",
        "replacement_required",
        "replaced_component_ids",
    }
)
_ARCHITECTURE_CANDIDATE_LIST_FIELDS = (
    "owned_source_roots",
    "public_contract_roots",
    "test_roots",
    "generated_roots",
    "shared_paths",
    "replaced_component_ids",
)
_AUXILIARY_DOC = re.compile(r"^(?:README|INSTALL|CHANGELOG)(?:\..*)?$", re.IGNORECASE)
_IMPERATIVE_STEP = re.compile(
    r"^(?:\d+\.\s+|-\s+)(?:"
    r"Add|Ask|Assign|Attach|Avoid|Bound|Capture|Change|Check|Choose|Compare|"
    r"Confirm|Create|Decide|Define|Distinguish|Execute|Extend|Generate|Identify|"
    r"Include|Inspect|Keep|Label|List|Map|Mark|Never|Parameterize|Persist|Prefer|"
    r"Preserve|Present|Preview|Put|Query|Read|Record|Reject|Report|Require|Resolve|"
    r"Re-run|Reuse|Run|Sample|Search|Select|Separate|Specify|Start|State|Stop|"
    r"Track|Treat|Update|Use|Validate|Verify"
    r")\b",
    re.MULTILINE,
)


def _relative(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _architecture_canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def architecture_owner_manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the digest of the owner declaration excluding its integrity block."""

    payload = {key: value for key, value in manifest.items() if key != "integrity"}
    return (
        "sha256:" + hashlib.sha256(_architecture_canonical_bytes(payload)).hexdigest()
    )


def _architecture_canonical_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = Path(value)
    return bool(
        not path.is_absolute()
        and value == path.as_posix()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _architecture_paths_overlap(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return bool(
        left == right
        or left.startswith(f"{right.rstrip('/')}/")
        or right.startswith(f"{left.rstrip('/')}/")
    )


def _architecture_revision_exists(revision: str) -> bool:
    if _ARCHITECTURE_SOURCE_REVISION.fullmatch(revision) is None:
        return False
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(PACKAGE_ROOT.parent),
                "cat-file",
                "-e",
                f"{revision}^{{commit}}",
            ],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _architecture_manifest_component_errors(
    component: Any, canonical_source: dict[str, Any], source_digest: str
) -> list[str]:
    if not isinstance(component, dict):
        return ["owner_manifest_component_invalid"]
    if set(component) != _ARCHITECTURE_OWNER_COMPONENT_FIELDS:
        return ["owner_manifest_component_fields_invalid"]
    expected = {
        "source_workspace_manifest": canonical_source["workspace_manifest"],
        "source_repository_id": canonical_source["repository_id"],
        "source_repository_path": canonical_source["repository_path"],
        "source_manifest_path": canonical_source["manifest_path"],
        "source_revision": canonical_source["source_revision"],
    }
    errors = [
        f"owner_manifest_{field}_disagreement"
        for field, value in expected.items()
        if component.get(field) != value
    ]
    candidate = dict(component)
    candidate.pop("status", None)
    candidate["source_digest"] = source_digest
    errors.extend(
        _validate_architecture_candidate(
            {"architecture_candidate": candidate}, "owner_manifest"
        )
    )
    if component.get("status") != _ARCHITECTURE_ACTIVE_STATUS:
        errors.append("owner_manifest_status_invalid")
    return errors


def _read_architecture_owner_manifest() -> dict[str, Any]:
    """Read the checked-in owner declaration without following a symlink."""

    if (
        not _ARCHITECTURE_OWNER_MANIFEST.is_file()
        or _ARCHITECTURE_OWNER_MANIFEST.is_symlink()
    ):
        raise ValueError("owner_manifest_missing")
    try:
        manifest = yaml.safe_load(
            _ARCHITECTURE_OWNER_MANIFEST.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("owner_manifest_unreadable") from exc
    if not isinstance(manifest, dict):
        raise ValueError("owner_manifest_fields_invalid")
    return manifest


def _validate_architecture_manifest_header(manifest: dict[str, Any]) -> None:
    """Require the owner manifest's exact schema and generator authority."""

    if set(manifest) != _ARCHITECTURE_MANIFEST_FIELDS:
        raise ValueError("owner_manifest_fields_invalid")
    expected = {
        "schema": _ARCHITECTURE_MANIFEST_SCHEMA,
        "owner_repository": _ARCHITECTURE_SOURCE_REPOSITORY_ID,
        "manifest_version": 1,
        "generator": _ARCHITECTURE_MANIFEST_GENERATOR,
    }
    errors = {
        "schema": "owner_manifest_schema_invalid",
        "owner_repository": "owner_manifest_repository_invalid",
        "manifest_version": "owner_manifest_version_invalid",
        "generator": "owner_manifest_generator_invalid",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(errors[field])


def _validate_architecture_canonical_source(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Require the source repository, manifest path, and real Git revision."""

    source = manifest.get("canonical_source")
    if (
        not isinstance(source, dict)
        or set(source) != _ARCHITECTURE_CANONICAL_SOURCE_FIELDS
    ):
        raise ValueError("owner_manifest_source_fields_invalid")
    expected = {
        "workspace_manifest": _ARCHITECTURE_SOURCE_WORKSPACE_MANIFEST,
        "repository_id": _ARCHITECTURE_SOURCE_REPOSITORY_ID,
        "repository_path": _ARCHITECTURE_SOURCE_REPOSITORY_PATH,
        "manifest_path": _ARCHITECTURE_MANIFEST_PATH,
        "source_authority": _ARCHITECTURE_SOURCE_AUTHORITY,
        "authority_state": _ARCHITECTURE_AUTHORITY_STATE,
    }
    errors = {
        "workspace_manifest": "owner_manifest_workspace_manifest_invalid",
        "repository_id": "owner_manifest_source_repository_invalid",
        "repository_path": "owner_manifest_source_repository_path_invalid",
        "manifest_path": "owner_manifest_source_manifest_path_invalid",
        "source_authority": "owner_manifest_source_authority_invalid",
        "authority_state": "owner_manifest_authority_state_invalid",
    }
    for field, value in expected.items():
        if source.get(field) != value:
            raise ValueError(errors[field])
    revision = source.get("source_revision")
    if not isinstance(revision, str) or not _architecture_revision_exists(revision):
        raise ValueError("owner_manifest_source_revision_invalid")
    return source


def _validate_architecture_manifest_integrity(manifest: dict[str, Any]) -> str:
    """Verify the SHA-256 binding over the canonical owner declaration."""

    integrity = manifest.get("integrity")
    if (
        not isinstance(integrity, dict)
        or set(integrity) != _ARCHITECTURE_INTEGRITY_FIELDS
    ):
        raise ValueError("owner_manifest_integrity_fields_invalid")
    if integrity.get("canonicalization") != _ARCHITECTURE_MANIFEST_CANONICALIZATION:
        raise ValueError("owner_manifest_canonicalization_invalid")
    source_digest = integrity.get("source_digest")
    if not isinstance(source_digest, str) or _DIGEST.fullmatch(source_digest) is None:
        raise ValueError("owner_manifest_source_digest_invalid")
    if source_digest != architecture_owner_manifest_digest(manifest):
        raise ValueError("owner_manifest_source_digest_mismatch")
    return source_digest


def _validate_architecture_manifest_components(
    manifest: dict[str, Any], canonical_source: dict[str, Any], source_digest: str
) -> None:
    """Validate each owner component and reject duplicate authorities."""

    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("owner_manifest_components_invalid")
    component_errors: list[str] = []
    component_ids: list[str] = []
    capability_ids: list[str] = []
    for component in components:
        component_errors.extend(
            _architecture_manifest_component_errors(
                component, canonical_source, source_digest
            )
        )
        if isinstance(component, dict):
            component_ids.append(str(component.get("component_id") or ""))
            capability_ids.append(str(component.get("capability_id") or ""))
    if len(component_ids) != len(set(component_ids)):
        component_errors.append("owner_manifest_component_duplicate")
    if len(capability_ids) != len(set(capability_ids)):
        component_errors.append("owner_manifest_capability_duplicate")
    if component_errors:
        raise ValueError(component_errors[0])


def load_architecture_owner_manifest() -> dict[str, Any]:
    """Load and verify the canonical owner declaration used by RF-021."""

    return _validate_architecture_owner_manifest_data(
        _read_architecture_owner_manifest()
    )


def _validate_architecture_owner_manifest_data(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify an in-memory owner declaration before it can generate evidence."""

    _validate_architecture_manifest_header(manifest)
    canonical_source = _validate_architecture_canonical_source(manifest)
    source_digest = _validate_architecture_manifest_integrity(manifest)
    _validate_architecture_manifest_components(
        manifest, canonical_source, source_digest
    )
    return manifest


def architecture_candidate_from_owner_manifest(
    manifest: dict[str, Any] | None = None, *, component_id: str | None = None
) -> dict[str, Any]:
    """Generate one runtime candidate projection from the canonical owner manifest."""

    owner_manifest = (
        load_architecture_owner_manifest()
        if manifest is None
        else _validate_architecture_owner_manifest_data(manifest)
    )
    components = owner_manifest["components"]
    selected = next(
        (
            component
            for component in components
            if isinstance(component, dict)
            and (component_id is None or component.get("component_id") == component_id)
        ),
        None,
    )
    if not isinstance(selected, dict):
        raise ValueError("owner_manifest_component_missing")
    candidate = dict(selected)
    candidate.pop("status", None)
    candidate["source_digest"] = owner_manifest["integrity"]["source_digest"]
    return candidate


def _frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return {}, text
    data = yaml.safe_load(match.group(1)) or {}
    return (data if isinstance(data, dict) else {}), match.group(2)


def _validate_skill_frontmatter(name: str, frontmatter: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(frontmatter) != {"name", "description", "skill_type"}:
        errors.append(
            f"{name}: SKILL.md frontmatter must contain only name, description, "
            "and skill_type"
        )
    if frontmatter.get("name") != name:
        errors.append(f"{name}: frontmatter name must match directory")
    if frontmatter.get("skill_type") != "skill":
        errors.append(f"{name}: frontmatter skill_type must be 'skill'")
    if not str(frontmatter.get("description") or "").strip():
        errors.append(f"{name}: description is empty")
    return errors


def _validate_skill_body(name: str, skill_md: Path, body: str) -> list[str]:
    errors: list[str] = []
    if len(skill_md.read_text(encoding="utf-8").splitlines()) >= 500:
        errors.append(f"{name}: SKILL.md must remain under 500 lines")
    if "TODO" in body:
        errors.append(f"{name}: unresolved TODO in SKILL.md")
    if "## Workflow" not in body:
        errors.append(f"{name}: SKILL.md must contain a Workflow section")
    if len(_IMPERATIVE_STEP.findall(body)) < 3:
        errors.append(f"{name}: body must contain at least three imperative steps")
    if not re.search(r"\beconom(?:y|ical)\b", body, re.IGNORECASE):
        errors.append(f"{name}: missing economy-model guidance")
    if "direct" not in body.lower() or "delegat" not in body.lower():
        errors.append(f"{name}: must explain direct and delegated execution")
    return errors


def _validate_skill_workflow_terms(name: str, skill_md: Path) -> list[str]:
    lowered = skill_md.read_text(encoding="utf-8").lower()
    missing_terms = sorted(
        term
        for term in _REQUIRED_WORKFLOW_TERMS.get(name, frozenset())
        if term not in lowered
    )
    if missing_terms:
        return [f"{name}: missing retained workflow coverage terms {missing_terms}"]
    return []


def _validate_skill_openai_interface(name: str, interface: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {"display_name", "short_description", "default_prompt"}
    if set(interface) != required:
        errors.append(f"{name}: OpenAI interface keys must be {sorted(required)}")
    short = str(interface.get("short_description") or "")
    if not 25 <= len(short) <= 64:
        errors.append(f"{name}: short_description must be 25-64 characters")
    if f"${name}" not in str(interface.get("default_prompt") or ""):
        errors.append(f"{name}: default_prompt must mention ${name}")
    return errors


def _validate_skill_openai_sidecar(name: str, openai: Path) -> list[str]:
    if not openai.is_file():
        return [f"{name}: missing agents/openai.yaml"]
    data = yaml.safe_load(openai.read_text(encoding="utf-8")) or {}
    interface = data.get("interface") if isinstance(data, dict) else None
    if set(data) != {"interface"} or not isinstance(interface, dict):
        return [f"{name}: OpenAI sidecar must contain only interface"]
    return _validate_skill_openai_interface(name, interface)


def _validate_skill_graph_os_sidecar(name: str, graph_os: Path) -> list[str]:
    if not graph_os.is_file():
        return [f"{name}: missing agents/graph-os.yaml"]
    meta = parse_graph_os_sidecar(graph_os, skill_name=name)
    errors = [f"{name}: {error}" for error in meta.errors]
    missing_routes = sorted(
        _REQUIRED_WORKFLOW_ROUTES.get(name, frozenset()) - set(meta.wraps)
    )
    if missing_routes:
        errors.append(f"{name}: missing retained workflow routes {missing_routes}")
    return errors


def _validate_skill_files(skill_dir: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        if _AUXILIARY_DOC.fullmatch(path.name):
            errors.append(
                f"{_relative(path)}: auxiliary skill documentation is forbidden"
            )
        text = path.read_text(encoding="utf-8")
        for label, pattern in _PRIVATE_PATTERNS:
            if pattern.search(text):
                errors.append(f"{_relative(path)}: contains {label}")
    return errors


def _validate_skill(skill_dir: Path) -> list[str]:
    errors: list[str] = []
    name = skill_dir.name
    skill_md = skill_dir / "SKILL.md"
    openai = skill_dir / "agents" / "openai.yaml"
    graph_os = skill_dir / "agents" / "graph-os.yaml"

    if not _SKILL_NAME.fullmatch(name):
        errors.append(f"{name}: directory name must use lowercase hyphenation")
    if not skill_md.is_file():
        return [f"{name}: missing SKILL.md"]
    frontmatter, body = _frontmatter(skill_md)
    errors.extend(_validate_skill_frontmatter(name, frontmatter))
    errors.extend(_validate_skill_body(name, skill_md, body))
    errors.extend(_validate_skill_workflow_terms(name, skill_md))
    errors.extend(_validate_skill_openai_sidecar(name, openai))
    errors.extend(_validate_skill_graph_os_sidecar(name, graph_os))
    errors.extend(_validate_skill_files(skill_dir))
    return errors


def _forward_matrix_domain_wraps() -> tuple[dict[str, set[str]], set[str]]:
    domain_wraps: dict[str, set[str]] = {}
    for skill in EXPECTED_SKILLS:
        meta = parse_graph_os_sidecar(
            SKILLS_ROOT / skill / "agents" / "graph-os.yaml", skill_name=skill
        )
        if meta.tier == "domain" and not meta.errors:
            domain_wraps[skill] = set(meta.wraps)
    all_domain_wraps = set().union(*domain_wraps.values()) if domain_wraps else set()
    return domain_wraps, all_domain_wraps


def _validate_forward_matrix_defaults(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != 2:
        errors.append("forward matrix: schema_version must be 2")
    defaults = data.get("runtime_defaults")
    if not isinstance(defaults, dict):
        errors.append("forward matrix: runtime_defaults must be a mapping")
        return errors
    max_steps = defaults.get("max_steps")
    token_budget = defaults.get("token_budget")
    trace_timeout = defaults.get("trace_timeout_seconds")
    if not isinstance(max_steps, int) or not 1 <= max_steps <= 8:
        errors.append("forward matrix: max_steps must be between 1 and 8")
    if not isinstance(token_budget, int) or not 256 <= token_budget <= 16384:
        errors.append("forward matrix: token_budget must be between 256 and 16384")
    if not isinstance(trace_timeout, int) or not 1 <= trace_timeout <= 60:
        errors.append("forward matrix: trace_timeout_seconds must be between 1 and 60")
    if defaults.get("sequential") is not True:
        errors.append("forward matrix: runtime validation must be sequential")
    return errors


def _validate_forward_matrix_case_domain_routes(
    case_id: str, mode: str, route_set: set[str], owned: set[str]
) -> list[str]:
    errors: list[str] = []
    invalid = route_set - owned - {"graph_orchestrate"}
    if invalid:
        errors.append(
            f"{case_id}: routes not owned by the domain skill: {sorted(invalid)}"
        )
    if mode == "direct" and "graph_orchestrate" in route_set:
        errors.append(f"{case_id}: a direct case cannot use graph_orchestrate")
    if mode == "delegated" and "graph_orchestrate" not in route_set:
        errors.append(f"{case_id}: a delegated domain case must use graph_orchestrate")
    return errors


def _validate_forward_matrix_case_routes(
    case_id: str,
    mode: str,
    skill: str,
    routes: Any,
    domain_wraps: dict[str, set[str]],
) -> list[str]:
    if not isinstance(routes, list) or not routes:
        return [f"{case_id}: expected_routes must be non-empty"]
    if skill in domain_wraps:
        return _validate_forward_matrix_case_domain_routes(
            case_id, mode, set(routes), domain_wraps[skill]
        )
    if mode == "delegated" and (
        not isinstance(routes, list) or "graph_orchestrate" not in routes
    ):
        return [f"{case_id}: delegated cases must use graph_orchestrate"]
    return []


def _validate_forward_matrix_case_delegated_tools(
    case_id: str, allowed_tools: list[Any], all_domain_wraps: set[str]
) -> list[str]:
    errors: list[str] = []
    if not allowed_tools or not all(
        isinstance(tool, str) and tool for tool in allowed_tools
    ):
        errors.append(f"{case_id}: delegated allowed_tools must be non-empty")
        return errors
    if len(allowed_tools) != len(set(allowed_tools)):
        errors.append(f"{case_id}: allowed_tools must not contain duplicates")
    if allowed_tools != sorted(allowed_tools):
        errors.append(f"{case_id}: allowed_tools must be sorted")
    unknown_tools = set(allowed_tools) - all_domain_wraps
    if unknown_tools:
        errors.append(
            f"{case_id}: allowed_tools contain unknown Graph-OS verbs: "
            f"{sorted(unknown_tools)}"
        )
    if "graph_orchestrate" in allowed_tools:
        errors.append(f"{case_id}: delegated child cannot recursively orchestrate")
    return errors


def _validate_forward_matrix_case_tools(
    case_id: str, mode: str, allowed_tools: Any, all_domain_wraps: set[str]
) -> list[str]:
    if not isinstance(allowed_tools, list):
        return [f"{case_id}: allowed_tools must be a list"]
    if mode == "direct":
        if allowed_tools:
            return [f"{case_id}: direct semantic cases cannot receive tools"]
        return []
    return _validate_forward_matrix_case_delegated_tools(
        case_id, allowed_tools, all_domain_wraps
    )


def _validate_forward_matrix_case_identity(
    case_id: str,
    skill: str,
    mode: str,
    ids: set[str],
    seen: set[tuple[str, str]],
) -> list[str]:
    errors: list[str] = []
    if not case_id or case_id in ids:
        errors.append(f"forward matrix: duplicate or empty id {case_id!r}")
    ids.add(case_id)
    if skill not in EXPECTED_SKILLS:
        errors.append(f"{case_id}: unknown skill {skill!r}")
    if mode not in {"direct", "delegated"}:
        errors.append(f"{case_id}: mode must be direct or delegated")
    seen.add((skill, mode))
    return errors


def _validate_forward_matrix_case_content(
    case: dict[str, Any], case_id: str, skill: str
) -> list[str]:
    errors: list[str] = []
    if "skill_path" in case:
        errors.append(f"{case_id}: filesystem skill_path is forbidden")
    task = str(case.get("task") or "")
    if f"${skill}" not in task or f"skill://{skill}" not in task:
        errors.append(
            f"{case_id}: task must identify the skill by neutral skill:// reference"
        )
    if case.get("model_class") not in {"economy", "standard"}:
        errors.append(f"{case_id}: unsupported model_class")
    if case.get("read_only") is not True:
        errors.append(f"{case_id}: validation cases must be read-only")
    return errors


def _validate_architecture_candidate(case: dict[str, Any], case_id: str) -> list[str]:
    """Validate the closed candidate identity required by development cases."""

    candidate = case.get("architecture_candidate")
    if not isinstance(candidate, dict):
        return [f"{case_id}: architecture_candidate must be a mapping"]
    errors: list[str] = []
    if set(candidate) != _ARCHITECTURE_CANDIDATE_FIELDS:
        errors.append(f"{case_id}: architecture_candidate fields are not exact")
    errors.extend(_validate_architecture_candidate_identity(candidate, case_id))
    errors.extend(_validate_architecture_candidate_digests(candidate, case_id))
    errors.extend(_validate_architecture_candidate_lists(candidate, case_id))
    return errors


def _validate_architecture_candidate_identity(
    candidate: dict[str, Any], case_id: str
) -> list[str]:
    """Validate canonical source and target identity fields."""

    errors = _validate_architecture_candidate_kind(candidate, case_id)
    errors.extend(_validate_architecture_candidate_revision(candidate, case_id))
    errors.extend(_validate_architecture_candidate_repository(candidate, case_id))
    errors.extend(_validate_architecture_candidate_target(candidate, case_id))
    return errors


def _validate_architecture_candidate_kind(
    candidate: dict[str, Any], case_id: str
) -> list[str]:
    """Require the implementation kind and owner-manifest identity."""

    errors: list[str] = []
    if candidate.get("component_kind") != "implementation_component":
        errors.append(f"{case_id}: candidate must be an implementation_component")
    if (
        candidate.get("source_workspace_manifest")
        != _ARCHITECTURE_SOURCE_WORKSPACE_MANIFEST
    ):
        errors.append(f"{case_id}: candidate workspace manifest is not canonical")
    if candidate.get("source_manifest_path") != _ARCHITECTURE_MANIFEST_PATH:
        errors.append(f"{case_id}: candidate owner manifest path is not canonical")

    return errors


def _validate_architecture_candidate_revision(
    candidate: dict[str, Any], case_id: str
) -> list[str]:
    """Require a real Git commit revision, not a sentinel or arbitrary hash."""

    source_revision = str(candidate.get("source_revision") or "")
    if _ARCHITECTURE_SOURCE_REVISION.fullmatch(source_revision) is None:
        return [f"{case_id}: candidate source_revision is not a Git revision"]
    if not _architecture_revision_exists(source_revision):
        return [f"{case_id}: candidate source_revision does not exist"]
    return []


def _validate_architecture_candidate_repository(
    candidate: dict[str, Any], case_id: str
) -> list[str]:
    """Require the canonical AU repository identity and relative path."""

    errors: list[str] = []
    if candidate.get("source_repository_id") != _ARCHITECTURE_SOURCE_REPOSITORY_ID:
        errors.append(f"{case_id}: candidate source repository is not canonical")
    source_path = candidate.get("source_repository_path")
    if not _architecture_canonical_relative_path(source_path):
        errors.append(f"{case_id}: candidate source_repository_path is not relative")
    elif source_path != _ARCHITECTURE_SOURCE_REPOSITORY_PATH:
        errors.append(f"{case_id}: candidate source repository path is not canonical")
    return errors


def _validate_architecture_candidate_target(
    candidate: dict[str, Any], case_id: str
) -> list[str]:
    """Require a canonical content-addressed target-inventory reference."""

    if not _ARCHITECTURE_TARGET_REF.fullmatch(
        str(candidate.get("target_inventory_ref") or "")
    ):
        return [f"{case_id}: candidate target_inventory_ref is not canonical"]
    return []


def _validate_architecture_candidate_digests(
    candidate: dict[str, Any], case_id: str
) -> list[str]:
    """Validate the externally supplied candidate content identities."""

    errors: list[str] = []
    for field in (
        "source_digest",
        "authority_signature",
        "behavioral_signature",
        "dependency_signature",
        "identity_policy_digest",
    ):
        if _DIGEST.fullmatch(str(candidate.get(field) or "")) is None:
            errors.append(f"{case_id}: candidate {field} is not a digest")
    return errors


def _validate_architecture_candidate_lists(
    candidate: dict[str, Any], case_id: str
) -> list[str]:
    """Validate collection-shaped fields before runtime model parsing."""

    errors: list[str] = []
    for field in _ARCHITECTURE_CANDIDATE_LIST_FIELDS:
        if not isinstance(candidate.get(field), list):
            errors.append(f"{case_id}: candidate {field} must be a list")
    root_groups = _architecture_candidate_root_groups(candidate)
    errors.extend(_validate_architecture_candidate_root_groups(root_groups, case_id))
    errors.extend(_validate_architecture_candidate_shared_shapes(candidate, case_id))
    return errors


def _architecture_candidate_root_groups(
    candidate: dict[str, Any],
) -> list[tuple[Any, ...]]:
    """Return only list-shaped owner root groups for safe structural checks."""

    return [
        tuple(candidate[field])
        for field in (
            "owned_source_roots",
            "public_contract_roots",
            "test_roots",
            "generated_roots",
        )
        if isinstance(candidate.get(field), list)
    ]


def _validate_architecture_candidate_root_groups(
    root_groups: list[tuple[Any, ...]], case_id: str
) -> list[str]:
    """Reject malformed, duplicate, or cross-role owner roots."""

    errors: list[str] = []
    for group in root_groups:
        errors.extend(_validate_architecture_candidate_root_group(group, case_id))
    errors.extend(
        _validate_architecture_candidate_cross_role_roots(root_groups, case_id)
    )
    return errors


def _validate_architecture_candidate_root_group(
    group: tuple[Any, ...], case_id: str
) -> list[str]:
    """Validate one owner-root role without hashing malformed values."""

    errors: list[str] = []
    if any(not _architecture_canonical_relative_path(root) for root in group):
        errors.append(f"{case_id}: candidate owner root is not relative")
    if all(isinstance(root, str) for root in group) and len(group) != len(set(group)):
        errors.append(f"{case_id}: candidate owner root is duplicated")
    return errors


def _validate_architecture_candidate_cross_role_roots(
    root_groups: list[tuple[Any, ...]], case_id: str
) -> list[str]:
    """Reject identical or nested roots assigned to different roles."""

    errors: list[str] = []
    for index, left_group in enumerate(root_groups):
        for right_group in root_groups[index + 1 :]:
            if any(
                _architecture_paths_overlap(left, right)
                for left in left_group
                for right in right_group
            ):
                errors.append(f"{case_id}: candidate owner roots overlap across roles")
                break
    return errors


def _validate_architecture_candidate_shared_shapes(
    candidate: dict[str, Any], case_id: str
) -> list[str]:
    """Reject malformed shared-path owner identifiers without hashing values."""

    errors: list[str] = []
    shared_paths = candidate.get("shared_paths")
    if isinstance(shared_paths, list):
        for shared in shared_paths:
            if not isinstance(shared, dict):
                errors.append(f"{case_id}: shared path must be a mapping")
                continue
            owners = shared.get("owner_component_ids")
            if not isinstance(owners, list) or any(
                not isinstance(owner, str)
                or _ARCHITECTURE_COMPONENT_ID.fullmatch(owner) is None
                for owner in owners
            ):
                errors.append(f"{case_id}: shared-path owner ID is invalid")
    return errors


def _validate_architecture_owner_manifest(data: dict[str, Any]) -> list[str]:
    """Require the matrix candidate to come from the canonical owner declaration."""

    try:
        manifest = load_architecture_owner_manifest()
        architecture_candidate_from_owner_manifest(manifest)
    except ValueError as exc:
        return [f"architecture owner manifest: {exc}"]
    cases = data.get("cases")
    if not isinstance(cases, list):
        return []
    errors: list[str] = []
    for case in cases:
        if (
            not isinstance(case, dict)
            or case.get("skill") != "agent-utilities-development"
        ):
            continue
        candidate = case.get("architecture_candidate")
        if not isinstance(candidate, dict):
            continue
        try:
            expected = architecture_candidate_from_owner_manifest(
                manifest, component_id=str(candidate.get("component_id") or "")
            )
        except ValueError:
            errors.append(
                f"{case.get('id')}: architecture_candidate component is not declared"
            )
            continue
        if candidate != expected:
            errors.append(
                f"{case.get('id')}: architecture_candidate is not bound to owner manifest"
            )
    return errors


def _validate_forward_matrix_case(
    case: Any,
    ids: set[str],
    seen: set[tuple[str, str]],
    domain_wraps: dict[str, set[str]],
    all_domain_wraps: set[str],
) -> list[str]:
    if not isinstance(case, dict):
        return ["forward matrix: every case must be a mapping"]
    case_id = str(case.get("id") or "")
    skill = str(case.get("skill") or "")
    mode = str(case.get("mode") or "")
    errors = _validate_forward_matrix_case_identity(case_id, skill, mode, ids, seen)
    errors.extend(_validate_forward_matrix_case_content(case, case_id, skill))
    errors.extend(_validate_architecture_case_assignment(case, case_id, skill))
    errors.extend(
        _validate_forward_matrix_case_routes(
            case_id, mode, skill, case.get("expected_routes"), domain_wraps
        )
    )
    errors.extend(
        _validate_forward_matrix_case_tools(
            case_id, mode, case.get("allowed_tools"), all_domain_wraps
        )
    )
    return errors


def _validate_architecture_case_assignment(
    case: dict[str, Any], case_id: str, skill: str
) -> list[str]:
    """Require the candidate only on development-skill matrix cases."""

    if skill == "agent-utilities-development":
        return _validate_architecture_candidate(case, case_id)
    if "architecture_candidate" in case:
        return [f"{case_id}: architecture_candidate is development-only"]
    return []


def _validate_forward_matrix_privacy(
    data: dict[str, Any], raw_matrix: str
) -> list[str]:
    errors: list[str] = []
    privacy = data.get("privacy_assertions") or {}
    forbidden = set(privacy.get("forbid_persisted") or [])
    required = {
        "credential",
        "internal_endpoint",
        "local_filesystem_path",
        "personal_name",
        "raw_model_output",
        "raw_trace_id",
    }
    if forbidden != required:
        errors.append("forward matrix: privacy assertions are incomplete")
    if privacy.get("require_reference_scheme") != "skill://":
        errors.append("forward matrix: skill:// references must be required")
    if privacy.get("require_synthetic_inputs") is not True:
        errors.append("forward matrix: synthetic inputs must be required")
    if privacy.get("require_metadata_only_observability") is not True:
        errors.append("forward matrix: metadata-only observability must be required")
    for label, pattern in _PRIVATE_PATTERNS:
        if pattern.search(raw_matrix):
            errors.append(f"forward matrix: contains {label}")
    return errors


def _validate_forward_matrix() -> list[str]:
    errors: list[str] = []
    domain_wraps, all_domain_wraps = _forward_matrix_domain_wraps()
    data = yaml.safe_load(FORWARD_MATRIX.read_text(encoding="utf-8")) or {}
    errors.extend(_validate_forward_matrix_defaults(data))
    errors.extend(_validate_architecture_owner_manifest(data))
    cases = data.get("cases")
    if not isinstance(cases, list):
        return [*errors, "forward matrix: cases must be a list"]

    seen: set[tuple[str, str]] = set()
    ids: set[str] = set()
    for case in cases:
        errors.extend(
            _validate_forward_matrix_case(
                case, ids, seen, domain_wraps, all_domain_wraps
            )
        )

    expected_pairs = {
        (skill, mode) for skill in EXPECTED_SKILLS for mode in ("direct", "delegated")
    }
    if seen != expected_pairs:
        errors.append(
            "forward matrix: each skill needs one direct and one delegated case"
        )

    raw_matrix = FORWARD_MATRIX.read_text(encoding="utf-8")
    errors.extend(_validate_forward_matrix_privacy(data, raw_matrix))
    return errors


def _validate_expert_prompt() -> list[str]:
    """Keep persisted expert metadata aligned with the public workflow surface."""
    errors: list[str] = []
    raw = EXPERT_PROMPT.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [f"expert prompt: invalid JSON ({type(exc).__name__})"]
    if set(data.get("skills") or []) != EXPECTED_SKILLS:
        errors.append("expert prompt: skills must match the retained taxonomy")
    directive = str((data.get("instructions") or {}).get("core_directive") or "")
    if "skill://<skill-name>" not in directive:
        errors.append("expert prompt: neutral skill:// persistence rule is missing")
    for label, pattern in _PRIVATE_PATTERNS:
        if pattern.search(raw):
            errors.append(f"expert prompt: contains {label}")
    return errors


def _validate_skill_inventory() -> tuple[set[str], list[str]]:
    actual = {
        path.parent.name for path in SKILLS_ROOT.glob("*/SKILL.md") if path.is_file()
    }
    errors: list[str] = []
    if len(EXPECTED_SKILLS) != 13:
        errors.append("canonical taxonomy must contain exactly 13 workflow skills")
    if actual != EXPECTED_SKILLS:
        errors.append(
            "skill inventory mismatch: "
            f"missing={sorted(EXPECTED_SKILLS - actual)} "
            f"unexpected={sorted(actual - EXPECTED_SKILLS)}"
        )
    return actual, errors


def _validate_skill_tree(actual: set[str]) -> list[str]:
    errors: list[str] = []
    # Scoped to the canonical 13-skill subtree only: a SKILL.md nested under one
    # of EXPECTED_SKILLS would be a real violation (that skill must be a flat
    # <name>/SKILL.md directory), but agent_utilities/skills/ also legitimately
    # hosts other, differently-shaped content outside this taxonomy — the
    # agent-os-genesis workflow skill (skills/workflows/) and the agent-utilities
    # skill-graph package (skills/skill_graphs/) — which this validator does not
    # own and must not flag.
    nested = [
        path
        for path in SKILLS_ROOT.rglob("SKILL.md")
        if path.parent.parent != SKILLS_ROOT
        and path.relative_to(SKILLS_ROOT).parts[0] in EXPECTED_SKILLS
    ]
    if nested:
        errors.append(
            "nested bundled skills are not allowed: "
            + ", ".join(_relative(path) for path in nested)
        )
    for name in sorted(actual):
        errors.extend(_validate_skill(SKILLS_ROOT / name))
    return errors


def validate() -> list[str]:
    """Return deterministic validation errors for the retained suite."""
    actual, errors = _validate_skill_inventory()
    errors.extend(_validate_skill_tree(actual))
    errors.extend(_validate_forward_matrix())
    errors.extend(_validate_expert_prompt())
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Pre-bundled skill validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(
        f"Pre-bundled skill validation OK — {len(EXPECTED_SKILLS)} skills, "
        f"{2 * len(EXPECTED_SKILLS)} synthetic forward cases."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
