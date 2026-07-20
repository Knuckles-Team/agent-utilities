#!/usr/bin/env python3
"""Validate the consolidated agent-utilities pre-bundled skill suite."""

from __future__ import annotations

import json
import re
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
        "network endpoint",
        re.compile(r"\b(?:https?|wss?|tcp|udp|ssh|unix)://", re.IGNORECASE),
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


def _frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return {}, text
    data = yaml.safe_load(match.group(1)) or {}
    return (data if isinstance(data, dict) else {}), match.group(2)


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
    if set(frontmatter) != {"name", "description"}:
        errors.append(
            f"{name}: SKILL.md frontmatter must contain only name and description"
        )
    if frontmatter.get("name") != name:
        errors.append(f"{name}: frontmatter name must match directory")
    if not str(frontmatter.get("description") or "").strip():
        errors.append(f"{name}: description is empty")
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
    lowered = skill_md.read_text(encoding="utf-8").lower()
    missing_terms = sorted(
        term
        for term in _REQUIRED_WORKFLOW_TERMS.get(name, frozenset())
        if term not in lowered
    )
    if missing_terms:
        errors.append(
            f"{name}: missing retained workflow coverage terms {missing_terms}"
        )

    if not openai.is_file():
        errors.append(f"{name}: missing agents/openai.yaml")
    else:
        data = yaml.safe_load(openai.read_text(encoding="utf-8")) or {}
        interface = data.get("interface") if isinstance(data, dict) else None
        if set(data) != {"interface"} or not isinstance(interface, dict):
            errors.append(f"{name}: OpenAI sidecar must contain only interface")
        else:
            required = {"display_name", "short_description", "default_prompt"}
            if set(interface) != required:
                errors.append(
                    f"{name}: OpenAI interface keys must be {sorted(required)}"
                )
            short = str(interface.get("short_description") or "")
            if not 25 <= len(short) <= 64:
                errors.append(f"{name}: short_description must be 25-64 characters")
            if f"${name}" not in str(interface.get("default_prompt") or ""):
                errors.append(f"{name}: default_prompt must mention ${name}")

    if not graph_os.is_file():
        errors.append(f"{name}: missing agents/graph-os.yaml")
    else:
        meta = parse_graph_os_sidecar(graph_os, skill_name=name)
        errors.extend(f"{name}: {error}" for error in meta.errors)
        missing_routes = sorted(
            _REQUIRED_WORKFLOW_ROUTES.get(name, frozenset()) - set(meta.wraps)
        )
        if missing_routes:
            errors.append(f"{name}: missing retained workflow routes {missing_routes}")

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


def _validate_forward_matrix() -> list[str]:
    errors: list[str] = []
    domain_wraps: dict[str, set[str]] = {}
    for skill in EXPECTED_SKILLS:
        meta = parse_graph_os_sidecar(
            SKILLS_ROOT / skill / "agents" / "graph-os.yaml", skill_name=skill
        )
        if meta.tier == "domain" and not meta.errors:
            domain_wraps[skill] = set(meta.wraps)
    all_domain_wraps = set().union(*domain_wraps.values()) if domain_wraps else set()
    data = yaml.safe_load(FORWARD_MATRIX.read_text(encoding="utf-8")) or {}
    if data.get("schema_version") != 2:
        errors.append("forward matrix: schema_version must be 2")
    defaults = data.get("runtime_defaults")
    if not isinstance(defaults, dict):
        errors.append("forward matrix: runtime_defaults must be a mapping")
    else:
        max_steps = defaults.get("max_steps")
        token_budget = defaults.get("token_budget")
        trace_timeout = defaults.get("trace_timeout_seconds")
        if not isinstance(max_steps, int) or not 1 <= max_steps <= 8:
            errors.append("forward matrix: max_steps must be between 1 and 8")
        if not isinstance(token_budget, int) or not 256 <= token_budget <= 16384:
            errors.append("forward matrix: token_budget must be between 256 and 16384")
        if not isinstance(trace_timeout, int) or not 1 <= trace_timeout <= 60:
            errors.append(
                "forward matrix: trace_timeout_seconds must be between 1 and 60"
            )
        if defaults.get("sequential") is not True:
            errors.append("forward matrix: runtime validation must be sequential")
    cases = data.get("cases")
    if not isinstance(cases, list):
        return [*errors, "forward matrix: cases must be a list"]

    seen: set[tuple[str, str]] = set()
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            errors.append("forward matrix: every case must be a mapping")
            continue
        case_id = str(case.get("id") or "")
        skill = str(case.get("skill") or "")
        mode = str(case.get("mode") or "")
        if not case_id or case_id in ids:
            errors.append(f"forward matrix: duplicate or empty id {case_id!r}")
        ids.add(case_id)
        if skill not in EXPECTED_SKILLS:
            errors.append(f"{case_id}: unknown skill {skill!r}")
        if mode not in {"direct", "delegated"}:
            errors.append(f"{case_id}: mode must be direct or delegated")
        seen.add((skill, mode))
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
        routes = case.get("expected_routes")
        if not isinstance(routes, list) or not routes:
            errors.append(f"{case_id}: expected_routes must be non-empty")
        elif skill in domain_wraps:
            route_set = set(routes)
            invalid = route_set - domain_wraps[skill] - {"graph_orchestrate"}
            if invalid:
                errors.append(
                    f"{case_id}: routes not owned by the domain skill: {sorted(invalid)}"
                )
            if mode == "direct" and "graph_orchestrate" in route_set:
                errors.append(f"{case_id}: a direct case cannot use graph_orchestrate")
            if mode == "delegated" and "graph_orchestrate" not in route_set:
                errors.append(
                    f"{case_id}: a delegated domain case must use graph_orchestrate"
                )
        elif mode == "delegated" and (
            not isinstance(routes, list) or "graph_orchestrate" not in routes
        ):
            errors.append(f"{case_id}: delegated cases must use graph_orchestrate")

        allowed_tools = case.get("allowed_tools")
        if not isinstance(allowed_tools, list):
            errors.append(f"{case_id}: allowed_tools must be a list")
        elif mode == "direct":
            if allowed_tools:
                errors.append(f"{case_id}: direct semantic cases cannot receive tools")
        else:
            if not allowed_tools or not all(
                isinstance(tool, str) and tool for tool in allowed_tools
            ):
                errors.append(f"{case_id}: delegated allowed_tools must be non-empty")
                continue
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
                errors.append(
                    f"{case_id}: delegated child cannot recursively orchestrate"
                )

    expected_pairs = {
        (skill, mode) for skill in EXPECTED_SKILLS for mode in ("direct", "delegated")
    }
    if seen != expected_pairs:
        errors.append(
            "forward matrix: each skill needs one direct and one delegated case"
        )

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
    raw_matrix = FORWARD_MATRIX.read_text(encoding="utf-8")
    for label, pattern in _PRIVATE_PATTERNS:
        if pattern.search(raw_matrix):
            errors.append(f"forward matrix: contains {label}")
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


def validate() -> list[str]:
    """Return deterministic validation errors for the retained suite."""
    actual = {
        path.parent.name for path in SKILLS_ROOT.glob("*/SKILL.md") if path.is_file()
    }
    errors: list[str] = []
    if len(EXPECTED_SKILLS) != 10:
        errors.append("canonical taxonomy must contain exactly 10 workflow skills")
    if actual != EXPECTED_SKILLS:
        errors.append(
            "skill inventory mismatch: "
            f"missing={sorted(EXPECTED_SKILLS - actual)} "
            f"unexpected={sorted(actual - EXPECTED_SKILLS)}"
        )
    nested = [
        path
        for path in SKILLS_ROOT.rglob("SKILL.md")
        if path.parent.parent != SKILLS_ROOT
    ]
    if nested:
        errors.append(
            "nested bundled skills are not allowed: "
            + ", ".join(_relative(path) for path in nested)
        )
    for name in sorted(actual):
        errors.extend(_validate_skill(SKILLS_ROOT / name))
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
    print("Pre-bundled skill validation OK — 10 skills, 20 synthetic forward cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
