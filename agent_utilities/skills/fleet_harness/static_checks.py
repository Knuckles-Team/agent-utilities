"""Static (no-services) validation layer.

Every check is independent, explains itself (rule id, expected vs actual),
and never masks a violation — a broken skill reports FAIL with the reason,
full stop. No ``try/except: pass``, no trivial assertions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from agent_utilities.skills.fleet_harness.discovery import SkillRecord

Status = Literal["PASS", "FAIL", "WARN"]

#: `skill_type` is AUTHORITATIVE (frontmatter field, not the directory path) —
#: see universal-skills AGENTS.md "The Atomicity Edict".
_VALID_SKILL_TYPES = frozenset({"skill", "workflow", "graph"})
_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_DESCRIPTION_CHARS = 1024

#: A backtick span naming a graph-os tool by the fleet's own naming
#: convention (``graph_*`` / ``engine_*`` / ``ontology_*`` / ``object_*``).
#: Deliberately narrower than "any backtick word" — English verbs like `ask`
#: appear constantly in prose and are not tool-name assertions.
GRAPHOS_TOOL_REF_RE = re.compile(r"`((?:graph|engine|ontology|object)_[a-z0-9_]+)`")

#: `- \`tool_name\`: description` bullets under a "## Tools" section — the
#: fleet convention for a skill to enumerate the MCP tools it drives (see
#: e.g. fan-manager-control's `set_fan_speed`).
_TOOLS_SECTION_RE = re.compile(r"^##\s+Tools\s*$", re.MULTILINE)
_TOOL_BULLET_RE = re.compile(r"^\s*-\s+`([a-zA-Z_][a-zA-Z0-9_]*)`", re.MULTILINE)

#: Relative-path references worth existence-checking: a markdown link target,
#: or a backtick span that looks like a repo-relative file (has a slash and
#: either a known asset prefix or a recognizable extension).
_MD_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
_BACKTICK_PATH_RE = re.compile(r"`([A-Za-z0-9_./-]+/[A-Za-z0-9_./-]+)`")
_ASSET_PREFIXES = (
    "scripts/",
    "references/",
    "assets/",
    "./scripts/",
    "./references/",
    "./assets/",
)
_PATH_LIKE_EXTENSIONS = (
    ".py",
    ".sh",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".csv",
    ".sql",
)


def resolvable_graphos_tool_names() -> frozenset[str]:
    """The canonical graph-os tool surface (import kept lazy/optional).

    Only consulted for skills that live under an ``agent-utilities`` or
    ``epistemic-graph`` repo root (the two repos that own the graph-os
    surface) — fleet ``agents/*`` skills wrap their own package's MCP tools
    and are resolved via :func:`_resolve_package_tool` instead.
    """
    from agent_utilities.mcp.tool_specs import INTENT_VERBS, TOOL_VERBS

    return frozenset(TOOL_VERBS) | frozenset(INTENT_VERBS)


@dataclass
class CheckResult:
    """One rule's verdict for one skill — always self-explanatory."""

    rule: str
    status: Status
    message: str


@dataclass
class SkillStaticReport:
    record: SkillRecord
    name: str | None
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def status(self) -> Status:
        if any(c.status == "FAIL" for c in self.checks):
            return "FAIL"
        return "PASS"

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "FAIL"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "WARN"]


def _split_frontmatter(text: str) -> tuple[str, str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[1], parts[2]
    return "", text


def parse_frontmatter(text: str) -> tuple[dict[str, Any] | None, str, str | None]:
    """Return ``(data, body, parse_error)``.

    ``data`` is ``None`` (with a populated ``parse_error``) when there is no
    frontmatter block, the YAML fails to parse, or it doesn't parse to a
    mapping.
    """
    fm_text, body = _split_frontmatter(text)
    if not fm_text.strip():
        return None, text, "no `---` frontmatter block found"
    try:
        data = yaml.safe_load(fm_text)
    except yaml.YAMLError as exc:
        return None, body, f"frontmatter is not valid YAML: {type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return None, body, "frontmatter YAML did not parse to a mapping"
    return data, body, None


def _check_frontmatter_schema(
    record: SkillRecord, text: str
) -> tuple[list[CheckResult], dict[str, Any] | None, str]:
    checks: list[CheckResult] = []
    data, body, parse_error = parse_frontmatter(text)
    if data is None:
        checks.append(
            CheckResult(
                "frontmatter.parses", "FAIL", parse_error or "unknown parse error"
            )
        )
        return checks, None, body
    checks.append(
        CheckResult("frontmatter.parses", "PASS", "frontmatter is valid YAML mapping")
    )

    name = data.get("name")
    if not name or not str(name).strip():
        checks.append(
            CheckResult(
                "frontmatter.name_present", "FAIL", "`name` field is missing or empty"
            )
        )
    else:
        checks.append(CheckResult("frontmatter.name_present", "PASS", f"name={name!r}"))
        name_s = str(name)
        if not _KEBAB_RE.match(name_s):
            checks.append(
                CheckResult(
                    "frontmatter.name_kebab_case",
                    "FAIL",
                    f"name {name_s!r} is not lowercase-hyphenated (expected pattern {_KEBAB_RE.pattern!r})",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "frontmatter.name_kebab_case",
                    "PASS",
                    f"name={name_s!r} is kebab-case",
                )
            )

        skill_type = data.get("skill_type")
        if skill_type != "graph" and name_s != record.directory_name:
            checks.append(
                CheckResult(
                    "frontmatter.name_matches_directory",
                    "FAIL",
                    f"name {name_s!r} != directory name {record.directory_name!r} "
                    f"(only skill_type=graph nodes may diverge)",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "frontmatter.name_matches_directory",
                    "PASS",
                    "name matches directory",
                )
            )

    description = data.get("description")
    desc_s = str(description).strip() if description is not None else ""
    if not desc_s:
        checks.append(
            CheckResult(
                "frontmatter.description_present",
                "FAIL",
                "`description` field is missing or empty",
            )
        )
    else:
        checks.append(
            CheckResult(
                "frontmatter.description_present", "PASS", f"{len(desc_s)} chars"
            )
        )
        if len(desc_s) > _MAX_DESCRIPTION_CHARS:
            checks.append(
                CheckResult(
                    "frontmatter.description_portable_length",
                    "WARN",
                    f"description is {len(desc_s)} chars (> {_MAX_DESCRIPTION_CHARS} — "
                    "exceeds the Codex cross-tool portability limit)",
                )
            )
        if "<" in desc_s or ">" in desc_s:
            checks.append(
                CheckResult(
                    "frontmatter.description_portable_chars",
                    "WARN",
                    "description contains `<`/`>` (rejected by the Codex frontmatter adapter)",
                )
            )

    skill_type = data.get("skill_type")
    if skill_type is None:
        checks.append(
            CheckResult(
                "frontmatter.skill_type_present",
                "FAIL",
                "`skill_type` field is missing — it is AUTHORITATIVE and required "
                f"(one of {sorted(_VALID_SKILL_TYPES)})",
            )
        )
    elif skill_type not in _VALID_SKILL_TYPES:
        checks.append(
            CheckResult(
                "frontmatter.skill_type_valid",
                "FAIL",
                f"skill_type {skill_type!r} is not one of {sorted(_VALID_SKILL_TYPES)}",
            )
        )
    else:
        checks.append(
            CheckResult(
                "frontmatter.skill_type_valid", "PASS", f"skill_type={skill_type!r}"
            )
        )

    return checks, data, body


def _extract_reference_candidates(body: str) -> list[str]:
    candidates: list[str] = []
    for target in _MD_LINK_RE.findall(body):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        candidates.append(target)
    for target in _BACKTICK_PATH_RE.findall(body):
        if target.startswith(_ASSET_PREFIXES) or target.endswith(_PATH_LIKE_EXTENSIONS):
            candidates.append(target)
    return candidates


def _check_structural_integrity(record: SkillRecord, body: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    candidates = sorted(set(_extract_reference_candidates(body)))
    missing: list[str] = []
    for candidate in candidates:
        cleaned = candidate.split("#", 1)[0].strip()
        if not cleaned or "*" in cleaned or "<" in cleaned or ">" in cleaned:
            continue
        if cleaned.startswith("/"):
            # absolute paths are never portable skill references; the
            # existing `agent_utilities.skills.validation` private-pattern
            # gate already flags these — skip here to avoid double-counting.
            continue
        candidate_path = (record.skill_dir / cleaned).resolve()
        try:
            candidate_path.relative_to(record.skill_dir.resolve())
        except ValueError:
            # escapes the skill directory (e.g. `../other-skill/x`) — not
            # this gate's concern.
            continue
        if not candidate_path.exists():
            missing.append(cleaned)
    if missing:
        checks.append(
            CheckResult(
                "structure.referenced_files_exist",
                "FAIL",
                f"referenced file(s) do not exist on disk relative to the skill directory: {missing}",
            )
        )
    else:
        checks.append(
            CheckResult(
                "structure.referenced_files_exist",
                "PASS",
                f"{len(candidates)} referenced path(s) resolved"
                if candidates
                else "no relative file references in body",
            )
        )
    return checks


def _package_root_for(record: SkillRecord) -> Path | None:
    """Best-effort owning-package root for a fleet ``agents/*`` skill.

    Walks up from the skill directory looking for the nearest ancestor that
    contains a ``pyproject.toml`` — that is the package whose source tree a
    "## Tools" bullet's tool name should resolve against.
    """
    current = record.skill_dir
    for _ in range(8):
        if (current / "pyproject.toml").is_file():
            return current
        if current.parent == current:
            return None
        current = current.parent
    return None


def _check_declared_tools(record: SkillRecord, body: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    graphos_refs = sorted(set(GRAPHOS_TOOL_REF_RE.findall(body)))
    if record.repo_name in {"agent-utilities", "epistemic-graph"} and graphos_refs:
        known = resolvable_graphos_tool_names()
        unknown = [t for t in graphos_refs if t not in known]
        if unknown:
            checks.append(
                CheckResult(
                    "tools.graphos_references_resolve",
                    "FAIL",
                    f"referenced graph-os tool(s) not in the canonical {len(known)}-tool "
                    f"surface (`agent_utilities.mcp.tool_specs`): {unknown}",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "tools.graphos_references_resolve",
                    "PASS",
                    f"all {len(graphos_refs)} referenced graph-os tool(s) are registered",
                )
            )

    tools_section = _TOOLS_SECTION_RE.search(body)
    if tools_section:
        section_body = body[tools_section.end() :]
        next_h2 = re.search(r"^##\s+", section_body, re.MULTILINE)
        if next_h2:
            section_body = section_body[: next_h2.start()]
        declared = sorted(set(_TOOL_BULLET_RE.findall(section_body)))
        package_root = _package_root_for(record)
        if declared and package_root is not None:
            unresolved: list[str] = []
            for tool_name in declared:
                pattern = re.compile(r"\b" + re.escape(tool_name) + r"\b")
                found = False
                for py_file in package_root.rglob("*.py"):
                    parts = py_file.relative_to(package_root).parts
                    if any(
                        p in {".venv", "venv", "tests", "test", "__pycache__"}
                        for p in parts
                    ):
                        continue
                    try:
                        text = py_file.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    if pattern.search(text):
                        found = True
                        break
                if not found:
                    unresolved.append(tool_name)
            if unresolved:
                checks.append(
                    CheckResult(
                        "tools.package_references_resolve",
                        "FAIL",
                        f"tool(s) declared under '## Tools' have no matching identifier "
                        f"anywhere in the owning package ({package_root.name}): {unresolved}",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        "tools.package_references_resolve",
                        "PASS",
                        f"all {len(declared)} declared tool(s) resolve within {package_root.name}",
                    )
                )
    return checks


def validate_skill(record: SkillRecord) -> SkillStaticReport:
    """Run every static rule against one skill and return its report."""
    text = record.skill_md.read_text(encoding="utf-8", errors="replace")
    schema_checks, data, body = _check_frontmatter_schema(record, text)
    name = str(data.get("name")) if data and data.get("name") else None
    checks = list(schema_checks)
    checks.extend(_check_structural_integrity(record, body))
    checks.extend(_check_declared_tools(record, body))
    return SkillStaticReport(record=record, name=name, checks=checks)


def _check_name_uniqueness(reports: list[SkillStaticReport]) -> None:
    """Mutates ``reports`` in place, appending a uniqueness verdict to each."""
    by_name: dict[str, list[SkillStaticReport]] = {}
    for report in reports:
        if report.name:
            by_name.setdefault(report.name, []).append(report)
    for report in reports:
        if not report.name:
            continue
        siblings = by_name[report.name]
        if len(siblings) > 1:
            others = [s.record.relative_path for s in siblings if s is not report]
            report.checks.append(
                CheckResult(
                    "frontmatter.name_unique",
                    "FAIL",
                    f"name {report.name!r} is used by {len(siblings)} skills fleet-wide; "
                    f"also declared at: {others}",
                )
            )
        else:
            report.checks.append(
                CheckResult(
                    "frontmatter.name_unique",
                    "PASS",
                    "name is unique across the scanned fleet",
                )
            )


def run_static_checks(records: list[SkillRecord]) -> list[SkillStaticReport]:
    """Run the static layer over every discovered skill.

    Name uniqueness is a fleet-wide (cross-skill) rule, so it runs as a
    second pass once every skill's frontmatter has been parsed once.
    """
    reports = [validate_skill(record) for record in records]
    _check_name_uniqueness(reports)
    return reports
