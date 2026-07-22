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

from agent_utilities.skills.fleet_harness.discovery import SkillRecord, _is_noise

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

#: Immediately after a matched span's closing backtick, optionally through one
#: closing paren, an enum/set-membership declaration (`` `ontology_host` ∈
#: {stardog, ...} ``) — a config-schema FIELD being defined, not a tool call.
_ENUM_FIELD_DECLARATION_RE = re.compile(r"^\)?\s*(?:∈|in)\s*\{")

#: A backtick-wrapped structured-config filename (`genesis.yaml`, `foo.json`,
#: ...). Naming-convention matches that follow one closely (see
#: ``_looks_like_config_key`` below) are read as that file's field names, not
#: graph-os tool names.
_CONFIG_FILENAME_RE = re.compile(r"`[\w.-]+\.(?:ya?ml|json|toml|ini|env)`")

#: How far back (chars, no intervening blank line) a config-filename mention
#: still plausibly governs a later naming-convention match, e.g. "...see
#: `genesis.yaml` `engine` + per-profile `engine_tier`)." — `engine_tier` is a
#: field in that file, referenced ~25 chars after it names the file.
_CONFIG_KEY_PROXIMITY_WINDOW = 200


def _looks_like_config_key_or_path(body: str, match: re.Match[str]) -> bool:
    """True when a ``GRAPHOS_TOOL_REF_RE`` match is better read as a
    config-schema field name or a filesystem/storage-volume path fragment
    than as a graph-os tool-name assertion.

    The naming convention (``graph_*``/``engine_*``/``ontology_*``/
    ``object_*``) is also just... common English/technical vocabulary, so
    prose legitimately uses it for other things: a redb storage-volume name
    (``epistemic-graph-migrations``), a genesis run-plan YAML field
    (``agent-os-genesis``'s ``engine_tier``/``ontology_host``). Rather than a
    hardcoded ignore-list of those specific strings (which would silently
    stop covering the NEXT tool-shaped config key or path fragment some other
    skill introduces), this checks the syntax immediately around the match
    for the two shapes that reliably signal "not a tool claim":

    1. Glued to a preceding ``/`` with no whitespace — a path/volume
       fragment, e.g. ``redb/`graph_snapshots` volume``.
    2. Followed by set-membership/enum notation (``∈ {...}`` / ``in
       {...}``) — a config field's declared value set, e.g.
       ``\\`ontology_host\\` ∈ {stardog, apache-jena, local}``.
    3. Preceded, within a short no-blank-line window, by a backtick-wrapped
       structured-config filename (``genesis.yaml``, ...) — subsequent
       naming-convention matches in that span read as that file's own field
       names, e.g. ``see \\`genesis.yaml\\` ... \\`engine_tier\\``.

    This intentionally does NOT try to resolve every ambiguous case (a
    heuristic that consulted a real non-tool registry — e.g. epistemic-
    graph's actual redb table list — would be more precise for case 1, but
    that registry lives in a different repo/language and isn't reachable
    here). Prose that names a non-tool identifier with none of these three
    shapes will still register a FAIL; that residual gap is the accepted,
    documented false-positive mode, not silently masked.
    """
    start = match.start()
    if body[:start].endswith("/"):
        return True
    after = body[match.end() :]
    if _ENUM_FIELD_DECLARATION_RE.match(after):
        return True
    window_start = max(0, start - _CONFIG_KEY_PROXIMITY_WINDOW)
    window = body[window_start:start]
    if "\n\n" in window:
        window = window[window.rindex("\n\n") + 2 :]
    if _CONFIG_FILENAME_RE.search(window):
        return True
    return False


def graphos_tool_reference_occurrences(body: str) -> set[str]:
    """Backtick spans in ``body`` read as graph-os tool-name assertions.

    Applies :func:`_looks_like_config_key_or_path` per OCCURRENCE (not per
    unique string) so a term used legitimately as a tool name elsewhere in
    the same skill is still caught even if one occurrence of the identical
    string is excluded here as a config-key/path mention.
    """
    return {
        match.group(1)
        for match in GRAPHOS_TOOL_REF_RE.finditer(body)
        if not _looks_like_config_key_or_path(body, match)
    }


#: `- \`tool_name\`: description` bullets under a "## Tools" section — the
#: fleet convention for a skill to enumerate the MCP tools it drives (see
#: e.g. fan-manager-control's `set_fan_speed`).
_TOOLS_SECTION_RE = re.compile(r"^##\s+Tools\s*$", re.MULTILINE)
_TOOL_BULLET_RE = re.compile(r"^\s*-\s+`([a-zA-Z_][a-zA-Z0-9_]*)`", re.MULTILINE)

#: A backtick span naming a file the skill BUNDLES with itself — scoped
#: strictly to the `scripts/`/`references/`/`assets/` asset-directory
#: convention (see universal-skills AGENTS.md). Deliberately narrower than
#: "any repo path with a slash": AU/EG skill bodies also cite files
#: elsewhere in the monorepo for context (e.g. `docs/concept_map.md`,
#: `agent_utilities/knowledge_graph/core/task_lanes.py`) — those are
#: cross-references, not a claim that the skill ships the file, and are
#: anchored at an ambiguous root (repo root vs workspace root vs the citing
#: doc's own root) that this check cannot resolve reliably. Only a skill's
#: own bundled-asset claim is unambiguous: it is always skill-dir-relative.
_BACKTICK_PATH_RE = re.compile(r"`([A-Za-z0-9_./-]+/[A-Za-z0-9_./-]+)`")
_ASSET_PREFIXES = (
    "scripts/",
    "references/",
    "assets/",
    "./scripts/",
    "./references/",
    "./assets/",
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
    skill_type: str | None = None
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
    return [t for t in _BACKTICK_PATH_RE.findall(body) if t.startswith(_ASSET_PREFIXES)]


def _resolves_elsewhere_in_repo(record: SkillRecord, relative_path: str) -> bool:
    """True if ``relative_path`` exists anywhere under the skill's repo root.

    Many skills legitimately cite a file owned by a SIBLING skill or a
    declared package dependency rather than claiming to bundle it
    themselves — the fleet convention is an explicit pointer like
    ``` `other-skill` -> `references/x.md` ``` or ``` `other-skill`'s
    `scripts/y.py` ``` (see e.g. servicenow-workflow-studio ->
    servicenow-sdk-docs, github-org-remediation-loop -> github-triage-
    resolver, genius-web-crawl -> universal-skills' declared
    ``web-crawler`` extra). Requiring a skill-dir-relative match alone
    flags every one of those as a missing file. This does not weaken
    detection of a genuinely missing/renamed file: it only accepts a
    candidate that resolves to a REAL file/dir somewhere the harness's
    scan root can see, so a truly absent reference (nothing bundled,
    nothing cited elsewhere) still fails.
    """
    for match in record.repo_root.rglob(relative_path):
        if not _is_noise(match.relative_to(record.repo_root)) and match.exists():
            return True
    return False


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
        if not candidate_path.exists() and not _resolves_elsewhere_in_repo(
            record, cleaned.rstrip("/")
        ):
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
    graphos_refs = sorted(graphos_tool_reference_occurrences(body))
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
    skill_type = data.get("skill_type") if data else None
    checks = list(schema_checks)
    checks.extend(_check_structural_integrity(record, body))
    checks.extend(_check_declared_tools(record, body))
    return SkillStaticReport(
        record=record, name=name, skill_type=skill_type, checks=checks
    )


def _exempt_from_uniqueness(report: SkillStaticReport) -> bool:
    """``skill_graphs``/``skill-graphs`` reference corpora AND ``skill_type=graph``
    nodes are exempt from the global name pool.

    This mirrors the repo's own authoritative scoping
    (``scripts/check_skill_name_collision.py``): a skill-graph is a
    KG-ingestion reference manual, not an installable skill, so the same
    topic legitimately recurs across many bundles/pages (e.g. the
    "deployment" page of a skill-graph documenting the atomic
    ``agent-utilities-deployment`` skill is itself named
    ``agent-utilities-deployment``). The ``skill_type`` check is kept as a
    belt-and-suspenders match for a graph node authored outside a
    ``skill_graphs`` directory.
    """
    return report.record.in_reference_corpus or report.skill_type == "graph"


def _check_name_uniqueness(reports: list[SkillStaticReport]) -> None:
    """Mutates ``reports`` in place, appending a uniqueness verdict to each."""
    by_name: dict[str, list[SkillStaticReport]] = {}
    for report in reports:
        if report.name and not _exempt_from_uniqueness(report):
            by_name.setdefault(report.name, []).append(report)
    for report in reports:
        if not report.name:
            continue
        if _exempt_from_uniqueness(report):
            report.checks.append(
                CheckResult(
                    "frontmatter.name_unique",
                    "PASS",
                    "skill-graph reference-corpus node — exempt from fleet-wide name uniqueness",
                )
            )
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
