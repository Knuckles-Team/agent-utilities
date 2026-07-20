#!/usr/bin/env python3
"""Validate the Agent Utilities contract across the declared provider fleet.

The repository-manager workspace is the source of truth for provider membership.
The gate validates publishable dependency bounds, rejects workspace-local source
overrides, and checks documentation language without emitting machine paths or
matched content.
"""

from __future__ import annotations

import argparse
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

EXPECTED_PROVIDER_COUNT = 65
REQUIRED_SPECIFIERS = frozenset({(">=", "1.27.1"), ("<", "2.0.0")})
_RETIRED_AGENT_UTILITIES_EXTRAS = frozenset({"agent", "engine"})

_PROVIDER_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_STALE_DOCKERFILE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "retired_agent_utilities_extra",
        re.compile(
            r"agent[-_]utilities\s*\[(?:[^\]]*,)?\s*(?:agent|engine)\s*(?:,|\])",
            re.IGNORECASE,
        ),
    ),
    (
        "agent_utilities_dspy_claim",
        re.compile(
            r"agent[-_]utilities(?:\s*\[[^\]]+\])?.{0,200}\bdspy\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "mcp_claims_no_engine",
        re.compile(
            r"agent[-_]utilities\s*\[[^\]]*\bmcp\b[^\]]*\]"
            r".{0,160}\b(?:no|without)\s+(?:the\s+)?engine\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)

_STALE_DOC_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "mcp_excludes_graph_engine",
        re.compile(
            r"(?:\bexcludes?\b.{0,240}epistemic-graph"
            r"|epistemic-graph.{0,240}\bexcludes?\b)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "mcp_fastapi_only",
        re.compile(
            r"(?:pulls|pulling)\s+only\s+the\s+FastMCP\s*/\s*FastAPI\s+tooling",
            re.IGNORECASE,
        ),
    ),
    (
        "mcp_container_without_engine",
        re.compile(r"\*\*slim\*\*,\s*no\s+engine/", re.IGNORECASE),
    ),
    (
        "agent_extra_owns_engine",
        re.compile(
            r"The\s+\*\*full agent\*\*[^\n]*embeds\s+the\s+"
            r"\*\*epistemic-graph\*\*\s+engine"
            r"|transitively\s+via\s+`agent-utilities\[agent\]`",
            re.IGNORECASE,
        ),
    ),
    (
        "mcp_declared_without_engine",
        re.compile(
            r"Slim\s+MCP\s+server(?:\s+only)?\s*"
            r"\(`agent-utilities\[mcp\]`\s+—\s+FastMCP/FastAPI\)"
            r"(?![^\n]*epistemic-graph)",
            re.IGNORECASE,
        ),
    ),
    (
        "mcp_declared_without_database",
        re.compile(
            r"(?:slim\s+)?`?\[mcp\]`?\s+server"
            r"(?:\s+and\s+the\s+core\s+(?:client|CLI))?\s+"
            r"does\s+\*\*not\*\*\s+require\s+the\s+database",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True, order=True)
class Finding:
    """A privacy-safe contract violation."""

    package: str
    path: str
    line: int
    rule: str


@dataclass(frozen=True)
class FleetStats:
    providers: int = 0
    pyproject_requirements: int = 0
    requirements_file_requirements: int = 0
    documentation_files: int = 0


def _provider_names(workspace: dict[str, Any]) -> list[str]:
    """Return normalized provider names from the agents workspace branch."""

    try:
        repositories = workspace["subdirectories"]["agent-packages"]["subdirectories"][
            "agents"
        ]["repositories"]
    except (KeyError, TypeError) as exc:
        raise ValueError("workspace is missing the provider repository branch") from exc
    if not isinstance(repositories, list):
        raise ValueError("provider repositories must be a list")

    providers: list[str] = []
    for entry in repositories:
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            raise ValueError("every provider repository needs a URL")
        name = entry["url"].rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
        if not _PROVIDER_NAME.fullmatch(name):
            raise ValueError("provider URL has an invalid repository name")
        providers.append(name)
    if len(providers) != len(set(providers)):
        raise ValueError("workspace contains duplicate provider repositories")
    return providers


def load_workspace(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Load a repository-manager workspace and its provider inventory."""

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("workspace root must be a mapping")
    return data, _provider_names(data)


def _dependency_values(project: dict[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    metadata = project.get("project", {})
    for index, value in enumerate(metadata.get("dependencies", [])):
        if isinstance(value, str):
            values.append((f"project.dependencies[{index}]", value))
    for group, dependencies in metadata.get("optional-dependencies", {}).items():
        if not isinstance(dependencies, list):
            continue
        for index, value in enumerate(dependencies):
            if isinstance(value, str):
                values.append(
                    (f"project.optional-dependencies.{group}[{index}]", value)
                )
    for group, dependencies in project.get("dependency-groups", {}).items():
        if not isinstance(dependencies, list):
            continue
        for index, value in enumerate(dependencies):
            if isinstance(value, str):
                values.append((f"dependency-groups.{group}[{index}]", value))
    return values


def _agent_utilities_requirement(value: str) -> Requirement | None:
    try:
        requirement = Requirement(value)
    except InvalidRequirement:
        if re.search(r"\bagent[-_]utilities\b", value, re.IGNORECASE):
            raise
        return None
    if canonicalize_name(requirement.name) != "agent-utilities":
        return None
    return requirement


def _has_required_bounds(requirement: Requirement) -> bool:
    actual = {(item.operator, item.version) for item in requirement.specifier}
    return REQUIRED_SPECIFIERS.issubset(actual)


def _requirement_lines(path: Path) -> list[tuple[int, str]]:
    if not path.is_file():
        return []
    values: list[tuple[int, str]] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        value = line.split("#", 1)[0].strip()
        if value and not value.startswith(
            ("-r", "--requirement", "-c", "--constraint")
        ):
            values.append((number, value))
    return values


def _documentation_paths(provider: Path) -> list[Path]:
    paths = {
        provider / "README.md",
        provider / "mkdocs.yml",
        provider / "Dockerfile",
        provider / "docker" / "Dockerfile",
    }
    docs = provider / "docs"
    if docs.is_dir():
        paths.update(docs.rglob("*.md"))
    return sorted(path for path in paths if path.is_file())


def _validate_provider(provider: Path, name: str) -> tuple[list[Finding], FleetStats]:
    findings: list[Finding] = []
    pyproject_path = provider / "pyproject.toml"
    if not provider.is_dir():
        return [Finding(name, "provider", 0, "provider_checkout_missing")], FleetStats()
    if not pyproject_path.is_file():
        return [Finding(name, "pyproject.toml", 0, "pyproject_missing")], FleetStats()

    try:
        project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return [Finding(name, "pyproject.toml", 0, "pyproject_invalid")], FleetStats()

    metadata = project.get("project")
    if not isinstance(metadata, dict):
        findings.append(Finding(name, "pyproject.toml", 0, "project_metadata_invalid"))
        metadata = {}
    authors = metadata.get("authors")
    if (
        not isinstance(authors, list)
        or not authors
        or any(
            not isinstance(author, dict)
            or bool(set(author) - {"name", "email"})
            or not any(isinstance(author.get(key), str) for key in ("name", "email"))
            for author in authors
        )
    ):
        findings.append(Finding(name, "pyproject.toml", 0, "project_authors_invalid"))
    if not isinstance(metadata.get("license"), (str, dict)):
        findings.append(Finding(name, "pyproject.toml", 0, "project_license_invalid"))

    declarations = 0
    for context, value in _dependency_values(project):
        try:
            requirement = _agent_utilities_requirement(value)
        except InvalidRequirement:
            findings.append(Finding(name, "pyproject.toml", 0, "dependency_invalid"))
            continue
        if requirement is None:
            continue
        declarations += 1
        if not _has_required_bounds(requirement):
            findings.append(
                    Finding(name, "pyproject.toml", 0, f"dependency_bounds:{context}")
                )
        for extra in sorted(
            _RETIRED_AGENT_UTILITIES_EXTRAS.intersection(requirement.extras)
        ):
            findings.append(
                Finding(
                    name,
                    "pyproject.toml",
                    0,
                    f"retired_agent_utilities_extra:{extra}:{context}",
                )
            )
    if declarations == 0:
        findings.append(Finding(name, "pyproject.toml", 0, "dependency_missing"))

    sources = project.get("tool", {}).get("uv", {}).get("sources", {})
    if sources:
        findings.append(Finding(name, "pyproject.toml", 0, "local_source_forbidden"))

    requirement_declarations = 0
    requirements_path = provider / "requirements.txt"
    for number, value in _requirement_lines(requirements_path):
        try:
            requirement = _agent_utilities_requirement(value)
        except InvalidRequirement:
            findings.append(
                Finding(name, "requirements.txt", number, "dependency_invalid")
            )
            continue
        if requirement is None:
            continue
        requirement_declarations += 1
        if not _has_required_bounds(requirement):
            findings.append(
                Finding(name, "requirements.txt", number, "dependency_bounds")
            )
        for extra in sorted(
            _RETIRED_AGENT_UTILITIES_EXTRAS.intersection(requirement.extras)
        ):
            findings.append(
                Finding(
                    name,
                    "requirements.txt",
                    number,
                    f"retired_agent_utilities_extra:{extra}",
                )
            )

    documentation = _documentation_paths(provider)
    for path in documentation:
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(provider).as_posix()
        rules = _STALE_DOC_RULES
        if path.name == "Dockerfile":
            rules += _STALE_DOCKERFILE_RULES
        for rule, pattern in rules:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(Finding(name, relative, line, rule))

    return sorted(findings), FleetStats(
        providers=1,
        pyproject_requirements=declarations,
        requirements_file_requirements=requirement_declarations,
        documentation_files=len(documentation),
    )


def validate_fleet(
    workspace_path: Path,
    providers_root: Path,
    *,
    expected_provider_count: int = EXPECTED_PROVIDER_COUNT,
) -> tuple[list[Finding], FleetStats]:
    """Validate every provider named by the repository-manager workspace."""

    try:
        _, providers = load_workspace(workspace_path)
    except (OSError, ValueError, yaml.YAMLError):
        return [
            Finding("workspace", "workspace.yml", 0, "workspace_invalid")
        ], FleetStats()

    findings: list[Finding] = []
    if expected_provider_count and len(providers) != expected_provider_count:
        findings.append(
            Finding("workspace", "workspace.yml", 0, "provider_count_mismatch")
        )

    stats = FleetStats()
    for name in providers:
        package_findings, package_stats = _validate_provider(
            providers_root / name, name
        )
        findings.extend(package_findings)
        stats = FleetStats(
            providers=stats.providers + package_stats.providers,
            pyproject_requirements=(
                stats.pyproject_requirements + package_stats.pyproject_requirements
            ),
            requirements_file_requirements=(
                stats.requirements_file_requirements
                + package_stats.requirements_file_requirements
            ),
            documentation_files=(
                stats.documentation_files + package_stats.documentation_files
            ),
        )
    return sorted(findings), stats


def _default_workspace() -> Path:
    configured = os.environ.get("AGENT_UTILITIES_WORKSPACE_CONFIG")
    if configured:
        return Path(configured).expanduser()
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    installed = xdg / "agent-utilities" / "workspace.yml"
    if installed.is_file():
        return installed
    return (
        Path(__file__).resolve().parents[2]
        / "agents"
        / "repository-manager"
        / "repository_manager"
        / "workspace.yml"
    )


def _default_providers_root(workspace_path: Path) -> Path:
    configured = os.environ.get("AGENT_UTILITIES_PROVIDERS_ROOT")
    if configured:
        return Path(configured).expanduser()
    local = Path(__file__).resolve().parents[2] / "agents"
    if local.is_dir():
        return local
    try:
        workspace, _ = load_workspace(workspace_path)
    except (OSError, ValueError, yaml.YAMLError):
        return local
    root = os.path.expandvars(str(workspace.get("path", "")))
    if root and "${" not in root:
        return Path(root).expanduser() / "agent-packages" / "agents"
    return local


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--providers-root", type=Path)
    parser.add_argument(
        "--expected-provider-count", type=int, default=EXPECTED_PROVIDER_COUNT
    )
    args = parser.parse_args()

    workspace = args.workspace or _default_workspace()
    providers_root = args.providers_root or _default_providers_root(workspace)
    findings, stats = validate_fleet(
        workspace,
        providers_root,
        expected_provider_count=args.expected_provider_count,
    )
    for finding in findings:
        location = f"{finding.package}/{finding.path}"
        if finding.line:
            location += f":{finding.line}"
        print(f"{location}: {finding.rule}")
    if findings:
        print(f"provider fleet contract: FAIL ({len(findings)} finding(s))")
        return 1
    print(
        "provider fleet contract: PASS "
        f"({stats.providers} providers, "
        f"{stats.pyproject_requirements} pyproject declaration(s), "
        f"{stats.requirements_file_requirements} requirements declaration(s), "
        f"{stats.documentation_files} documentation file(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
