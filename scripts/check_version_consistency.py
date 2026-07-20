#!/usr/bin/env python3
"""Verify that every release surface derives from one package version."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import InvalidRequirement, Requirement

ROOT = Path(__file__).resolve().parents[1]
VERSION_ATTRIBUTE = "agent_utilities._version.__version__"
VERSION_IMPORT = "from agent_utilities._version import __version__"
VERSION_CONSUMERS = (
    "agent_utilities/__init__.py",
    "agent_utilities/__main__.py",
    "agent_utilities/api_utilities.py",
    "agent_utilities/base_utilities.py",
    "agent_utilities/core/embedding_utilities.py",
    "agent_utilities/gateway/__init__.py",
    "agent_utilities/mcp/harness_server.py",
    "agent_utilities/mcp/kg_server.py",
    "agent_utilities/mcp/server_factory.py",
    "agent_utilities/tools/tool_registry.py",
)
SEMVER = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
RELEASE_ORDER = (
    "epistemic-operations-protocol",
    "epistemic-graph",
    "agent-utilities",
    "langfuse-agent",
    "connector-bundles",
    "prebundled-skills",
    "ontology-lock",
    "index-migrations",
)


def _source_version(root: Path) -> str:
    tree = ast.parse(
        (root / "agent_utilities" / "_version.py").read_text(encoding="utf-8")
    )
    values: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            values.append(value.value)
    if len(values) != 1 or SEMVER.fullmatch(values[0]) is None:
        raise ValueError("invalid version authority")
    return values[0]


def _provider_floor(root: Path) -> frozenset[tuple[str, str]]:
    tree = ast.parse(
        (root / "scripts" / "check_provider_fleet_contract.py").read_text(
            encoding="utf-8"
        )
    )
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SPECIFIERS"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Call) or len(node.value.args) != 1:
            break
        value: Any = ast.literal_eval(node.value.args[0])
        return frozenset(value)
    raise ValueError("invalid provider floor")


def _duplicate_version_authorities(root: Path) -> list[str]:
    """Find tracked duplicate assignments without recursively statting the tree."""
    try:
        result = subprocess.run(  # noqa: S603
            [
                "git",
                "grep",
                "-n",
                "-E",
                r"^[[:space:]]*__version__[[:space:]]*(:[^=]+)?=",
                "--",
                "agent_utilities",
            ],  # noqa: S607
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        result = None
    if result is not None and result.returncode in {0, 1}:
        return sorted(
            {
                line.split(":", 1)[0]
                for line in result.stdout.splitlines()
                if line and line.split(":", 1)[0] != "agent_utilities/_version.py"
            }
        )

    assignment = re.compile(r"(?m)^\s*__version__\s*(?::[^=]+)?=")
    duplicates: list[str] = []
    for relative_path in VERSION_CONSUMERS:
        try:
            if assignment.search((root / relative_path).read_text(encoding="utf-8")):
                duplicates.append(relative_path)
        except OSError:
            continue
    return duplicates


def validate(root: Path = ROOT) -> list[str]:
    """Return stable finding identifiers; an empty list means consistency."""
    findings: list[str] = []
    try:
        version = _source_version(root)
    except (OSError, SyntaxError, ValueError):
        return ["version-authority"]

    project: dict[str, Any] | None = None
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        metadata = project["project"]
        dynamic = project["tool"]["setuptools"]["dynamic"]
        if "version" in metadata or "version" not in metadata.get("dynamic", []):
            findings.append("project-version-mode")
        if dynamic.get("version", {}).get("attr") != VERSION_ATTRIBUTE:
            findings.append("project-version-authority")
    except (KeyError, OSError, tomllib.TOMLDecodeError):
        findings.append("project-version-metadata")

    try:
        lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
        packages = [
            package
            for package in lock.get("package", [])
            if package.get("name") == "agent-utilities"
        ]
        if (
            len(packages) != 1
            or packages[0].get("source") != {"editable": "."}
            or packages[0].get("version", version) != version
        ):
            findings.append("lock-version")
    except (OSError, tomllib.TOMLDecodeError):
        findings.append("lock-metadata")

    expected_text = {
        "README.md": f"*Version: {version}*",
        "CHANGELOG.md": f"## [{version}] - ",
        "scripts/install.sh": f"AGENT_UTILITIES_VERSION:-{version}",
        "scripts/install.ps1": f"AGENT_UTILITIES_VERSION) {{ $env:AGENT_UTILITIES_VERSION }} else {{ '{version}' }}",
        "docs/ecosystem.md": f"agent-utilities>={version},<2.0.0",
        "docs/guides/creating-an-agent.md": f"agent-utilities[agent-runtime]>={version},<2.0.0",
        "docs/guides/building-mcp-servers.md": f"agent-utilities[agent-runtime]>={version},<2.0.0",
    }
    for relative_path, marker in expected_text.items():
        try:
            present = marker in (root / relative_path).read_text(encoding="utf-8")
        except OSError:
            present = False
        if not present:
            findings.append(f"release-surface:{relative_path}")

    server_markers = {
        "agent_utilities/mcp/server_factory.py": (
            "version: str = __version__",
            "FastMCP(name, version=version",
        ),
        "agent_utilities/mcp/kg_server.py": ("version=__version__",),
        "agent_utilities/mcp/harness_server.py": ("version=__version__",),
    }
    for relative_path, markers in server_markers.items():
        try:
            source = (root / relative_path).read_text(encoding="utf-8")
        except OSError:
            source = ""
        if any(marker not in source for marker in markers):
            findings.append(f"server-version:{relative_path}")

    try:
        if _provider_floor(root) != frozenset({(">=", version), ("<", "2.0.0")}):
            findings.append("provider-release-floor")
    except (OSError, SyntaxError, ValueError):
        findings.append("provider-release-floor")

    try:
        compatibility = yaml.safe_load(
            (root / "deploy/release/compatibility-matrix.yml").read_text(
                encoding="utf-8"
            )
        )
        components = compatibility["components"]
        release_order = tuple(compatibility["releaseTrain"]["assemblyOrder"])
        exact_versions = {
            name: str(component["version"])
            for name, component in components.items()
        }
        if (
            set(components) != set(RELEASE_ORDER)
            or release_order != RELEASE_ORDER
            or exact_versions["agent-utilities"] != f"=={version}"
            or any(not value.startswith("==") for value in exact_versions.values())
            or components["connector-bundles"].get("exactEntries") != 65
            or "langfuse-agent" not in components
        ):
            findings.append("compatibility-matrix-version")
        for name, component in components.items():
            dependencies = component.get("dependsOn") or {}
            if any(
                specifier != components[dependency]["version"]
                for dependency, specifier in dependencies.items()
            ):
                findings.append(f"compatibility-matrix-dependency:{name}")
        engine_version = exact_versions["epistemic-graph"].removeprefix("==")
        declarations = project["project"].get("dependencies", []) if project else []
        engine_requirements = []
        for declaration in declarations:
            try:
                requirement = Requirement(declaration)
            except InvalidRequirement:
                continue
            if requirement.name.casefold() == "epistemic-graph":
                engine_requirements.append(requirement)
        if (
            len(engine_requirements) != 1
            or engine_requirements[0].extras != {"full"}
            or str(engine_requirements[0].specifier)
            != f"<3.0.0,>={engine_version}"
        ):
            findings.append("full-engine-dependency")
    except (KeyError, OSError, TypeError, yaml.YAMLError):
        findings.append("compatibility-matrix-version")

    for relative_path in VERSION_CONSUMERS:
        try:
            source = (root / relative_path).read_text(encoding="utf-8")
        except OSError:
            findings.append(f"version-consumer:{relative_path}")
            continue
        if VERSION_IMPORT not in source:
            findings.append(f"version-consumer:{relative_path}")

    findings.extend(
        f"duplicate-version-authority:{relative_path}"
        for relative_path in _duplicate_version_authorities(root)
    )

    return sorted(set(findings))


def main() -> int:
    findings = validate()
    if findings:
        print("version consistency: FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"version consistency: PASS ({_source_version(ROOT)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
