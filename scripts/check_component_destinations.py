"""Check AU changes against source-declared destination ownership (AUD-31).

The owner manifests are the inputs. The plans registry is a generated read view.
This delta gate permits an unchanged pre-cutover baseline while rejecting new
AU edits in a destination already assigned to graph-os or the connector SDK.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath

import yaml


def _manifest(path: Path, repository: str) -> list[dict[str, object]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("schema") != "architecture-destination-owner/v1"
    ):
        raise ValueError(f"{path}: invalid destination schema")
    if data.get("owner_repository") != repository:
        raise ValueError(f"{path}: wrong repository owner")
    components = data.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError(f"{path}: missing components")
    ids: set[str] = set()
    for component in components:
        if not isinstance(component, dict) or set(component) != {
            "component_id",
            "parent_layer",
            "capability_ids",
            "owned_source_roots",
            "legacy_au_source_roots",
            "console_scripts",
        }:
            raise ValueError(f"{path}: malformed component")
        identifier = component["component_id"]
        if not isinstance(identifier, str) or identifier in ids:
            raise ValueError(f"{path}: duplicate component")
        ids.add(identifier)
        for field in (
            "capability_ids",
            "owned_source_roots",
            "legacy_au_source_roots",
            "console_scripts",
        ):
            values = component[field]
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{path}: invalid {field}")
        for field in ("owned_source_roots", "legacy_au_source_roots"):
            for value in component[field]:
                if (
                    PurePosixPath(value).is_absolute()
                    or ".." in PurePosixPath(value).parts
                ):
                    raise ValueError(f"{path}: non-relative {field}")
    return components


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _scripts(text: str) -> dict[str, str]:
    return tomllib.loads(text).get("project", {}).get("scripts", {})


def _belongs_to_root(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root.rstrip('/')}/")


def validate_changes(
    *,
    au_repo: Path,
    graph_manifest: Path,
    sdk_manifest: Path,
    au_manifest: Path,
    base: str,
    changed_paths: list[str] | None = None,
) -> list[str]:
    """Return violations for edits and console entrypoints added since *base*."""

    owners = {
        "agent-utilities": _manifest(au_manifest, "agent-utilities"),
        "graph-os": _manifest(graph_manifest, "graph-os"),
        "agent-connector-sdk": _manifest(sdk_manifest, "agent-connector-sdk"),
    }
    all_ids = [
        str(component["component_id"])
        for entries in owners.values()
        for component in entries
    ]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate component authority")
    capability_ids = [
        capability
        for entries in owners.values()
        for component in entries
        for capability in component["capability_ids"]
    ]
    if len(capability_ids) != len(set(capability_ids)):
        raise ValueError("duplicate capability authority")
    if changed_paths is None:
        changed_paths = _git(
            au_repo, "diff", "--diff-filter=ACMRT", "--name-only", base, "--"
        ).splitlines()
        changed_paths += _git(
            au_repo, "ls-files", "--others", "--exclude-standard"
        ).splitlines()
    violations: list[str] = []
    current = _scripts((au_repo / "pyproject.toml").read_text(encoding="utf-8"))
    previous = _scripts(_git(au_repo, "show", f"{base}:pyproject.toml"))
    for repository in ("graph-os", "agent-connector-sdk"):
        for component in owners[repository]:
            identifier = str(component["component_id"])
            for path in changed_paths:
                if any(
                    _belongs_to_root(path, root)
                    for root in component["legacy_au_source_roots"]
                ):
                    violations.append(f"{path}: destination is {identifier}")
            for name in component["console_scripts"]:
                if name in current and current[name] != previous.get(name):
                    violations.append(
                        f"project.scripts.{name}: destination is {identifier}"
                    )
    return sorted(set(violations))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--au-repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--graph-manifest", type=Path, required=True)
    parser.add_argument("--sdk-manifest", type=Path, required=True)
    parser.add_argument("--base", default="origin/main")
    arguments = parser.parse_args(argv)
    try:
        violations = validate_changes(
            au_repo=arguments.au_repo,
            au_manifest=arguments.au_repo
            / "architecture/component-registry.destinations.yml",
            graph_manifest=arguments.graph_manifest,
            sdk_manifest=arguments.sdk_manifest,
            base=arguments.base,
        )
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        tomllib.TOMLDecodeError,
        yaml.YAMLError,
    ) as error:
        print(f"component destinations: {error}", file=sys.stderr)
        return 2
    for violation in violations:
        print(violation, file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
