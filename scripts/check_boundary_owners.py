#!/usr/bin/env python3
"""AU-BOUNDARY-R038: reject a module or console script living in the wrong owner.

``architecture/component-registry-boundary-seeds.yml`` lists module path
prefixes that ``requirements.md``/``coverage.md`` name as moving to graph-os
or agent-connector-sdk (see that file's header for why boundary seeds live
there rather than in the signed ``component-registry.yml``). This gate fails
when a real module file, or a ``[project.scripts]`` console-script target, in
this repository matches one of those seeds AND the covering row in
``specs/au-boundary-deconstruction/coverage.md`` does not record it as
"relocate to <owner>" -- coverage.md's own phrasing for "moving, still
present". As each path is actually cut over, its row stops saying that, and a
reappearance of the path is then rejected rather than silently tolerated.

This gate is hermetic: it reads only the checked-in seed manifest, the
checked-in inventory, the repository's own tracked Python files, and
``pyproject.toml``. No live service, no network.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._git_scan import repo_root_of, tracked_or_walked  # noqa: E402
from scripts.boundary_inventory import (  # noqa: E402
    COVERAGE_PATH,
    CoverageParseError,
    CoverageRow,
    best_matching_row,
    directory_rows_by_path,
    load_coverage_rows,
    owner_recorded_present,
)

SEEDS_PATH = Path("architecture/component-registry-boundary-seeds.yml")
PYPROJECT_PATH = Path("pyproject.toml")
PACKAGE = "agent_utilities"


class SeedManifestError(ValueError):
    """The boundary-seed manifest could not be read strictly."""


@dataclass(frozen=True, slots=True)
class Seed:
    """One owner-seed: a module path prefix owned by another repository."""

    owner_repository: str
    requirement_id: str
    path: str


def load_seeds(seeds_file: Path) -> list[Seed]:
    """Strictly parse the boundary-seed manifest."""
    try:
        manifest: Any = yaml.safe_load(seeds_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SeedManifestError(f"cannot read {seeds_file}: {exc}") from exc
    entries = manifest.get("seeds") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise SeedManifestError(f"{seeds_file}: missing or empty 'seeds' list")
    seeds: list[Seed] = []
    for entry in entries:
        seeds.extend(_seeds_from_entry(entry, seeds_file))
    return seeds


def _seeds_from_entry(entry: Any, seeds_file: Path) -> list[Seed]:
    if not isinstance(entry, dict):
        raise SeedManifestError(f"{seeds_file}: seed entry is not a mapping: {entry!r}")
    owner = entry.get("owner_repository")
    requirement_id = entry.get("requirement_id")
    paths = entry.get("owned_module_paths")
    if (
        not isinstance(owner, str)
        or not isinstance(requirement_id, str)
        or not isinstance(paths, list)
        or not paths
    ):
        raise SeedManifestError(f"{seeds_file}: malformed seed entry {entry!r}")
    return [
        Seed(owner, requirement_id, path) for path in paths if isinstance(path, str)
    ]


def _canonical(path: str) -> str:
    """Drop a ``.py`` suffix so a file seed and a dotted module path compare equal."""
    return path[:-3] if path.endswith(".py") else path


def matching_seed(module_path: str, seeds: list[Seed]) -> Seed | None:
    """Return the longest-matching seed covering ``module_path``, if any."""
    candidate = _canonical(module_path)
    best: Seed | None = None
    best_len = -1
    for seed in seeds:
        seed_path = _canonical(seed.path)
        if candidate != seed_path and not candidate.startswith(seed_path + "/"):
            continue
        if len(seed_path) > best_len:
            best, best_len = seed, len(seed_path)
    return best


def _package_python_files(root: Path) -> list[Path]:
    package_root = root / PACKAGE
    repo_root = repo_root_of(package_root) or root
    return tracked_or_walked(package_root, "*.py", root=repo_root)


def _console_script_modules(root: Path) -> dict[str, str]:
    pyproject = root / PYPROJECT_PATH
    if not pyproject.is_file():
        return {}
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    return {
        name: target.split(":", 1)[0]
        for name, target in scripts.items()
        if isinstance(target, str)
    }


def _ownership_finding(
    module_path: str,
    seed: Seed,
    by_path: dict[str, list[CoverageRow]],
    *,
    origin: str,
) -> str | None:
    row = best_matching_row(module_path, by_path)
    if row is not None and owner_recorded_present(
        row.disposition, seed.owner_repository
    ):
        return None
    return (
        f"{origin} at '{module_path}' matches a {seed.owner_repository} seed "
        f"({seed.requirement_id}) but {COVERAGE_PATH.as_posix()} does not record it "
        "as relocating there"
    )


def _module_findings(
    root: Path, seeds: list[Seed], by_path: dict[str, list[CoverageRow]]
) -> list[str]:
    findings = []
    for file_path in sorted(_package_python_files(root)):
        module_path = file_path.relative_to(root).as_posix()
        seed = matching_seed(module_path, seeds)
        if seed is None:
            continue
        finding = _ownership_finding(module_path, seed, by_path, origin="module")
        if finding:
            findings.append(finding)
    return findings


def _console_script_findings(
    root: Path, seeds: list[Seed], by_path: dict[str, list[CoverageRow]]
) -> list[str]:
    findings = []
    for script_name, module in sorted(_console_script_modules(root).items()):
        module_path = module.replace(".", "/")
        seed = matching_seed(module_path, seeds)
        if seed is None:
            continue
        origin = f"console script '{script_name}'"
        finding = _ownership_finding(module_path, seed, by_path, origin=origin)
        if finding:
            findings.append(finding)
    return findings


def check(root: Path) -> list[str]:
    """Return every wrong-owner module or console-script finding for ``root``."""
    try:
        seeds = load_seeds(root / SEEDS_PATH)
    except SeedManifestError as exc:
        return [f"boundary seed manifest is unreadable: {exc}"]
    try:
        rows = load_coverage_rows(root / COVERAGE_PATH)
    except CoverageParseError as exc:
        return [f"inventory table is unparseable: {exc}"]
    by_path = directory_rows_by_path(rows)
    findings = _module_findings(root, seeds, by_path)
    findings += _console_script_findings(root, seeds, by_path)
    return sorted(set(findings))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]).resolve() if args else ROOT
    findings = check(root)
    if findings:
        print("Boundary ownership gate failed (AU-BOUNDARY-R038):", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("Boundary ownership gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
