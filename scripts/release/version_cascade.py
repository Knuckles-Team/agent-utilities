#!/usr/bin/env python3
"""Plan and (optionally) apply the cross-project release version cascade.

See docs/architecture/drift_proof_release.md for the drift-class rationale.

The release train couples three tiers of version, in this order:

  TIER 1  ``epistemic-graph`` ``X.Y.Z``        (the Rust engine, an external project)
     -> TIER 2  this repo's ``pyproject.toml`` dependency
                ``epistemic-graph[full]>=X.Y.Z,<{X+1}.0.0`` and this repo's own
                version ``A.B.C`` in ``agent_utilities/_version.py``
        -> TIER 3  every provider package under ``agent-packages/agents/<pkg>/``
                   pins ``agent-utilities[<extras>]>={A}.0.0,<{A+1}.0.0`` — a
                   MAJOR band, never a patch pin.

``deploy/release/compatibility-matrix.yml`` pins the *exact* tier-1/tier-2
versions and every dependant component's ``dependsOn`` entry must equal the
version of the component it names — see ``scripts/check_version_consistency.py``,
which is the gate this module keeps satisfied.

The severity of a bump determines its blast radius:

* a **patch** or **minor** ``agent-utilities`` bump changes nothing below tier 2 —
  the existing ``>=A.0.0,<{A+1}.0.0`` provider band already covers it.
* a **major** ``agent-utilities`` bump invalidates every provider's ceiling and the
  cascade must rewrite all of them (and flags the consequence in
  ``breaking_notes``).
* the same floor/ceiling logic applies one tier up for the engine: any engine bump
  moves the exact floor in tier 2, a major engine bump also moves its ceiling.

``plan_cascade`` is a pure function — it only reads files and returns a plan.
``apply_plan`` performs the (textual, targeted) writes. Neither one touches git.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from packaging.requirements import InvalidRequirement, Requirement

_SEMVER = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)$")
_BUMP_LEVELS = ("major", "minor", "patch")
_SEVERITY_RANK = {"none": 0, "patch": 1, "minor": 2, "major": 3}

_ENGINE_REQUIREMENT = re.compile(r'"(epistemic-graph\[full\][^"]*)"')
_VERSION_ASSIGNMENT = re.compile(
    r'(?P<prefix>__version__\s*=\s*")(?P<value>[^"]+)(?P<suffix>")'
)
_PROVIDER_REQUIREMENT = re.compile(r'"(agent-utilities(?:\[[^\]]*\])?[^"]*)"')

_DRIFTED_PROVIDERS_NOTE = (
    "every provider package's agent-utilities ceiling is invalidated; "
    "providers must be re-released"
)
_ENGINE_DRIFTED_NOTE = (
    "agent-utilities' epistemic-graph[full] floor and ceiling both move; "
    "agent-utilities must be re-released against the new engine major"
)


class CascadeError(ValueError):
    """The cascade cannot be planned or applied as requested."""


@dataclass(frozen=True)
class FileEdit:
    """One targeted, textual replacement inside one file."""

    path: Path
    kind: str
    current: str
    proposed: str
    tier: int


@dataclass(frozen=True)
class CascadePlan:
    """The full, ordered set of edits one cascade run produces."""

    engine_current: str
    engine_proposed: str
    au_current: str
    au_proposed: str
    severity: str
    edits: tuple[FileEdit, ...] = field(default_factory=tuple)
    breaking_notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def engine_changed(self) -> bool:
        return self.engine_current != self.engine_proposed

    @property
    def au_changed(self) -> bool:
        return self.au_current != self.au_proposed

    @property
    def is_empty(self) -> bool:
        return not self.edits and not self.engine_changed and not self.au_changed


# --------------------------------------------------------------------------- #
# version arithmetic
# --------------------------------------------------------------------------- #


def _parse_semver(value: str) -> tuple[int, int, int]:
    match = _SEMVER.match(value)
    if match is None:
        raise CascadeError(f"not a semantic version: {value!r}")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def _bump(current: str, level: str) -> str:
    if level not in _BUMP_LEVELS:
        raise CascadeError(f"unknown bump level: {level!r}")
    major, minor, patch = _parse_semver(current)
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def _bump_level(current: str, proposed: str) -> str:
    """Classify the ``current`` -> ``proposed`` change as none/patch/minor/major."""
    if current == proposed:
        return "none"
    current_parts = _parse_semver(current)
    proposed_parts = _parse_semver(proposed)
    for level, index in (("major", 0), ("minor", 1), ("patch", 2)):
        if current_parts[index] != proposed_parts[index]:
            return level
    return "none"


def _resolve_target(
    *, current: str, explicit: str | None, bump: str | None, label: str
) -> str:
    if explicit is not None and bump is not None:
        raise CascadeError(f"pass at most one of --{label}-version/--{label}-bump")
    if explicit is not None:
        _parse_semver(explicit)
        return explicit
    if bump is not None:
        return _bump(current, bump)
    return current


def _major(version: str) -> int:
    return _parse_semver(version)[0]


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CascadeError(f"cannot read {path}: {exc}") from exc


def read_au_version(root: Path) -> str:
    """Read the authoritative agent-utilities version from ``_version.py``."""
    version_file = root / "agent_utilities" / "_version.py"
    source = _read_text(version_file)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise CascadeError(f"cannot parse {version_file}: {exc}") from exc
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
    if len(values) != 1:
        raise CascadeError(
            f"expected exactly one __version__ authority in {version_file}"
        )
    _parse_semver(values[0])
    return values[0]


def read_engine_version(root: Path) -> str:
    """Read the authoritative engine version from the compatibility matrix."""
    matrix_path = root / "deploy" / "release" / "compatibility-matrix.yml"
    matrix = _load_compatibility_matrix(matrix_path)
    try:
        pinned = str(matrix["components"]["epistemic-graph"]["version"])
    except (KeyError, TypeError) as exc:
        raise CascadeError(f"no epistemic-graph version in {matrix_path}") from exc
    version = pinned.removeprefix("==")
    _parse_semver(version)
    return version


def _load_compatibility_matrix(path: Path) -> dict:
    try:
        loaded = yaml.safe_load(_read_text(path))
    except yaml.YAMLError as exc:
        raise CascadeError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise CascadeError(f"{path} does not contain a mapping")
    return loaded


# --------------------------------------------------------------------------- #
# tier 2 — this repo
# --------------------------------------------------------------------------- #


def _engine_requirement_edit(
    root: Path, *, current: str, proposed: str, severity: str
) -> FileEdit:
    pyproject = root / "pyproject.toml"
    text = _read_text(pyproject)
    match = _ENGINE_REQUIREMENT.search(text)
    if match is None:
        raise CascadeError(f"no epistemic-graph[full] dependency found in {pyproject}")
    current_text = match.group(1)
    try:
        requirement = Requirement(current_text)
    except InvalidRequirement as exc:
        raise CascadeError(f"invalid requirement {current_text!r}: {exc}") from exc
    if requirement.extras != {"full"}:
        raise CascadeError(f"expected epistemic-graph[full], found {current_text!r}")
    ceiling_major = _major(proposed) + 1 if severity == "major" else _major(current) + 1
    proposed_text = f"epistemic-graph[full]>={proposed},<{ceiling_major}.0.0"
    return FileEdit(
        path=pyproject,
        kind="engine-dependency",
        current=current_text,
        proposed=proposed_text,
        tier=2,
    )


def _au_version_edit(root: Path, *, current: str, proposed: str) -> FileEdit:
    version_file = root / "agent_utilities" / "_version.py"
    text = _read_text(version_file)
    match = _VERSION_ASSIGNMENT.search(text)
    if match is None:
        raise CascadeError(f"no __version__ assignment found in {version_file}")
    current_text = match.group(0)
    proposed_text = f"{match.group('prefix')}{proposed}{match.group('suffix')}"
    if match.group("value") != current:
        raise CascadeError(
            f"{version_file} version {match.group('value')!r} does not match "
            f"expected current {current!r}"
        )
    return FileEdit(
        path=version_file,
        kind="au-version",
        current=current_text,
        proposed=proposed_text,
        tier=2,
    )


def _compatibility_matrix_edits(
    root: Path,
    *,
    engine_current: str,
    engine_proposed: str,
    au_current: str,
    au_proposed: str,
) -> list[FileEdit]:
    matrix_path = root / "deploy" / "release" / "compatibility-matrix.yml"
    text = _read_text(matrix_path)
    matrix = _load_compatibility_matrix(matrix_path)
    components = matrix.get("components")
    if not isinstance(components, dict):
        raise CascadeError(f"{matrix_path} has no components mapping")

    own_version_targets = {
        "epistemic-graph": (engine_current, engine_proposed),
        "agent-utilities": (au_current, au_proposed),
    }
    dependency_targets = {
        "epistemic-graph": (engine_current, engine_proposed),
        "agent-utilities": (au_current, au_proposed),
    }

    edits: list[FileEdit] = []
    for name in sorted(components):
        component = components[name]
        if not isinstance(component, dict):
            continue
        if name in own_version_targets:
            old, new = own_version_targets[name]
            if old != new:
                edits.append(
                    _matrix_component_version_edit(matrix_path, text, name, old, new)
                )
        depends_on = component.get("dependsOn")
        if not isinstance(depends_on, dict):
            continue
        for dependency in sorted(depends_on):
            if dependency not in dependency_targets:
                continue
            old, new = dependency_targets[dependency]
            if old == new:
                continue
            edits.append(
                _matrix_dependency_edit(matrix_path, text, name, dependency, old, new)
            )
    return edits


def _matrix_component_version_edit(
    path: Path, text: str, name: str, old: str, new: str
) -> FileEdit:
    current = f'  {name}:\n    version: "=={old}"'
    proposed = f'  {name}:\n    version: "=={new}"'
    if current not in text:
        raise CascadeError(f"expected component block {current!r} in {path}")
    return FileEdit(
        path=path,
        kind=f"compatibility-matrix:{name}.version",
        current=current,
        proposed=proposed,
        tier=2,
    )


def _matrix_dependency_edit(
    path: Path, text: str, component: str, dependency: str, old: str, new: str
) -> FileEdit:
    current = f'      {dependency}: "=={old}"'
    proposed = f'      {dependency}: "=={new}"'
    if current not in text:
        raise CascadeError(
            f"expected dependsOn.{dependency} == {old!r} for {component} in {path}"
        )
    return FileEdit(
        path=path,
        kind=f"compatibility-matrix:{component}.dependsOn.{dependency}",
        current=current,
        proposed=proposed,
        tier=2,
    )


# --------------------------------------------------------------------------- #
# tier 3 — providers
# --------------------------------------------------------------------------- #


def _discover_providers(providers_root: Path) -> list[Path]:
    if not providers_root.is_dir():
        raise CascadeError(f"providers root does not exist: {providers_root}")
    providers = [
        candidate / "pyproject.toml"
        for candidate in providers_root.iterdir()
        if candidate.is_dir() and (candidate / "pyproject.toml").is_file()
    ]
    return sorted(providers, key=lambda path: path.parent.name)


def _split_requirement(requirement_text: str) -> tuple[str | None, str]:
    """Split ``agent-utilities[extras]>=..`` into (extras text or None, specifier)."""
    match = re.match(r"^agent-utilities(\[[^\]]*\])?(.*)$", requirement_text)
    if match is None:
        raise CascadeError(f"not an agent-utilities requirement: {requirement_text!r}")
    return match.group(1), match.group(2)


def _provider_edits(pyproject: Path, *, au_target_major: int) -> list[FileEdit]:
    text = _read_text(pyproject)
    edits: list[FileEdit] = []
    for match in _PROVIDER_REQUIREMENT.finditer(text):
        requirement_text = match.group(1)
        try:
            Requirement(requirement_text)
        except InvalidRequirement:
            continue
        extras, specifier = _split_requirement(requirement_text)
        extras_text = extras or ""
        has_ceiling = "<" in specifier
        proposed_text = (
            f"agent-utilities{extras_text}>={au_target_major}.0.0,"
            f"<{au_target_major + 1}.0.0"
        )
        if proposed_text == requirement_text:
            continue
        kind = "missing-ceiling" if not has_ceiling else "provider-major-ceiling"
        edits.append(
            FileEdit(
                path=pyproject,
                kind=kind,
                current=requirement_text,
                proposed=proposed_text,
                tier=3,
            )
        )
    return edits


def _plan_provider_edits(
    providers_root: Path, *, au_severity: str, au_target_major: int
) -> list[FileEdit]:
    edits: list[FileEdit] = []
    for pyproject in _discover_providers(providers_root):
        for edit in _provider_edits(pyproject, au_target_major=au_target_major):
            if edit.kind == "missing-ceiling" or au_severity == "major":
                edits.append(edit)
    return edits


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def plan_cascade(
    *,
    root: Path,
    providers_root: Path,
    engine_version: str | None = None,
    engine_bump: str | None = None,
    au_version: str | None = None,
    au_bump: str | None = None,
) -> CascadePlan:
    """Compute the full cascade plan. Pure — reads files, writes nothing."""
    root = Path(root)
    providers_root = Path(providers_root)

    engine_current = read_engine_version(root)
    au_current = read_au_version(root)

    engine_proposed = _resolve_target(
        current=engine_current,
        explicit=engine_version,
        bump=engine_bump,
        label="engine",
    )
    au_proposed = _resolve_target(
        current=au_current,
        explicit=au_version,
        bump=au_bump,
        label="agent-utilities",
    )

    engine_severity = _bump_level(engine_current, engine_proposed)
    au_severity = _bump_level(au_current, au_proposed)
    severity = max(
        (engine_severity, au_severity), key=lambda level: _SEVERITY_RANK[level]
    )

    edits: list[FileEdit] = []
    breaking_notes: list[str] = []

    if engine_severity != "none":
        edits.append(
            _engine_requirement_edit(
                root,
                current=engine_current,
                proposed=engine_proposed,
                severity=engine_severity,
            )
        )
        if engine_severity == "major":
            breaking_notes.append(_ENGINE_DRIFTED_NOTE)

    if au_severity != "none":
        edits.append(_au_version_edit(root, current=au_current, proposed=au_proposed))

    if engine_severity != "none" or au_severity != "none":
        edits.extend(
            _compatibility_matrix_edits(
                root,
                engine_current=engine_current,
                engine_proposed=engine_proposed,
                au_current=au_current,
                au_proposed=au_proposed,
            )
        )

    if au_severity != "none":
        provider_edits = _plan_provider_edits(
            providers_root,
            au_severity=au_severity,
            au_target_major=_major(au_proposed),
        )
        edits.extend(provider_edits)
        if au_severity == "major" and any(
            edit.kind == "provider-major-ceiling" for edit in provider_edits
        ):
            breaking_notes.append(_DRIFTED_PROVIDERS_NOTE)

    return CascadePlan(
        engine_current=engine_current,
        engine_proposed=engine_proposed,
        au_current=au_current,
        au_proposed=au_proposed,
        severity=severity,
        edits=tuple(edits),
        breaking_notes=tuple(breaking_notes),
    )


# --------------------------------------------------------------------------- #
# rendering + apply
# --------------------------------------------------------------------------- #


def render_plan(plan: CascadePlan) -> str:
    """Deterministic, human-readable dry-run rendering of ``plan``."""
    lines = [
        f"version cascade: severity={plan.severity}",
        f"  tier 1 epistemic-graph: {plan.engine_current} -> {plan.engine_proposed}",
        f"  tier 2 agent-utilities: {plan.au_current} -> {plan.au_proposed}",
    ]
    if not plan.edits:
        lines.append("no edits required")
    for tier, label in ((2, "tier 2 edits"), (3, "tier 3 edits")):
        tier_edits = [edit for edit in plan.edits if edit.tier == tier]
        if not tier_edits:
            continue
        lines.append(f"{label} ({len(tier_edits)}):")
        for edit in tier_edits:
            lines.append(f"  {edit.path}: {edit.current}  ->  {edit.proposed}")
    if plan.breaking_notes:
        lines.append("breaking notes:")
        for note in plan.breaking_notes:
            lines.append(f"  - {note}")
    return "\n".join(lines) + "\n"


def _plan_to_json(plan: CascadePlan) -> dict:
    payload = dataclasses.asdict(plan)
    for edit in payload["edits"]:
        edit["path"] = str(edit["path"])
    return payload


def apply_plan(plan: CascadePlan, *, root: Path, providers_root: Path) -> list[Path]:
    """Write every edit in ``plan``. Never touches git. Fails closed."""
    del root, providers_root  # paths are already resolved on each FileEdit
    written: list[Path] = []
    edits_by_path: dict[Path, list[FileEdit]] = {}
    for edit in plan.edits:
        edits_by_path.setdefault(edit.path, []).append(edit)

    for path in sorted(edits_by_path, key=str):
        text = _read_text(path)
        for edit in edits_by_path[path]:
            if edit.current not in text:
                raise CascadeError(
                    f"{path} no longer contains the expected text for a "
                    f"{edit.kind} edit; refusing to apply (concurrent modification?)"
                )
            text = text.replace(edit.current, edit.proposed, 1)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="version-cascade",
        description=(
            "Plan (and optionally apply) the epistemic-graph -> agent-utilities -> "
            "provider version cascade."
        ),
    )
    engine_group = parser.add_mutually_exclusive_group()
    engine_group.add_argument("--engine-version")
    engine_group.add_argument("--engine-bump", choices=_BUMP_LEVELS)
    au_group = parser.add_mutually_exclusive_group()
    au_group.add_argument("--agent-utilities-version")
    au_group.add_argument("--agent-utilities-bump", choices=_BUMP_LEVELS)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="agent-utilities repository root (default: two parents up from here)",
    )
    parser.add_argument(
        "--providers-root",
        type=Path,
        default=None,
        help="agent-packages/agents root (default: <root>/../agents)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the planned edits (default is a dry run)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the plan as JSON instead of text",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if (
        args.engine_version is None
        and args.engine_bump is None
        and args.agent_utilities_version is None
        and args.agent_utilities_bump is None
    ):
        print(
            "no change requested: pass an engine and/or agent-utilities target",
            file=sys.stderr,
        )
        return 2

    root = args.root.resolve()
    providers_root = (
        args.providers_root.resolve()
        if args.providers_root is not None
        else (root.parent / "agents").resolve()
    )

    try:
        plan = plan_cascade(
            root=root,
            providers_root=providers_root,
            engine_version=args.engine_version,
            engine_bump=args.engine_bump,
            au_version=args.agent_utilities_version,
            au_bump=args.agent_utilities_bump,
        )
    except CascadeError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2

    if plan.is_empty:
        print("empty plan: requested target already matches the current version")
        if args.json:
            print(json.dumps(_plan_to_json(plan), indent=2, sort_keys=True))
        else:
            print(render_plan(plan))
        return 1

    if args.json:
        print(json.dumps(_plan_to_json(plan), indent=2, sort_keys=True))
    else:
        print(render_plan(plan), end="")

    if args.apply:
        written = apply_plan(plan, root=root, providers_root=providers_root)
        print(f"applied {len(written)} file(s)")
    else:
        print("DRY RUN — no files written")

    return 0


if __name__ == "__main__":
    sys.exit(main())
