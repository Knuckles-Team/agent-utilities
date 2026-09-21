"""Dependency-census scan orchestration."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from scripts.layer_direction.imports import ImportWalker, product_imports
from scripts.layer_direction.model import (
    UNASSIGNED,
    BoundaryViolation,
    ScanIncomplete,
    Violation,
    boundary_reason,
    classify,
    violation_kind,
)
from scripts.layer_direction.resolver import (
    known_modules,
    module_name,
    python_paths,
    targets,
)


@dataclass(frozen=True, slots=True)
class ImportContext:
    """Stable file-level inputs used for each import statement."""

    relative_path: str
    source_group: str
    source_layer: str
    package_name: str
    current_package: str
    known_modules: frozenset[str]


def import_findings(
    node: ast.Import | ast.ImportFrom,
    eager: bool,
    context: ImportContext,
) -> tuple[list[Violation], Counter]:
    """Classify the unique internal targets of one import statement."""
    violations: list[Violation] = []
    unassigned: Counter = Counter()
    resolved = targets(
        node,
        package_name=context.package_name,
        current_package=context.current_package,
        known_modules=context.known_modules,
    )
    for target_group, target_layer in unique_classifications(resolved):
        if target_group == context.source_group:
            continue
        if UNASSIGNED in (context.source_layer, target_layer):
            unassigned[f"{context.source_group} -> {target_group}"] += 1
            continue
        kind = violation_kind(
            src_group=context.source_group,
            src_layer=context.source_layer,
            tgt_group=target_group,
            tgt_layer=target_layer,
        )
        if kind:
            violations.append(
                Violation(
                    context.relative_path,
                    node.lineno,
                    context.source_group,
                    context.source_layer,
                    target_group,
                    target_layer,
                    kind,
                    eager,
                )
            )
    return violations, unassigned


def unique_classifications(target_names: list[str]) -> list[tuple[str, str]]:
    """Classify target names once each while retaining import order."""
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for target_name in target_names:
        classification = classify(target_name)
        if classification not in seen:
            seen.add(classification)
            result.append(classification)
    return result


def scan_path(
    path: Path, package_root: Path, discovered: frozenset[str]
) -> tuple[list[Violation], list[BoundaryViolation], Counter, tuple[str, str]]:
    """Scan one Python file and return findings plus source classification."""
    relative = path.relative_to(package_root)
    source_group, source_layer = classify(module_name(relative))
    try:
        tree = ast.parse(
            path.read_text(encoding="utf-8", errors="ignore"), filename=str(path)
        )
    except (OSError, SyntaxError) as exc:
        raise ScanIncomplete(f"could not parse {path}: {exc}") from exc
    walker = ImportWalker()
    walker.visit(tree)
    relative_path = f"{package_root.name}/{relative.as_posix()}"
    context = ImportContext(
        relative_path=relative_path,
        source_group=source_group,
        source_layer=source_layer,
        package_name=package_root.name,
        current_package=".".join(
            (package_root.name, *relative.with_suffix("").parts[:-1])
        ),
        known_modules=discovered,
    )
    violations: list[Violation] = []
    unassigned: Counter = Counter()
    for node, eager in walker.found:
        found, edges = import_findings(node, eager, context)
        violations.extend(found)
        unassigned.update(edges)
    boundaries = [
        BoundaryViolation(relative_path, imported.lineno, imported.target, reason)
        for imported in product_imports(tree)
        if (reason := boundary_reason(imported.target)) is not None
    ]
    return violations, boundaries, unassigned, (source_group, source_layer)


def scan(
    package_root: Path,
) -> tuple[list[Violation], list[BoundaryViolation], Counter, Counter]:
    """Return internal debt, blocking product-boundary findings, and coverage."""
    paths = python_paths(package_root)
    discovered = known_modules(paths, package_root)
    violations: list[Violation] = []
    boundaries: list[BoundaryViolation] = []
    unassigned_files: Counter = Counter()
    unassigned_edges: Counter = Counter()
    for path in paths:
        found, boundary_findings, edges, (source_group, source_layer) = scan_path(
            path, package_root, discovered
        )
        violations.extend(found)
        boundaries.extend(boundary_findings)
        unassigned_edges.update(edges)
        if source_layer == UNASSIGNED:
            unassigned_files[source_group] += 1
    return violations, boundaries, unassigned_files, unassigned_edges
