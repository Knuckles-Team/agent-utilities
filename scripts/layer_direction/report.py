"""Console reporting and command-line entrypoint."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from scripts.layer_direction.model import (
    CLASSIFICATION_EXCEPTIONS,
    LAYER_RULES,
    UNASSIGNED,
    BoundaryViolation,
    ScanIncomplete,
    Violation,
    longest_match,
)
from scripts.layer_direction.scan import scan

_ROOT = Path(__file__).resolve().parents[2]


def report(
    *,
    violations: list[Violation],
    unassigned_files: Counter,
    unassigned_edges: Counter,
    show_all: bool,
) -> None:
    """Print the real counts unconditionally so debt never goes quiet."""
    total = len(violations)
    print(f"layer-direction violations: {total} (report-only)")
    for kind in ("outward-import", "adapter-crosstalk"):
        report_kind(violations, kind)
    eager = sum(1 for violation in violations if violation.eager)
    print(
        f"  scope: {total} all-import statements, of which {eager} are eager "
        "(load-time). check-import-cycles measures the eager graph ONLY."
    )
    report_coverage(unassigned_files, unassigned_edges)
    if show_all:
        print("\nEvery violating import statement:\n")
        for violation in sorted(
            violations, key=lambda item: (item.kind, item.relpath, item.lineno)
        ):
            print(violation)


def report_kind(violations: list[Violation], kind: str) -> None:
    """Print one violation-kind summary."""
    pairs = Counter(item.pair for item in violations if item.kind == kind)
    print(
        f"  {kind:18s} {sum(pairs.values()):4d} import statements across "
        f"{len(pairs)} package pairs"
    )
    for pair, occurrences in pairs.most_common(10):
        print(f"      {occurrences:4d}  {pair}")
    if len(pairs) > 10:
        print(f"      ... {len(pairs) - 10} further pair(s)")


def report_coverage(unassigned_files: Counter, unassigned_edges: Counter) -> None:
    """Print unassigned areas and effective classification exceptions."""
    print(
        f"  coverage census: {len(unassigned_files)} module group(s) carry no "
        f"declared layer ({sum(unassigned_files.values())} files, "
        f"{sum(unassigned_edges.values())} cross-package imports NOT judged)."
    )
    covered = sorted({layer for _, layer in LAYER_RULES})
    print(f"      layers in use: {', '.join(covered)}")
    for prefix, effective in CLASSIFICATION_EXCEPTIONS:
        canonical = longest_match(prefix, LAYER_RULES)
        canonical_layer = canonical[1] if canonical else UNASSIGNED
        print(
            f"      classification exception: {prefix} -> {effective} "
            f"(canonical inventory: {canonical_layer})"
        )
    if unassigned_files:
        top = ", ".join(
            f"{group}({count})" for group, count in unassigned_files.most_common(12)
        )
        print(f"      unassigned: {top}")


def report_boundaries(violations: list[BoundaryViolation]) -> None:
    """Print the mandatory single-contract boundary verdict."""
    print(f"single-contract boundary violations: {len(violations)} (blocking)")
    for violation in sorted(
        violations, key=lambda item: (item.relpath, item.lineno, item.target)
    ):
        print(violation)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the intentionally small direction-gate CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Report AU's six-layer census and enforce its product dependency boundary."
        )
    )
    parser.add_argument(
        "package_root",
        nargs="?",
        default=None,
        help="Package directory to scan (default: agent_utilities).",
    )
    parser.add_argument(
        "--list", action="store_true", dest="show_all", help="Print every violation."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Return 1 for boundary reversal and 2 for an incomplete scan."""
    arguments = parse_args(argv)
    package_root = (
        Path(arguments.package_root).resolve()
        if arguments.package_root
        else _ROOT / "agent_utilities"
    )
    if not package_root.is_dir():
        print(
            f"check_layer_direction: CANNOT RUN: {package_root} not found",
            file=sys.stderr,
        )
        return 2
    try:
        violations, boundaries, unassigned_files, unassigned_edges = scan(package_root)
    except ScanIncomplete as exc:
        print(f"check_layer_direction: CANNOT RUN: {exc}", file=sys.stderr)
        return 2
    report(
        violations=violations,
        unassigned_files=unassigned_files,
        unassigned_edges=unassigned_edges,
        show_all=arguments.show_all,
    )
    report_boundaries(boundaries)
    if boundaries:
        print("FAIL: AU imports an implementation owned by EG, SDK, or GraphOS.")
        return 1
    print("PASS: accepted AU product dependency direction is intact.")
    return 0
