#!/usr/bin/env python3
"""Check current package directories against their recorded dispositions."""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath


def check(root: Path) -> list[str]:
    """Compare the first two package-directory levels with the inventory."""
    rows: dict[str, tuple[str, str]] = {}
    coverage = root / "specs/au-boundary-deconstruction/coverage.md"
    for line in coverage.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if (
            len(cells) != 3
            or not cells[0].startswith("`")
            or not cells[0].endswith("`")
        ):
            raise ValueError("invalid coverage row")
        path, disposition, destination = cells[0][1:-1], cells[1], cells[2]
        parts = PurePosixPath(path).parts
        if (
            not 2 <= len(parts) <= 3
            or parts[0] != "agent_utilities"
            or ".." in parts
            or str(PurePosixPath(path)) != path
            or path in rows
        ):
            raise ValueError(f"invalid or duplicate coverage path: {path}")
        if disposition not in {"keep", "delete", "relocate"}:
            raise ValueError(f"undecided disposition: {path}")
        if (
            (disposition == "keep" and destination != "meristem")
            or (
                disposition == "relocate"
                and destination
                not in {"arbora", "medulla", "scion-sdk", "repository-manager"}
            )
            or (disposition == "delete" and destination != "none")
        ):
            raise ValueError(f"invalid destination: {path}")
        rows[path] = (disposition, destination)

    package = root / "agent_utilities"
    if not package.is_dir():
        raise ValueError("package directory is missing")
    directories: set[str] = set()
    for first in package.iterdir():
        if (
            not first.is_dir()
            or first.name.startswith(".")
            or first.name == "__pycache__"
        ):
            continue
        directories.add(first.relative_to(root).as_posix())
        for second in first.iterdir():
            if (
                second.is_dir()
                and not second.name.startswith(".")
                and second.name != "__pycache__"
            ):
                directories.add(second.relative_to(root).as_posix())
    return [
        *(f"unlisted directory: {path}" for path in sorted(directories - rows.keys())),
        *(f"stale directory: {path}" for path in sorted(rows.keys() - directories)),
    ]


def main() -> int:
    try:
        findings = check(Path(__file__).resolve().parents[1])
    except (OSError, ValueError) as exc:
        print(f"boundary coverage invalid: {exc}", file=sys.stderr)
        return 1
    for finding in findings:
        print(finding, file=sys.stderr)
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
