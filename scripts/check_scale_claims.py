#!/usr/bin/env python3
"""Check the AU scaling claim register against repository source.

The register is intentionally small and dependency-free.  It prevents a
documentation status from outliving the AU source fragment it describes while
keeping deployment evidence separate from source and unit evidence.  This is
an AU-local checker: it does not import or duplicate epistemic-graph claims.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_REGISTER = Path("docs/scaling/scale_claims.md")
STATUSES = frozenset(
    {"DESIGNED", "IMPLEMENTED", "UNIT-PROVEN", "LAB-PROVEN", "LIVE", "1M-CERTIFIED"}
)
_ROW = re.compile(r"^\|.*\|$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == "`" and value[-1] == "`":
        return value[1:-1]
    return value


def _repo_relative(path: Path, repo_root: Path) -> tuple[Path | None, str | None]:
    if path.is_absolute():
        return None, "source path must be repository-relative"
    candidate = (repo_root / path).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        return None, "source path escapes the repository"
    return candidate, None


def check_claim_register(
    register_path: Path, repo_root: Path | None = None
) -> list[str]:
    """Return deterministic validation errors for one claim register."""

    register_path = Path(register_path)
    repo_root = (repo_root or register_path.parent.parent.parent).resolve()
    errors: list[str] = []
    try:
        text = register_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{register_path}: cannot read register: {exc}"]

    seen_ids: set[str] = set()
    rows = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        if not _ROW.match(line) or line.lstrip().startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip()[1:-1].split("|")]
        if (
            not cells
            or cells[0] == "ID"
            or all(set(cell) <= {"-", ":"} for cell in cells)
        ):
            continue
        if len(cells) != 5:
            errors.append(
                f"{register_path}:{line_no}: claim row must have five columns"
            )
            continue
        rows += 1
        claim_id, status, source, fragment, evidence = (
            _unquote(cell) for cell in cells
        )
        prefix = f"{register_path}:{line_no}"
        if not claim_id.startswith("AU-SCALE-"):
            errors.append(f"{prefix}: claim ID must start with AU-SCALE-")
        if claim_id in seen_ids:
            errors.append(f"{prefix}: duplicate claim ID {claim_id!r}")
        seen_ids.add(claim_id)
        if status not in STATUSES:
            errors.append(f"{prefix}: unsupported status {status!r}")
        if "#" not in source:
            errors.append(f"{prefix}: source must be path#anchor")
            continue
        source_name, anchor = source.split("#", 1)
        source_path, path_error = _repo_relative(Path(source_name), repo_root)
        if path_error:
            errors.append(f"{prefix}: {path_error}")
            continue
        assert source_path is not None
        if "epistemic-graph" in Path(source_name).parts:
            errors.append(f"{prefix}: AU claim cannot use an epistemic-graph source")
        try:
            source_text = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{prefix}: cannot read source {source_name!r}: {exc}")
            continue
        if anchor not in source_text:
            errors.append(
                f"{prefix}: source anchor {anchor!r} is absent from {source_name}"
            )
        if fragment not in source_text:
            errors.append(f"{prefix}: required fragment is absent from {source_name}")
        if status in {"LIVE", "1M-CERTIFIED"} and (
            "reports/" not in evidence or "not" in evidence.lower()
        ):
            errors.append(
                f"{prefix}: {status} requires a positive reports/ evidence reference"
            )
    if rows == 0:
        errors.append(f"{register_path}: no claim rows found")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("register", nargs="?", type=Path, default=DEFAULT_REGISTER)
    args = parser.parse_args(argv)
    errors = check_claim_register(args.register)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: {args.register} source anchors and status rules are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
