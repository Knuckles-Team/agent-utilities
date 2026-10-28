#!/usr/bin/env python3
"""Compare tracked package files with their explicit owner dispositions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

DESTINATIONS = {"arbora", "medulla", "scion-sdk", "repository-manager"}


def check(root: Path, tracked: Iterable[str]) -> list[str]:
    """Check a supplied tracked-file set against the checked-in inventory."""
    manifest: Any = json.loads((root / "architecture/file-ownership.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != "au-file-owners/v1":
        raise ValueError("invalid file-ownership schema")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("empty file-ownership inventory")
    recorded: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "disposition",
            "owner",
        }:
            raise ValueError("invalid file-ownership entry")
        path, disposition, owner = entry["path"], entry["disposition"], entry["owner"]
        if not isinstance(disposition, str) or not isinstance(owner, str):
            raise ValueError("invalid file disposition or owner")
        if (
            not isinstance(path, str)
            or not path.startswith("agent_utilities/")
            or ".." in PurePosixPath(path).parts
            or str(PurePosixPath(path)) != path
            or path in recorded
        ):
            raise ValueError("invalid or duplicate file path")
        if not (
            (disposition == "keep" and owner == "meristem")
            or (disposition == "relocate" and owner in DESTINATIONS)
            or (disposition == "delete" and owner == "none")
        ):
            raise ValueError(f"undecided or invalid file owner: {path}")
        recorded.add(path)
    actual = set(tracked)
    if not actual or any(not path.startswith("agent_utilities/") for path in actual):
        raise ValueError("invalid tracked package files")
    return [
        *(f"unowned file: {path}" for path in sorted(actual - recorded)),
        *(f"stale file: {path}" for path in sorted(recorded - actual)),
    ]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", "agent_utilities"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        tracked = [os.fsdecode(path) for path in result.stdout.split(b"\0") if path]
        findings = check(root, tracked)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"boundary file ownership invalid: {exc}", file=sys.stderr)
        return 1
    for finding in findings:
        print(finding, file=sys.stderr)
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
