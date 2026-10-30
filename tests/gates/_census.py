"""Shared removal-census helpers for the AU boundary cut gates."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "meristem"


def assert_absent(*relative: str) -> None:
    """Every path (relative to ``meristem/``) is gone from the tree."""
    present = [rel for rel in relative if (PACKAGE / rel).exists()]
    assert present == []


def _scan_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".yml", ".toml"}:
                continue
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def references(tokens: tuple[str, ...], *, include_tests: bool = False) -> list[str]:
    """Files under ``meristem/`` (and non-census tests) naming a token."""
    roots = [PACKAGE]
    if include_tests:
        roots.append(ROOT / "tests")
    hits: list[str] = []
    for path in _scan_files(*roots):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("tests/gates/") or rel.startswith("tests/test_retired"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits.extend(f"{rel}: {token}" for token in tokens if token in text)
    return hits


def console_scripts() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return dict(data["project"].get("scripts", {}))
