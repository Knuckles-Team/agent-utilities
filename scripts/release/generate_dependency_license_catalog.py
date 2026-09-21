#!/usr/bin/env python3
"""Reconcile the curated dependency-license catalog with package metadata."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_backend  # noqa: E402
from agent_utilities import release_catalogs  # noqa: E402

DEFAULT_PYPROJECT = ROOT / "pyproject.toml"
DEFAULT_OUTPUT = ROOT / "agent_utilities" / "dependency-license-catalog.json"


def _requirement_names(pyproject_path: Path) -> set[str]:
    """Return the exact normalized dependency set declared by the project."""

    try:
        pyproject: dict[str, Any] = tomllib.loads(
            pyproject_path.read_text(encoding="utf-8")
        )
        project = pyproject["project"]
        declarations = list(project.get("dependencies") or ())
        for values in project.get("optional-dependencies", {}).values():
            declarations.extend(values)
        for values in pyproject.get("dependency-groups", {}).values():
            declarations.extend(values)
        names = {
            build_backend._normalized_name(Requirement(value).name)
            for value in declarations
        }
        names.add(build_backend._normalized_name(project["name"]))
    except (
        InvalidRequirement,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        tomllib.TOMLDecodeError,
    ) as exc:
        raise release_catalogs.ReleaseCatalogError(
            "dependency_license_metadata_invalid"
        ) from exc
    return names


def render_catalog(*, pyproject_path: Path, catalog_path: Path) -> bytes:
    """Prune stale entries without ever inventing an SPDX expression."""

    required = _requirement_names(pyproject_path)
    try:
        existing = build_backend._license_catalog(catalog_path)
    except (OSError, ValueError) as exc:
        raise release_catalogs.ReleaseCatalogError(
            "dependency_license_catalog_invalid"
        ) from exc
    missing = required - set(existing)
    if missing:
        raise release_catalogs.ReleaseCatalogError(
            "dependency_license_catalog_missing_entries"
        )
    payload = {
        "version": 1,
        "licenses": {name: existing[name] for name in sorted(required)},
    }
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate-dependency-license-catalog")
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = render_catalog(
            pyproject_path=args.pyproject,
            catalog_path=args.output,
        )
        if not release_catalogs.check_or_write_catalog(
            args.output,
            payload,
            check=args.check,
            prefix=".dependency-license-catalog-",
        ):
            print(json.dumps({"error": "CatalogDrift", "ok": False}, sort_keys=True))
            return 1
    except (release_catalogs.ReleaseCatalogError, OSError, ValueError):
        print(json.dumps({"error": "CatalogInputInvalid", "ok": False}, sort_keys=True))
        return 1

    entries = len(json.loads(payload)["licenses"])
    print(json.dumps({"entries": entries, "ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
