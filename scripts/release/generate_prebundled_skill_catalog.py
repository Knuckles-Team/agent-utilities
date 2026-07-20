#!/usr/bin/env python3
"""Generate the deterministic exact prebundled-skill release catalog."""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.release_catalogs import (  # noqa: E402
    ReleaseCatalogError,
    content_digest,
    prebundled_skill_catalog_bytes,
    write_catalog,
)
from agent_utilities.skills import BUNDLED_SKILLS  # noqa: E402

DEFAULT_SKILLS_ROOT = ROOT / "agent_utilities" / "skills"
DEFAULT_MATRIX = ROOT / "deploy" / "release" / "compatibility-matrix.yml"
DEFAULT_OUTPUT = ROOT / "deploy" / "release" / "prebundled-skills.catalog.json"


def _matrix_expected_entries(matrix_path: Path) -> int:
    try:
        value: Any = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
        component = value["components"]["prebundled-skills"]
        expected = component["exactEntries"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ReleaseCatalogError("prebundled_skill_matrix_invalid") from exc
    if type(expected) is not int or expected != len(BUNDLED_SKILLS):
        raise ReleaseCatalogError("prebundled_skill_matrix_membership_invalid")
    return expected


def render_catalog(*, skills_root: Path, matrix_path: Path) -> bytes:
    """Return catalog bytes after binding the compatibility matrix count."""

    _matrix_expected_entries(matrix_path)
    return prebundled_skill_catalog_bytes(skills_root)


def _regular_bytes(path: Path) -> bytes | None:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            return None
        return path.read_bytes()
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate-prebundled-skill-catalog")
    parser.add_argument("--skills-root", type=Path, default=DEFAULT_SKILLS_ROOT)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = render_catalog(
            skills_root=args.skills_root,
            matrix_path=args.matrix,
        )
        if args.check:
            if _regular_bytes(args.output) != payload:
                print(json.dumps({"error": "CatalogDrift", "ok": False}, sort_keys=True))
                return 1
        else:
            write_catalog(args.output, payload, prefix=".skill-catalog-")
    except (ReleaseCatalogError, OSError, UnicodeError, ValueError):
        print(json.dumps({"error": "CatalogInputInvalid", "ok": False}, sort_keys=True))
        return 1

    print(
        json.dumps(
            {
                "digest": content_digest(payload),
                "entries": len(BUNDLED_SKILLS),
                "ok": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
