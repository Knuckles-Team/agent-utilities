#!/usr/bin/env python3
"""Keep public Pages download files byte-identical to their source authority."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    ("scripts/install.sh", "pages/install.sh"),
    ("scripts/install.ps1", "pages/install.ps1"),
    ("genesis.yaml", "pages/genesis.yaml"),
)


def sync(root: Path, *, write: bool) -> list[str]:
    drift: list[str] = []
    for source_name, public_name in PUBLIC_FILES:
        source = root / source_name
        public = root / public_name
        expected = source.read_bytes()
        if write:
            public.parent.mkdir(parents=True, exist_ok=True)
            public.write_bytes(expected)
        elif not public.is_file() or public.read_bytes() != expected:
            drift.append(f"{public_name} differs from {source_name}")
    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        drift = sync(args.root, write=args.write)
    except OSError as exc:
        print(f"Pages public files: CANNOT RUN: {exc}", file=sys.stderr)
        return 2
    for item in drift:
        print(f"Pages public files: {item}", file=sys.stderr)
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
