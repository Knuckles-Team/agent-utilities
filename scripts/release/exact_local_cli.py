"""Shared command-line arguments for exact-local release tooling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Keep callers that are executed directly bound to this checkout before they
# import the rest of the ``scripts.release`` package.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def exact_local_parser(*, prog: str, description: str) -> argparse.ArgumentParser:
    """Create an exact-local CLI parser with its shared release arguments."""

    parser = argparse.ArgumentParser(prog=prog, description=description)
    add_exact_local_release_arguments(parser)
    return parser


def add_exact_local_release_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the required release identity and source paths to an exact CLI."""

    parser.add_argument("--release-id", required=True)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--promotion-evidence", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
