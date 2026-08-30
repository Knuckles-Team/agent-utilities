"""Shared command-line arguments for exact-local release tooling."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_exact_local_release_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the required release identity and source paths to an exact CLI."""

    parser.add_argument("--release-id", required=True)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--promotion-evidence", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
