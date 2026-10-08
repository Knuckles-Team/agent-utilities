#!/usr/bin/env python3
"""Upload only files admitted by the pinned pipelines publication contract.

The workflow owns release identity, installed-wheel proof and approval. Pipelines
owns index parsing, staged-byte identity and retry policy; this caller only wires
its three phases to the existing pinned uv uploader.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CONTRACT = ".pipeline-contract/.github/actions/wheel-readiness/publication.py"


def publish(version: str, manifest: Path) -> None:
    arguments = [
        "--directory",
        "dist",
        "--package",
        "agent-utilities",
        "--version",
        version,
        "--manifest",
        str(manifest),
    ]

    def check(mode: str) -> str:
        return subprocess.run(
            [sys.executable, "-I", CONTRACT, mode, *arguments],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout

    check("preflight")
    missing = json.loads(check("missing"))
    if missing:
        subprocess.run(
            [
                "uv",
                "publish",
                "--publish-url",
                "https://upload.pypi.org/legacy/",
                *[str(Path("dist") / name) for name in missing],
            ],
            check=True,
        )
    check("postverify")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    publish(args.version, args.manifest)


if __name__ == "__main__":
    main()
