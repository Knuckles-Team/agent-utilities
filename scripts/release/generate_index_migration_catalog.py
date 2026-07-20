#!/usr/bin/env python3
"""Generate the deterministic current index-migration release catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.index_migrations import (  # noqa: E402
    index_migration_catalog,
)


def render_catalog() -> bytes:
    """Return canonical bytes suitable for hashing and catalog promotion."""

    return (
        json.dumps(index_migration_catalog(), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _write(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise ValueError("index migration catalog output symlinks are not accepted")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".index-catalog-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate-index-migration-catalog")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = render_catalog()
    if args.check:
        try:
            current = args.output.read_bytes()
        except OSError:
            current = b""
        if current != payload:
            print(json.dumps({"ok": False, "error": "CatalogDrift"}, sort_keys=True))
            return 1
    else:
        _write(args.output, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "entries": len(index_migration_catalog()["entries"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
