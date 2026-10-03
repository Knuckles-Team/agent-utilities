#!/usr/bin/env python3
"""Bind the test engine to the workflow's official artifact and wheel digest."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
from urllib.parse import parse_qs, urldefrag


def verify_artifact(
    path: Path, artifact_id: int, name: str, run_id: int, source: str
) -> None:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    run = artifact.get("workflow_run") or {}
    if (
        artifact.get("id") != artifact_id
        or artifact.get("name") != name
        or artifact.get("expired") is not False
        or run.get("id") != run_id
        or run.get("head_sha") != source
    ):
        raise ValueError("official artifact identity, source, or expiry mismatch")


def verify_wheel(path: Path, filename: str, digest: str) -> None:
    if path.name != filename or path.is_symlink() or not path.is_file():
        raise ValueError("expected regular wheel file is missing or misnamed")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError("test engine wheel sha256 mismatch")


def _receipt_hashes(archive: dict, fragment: str, digest: str) -> list[str | None]:
    """Collect archive hashes and reject a conflicting URL-fragment digest."""
    recorded = [archive.get("hashes", {}).get("sha256")]
    if "hash" in archive:
        recorded.append(archive["hash"].removeprefix("sha256="))
    if fragment:
        if parse_qs(fragment) != {"sha256": [digest]}:
            raise ValueError("installed engine URL has a mismatched digest")
        recorded.append(digest)
    return recorded


def _verify_installed_origin(
    distribution: importlib.metadata.Distribution, path: Path, digest: str
) -> None:
    """Require the exact wheel URL and consistent recorded SHA-256 evidence."""
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    url, fragment = urldefrag(direct.get("url", ""))
    recorded = _receipt_hashes(direct.get("archive_info", {}), fragment, digest)
    if (
        url != path.resolve().as_uri()
        or digest not in recorded
        or any(value is not None and value != digest for value in recorded)
    ):
        raise ValueError("installed engine is not from the verified wheel")


def verify_installed(path: Path, digest: str) -> None:
    verify_wheel(path, path.name, digest)
    distribution = importlib.metadata.distribution("epistemic-graph")
    _verify_installed_origin(distribution, path, digest)
    for name in ("epistemic_graph", "epistemic_graph.numeric"):
        module = importlib.import_module(name)
        location = Path(module.__file__).resolve()
        owned = {
            Path(distribution.locate_file(f)).resolve()
            for f in distribution.files or ()
        }
        if location not in owned:
            raise ValueError(f"{name} is shadowed outside the installed distribution")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    artifact = commands.add_parser("artifact")
    artifact.add_argument("path", type=Path)
    artifact.add_argument("--id", type=int, required=True)
    artifact.add_argument("--name", required=True)
    artifact.add_argument("--run", type=int, required=True)
    artifact.add_argument("--source", required=True)
    for mode in ("wheel", "installed"):
        command = commands.add_parser(mode)
        command.add_argument("path", type=Path)
        command.add_argument("--sha256", required=True)
        if mode == "wheel":
            command.add_argument("--filename", required=True)
    args = parser.parse_args()
    if args.mode == "artifact":
        verify_artifact(args.path, args.id, args.name, args.run, args.source)
    elif args.mode == "wheel":
        verify_wheel(args.path, args.filename, args.sha256)
    else:
        verify_installed(args.path, args.sha256)


if __name__ == "__main__":
    main()
