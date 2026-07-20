#!/usr/bin/env python3
"""Transactionally pin the authoritative connector fleet into ``ontology.lock``.

The explicit repository-manager workspace is the sole provider membership authority.
Every configured provider checkout and the required in-package native manifest are
verified and compiled before one atomic lockfile replacement. Stale
``agents/*/connector_manifest.yml`` entries are removed only in that successful
replacement; any missing, extra, malformed, or untrusted candidate leaves the original
lock bytes unchanged.

Usage:
  python3 scripts/update_ontology_lock.py --agents-root <path> --workspace <path> \
      --native-manifest <path> [--lock-path <path>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.integrations.connector_source_attestation import (  # noqa: E402
    source_attestation_violations,
)
from agent_utilities.knowledge_graph.ontology import ontology_integrity  # noqa: E402
from agent_utilities.knowledge_graph.ontology.connector_manifest import (  # noqa: E402
    ConnectorManifest,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (  # noqa: E402
    compile_manifest,
    export_manifest_ttl,
)

_PROVIDER_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PROVIDER_LOCK_KEY = re.compile(
    r"^agents/[a-z0-9][a-z0-9._-]*/connector_manifest\.yml$"
)
_NATIVE_CONNECTOR = "native-source-connectors"
_NATIVE_ARTIFACT_KEY = f"agents/{_NATIVE_CONNECTOR}/connector_manifest.yml"


def _configured_provider_names(workspace_path: Path) -> tuple[str, ...]:
    workspace = yaml.safe_load(workspace_path.read_text(encoding="utf-8")) or {}
    try:
        repositories = workspace["subdirectories"]["agent-packages"]["subdirectories"][
            "agents"
        ]["repositories"]
    except (KeyError, TypeError) as exc:
        raise ValueError("workspace is missing the provider repository branch") from exc
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("provider repository membership must be a non-empty list")

    providers: list[str] = []
    for entry in repositories:
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            raise ValueError("every provider repository needs a URL")
        name = entry["url"].rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
        if not _PROVIDER_NAME.fullmatch(name) or name == _NATIVE_CONNECTOR:
            raise ValueError("provider URL has an invalid repository name")
        providers.append(name)
    if len(providers) != len(set(providers)):
        raise ValueError("workspace contains duplicate provider repositories")
    return tuple(providers)


def _load(path: Path) -> ConnectorManifest:
    return ConnectorManifest.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def _certification_metadata(
    repo: Path,
    manifest: ConnectorManifest,
    *,
    trusted_public_keys: tuple[str, ...],
) -> dict[str, str]:
    matches = sorted(repo.glob("*/ontology/certification.json"))
    if len(matches) != 1:
        return {}
    path = matches[0]
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    if source_attestation_violations(document, manifest):
        raise ValueError("connector source attestation is invalid")
    certification_hash = ontology_integrity.canonical_signed_document_hash(document)
    if not ontology_integrity.verify_release_signature(
        certification_hash,
        document.get("signature"),
        signer_id=document.get("signer"),
        algorithm=document.get("signature_algorithm"),
        public_key=document.get("signing_public_key"),
        trusted_public_keys=trusted_public_keys,
    ):
        raise ValueError("certification release signature is invalid")
    return {
        "certification_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "certification_hash": certification_hash,
        "certification_signer": str(document.get("signer") or ""),
        "certification_signature_algorithm": str(
            document.get("signature_algorithm") or ""
        ),
        "certification_signing_public_key": str(
            document.get("signing_public_key") or ""
        ),
        "certification_signature": str(document.get("signature") or ""),
    }


def _trusted_public_keys() -> tuple[str, ...]:
    keys = list(ontology_integrity.release_trusted_public_keys())
    try:
        runtime_signer = ontology_integrity.ReleaseSigner.from_runtime()
    except ontology_integrity.ReleaseSigningError:
        runtime_signer = None
    if runtime_signer is not None:
        keys.append(runtime_signer.public_key)
    trusted = tuple(dict.fromkeys(keys))
    if not trusted:
        raise ValueError(
            "release pin update requires an explicit signer or trusted public key"
        )
    return trusted


def _lock_entry(
    manifest_path: Path,
    *,
    expected_connector: str,
    certification_repo: Path | None,
    trusted_public_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Verify and compile one candidate without mutating the release lock."""

    manifest = _load(manifest_path)
    if manifest.connector != expected_connector:
        raise ValueError("manifest connector identity differs")
    manifest_hash = ontology_integrity.canonical_manifest_hash(manifest)
    if not ontology_integrity.verify_release_signature(
        manifest_hash,
        manifest.provenance.signature,
        signer_id=manifest.provenance.signer,
        algorithm=manifest.provenance.signature_algorithm,
        public_key=manifest.provenance.signing_public_key,
        trusted_public_keys=trusted_public_keys,
    ):
        raise ValueError("manifest release signature is invalid")

    spec = compile_manifest(manifest)
    ttl = export_manifest_ttl(spec, source=manifest.resolved_ontology_source)
    import rdflib

    graph = rdflib.Graph()
    graph.parse(data=ttl, format="turtle")
    digest, triple_count = ontology_integrity.canonical_hash(graph)
    certification = (
        _certification_metadata(
            certification_repo,
            manifest,
            trusted_public_keys=trusted_public_keys,
        )
        if certification_repo is not None
        else {}
    )
    return {
        "hash": digest,
        "algorithm": "urdna2015-sha256",
        "triple_count": triple_count,
        "manifest_hash": manifest_hash,
        "signer": manifest.provenance.signer,
        "signature": manifest.provenance.signature,
        "signature_algorithm": manifest.provenance.signature_algorithm,
        "signing_public_key": manifest.provenance.signing_public_key,
        **certification,
    }


def _regular_contained_manifest(root: Path, connector: str) -> Path:
    repo = root / connector
    manifest = repo / "connector_manifest.yml"
    try:
        root_metadata = root.lstat()
        repo_metadata = repo.lstat()
        manifest_metadata = manifest.lstat()
        if (
            stat.S_ISLNK(root_metadata.st_mode)
            or not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_ISLNK(repo_metadata.st_mode)
            or not stat.S_ISDIR(repo_metadata.st_mode)
            or stat.S_ISLNK(manifest_metadata.st_mode)
            or not stat.S_ISREG(manifest_metadata.st_mode)
        ):
            raise ValueError
        resolved_root = root.resolve(strict=True)
        repo.resolve(strict=True).relative_to(resolved_root)
        manifest.resolve(strict=True).relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("provider checkout is not a regular contained bundle") from exc
    return manifest


def _owned_provider_names(agents_root: Path) -> set[str]:
    owned: set[str] = set()
    for candidate in agents_root.iterdir():
        manifest = candidate / "connector_manifest.yml"
        try:
            manifest.lstat()
        except FileNotFoundError:
            continue
        owned.add(candidate.name)
    return owned


def _save_lock_atomic(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Persist stable lock bytes through one same-directory atomic replacement."""

    parent = path.parent
    parent_descriptor = -1
    temporary_name = ""
    try:
        parent_metadata = parent.lstat()
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
            parent_metadata.st_mode
        ):
            raise ValueError
        parent.resolve(strict=True)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        parent_descriptor = os.open(parent, directory_flags)
        opened_parent = os.fstat(parent_descriptor)
        if (
            opened_parent.st_dev != parent_metadata.st_dev
            or opened_parent.st_ino != parent_metadata.st_ino
        ):
            raise ValueError
    except (OSError, RuntimeError, ValueError) as exc:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        raise ValueError("ontology lock parent is invalid") from exc

    try:
        try:
            existing_metadata = os.stat(
                path.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            mode = 0o644
        else:
            if stat.S_ISLNK(existing_metadata.st_mode) or not stat.S_ISREG(
                existing_metadata.st_mode
            ):
                raise ValueError("ontology lock is not a regular file")
            mode = stat.S_IMODE(existing_metadata.st_mode)

        ordered = {key: entries[key] for key in sorted(entries)}
        payload = yaml.safe_dump(
            ordered, sort_keys=True, default_flow_style=False
        ).encode("utf-8")
        temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_descriptor)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short ontology lock write")
                view = view[written:]
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = ""
        os.fsync(parent_descriptor)
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents-root", type=Path, required=True)
    ap.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="authoritative repository-manager workspace membership",
    )
    ap.add_argument(
        "--lock-path",
        type=Path,
        default=ROOT / "agent_utilities" / "knowledge_graph" / "ontology.lock",
    )
    ap.add_argument(
        "--native-manifest",
        type=Path,
        required=True,
        help="required in-package native connector manifest",
    )
    args = ap.parse_args()

    try:
        configured = set(_configured_provider_names(args.workspace))
        owned = _owned_provider_names(args.agents_root)
    except Exception:  # noqa: BLE001 - privacy-safe aggregate only
        print("release pin update could not resolve authoritative provider membership")
        return 1
    if configured != owned:
        print("release pin update provider membership differs from the workspace")
        return 1

    try:
        trusted_public_keys = _trusted_public_keys()
    except Exception:  # noqa: BLE001 - privacy-safe aggregate only
        print("release pin update has no valid trusted signing key")
        return 1

    pins: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for connector in sorted(configured):
        try:
            manifest_path = _regular_contained_manifest(args.agents_root, connector)
            pins[f"agents/{connector}/connector_manifest.yml"] = _lock_entry(
                manifest_path,
                expected_connector=connector,
                certification_repo=manifest_path.parent,
                trusted_public_keys=trusted_public_keys,
            )
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{connector}/connector_manifest.yml: {type(exc).__name__}")
    try:
        native_metadata = args.native_manifest.lstat()
        if stat.S_ISLNK(native_metadata.st_mode) or not stat.S_ISREG(
            native_metadata.st_mode
        ):
            raise ValueError
        pins[_NATIVE_ARTIFACT_KEY] = _lock_entry(
            args.native_manifest,
            expected_connector=_NATIVE_CONNECTOR,
            certification_repo=None,
            trusted_public_keys=trusted_public_keys,
        )
    except Exception as exc:  # noqa: BLE001
        failed.append(f"native connector manifest: {type(exc).__name__}")

    if failed:
        print(f"FAILED to compile {len(failed)} manifest(s):")
        for failure in failed:
            print(f"  ✗ {failure}")
        return 1

    try:
        current = ontology_integrity.load_lock(args.lock_path)
        next_lock = {
            key: value
            for key, value in current.items()
            if not _PROVIDER_LOCK_KEY.fullmatch(str(key))
        }
        next_lock.update(pins)
        _save_lock_atomic(args.lock_path, next_lock)
    except Exception:  # noqa: BLE001 - no candidate/path detail leaves this boundary
        print("release pin update could not publish ontology.lock")
        return 1

    print(f"pinned {len(pins)} artifact(s) into ontology.lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
