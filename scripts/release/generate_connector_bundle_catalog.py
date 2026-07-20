#!/usr/bin/env python3
"""Generate the deterministic exact connector-bundle release catalog."""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts"
for import_root in (ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from check_connector_capability_bundles import (  # noqa: E402
    _configured_provider_names,
    _is_regular_contained_file,
    _module,
    _provider_owned_names,
    check_one,
)

from agent_utilities.knowledge_graph.ontology.connector_manifest import (  # noqa: E402
    ConnectorManifest,
)
from agent_utilities.release_catalogs import (  # noqa: E402
    ReleaseCatalogError,
    canonical_json_bytes,
    canonical_value_digest,
    content_digest,
    write_catalog,
)
from agent_utilities.security.persistence_privacy import (  # noqa: E402
    PersistencePrivacyGuard,
)

DEFAULT_AGENTS_ROOT = ROOT.parent / "agents"
DEFAULT_WORKSPACE = (
    DEFAULT_AGENTS_ROOT
    / "repository-manager"
    / "repository_manager"
    / "workspace.yml"
)
DEFAULT_BUNDLED_ROOT = (
    ROOT
    / "agent_utilities"
    / "knowledge_graph"
    / "ontology"
    / "connector_manifests"
)
DEFAULT_LOCK_PATH = (
    ROOT / "agent_utilities" / "knowledge_graph" / "ontology.lock"
)
DEFAULT_MATRIX = ROOT / "deploy" / "release" / "compatibility-matrix.yml"
DEFAULT_OUTPUT = ROOT / "deploy" / "release" / "connector-bundles.catalog.json"


def _matrix_expected_entries(matrix_path: Path) -> int:
    try:
        value: Any = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
        expected = value["components"]["connector-bundles"]["exactEntries"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ReleaseCatalogError("connector_catalog_matrix_invalid") from exc
    if type(expected) is not int or expected != 65:
        raise ReleaseCatalogError("connector_catalog_matrix_membership_invalid")
    return expected


def _regular_bytes(root: Path, path: Path) -> bytes:
    if not _is_regular_contained_file(root, path):
        raise ReleaseCatalogError("connector_catalog_artifact_not_regular")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ReleaseCatalogError("connector_catalog_artifact_unreadable") from exc


def _entry(repo: Path) -> dict[str, Any]:
    manifest_path = repo / "connector_manifest.yml"
    try:
        module = _module(repo)
    except (OSError, ValueError) as exc:
        raise ReleaseCatalogError("connector_catalog_certification_not_unique") from exc
    certification_path = module / "ontology" / "certification.json"
    manifest_bytes = _regular_bytes(repo, manifest_path)
    certification_bytes = _regular_bytes(repo, certification_path)
    try:
        manifest = ConnectorManifest.model_validate(
            yaml.safe_load(manifest_bytes.decode("utf-8"))
        )
        certification = json.loads(certification_bytes.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise ReleaseCatalogError("connector_catalog_bundle_invalid") from exc
    if not isinstance(certification, dict):
        raise ReleaseCatalogError("connector_catalog_certification_invalid")
    artifacts = certification.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts or len(artifacts) > 4_096:
        raise ReleaseCatalogError("connector_catalog_artifact_ledger_invalid")

    artifact_entries: list[dict[str, str]] = []
    for name, digest in sorted(artifacts.items(), key=lambda item: str(item[0])):
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or not PurePosixPath(name).parts
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or "\\" in name
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ReleaseCatalogError("connector_catalog_artifact_ledger_invalid")
        artifact_entries.append({"name": name, "digest": f"sha256:{digest}"})

    manifest_digest = content_digest(manifest_bytes)
    certification_digest = content_digest(certification_bytes)
    source_preset_count = len(manifest.sync)
    if source_preset_count > 4_096:
        raise ReleaseCatalogError("connector_catalog_source_preset_count_exceeded")
    bundle_digest = canonical_value_digest(
        {
            "connector": repo.name,
            "manifestDigest": manifest_digest,
            "certificationDigest": certification_digest,
            "sourcePresetCount": source_preset_count,
            "artifacts": artifact_entries,
        }
    )
    return {
        "connector": repo.name,
        "manifestDigest": manifest_digest,
        "certificationDigest": certification_digest,
        "bundleDigest": bundle_digest,
        "artifactCount": len(artifact_entries),
        "sourcePresetCount": source_preset_count,
    }


def connector_bundle_catalog(
    *,
    agents_root: Path,
    workspace_path: Path,
    bundled_root: Path,
    lock_path: Path,
    matrix_path: Path,
) -> dict[str, Any]:
    """Validate and catalog the exact workspace-authoritative provider fleet."""

    expected_entries = _matrix_expected_entries(matrix_path)
    try:
        configured = tuple(sorted(_configured_provider_names(workspace_path)))
        provider_owned = _provider_owned_names(agents_root)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ReleaseCatalogError("connector_catalog_membership_unresolvable") from exc
    if (
        len(configured) != expected_entries
        or len(configured) != len(set(configured))
        or configured != provider_owned
    ):
        raise ReleaseCatalogError("connector_catalog_membership_not_exact")

    privacy = PersistencePrivacyGuard()
    entries: list[dict[str, Any]] = []
    failed = 0
    for name in configured:
        repo = agents_root / name
        try:
            violations = check_one(
                repo,
                bundled_root=bundled_root,
                lock_path=lock_path,
                privacy=privacy,
            )
            if violations:
                failed += 1
                continue
            entries.append(_entry(repo))
        except Exception:  # noqa: BLE001 - fail closed without leaking local details
            failed += 1
    if failed or len(entries) != expected_entries:
        raise ReleaseCatalogError("connector_catalog_bundle_validation_failed")

    return {
        "apiVersion": "graphos.io/v1",
        "kind": "ConnectorBundleCatalog",
        "catalogVersion": 1,
        "membershipDigest": canonical_value_digest(list(configured)),
        "entryCount": len(entries),
        "entries": entries,
    }


def render_catalog(
    *,
    agents_root: Path,
    workspace_path: Path,
    bundled_root: Path,
    lock_path: Path,
    matrix_path: Path,
) -> bytes:
    """Return the canonical retained connector catalog bytes."""

    return canonical_json_bytes(
        connector_bundle_catalog(
            agents_root=agents_root,
            workspace_path=workspace_path,
            bundled_root=bundled_root,
            lock_path=lock_path,
            matrix_path=matrix_path,
        )
    )


def _retained_bytes(path: Path) -> bytes | None:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            return None
        return path.read_bytes()
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate-connector-bundle-catalog")
    parser.add_argument("--agents-root", type=Path, default=DEFAULT_AGENTS_ROOT)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--bundled-root", type=Path, default=DEFAULT_BUNDLED_ROOT)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = render_catalog(
            agents_root=args.agents_root,
            workspace_path=args.workspace,
            bundled_root=args.bundled_root,
            lock_path=args.lock_path,
            matrix_path=args.matrix,
        )
        if args.check:
            if _retained_bytes(args.output) != payload:
                print(json.dumps({"error": "CatalogDrift", "ok": False}, sort_keys=True))
                return 1
        else:
            write_catalog(args.output, payload, prefix=".connector-catalog-")
    except (ReleaseCatalogError, OSError, UnicodeError, ValueError):
        print(json.dumps({"error": "CatalogInputInvalid", "ok": False}, sort_keys=True))
        return 1

    document = json.loads(payload)
    print(
        json.dumps(
            {
                "digest": content_digest(payload),
                "entries": document["entryCount"],
                "ok": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
