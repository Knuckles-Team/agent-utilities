#!/usr/bin/env python3
"""Generate connector-owned, privacy-safe capability certification bundles.

Each provider keeps its authoritative ``connector_manifest.yml`` at repository
root and generated bundle artifacts under ``<module>/ontology`` so they ship in
the provider wheel alongside the source ontology:

* ``shapes/connector.shacl.ttl``
* ``mappings/source.yaml``
* ``fixtures/records.json``
* ``migrations/manifest.json``
* ``certification.json`` (a signed offline source attestation)

The bundle attestation never claims that a live tool or fixture run occurred.
External-live evidence is produced only by the separate connector certification
harness. The bundle contains no endpoint, credential, response data, hostname, personal
identifier, or absolute path.  It records only structural contracts, synthetic
fixtures, relative artifact names, hashes, and a signature.  Providers are
processed sequentially. Release signing requires the explicit runtime Ed25519 key
contract used by ``generate_connector_manifests.py``; a missing key fails before any
artifact is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.integrations.connector_source_attestation import (  # noqa: E402
    build_source_attestation,
)
from agent_utilities.knowledge_graph.ontology import ontology_integrity  # noqa: E402
from agent_utilities.knowledge_graph.ontology.connector_manifest import (  # noqa: E402
    ConnectorManifest,
)
from agent_utilities.knowledge_graph.ontology.connector_manifest_gate import (  # noqa: E402
    check_manifest_bytes,
)
from agent_utilities.security.persistence_privacy import (  # noqa: E402
    PersistencePrivacyGuard,
)
from scripts.generate_connector_manifests import _to_yaml, build_manifest  # noqa: E402

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_]")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MAX_PUBLISH_FILE_BYTES = 8 * 1024 * 1024
_MAX_PUBLISH_TOTAL_BYTES = 128 * 1024 * 1024


class _Publication(NamedTuple):
    staged: Path
    target: Path
    containment_root: Path


class _PreparedPublication(NamedTuple):
    publication: _Publication
    temporary: Path
    original: bytes | None
    original_mode: int


def _configured_provider_names(workspace_path: Path) -> tuple[str, ...]:
    """Resolve the exact provider fleet from repository-manager membership."""

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
        if not _PROVIDER_NAME.fullmatch(name):
            raise ValueError("provider URL has an invalid repository name")
        providers.append(name)
    if len(providers) != len(set(providers)):
        raise ValueError("workspace contains duplicate provider repositories")
    return tuple(providers)


def _module_dir(repo: Path) -> Path:
    matches = sorted(path.parent for path in repo.glob("*/ontology"))
    if len(matches) != 1:
        raise RuntimeError("provider must ship exactly one ontology directory")
    return matches[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_path(target: dict[str, Any], dotted: str, value: Any) -> None:
    current = target
    parts = [part for part in dotted.split(".") if part]
    if not parts:
        return
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _fixture_record(sync: Any) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for field, value in (
        (sync.id_field, "fixture-record"),
        (sync.title_field, "Synthetic certification record"),
        (sync.text_field, "Synthetic content used only for contract validation."),
        (sync.updated_field, "2000-01-01T00:00:00Z"),
    ):
        if field:
            _set_path(record, str(field), value)
    record.setdefault("acl_public", False)
    return {
        "preset": sync.preset,
        "record": record,
        "expected": {
            "id": "fixture-record",
            "title": "Synthetic certification record",
            "acl_state": "quarantine",
        },
    }


def _shapes(manifest: ConnectorManifest) -> str:
    source = manifest.resolved_ontology_source
    lines = [
        "@prefix : <http://knuckles.team/kg#> .",
        f"@prefix shape: <http://knuckles.team/kg/{source}/shape#> .",
        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
    ]
    for resource in sorted(manifest.resources, key=lambda item: item.name):
        shape_name = _SAFE_NAME.sub("_", resource.name) + "GovernanceShape"
        lines.extend(
            [
                f"shape:{shape_name} a sh:NodeShape ;",
                f"    sh:targetClass :{resource.name} ;",
                "    sh:closed false ;",
                "    sh:property [",
                "        sh:path :sourceRecordRef ;",
                "        sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:string",
                "    ] ;",
                "    sh:property [",
                "        sh:path :tenantReference ;",
                "        sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:string",
                "    ] ;",
                "    sh:property [",
                "        sh:path :accessPolicyReference ;",
                "        sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:string",
                "    ] ;",
                "    sh:property [",
                "        sh:path :provenanceReference ;",
                "        sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:string",
                "    ] .",
                "",
            ]
        )
    if not manifest.resources:
        lines.extend(
            [
                "shape:ProviderBundleShape a sh:NodeShape ;",
                "    sh:targetClass :Document ;",
                "    sh:closed false .",
                "",
            ]
        )
    return "\n".join(lines)


def _mapping(manifest: ConnectorManifest) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "connector": manifest.connector,
        "ontology_source": manifest.resolved_ontology_source,
        "sync": [
            {
                "preset": sync.preset,
                "server": sync.server,
                "tool": sync.tool,
                "action": sync.action,
                "records_path": sync.records_path,
                "id_field": sync.id_field,
                "title_field": sync.title_field,
                "text_field": sync.text_field,
                "updated_field": sync.updated_field,
                "doc_type": sync.doc_type,
                "tool_schema_sha256": sync.tool_schema_sha256,
            }
            for sync in manifest.sync
        ],
        "governance": {
            "default_acl": "quarantine",
            "identity": "opaque-reference",
            "tenant": "required-at-materialization",
            "provenance": "required-at-materialization",
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _safe_source_artifact_paths(
    repo: Path, manifest: ConnectorManifest
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for value in manifest.provenance.source_artifacts:
        rendered = str(value)
        relative = PurePosixPath(rendered)
        if (
            not relative.parts
            or rendered != relative.as_posix()
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in rendered
        ):
            raise RuntimeError("provider source artifact path is invalid")
        paths.append(repo.joinpath(*relative.parts))
    return tuple(paths)


def _artifact_paths(
    repo: Path, module: Path, manifest: ConnectorManifest
) -> list[Path]:
    candidates = [repo / "connector_manifest.yml"]
    candidates.extend(_safe_source_artifact_paths(repo, manifest))
    candidates.extend(
        [
            module / "ontology" / "shapes" / "connector.shacl.ttl",
            module / "ontology" / "mappings" / "source.yaml",
            module / "ontology" / "fixtures" / "records.json",
            module / "ontology" / "migrations" / "manifest.json",
        ]
    )
    unique = list(dict.fromkeys(candidates))
    if any(not path.is_file() for path in unique):
        raise RuntimeError("provider artifact ledger contains a missing file")
    return unique


def _validate_written_bundle(
    repo: Path,
    module: Path,
    manifest: ConnectorManifest,
    artifact_paths: list[Path],
    *,
    release_signer: ontology_integrity.ReleaseSigner,
) -> None:
    """Execute every structural check before signing its passing state."""

    manifest_path = repo / "connector_manifest.yml"
    if check_manifest_bytes(manifest_path, require_signature=False):
        raise RuntimeError("provider manifest validation failed")
    manifest_hash = ontology_integrity.canonical_manifest_hash(manifest)
    if not ontology_integrity.verify_release_signature(
        manifest_hash,
        manifest.provenance.signature,
        signer_id=manifest.provenance.signer,
        algorithm=manifest.provenance.signature_algorithm,
        public_key=manifest.provenance.signing_public_key,
        trusted_public_keys=(release_signer.public_key,),
    ):
        raise RuntimeError("provider manifest candidate signature is invalid")
    if any(
        not isinstance(sync.tool_schema_sha256, str)
        or not _SHA256.fullmatch(sync.tool_schema_sha256)
        for sync in manifest.sync
    ):
        raise RuntimeError("provider tool schema declaration is invalid")

    fixture_path = module / "ontology" / "fixtures" / "records.json"
    fixture_document = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixtures = (
        fixture_document.get("fixtures") if isinstance(fixture_document, dict) else None
    )
    if (
        not isinstance(fixture_document, dict)
        or fixture_document.get("schema_version") != "1"
        or fixture_document.get("connector") != manifest.connector
        or not isinstance(fixtures, list)
        or len(fixtures) != len(manifest.sync)
        or any(
            not isinstance(fixture, dict)
            or fixture.get("expected", {}).get("acl_state") != "quarantine"
            for fixture in fixtures
        )
    ):
        raise RuntimeError("provider synthetic fixture contract is invalid")

    import rdflib

    shapes_path = module / "ontology" / "shapes" / "connector.shacl.ttl"
    shapes = rdflib.Graph()
    shapes.parse(str(shapes_path), format="turtle")

    privacy = PersistencePrivacyGuard()
    semantic_paths = {
        shapes_path,
        module / "ontology" / "mappings" / "source.yaml",
        fixture_path,
        module / "ontology" / "migrations" / "manifest.json",
    }
    if not semantic_paths.issubset(set(artifact_paths)):
        raise RuntimeError("provider semantic artifact ledger is incomplete")
    for path in sorted(semantic_paths):
        text = path.read_text(encoding="utf-8")
        clean, report = privacy.sanitize_text(text)
        if report.changed or clean != text:
            raise RuntimeError("provider source bundle privacy validation failed")


def _read_regular_contained(root: Path, path: Path) -> bytes:
    try:
        relative = path.relative_to(root)
        if not relative.parts or ".." in relative.parts:
            raise ValueError
        current = root
        for index, part in enumerate(relative.parts):
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError
            if index < len(relative.parts) - 1:
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError
            elif not stat.S_ISREG(metadata.st_mode):
                raise ValueError
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
        payload = current.read_bytes()
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("provider artifact is not a regular contained file") from exc
    if len(payload) > _MAX_PUBLISH_FILE_BYTES:
        raise RuntimeError("provider artifact exceeds the staging boundary")
    return payload


def _write_staged(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _stage_one(
    repo: Path,
    *,
    bundled_output: Path,
    staging_root: Path,
    now: datetime,
    release_signer: ontology_integrity.ReleaseSigner,
) -> tuple[_Publication, ...]:
    module = _module_dir(repo)
    manifest = build_manifest(repo, now=now, release_signer=release_signer)
    stage_repo = staging_root / "agents" / repo.name
    stage_module = stage_repo / module.relative_to(repo)

    for source in _safe_source_artifact_paths(repo, manifest):
        _write_staged(
            stage_repo / source.relative_to(repo),
            _read_regular_contained(repo, source),
        )

    manifest_text = _to_yaml(manifest)
    staged_manifest = stage_repo / "connector_manifest.yml"
    _write_staged(staged_manifest, manifest_text.encode("utf-8"))
    staged_bundled = staging_root / "bundled" / repo.name / "connector_manifest.yml"
    _write_staged(staged_bundled, manifest_text.encode("utf-8"))

    ontology = stage_module / "ontology"
    shapes_path = ontology / "shapes" / "connector.shacl.ttl"
    shapes_path.parent.mkdir(parents=True, exist_ok=True)
    shapes_path.write_text(_shapes(manifest), encoding="utf-8")

    mappings_path = ontology / "mappings" / "source.yaml"
    mappings_path.parent.mkdir(parents=True, exist_ok=True)
    mappings_path.write_text(
        yaml.safe_dump(_mapping(manifest), sort_keys=False, width=100),
        encoding="utf-8",
    )
    _write_json(
        ontology / "fixtures" / "records.json",
        {
            "schema_version": "1",
            "connector": manifest.connector,
            "fixtures": [_fixture_record(sync) for sync in manifest.sync],
        },
    )
    _write_json(
        ontology / "migrations" / "manifest.json",
        {
            "schema_version": "1",
            "connector": manifest.connector,
            "current_ontology_source": manifest.resolved_ontology_source,
            "migrations": [],
        },
    )

    artifact_paths = _artifact_paths(stage_repo, stage_module, manifest)
    _validate_written_bundle(
        stage_repo,
        stage_module,
        manifest,
        artifact_paths,
        release_signer=release_signer,
    )
    artifacts = {
        path.relative_to(stage_repo).as_posix(): _sha256(path)
        for path in artifact_paths
    }
    certification_path = ontology / "certification.json"
    _write_json(
        certification_path,
        build_source_attestation(
            manifest,
            artifacts=artifacts,
            now=now,
            release_signer=release_signer,
        ),
    )

    generated_relatives = (
        Path("connector_manifest.yml"),
        shapes_path.relative_to(stage_repo),
        mappings_path.relative_to(stage_repo),
        (ontology / "fixtures" / "records.json").relative_to(stage_repo),
        (ontology / "migrations" / "manifest.json").relative_to(stage_repo),
        certification_path.relative_to(stage_repo),
    )
    publications = [
        _Publication(
            staged=stage_repo / relative,
            target=repo / relative,
            containment_root=repo,
        )
        for relative in generated_relatives
    ]
    publications.append(
        _Publication(
            staged=staged_bundled,
            target=bundled_output / repo.name / "connector_manifest.yml",
            containment_root=bundled_output,
        )
    )
    return tuple(publications)


def _target_parent(publication: _Publication, created_directories: list[Path]) -> Path:
    root = publication.containment_root
    try:
        root_metadata = root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            raise ValueError
        relative = publication.target.relative_to(root)
        if not relative.parts or ".." in relative.parts:
            raise ValueError
        resolved_root = root.resolve(strict=True)
        current = root
        for part in relative.parts[:-1]:
            current = current / part
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                current.mkdir(mode=0o755)
                created_directories.append(current)
                metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError
        current.resolve(strict=True).relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("publication target is not contained") from exc
    return current


def _temporary_payload(target: Path, payload: bytes, mode: int) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _prepare_publication(
    publication: _Publication, created_directories: list[Path]
) -> _PreparedPublication:
    try:
        staged_metadata = publication.staged.lstat()
        if stat.S_ISLNK(staged_metadata.st_mode) or not stat.S_ISREG(
            staged_metadata.st_mode
        ):
            raise ValueError
        payload = publication.staged.read_bytes()
    except (OSError, ValueError) as exc:
        raise RuntimeError("staged publication is not a regular file") from exc
    if len(payload) > _MAX_PUBLISH_FILE_BYTES:
        raise RuntimeError("staged publication exceeds the file boundary")

    parent = _target_parent(publication, created_directories)
    target = publication.target
    if target.parent != parent:
        raise RuntimeError("publication target parent differs")
    try:
        target_metadata = target.lstat()
    except FileNotFoundError:
        original = None
        original_mode = 0o644
    else:
        if stat.S_ISLNK(target_metadata.st_mode) or not stat.S_ISREG(
            target_metadata.st_mode
        ):
            raise RuntimeError("publication target is not a regular file")
        original = target.read_bytes()
        if len(original) > _MAX_PUBLISH_FILE_BYTES:
            raise RuntimeError("publication target exceeds the rollback boundary")
        original_mode = stat.S_IMODE(target_metadata.st_mode)

    temporary = _temporary_payload(target, payload, original_mode)
    return _PreparedPublication(
        publication=publication,
        temporary=temporary,
        original=original,
        original_mode=original_mode,
    )


def _remove_created_directories(paths: list[Path]) -> None:
    for path in reversed(paths):
        try:
            path.rmdir()
        except OSError:
            pass


def _publish_transaction(publications: list[_Publication]) -> None:
    """Publish a fully validated fleet and roll back any ordinary write failure."""

    targets = [publication.target for publication in publications]
    if len(targets) != len(set(targets)):
        raise RuntimeError("publication plan contains duplicate targets")

    prepared: list[_PreparedPublication] = []
    committed: list[_PreparedPublication] = []
    created_directories: list[Path] = []
    total_bytes = 0
    try:
        for publication in publications:
            item = _prepare_publication(publication, created_directories)
            prepared.append(item)
            total_bytes += item.temporary.stat().st_size
            if total_bytes > _MAX_PUBLISH_TOTAL_BYTES:
                raise RuntimeError("fleet publication exceeds the aggregate boundary")

        for item in prepared:
            os.replace(item.temporary, item.publication.target)
            committed.append(item)
    except Exception as publish_error:
        rollback_errors = 0
        for item in reversed(committed):
            target = item.publication.target
            try:
                if item.original is None:
                    target.unlink(missing_ok=True)
                else:
                    rollback = _temporary_payload(
                        target, item.original, item.original_mode
                    )
                    try:
                        os.replace(rollback, target)
                    finally:
                        rollback.unlink(missing_ok=True)
            except Exception:
                rollback_errors += 1
        for item in prepared:
            item.temporary.unlink(missing_ok=True)
        _remove_created_directories(created_directories)
        if rollback_errors:
            raise RuntimeError(
                "fleet publication and rollback failed"
            ) from publish_error
        raise RuntimeError(
            "fleet publication failed and was rolled back"
        ) from publish_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents-root", type=Path, required=True)
    parser.add_argument("--bundled-output", type=Path, required=True)
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="authoritative repository-manager workspace membership",
    )
    parser.add_argument(
        "--connector",
        action="append",
        default=[],
        help="Generate only this connector (repeatable); defaults to the complete fleet.",
    )
    parser.add_argument(
        "--now", required=True, help="UTC timestamp: YYYY-MM-DDTHH:MM:SSZ"
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    now = datetime.strptime(args.now, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)

    selected = set(args.connector)
    try:
        configured = set(_configured_provider_names(args.workspace))
        release_signer = ontology_integrity.ReleaseSigner.from_runtime()
    except Exception:  # noqa: BLE001 - privacy-safe aggregate only
        print("capability bundle generation preflight failed")
        return 1
    unknown = selected - configured
    target_names = sorted(selected or configured)
    if unknown:
        print(
            "capability bundle generation could not resolve "
            f"{len(unknown)} configured provider(s)"
        )
        return 1

    repos: list[Path] = []
    failures = 0
    try:
        resolved_agents_root = args.agents_root.resolve(strict=True)
    except (OSError, RuntimeError):
        print("capability bundle generation could not resolve the provider fleet")
        return 1
    for name in target_names:
        repo = args.agents_root / name
        try:
            metadata = repo.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError
            repo.resolve(strict=True).relative_to(resolved_agents_root)
            pyproject = repo / "pyproject.toml"
            if not pyproject.is_file() or pyproject.is_symlink():
                raise ValueError
        except (OSError, RuntimeError, ValueError):
            failures += 1
            continue
        repos.append(repo)
    if failures:
        print(f"capability bundle generation failed for {failures} provider(s)")
        return 1

    publications: list[_Publication] = []
    with tempfile.TemporaryDirectory(prefix="agent-utilities-connector-fleet-") as raw:
        staging_root = Path(raw)
        failures = 0
        for repo in repos:
            try:
                publications.extend(
                    _stage_one(
                        repo,
                        bundled_output=args.bundled_output,
                        staging_root=staging_root,
                        now=now,
                        release_signer=release_signer,
                    )
                )
            except Exception:  # noqa: BLE001 - privacy-safe aggregate only
                failures += 1
        if failures:
            print(f"capability bundle generation failed for {failures} provider(s)")
            return 1
        if args.apply:
            try:
                _publish_transaction(publications)
            except Exception:  # noqa: BLE001 - privacy-safe aggregate only
                print("capability bundle fleet publication failed")
                return 1

    verb = "generated" if args.apply else "would generate"
    print(f"{verb} {len(repos)} capability bundle(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
