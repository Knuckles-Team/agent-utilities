#!/usr/bin/env python3
"""Fail closed when retained connector or skill catalogs drift from source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import secrets
import stat
import sys
import tomllib
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.release_catalogs import (  # noqa: E402
    canonical_value_digest,
    content_digest,
)
from scripts.release import generate_oci_acquisition_attestation  # noqa: E402
from scripts.release.check_compatibility import (  # noqa: E402
    file_digest,
    validate_compatibility_matrix,
)
from scripts.release.generate_connector_bundle_catalog import (  # noqa: E402
    DEFAULT_AGENTS_ROOT,
    DEFAULT_BUNDLED_ROOT,
    DEFAULT_LOCK_PATH,
    DEFAULT_MATRIX,
    DEFAULT_WORKSPACE,
)
from scripts.release.generate_connector_bundle_catalog import (
    DEFAULT_OUTPUT as CONNECTOR_OUTPUT,
)
from scripts.release.generate_connector_bundle_catalog import (
    render_catalog as render_connector_catalog,
)
from scripts.release.generate_prebundled_skill_catalog import (  # noqa: E402
    DEFAULT_OUTPUT as SKILL_OUTPUT,
)
from scripts.release.generate_prebundled_skill_catalog import (  # noqa: E402
    DEFAULT_SKILLS_ROOT,
)
from scripts.release.generate_prebundled_skill_catalog import (
    render_catalog as render_skill_catalog,
)

_RELEASE_ROOT = ROOT / "deploy" / "release"
_SCHEMA_BINDINGS = (
    (
        _RELEASE_ROOT / "connector-bundle-catalog.schema.json",
        CONNECTOR_OUTPUT,
    ),
    (
        _RELEASE_ROOT / "prebundled-skill-catalog.schema.json",
        SKILL_OUTPUT,
    ),
)
_ASSEMBLY_SCHEMAS = (
    _RELEASE_ROOT / "component-provenance.schema.json",
    _RELEASE_ROOT / "component-signature-bundle.schema.json",
    _RELEASE_ROOT / "component-source-evidence.schema.json",
    _RELEASE_ROOT / "release-assembly.schema.json",
)
_RESOURCE_CATALOG = _RELEASE_ROOT / "release-contract-resources.catalog.json"
_RESOURCE_PATHS = (
    "deploy/release/certification-campaign.schema.json",
    "deploy/release/certification-campaign.yml",
    "deploy/release/compatibility-matrix.schema.json",
    "deploy/release/compatibility-matrix.yml",
    "deploy/release/component-provenance.schema.json",
    "deploy/release/component-signature-bundle.schema.json",
    "deploy/release/component-source-evidence.schema.json",
    "deploy/release/connector-bundle-catalog.schema.json",
    "deploy/release/connector-bundles.catalog.json",
    "deploy/release/connector-live-certification-ledger.schema.json",
    "deploy/release/exact-artifact-closure-evidence.schema.json",
    "deploy/release/exact-local-gates-manifest.schema.json",
    "deploy/release/exact-local-release-evidence.schema.json",
    "deploy/release/exact-local-release-spec.schema.json",
    "deploy/release/index-migration-catalog.schema.json",
    "deploy/release/index-migrations.catalog.json",
    "deploy/release/oci-scanner-attestation.schema.json",
    "deploy/release/oci-vulnerability-database-attestation.schema.json",
    "deploy/release/oci-vulnerability-scan-evidence.schema.json",
    "deploy/release/operational-evidence.schema.json",
    "deploy/release/prebundled-skill-catalog.schema.json",
    "deploy/release/prebundled-skill-validation-evidence.schema.json",
    "deploy/release/prebundled-skills.catalog.json",
    "deploy/release/release-assembly.schema.json",
    "deploy/release/release-configuration.schema.json",
    "deploy/release/release-manifest.schema.json",
    "deploy/release/release-migration-plan.schema.json",
    "deploy/release/skill-validation-deployment-evidence.schema.json",
    "deploy/release/skill-validation-deployment.schema.json",
    "deploy/release/source-freeze-evidence.schema.json",
    "deploy/release/source-freeze-gates.json",
    "deploy/release/source-freeze-gates.schema.json",
    "scripts/scale/workload_contract.yml",
)
_CERTIFICATION_ENTRY_POINTS = {
    "graphos-certification-campaign": "scripts.certification.campaign:main",
    "graphos-certification-fault": "scripts.certification.fault_hook:main",
    "graphos-certification-load": "scripts.scale.loadgen:main",
    "graphos-certification-metrics": "scripts.certification.collect_metrics:main",
    "graphos-operational-evidence": "scripts.certification.evidence:main",
}
_CERTIFICATION_MODULES = (
    "scripts/certification/campaign.py",
    "scripts/certification/collect_metrics.py",
    "scripts/certification/evidence.py",
    "scripts/certification/fault_hook.py",
    "scripts/certification/subprocess_boundary.py",
    "scripts/scale/fake_engine.py",
    "scripts/scale/loadgen.py",
    "scripts/scale/workload_contract.py",
)


def _retained_bytes(path: Path) -> bytes | None:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            return None
        return path.read_bytes()
    except OSError:
        return None


def _json_object(path: Path) -> dict[str, object]:
    payload = _retained_bytes(path)
    if payload is None:
        raise ValueError("release document must be a regular file")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("release document root must be an object")
    return value


def _validate_release_documents() -> None:
    for schema_path, document_path in _SCHEMA_BINDINGS:
        schema = _json_object(schema_path)
        validator = Draft202012Validator(schema)
        validator.check_schema(schema)
        validator.validate(_json_object(document_path))
    for schema_path in _ASSEMBLY_SCHEMAS:
        schema = _json_object(schema_path)
        Draft202012Validator.check_schema(schema)


def _validate_acquisition_surface() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    if scripts.get("generate-oci-acquisition-attestation") != (
        "scripts.release.generate_oci_acquisition_attestation:main"
    ):
        raise ValueError("OCI acquisition entry point is unavailable")
    if not all(
        callable(value)
        for value in (
            generate_oci_acquisition_attestation.generate_scanner_attestation,
            generate_oci_acquisition_attestation.generate_database_attestation,
            generate_oci_acquisition_attestation.main,
        )
    ):
        raise ValueError("OCI acquisition producer surface is unavailable")
    source = (
        ROOT / "scripts" / "release" / "generate_oci_acquisition_attestation.py"
    ).read_text(encoding="utf-8")
    ast.parse(source)
    for forbidden in ("shell=True", "verify=False", "os.system(", "requests.", "httpx."):
        if forbidden in source:
            raise ValueError("OCI acquisition producer violates the offline boundary")


def _validate_certification_surface() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    if any(scripts.get(name) != target for name, target in _CERTIFICATION_ENTRY_POINTS.items()):
        raise ValueError("production certification entry points are unavailable")
    for relative in _CERTIFICATION_MODULES:
        source = (ROOT / relative).read_text(encoding="utf-8")
        ast.parse(source)
        if "spec_from_file_location" in source or "importlib.util" in source:
            raise ValueError("production certification uses a source-tree module loader")
    campaign = yaml.safe_load(
        (_RELEASE_ROOT / "certification-campaign.yml").read_text(encoding="utf-8")
    )
    schema = _json_object(_RELEASE_ROOT / "certification-campaign.schema.json")
    Draft202012Validator(schema).validate(campaign)


def _resource_catalog_bytes() -> bytes:
    resources: list[dict[str, str]] = []
    for relative in _RESOURCE_PATHS:
        payload = _retained_bytes(ROOT / relative)
        if payload is None:
            raise ValueError("release contract resource is unavailable")
        resources.append(
            {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}
        )
        if relative.endswith(".schema.json"):
            schema = json.loads(payload)
            Draft202012Validator.check_schema(schema)
    value = {"schema": "release-contract-resources/1", "resources": resources}
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _write_resource_catalog(payload: bytes) -> None:
    """Atomically replace only the canonical, repository-owned resource catalog."""

    parent = _RESOURCE_CATALOG.parent
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise ValueError("release resource catalog parent is unavailable") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
        parent_metadata.st_mode
    ):
        raise ValueError("release resource catalog parent is invalid")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory = os.open(parent, directory_flags)
    except OSError as exc:
        raise ValueError("release resource catalog parent is unavailable") from exc
    temporary_name = f".release-contract-resources.{secrets.token_hex(16)}.tmp"
    temporary_created = False
    try:
        try:
            existing = os.stat(
                _RESOURCE_CATALOG.name,
                dir_fd=directory,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1
        ):
            raise ValueError(
                "release resource catalog must be an unaliased regular file"
            )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory)
        temporary_created = True
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ValueError("release resource catalog write failed")
                view = view[written:]
            os.fchmod(descriptor, 0o644)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(
            temporary_name,
            _RESOURCE_CATALOG.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
        )
        temporary_created = False
        os.fsync(directory)
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=directory)
            except FileNotFoundError:
                pass
        os.close(directory)


def _validate_matrix() -> str:
    payload = _retained_bytes(DEFAULT_MATRIX)
    if payload is None:
        raise ValueError("compatibility matrix must be a regular file")
    matrix = yaml.safe_load(payload)
    if not isinstance(matrix, dict):
        raise ValueError("compatibility matrix root must be a mapping")
    validate_compatibility_matrix(matrix)
    return file_digest(DEFAULT_MATRIX)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate only the canonical release resource digest catalog.",
    )
    arguments = parser.parse_args(argv)
    try:
        matrix_digest = _validate_matrix()
        connector = render_connector_catalog(
            agents_root=DEFAULT_AGENTS_ROOT,
            workspace_path=DEFAULT_WORKSPACE,
            bundled_root=DEFAULT_BUNDLED_ROOT,
            lock_path=DEFAULT_LOCK_PATH,
            matrix_path=DEFAULT_MATRIX,
        )
        skill = render_skill_catalog(
            skills_root=DEFAULT_SKILLS_ROOT,
            matrix_path=DEFAULT_MATRIX,
        )
        resources = _resource_catalog_bytes()
        _validate_release_documents()
        _validate_acquisition_surface()
        _validate_certification_surface()
    except Exception:  # noqa: BLE001 - content-free source gate result
        print(json.dumps({"error": "CatalogInputInvalid", "ok": False}, sort_keys=True))
        return 1
    if _retained_bytes(CONNECTOR_OUTPUT) != connector or _retained_bytes(
        SKILL_OUTPUT
    ) != skill:
        print(json.dumps({"error": "CatalogDrift", "ok": False}, sort_keys=True))
        return 1
    if arguments.write:
        try:
            _write_resource_catalog(resources)
        except Exception:  # noqa: BLE001 - content-free source gate result
            print(
                json.dumps(
                    {"error": "CatalogWriteFailed", "ok": False}, sort_keys=True
                )
            )
            return 1
    if _retained_bytes(_RESOURCE_CATALOG) != resources:
        print(json.dumps({"error": "CatalogDrift", "ok": False}, sort_keys=True))
        return 1

    print(
        json.dumps(
            {
                "digest": canonical_value_digest(
                    [
                        matrix_digest,
                        content_digest(connector),
                        content_digest(skill),
                        content_digest(resources),
                    ]
                ),
                "entries": sum(
                    json.loads(payload)["entryCount"]
                    for payload in (connector, skill)
                ),
                "ok": True,
                "written": arguments.write,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
