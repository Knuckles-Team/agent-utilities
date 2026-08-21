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
import traceback
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
    "scripts/scale/live_contract.py",
    "scripts/scale/loadgen_source_authority.py",
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
    for forbidden in (
        "shell=True",
        "verify=False",
        "os.system(",
        "requests.",
        "httpx.",
    ):
        if forbidden in source:
            raise ValueError("OCI acquisition producer violates the offline boundary")


def _validate_certification_surface() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    if any(
        scripts.get(name) != target
        for name, target in _CERTIFICATION_ENTRY_POINTS.items()
    ):
        raise ValueError("production certification entry points are unavailable")
    for relative in _CERTIFICATION_MODULES:
        source = (ROOT / relative).read_text(encoding="utf-8")
        ast.parse(source)
        if "spec_from_file_location" in source or "importlib.util" in source:
            raise ValueError(
                "production certification uses a source-tree module loader"
            )
    campaign = yaml.safe_load(
        (_RELEASE_ROOT / "certification-campaign.yml").read_text(encoding="utf-8")
    )
    schema = _json_object(_RELEASE_ROOT / "certification-campaign.schema.json")
    Draft202012Validator(schema).validate(campaign)


def _validate_release_resources() -> None:
    """Every declared release-contract resource exists and, if it is a schema,
    is a valid JSON Schema.

    This replaced a catalog of their sha256 digests, written to
    `deploy/release/release-contract-resources.catalog.json` through ~100 lines
    of O_NOFOLLOW/dir_fd/temp-and-rename machinery. Inside one git repository a
    hash ledger over that repository's own files re-proves what the commit
    already proves, and the copy that DID matter -- the one shipped in the wheel
    -- is now compared directly against these files by
    `scripts/release/check_release_wheel.py`.

    The removal was not only about redundancy. `compatibility-matrix.yml` is one
    of these resources, so every version bump rewrote it and staled the catalog,
    while this very gate (a default-stage pre-commit hook) refused the stale
    catalog and blocked the bump commit that would have refreshed it. A derived
    artifact whose refresh is gated on itself is a deadlock, and agent-utilities
    has not published since 1.26.4.

    What survives is the part git cannot do: asserting the files are present and
    that each `.schema.json` actually compiles as a schema.
    """

    for relative in _RESOURCE_PATHS:
        payload = _retained_bytes(ROOT / relative)
        if payload is None:
            raise ValueError("release contract resource is unavailable")
        if relative.endswith(".schema.json"):
            Draft202012Validator.check_schema(json.loads(payload))


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
    # `--write` is gone with the resource catalog it regenerated. Nothing here
    # writes any more: this gate only reads.
    parser.parse_args(argv)
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
        _validate_release_resources()
        _validate_release_documents()
        _validate_acquisition_surface()
        _validate_certification_surface()
    except Exception as exc:  # noqa: BLE001 — STDOUT stays content-free on purpose (the JSON result is the attested gate contract and must not leak paths), but the cause is NOT discarded: it goes to stderr, which no consumer parses. An opaque "CatalogInputInvalid" with no reason anywhere cost two lanes real time.
        print(json.dumps({"error": "CatalogInputInvalid", "ok": False}, sort_keys=True))
        print(f"CatalogInputInvalid: {exc!r}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    if (
        _retained_bytes(CONNECTOR_OUTPUT) != connector
        or _retained_bytes(SKILL_OUTPUT) != skill
    ):
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
                    ]
                ),
                "entries": sum(
                    json.loads(payload)["entryCount"] for payload in (connector, skill)
                ),
                "ok": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
