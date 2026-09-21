#!/usr/bin/env python3
"""Fail closed when retained connector or skill catalogs drift from source."""

from __future__ import annotations

import argparse
import ast
import json
import re
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
    read_retained_bytes,
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
from scripts.release.generate_dependency_license_catalog import (  # noqa: E402
    DEFAULT_OUTPUT as DEPENDENCY_LICENSE_OUTPUT,
)
from scripts.release.generate_dependency_license_catalog import (  # noqa: E402
    DEFAULT_PYPROJECT,
)
from scripts.release.generate_dependency_license_catalog import (
    render_catalog as render_dependency_license_catalog,
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

# Stable test surfaces (`test` and `test-backends`) are intentional package
# extras.  Anything marked as disposable, fixture-only, or explicitly
# test-only is not a release surface and must never reach wheel/lock metadata.
_RELEASE_EXTRA_ALLOWED_TEST_NAMES = frozenset({"test", "test-backends"})
_RELEASE_EXTRA_EPHEMERAL_TOKENS = frozenset(
    {"dummy", "fixture", "scratch", "temp", "temporary", "throwaway"}
)


def _ephemeral_extra(name: str) -> bool:
    normalized = re.sub(r"[._-]+", "-", name.casefold()).strip("-")
    if not normalized or normalized in _RELEASE_EXTRA_ALLOWED_TEST_NAMES:
        return False
    tokens = set(normalized.split("-"))
    return (
        bool(tokens & _RELEASE_EXTRA_EPHEMERAL_TOKENS)
        or {
            "test",
            "only",
        }
        <= tokens
    )


def _project_extra_names(pyproject_path: Path) -> tuple[str, ...]:
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    optional = project.get("project", {}).get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise ValueError("project optional-dependencies must be a table")
    return tuple(str(name) for name in optional)


def _lock_root_package(lock_path: Path) -> dict[str, object]:
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = [
        package
        for package in lock.get("package", [])
        if isinstance(package, dict)
        and package.get("name") == "agent-utilities"
        and package.get("source") == {"editable": "."}
    ]
    if len(packages) != 1:
        raise ValueError("uv.lock must contain one editable agent-utilities root")
    return packages[0]


def _lock_extra_names(lock_path: Path) -> set[str]:
    root_package = _lock_root_package(lock_path)
    lock_optional = root_package.get("optional-dependencies", {})
    metadata = root_package.get("metadata", {})
    provided = metadata.get("provides-extras", []) if isinstance(metadata, dict) else []
    names = set(lock_optional) if isinstance(lock_optional, dict) else set()
    if isinstance(provided, list):
        names.update(str(name) for name in provided)
    return {str(name) for name in names}


def _reject_ephemeral_extras(names: tuple[str, ...] | set[str], *, source: str) -> None:
    bad = sorted(name for name in names if _ephemeral_extra(name))
    if bad:
        raise ValueError(
            f"ephemeral dependency extras in {source} are not releaseable: "
            + ", ".join(bad)
        )


def _validate_dependency_extras(
    *,
    pyproject_path: Path = ROOT / "pyproject.toml",
    lock_path: Path = ROOT / "uv.lock",
) -> None:
    """Reject disposable extras before they become release metadata.

    ``uv.lock`` records the root package's optional-dependency projection and
    ``provides-extras`` list.  Checking both the source manifest and that
    generated projection catches the exact failure mode where a temporary
    dependency is removed from ``pyproject.toml`` but remains in the committed
    lock (and therefore propagates to downstream workspace locks).
    """

    _reject_ephemeral_extras(
        _project_extra_names(pyproject_path), source="pyproject.toml"
    )
    _reject_ephemeral_extras(_lock_extra_names(lock_path), source="uv.lock")


def _json_object(path: Path) -> dict[str, object]:
    payload = read_retained_bytes(path)
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
        payload = read_retained_bytes(ROOT / relative)
        if payload is None:
            raise ValueError("release contract resource is unavailable")
        if relative.endswith(".schema.json"):
            Draft202012Validator.check_schema(json.loads(payload))


def _validate_matrix() -> str:
    payload = read_retained_bytes(DEFAULT_MATRIX)
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
        _validate_dependency_extras()
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
        dependency_licenses = render_dependency_license_catalog(
            pyproject_path=DEFAULT_PYPROJECT,
            catalog_path=DEPENDENCY_LICENSE_OUTPUT,
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
    if any(
        (
            read_retained_bytes(CONNECTOR_OUTPUT) != connector,
            read_retained_bytes(SKILL_OUTPUT) != skill,
            read_retained_bytes(DEPENDENCY_LICENSE_OUTPUT) != dependency_licenses,
        )
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
                        content_digest(dependency_licenses),
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
