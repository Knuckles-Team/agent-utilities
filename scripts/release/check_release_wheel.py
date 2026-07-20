#!/usr/bin/env python3
"""Fail closed unless a built wheel carries the exact local release surface."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import stat
import sys
import zipfile
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

_MAX_WHEEL_BYTES = 4 * 1024 * 1024 * 1024
_MAX_CONTRACT_MEMBER_BYTES = 1024 * 1024
_MAX_MEMBERS = 400_000
_RELEASE_RESOURCE_CATALOG = (
    "deploy/release/release-contract-resources.catalog.json"
)
_RELEASE_RESOURCE_CATALOG_SHA256 = (
    "fc69c757fa7355efaedb418002cf84a3125618746e0514e69ee1c043a00b2fe9"
)
_RELEASE_RESOURCE_PATHS = (
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
_BUNDLED_SKILL_NAMES = (
    "agent-utilities-deployment",
    "agent-utilities-development",
    "agent-utilities-evolution",
    "graph-engine-and-modalities",
    "graph-ingestion-and-integration",
    "graph-modeling-and-mutation",
    "graph-orchestration-and-automation",
    "graph-query-and-explanation",
    "graph-research-and-analysis",
    "graph-runtime-and-governance",
)
_BUNDLED_SKILL_ASSETS = {
    asset
    for skill in _BUNDLED_SKILL_NAMES
    for asset in (
        f"agent_utilities/skills/{skill}/SKILL.md",
        f"agent_utilities/skills/{skill}/agents/graph-os.yaml",
        f"agent_utilities/skills/{skill}/agents/openai.yaml",
    )
}
_REQUIRED_MEMBERS = {
    "agent_utilities/deployment/certification_oidc.py",
    "agent_utilities/deployment/skill_validation.py",
    "agent_utilities/deployment/skill_validation_assets.py",
    "agent_utilities/prompts/agent-utilities-expert.json",
    "agent_utilities/skills/__init__.py",
    "agent_utilities/skills/runtime_validation.py",
    "agent_utilities/skills/runtime_validation.yaml",
    "agent_utilities/skills/validation.py",
    "deploy/__init__.py",
    "deploy/release/__init__.py",
    "deploy/release/exact-local-release-evidence.schema.json",
    "deploy/release/exact-local-release-spec.schema.json",
    "deploy/release/oci-scanner-attestation.schema.json",
    "deploy/release/oci-vulnerability-database-attestation.schema.json",
    "deploy/release/oci-vulnerability-scan-evidence.schema.json",
    "deploy/release/prebundled-skill-validation-evidence.schema.json",
    "deploy/release/skill-validation-deployment-evidence.schema.json",
    "deploy/release/skill-validation-deployment.schema.json",
    _RELEASE_RESOURCE_CATALOG,
    "scripts/__init__.py",
    "scripts/release/assemble_manifest.py",
    "scripts/release/__init__.py",
    "scripts/release/assemble_exact_local_release.py",
    "scripts/release/check_compatibility.py",
    "scripts/release/exact_artifact_closure.py",
    "scripts/release/exact_local_gates_manifest.py",
    "scripts/release/generate_component_evidence.py",
    "scripts/release/generate_oci_acquisition_attestation.py",
    "scripts/release/generate_oci_vulnerability_scan_evidence.py",
    "scripts/release/generate_release_inputs.py",
    "scripts/release/generate_release_assembly.py",
    "scripts/release/materialize_component_wheelhouse.py",
    "scripts/release/promote_local_release.py",
    "scripts/certification/__init__.py",
    "scripts/certification/campaign.py",
    "scripts/certification/collect_metrics.py",
    "scripts/certification/evidence.py",
    "scripts/certification/fault_hook.py",
    "scripts/certification/subprocess_boundary.py",
    "scripts/scale/__init__.py",
    "scripts/scale/fake_engine.py",
    "scripts/scale/loadgen.py",
    "scripts/scale/workload_contract.py",
} | _BUNDLED_SKILL_ASSETS | set(_RELEASE_RESOURCE_PATHS)
_SCHEMA_ID_SUFFIXES = {
    "deploy/release/certification-campaign.schema.json": (
        "certification-campaign-v1.json"
    ),
    "deploy/release/compatibility-matrix.schema.json": (
        "compatibility-matrix-v2.json"
    ),
    "deploy/release/component-provenance.schema.json": (
        "component-provenance-v1.json"
    ),
    "deploy/release/component-signature-bundle.schema.json": (
        "component-signature-bundle-v2.json"
    ),
    "deploy/release/component-source-evidence.schema.json": (
        "component-source-evidence-v1.json"
    ),
    "deploy/release/connector-bundle-catalog.schema.json": (
        "connector-bundle-catalog-v1.json"
    ),
    "deploy/release/connector-live-certification-ledger.schema.json": (
        "connector-live-certification-ledger-v1.json"
    ),
    "deploy/release/exact-artifact-closure-evidence.schema.json": (
        "exact-artifact-closure-evidence-v1.json"
    ),
    "deploy/release/exact-local-gates-manifest.schema.json": (
        "exact-local-gates-manifest-v1.json"
    ),
    "deploy/release/exact-local-release-evidence.schema.json": "-v2.json",
    "deploy/release/exact-local-release-spec.schema.json": "-v2.json",
    "deploy/release/index-migration-catalog.schema.json": (
        "index-migration-catalog-v1.json"
    ),
    "deploy/release/oci-scanner-attestation.schema.json": (
        "oci-scanner-attestation-v1.json"
    ),
    "deploy/release/oci-vulnerability-database-attestation.schema.json": (
        "oci-vulnerability-database-attestation-v1.json"
    ),
    "deploy/release/oci-vulnerability-scan-evidence.schema.json": (
        "oci-vulnerability-scan-evidence-v1.json"
    ),
    "deploy/release/operational-evidence.schema.json": (
        "operational-evidence-v1.json"
    ),
    "deploy/release/prebundled-skill-catalog.schema.json": (
        "prebundled-skill-catalog-v1.json"
    ),
    "deploy/release/prebundled-skill-validation-evidence.schema.json": (
        "prebundled-skill-validation-evidence-v2.json"
    ),
    "deploy/release/release-assembly.schema.json": "release-assembly-v1.json",
    "deploy/release/release-configuration.schema.json": (
        "release-configuration-v1.json"
    ),
    "deploy/release/release-manifest.schema.json": "release-manifest-v1.json",
    "deploy/release/release-migration-plan.schema.json": (
        "release-migration-plan-v1.json"
    ),
    "deploy/release/skill-validation-deployment-evidence.schema.json": (
        "skill-validation-lifecycle-evidence-v2.json"
    ),
    "deploy/release/skill-validation-deployment.schema.json": (
        "skill-validation-deployment-v2.json"
    ),
    "deploy/release/source-freeze-evidence.schema.json": (
        "source-freeze-evidence:1"
    ),
    "deploy/release/source-freeze-gates.schema.json": "source-freeze-gates:1",
}
_ENTRY_POINTS = {
    "agent-utilities-validate-skills": (
        "agent_utilities.skills.runtime_validation:main"
    ),
    "assemble-exact-local-release": (
        "scripts.release.assemble_exact_local_release:main"
    ),
    "assemble-graphos-release": "scripts.release.assemble_manifest:main",
    "check-graphos-compatibility": "scripts.release.check_compatibility:main",
    "generate-graphos-component-evidence": (
        "scripts.release.generate_component_evidence:main"
    ),
    "generate-graphos-release-input": (
        "scripts.release.generate_release_inputs:main"
    ),
    "generate-graphos-release-assembly": (
        "scripts.release.generate_release_assembly:main"
    ),
    "generate-oci-acquisition-attestation": (
        "scripts.release.generate_oci_acquisition_attestation:main"
    ),
    "generate-oci-vulnerability-scan-evidence": (
        "scripts.release.generate_oci_vulnerability_scan_evidence:main"
    ),
    "generate-exact-local-gates-manifest": (
        "scripts.release.exact_local_gates_manifest:main"
    ),
    "bind-exact-local-release-evidence": (
        "scripts.release.exact_artifact_closure:main"
    ),
    "materialize-component-wheelhouse": (
        "scripts.release.materialize_component_wheelhouse:main"
    ),
    "graph-os-certify-skills": (
        "agent_utilities.deployment.skill_validation:main"
    ),
    "graph-os-generate-skill-certification": (
        "agent_utilities.deployment.skill_validation_assets:generator_main"
    ),
    "graph-os-generate-skill-runtime-profile": (
        "agent_utilities.deployment.skill_validation_assets:profile_main"
    ),
    "graph-os-skill-readiness": (
        "agent_utilities.deployment.skill_validation_assets:readiness_main"
    ),
    "graph-os-verify-skill-certification": (
        "agent_utilities.deployment.skill_validation_assets:verifier_main"
    ),
    "graphos-certification-campaign": "scripts.certification.campaign:main",
    "graphos-certification-fault": "scripts.certification.fault_hook:main",
    "graphos-certification-load": "scripts.scale.loadgen:main",
    "graphos-certification-metrics": "scripts.certification.collect_metrics:main",
    "graphos-operational-evidence": "scripts.certification.evidence:main",
    "promote-local-graphos-release": "scripts.release.promote_local_release:main",
    "verify-local-graphos-release-evidence": (
        "scripts.release.promote_local_release:verify_main"
    ),
}


class WheelContractError(RuntimeError):
    """Stable, path-free wheel contract rejection."""


def check_wheel(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise WheelContractError("wheel-unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > _MAX_WHEEL_BYTES
    ):
        raise WheelContractError("wheel-invalid")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if (
                len(names) > _MAX_MEMBERS
                or len(names) != len(set(names))
                or not _REQUIRED_MEMBERS <= set(names)
            ):
                raise WheelContractError("release-surface-missing")
            catalog_info = archive.getinfo(_RELEASE_RESOURCE_CATALOG)
            if catalog_info.file_size > _MAX_CONTRACT_MEMBER_BYTES:
                raise WheelContractError("release-resource-catalog-invalid")
            catalog_payload = archive.read(_RELEASE_RESOURCE_CATALOG)
            if (
                hashlib.sha256(catalog_payload).hexdigest()
                != _RELEASE_RESOURCE_CATALOG_SHA256
            ):
                raise WheelContractError("release-resource-catalog-invalid")
            catalog = json.loads(catalog_payload)
            resources = catalog.get("resources") if isinstance(catalog, dict) else None
            if (
                not isinstance(catalog, dict)
                or set(catalog) != {"schema", "resources"}
                or catalog.get("schema") != "release-contract-resources/1"
                or not isinstance(resources, list)
                or len(resources) != len(_RELEASE_RESOURCE_PATHS)
            ):
                raise WheelContractError("release-resource-catalog-invalid")
            resource_digests: dict[str, str] = {}
            for resource in resources:
                if (
                    not isinstance(resource, dict)
                    or set(resource) != {"path", "sha256"}
                    or not isinstance(resource.get("path"), str)
                    or not isinstance(resource.get("sha256"), str)
                    or resource["path"] in resource_digests
                ):
                    raise WheelContractError("release-resource-catalog-invalid")
                resource_digests[resource["path"]] = resource["sha256"]
            if tuple(resource_digests) != _RELEASE_RESOURCE_PATHS:
                raise WheelContractError("release-resource-catalog-invalid")
            for resource_name, expected_digest in resource_digests.items():
                info = archive.getinfo(resource_name)
                if (
                    info.file_size > _MAX_CONTRACT_MEMBER_BYTES
                    or hashlib.sha256(archive.read(resource_name)).hexdigest()
                    != expected_digest
                ):
                    raise WheelContractError("release-resource-digest-mismatch")
            packaged_skill_assets = {
                name
                for name in names
                if name.startswith("agent_utilities/skills/")
                and len(name.split("/")) >= 4
                and not name.endswith("/")
            }
            if packaged_skill_assets != _BUNDLED_SKILL_ASSETS:
                raise WheelContractError("skill-surface-invalid")
            entry_points = [
                name for name in names if name.endswith(".dist-info/entry_points.txt")
            ]
            if len(entry_points) != 1:
                raise WheelContractError("entry-points-missing")
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise WheelContractError("metadata-missing")
            if (
                archive.getinfo(metadata_names[0]).file_size
                > _MAX_CONTRACT_MEMBER_BYTES
            ):
                raise WheelContractError("metadata-invalid")
            metadata_message = BytesParser(policy=email_policy).parsebytes(
                archive.read(metadata_names[0])
            )
            try:
                engine_requirements = [
                    Requirement(value)
                    for value in metadata_message.get_all("Requires-Dist") or ()
                    if canonicalize_name(Requirement(value).name) == "epistemic-graph"
                ]
            except InvalidRequirement as exc:
                raise WheelContractError("metadata-invalid") from exc
            if (
                len(engine_requirements) != 1
                or engine_requirements[0].extras != {"full"}
                or engine_requirements[0].url is not None
                or engine_requirements[0].marker is not None
                or engine_requirements[0].specifier != SpecifierSet(">=2.23.1,<3.0.0")
            ):
                raise WheelContractError("full-engine-requirement-invalid")
            if archive.getinfo(entry_points[0]).file_size > _MAX_CONTRACT_MEMBER_BYTES:
                raise WheelContractError("entry-points-invalid")
            parser = configparser.ConfigParser(interpolation=None)
            parser.optionxform = str
            parser.read_string(archive.read(entry_points[0]).decode("utf-8"))
            if (
                not parser.has_section("console_scripts")
                or {
                    name: parser.get("console_scripts", name, fallback="")
                    for name in _ENTRY_POINTS
                }
                != _ENTRY_POINTS
            ):
                raise WheelContractError("entry-points-invalid")
            schemas: dict[str, dict[str, object]] = {}
            for schema_name, id_suffix in _SCHEMA_ID_SUFFIXES.items():
                if archive.getinfo(schema_name).file_size > _MAX_CONTRACT_MEMBER_BYTES:
                    raise WheelContractError("release-schema-invalid")
                schema = json.loads(archive.read(schema_name))
                if (
                    not isinstance(schema, dict)
                    or schema.get("$schema")
                    != "https://json-schema.org/draft/2020-12/schema"
                    or not str(schema.get("$id", "")).endswith(id_suffix)
                ):
                    raise WheelContractError("release-schema-invalid")
                Draft202012Validator.check_schema(schema)
                schemas[schema_name] = schema
            source_freeze = json.loads(
                archive.read("deploy/release/source-freeze-gates.json")
            )
            Draft202012Validator(
                schemas["deploy/release/source-freeze-gates.schema.json"]
            ).validate(source_freeze)
    except WheelContractError:
        raise
    except (
        OSError,
        SchemaError,
        UnicodeError,
        ValidationError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        raise WheelContractError("wheel-invalid") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args(argv)
    try:
        check_wheel(arguments.wheel)
    except WheelContractError as exc:
        print(f"release-wheel-contract: failed ({exc})", file=sys.stderr)
        return 1
    print("release-wheel-contract: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
