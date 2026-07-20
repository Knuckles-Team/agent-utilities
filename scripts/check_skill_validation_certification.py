#!/usr/bin/env python3
"""Source gate for the exact bundled-skill certification contracts."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_utilities.deployment.skill_validation import (
    SkillValidationDeployment,
    run_deployment,
)
from agent_utilities.deployment.skill_validation_assets import (
    attest_installed_release_binding,
    derive_model_registry_proof,
    generate_deployment,
    generate_runtime_profile,
    probe_readiness,
    prove_model_registry_runtime,
    verify_certification_documents,
)
from agent_utilities.skills import BUNDLED_SKILLS
from agent_utilities.skills.runtime_validation import load_matrix
from scripts.release import assemble_manifest, check_compatibility
from scripts.release.generate_release_assembly import generate as generate_assembly

_CERTIFICATION_FIELDS = {
    "connectorLiveCertificationLedger",
    "prebundledSkillValidationMatrix",
    "skillValidationDeployment",
    "skillValidationLifecycleEvidence",
    "exactArtifactClosureEvidence",
    "ociVulnerabilityScanEvidence",
}


def _nested(value: object, *keys: str) -> object:
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError("certification_release_schema_binding_invalid")
        current = current[key]
    return current


def main() -> int:
    schema_names = (
        "prebundled-skill-validation-evidence.schema.json",
        "skill-validation-deployment.schema.json",
        "skill-validation-deployment-evidence.schema.json",
        "release-assembly.schema.json",
        "release-manifest.schema.json",
        "operational-evidence.schema.json",
        "oci-vulnerability-scan-evidence.schema.json",
    )
    try:
        schemas: dict[str, dict[str, object]] = {}
        for name in schema_names:
            schema = json.loads(
                (ROOT / "deploy" / "release" / name).read_text(encoding="utf-8")
            )
            Draft202012Validator.check_schema(schema)
            schemas[name] = schema
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = project["project"]["scripts"]
        if scripts.get("agent-utilities-validate-skills") != (
            "agent_utilities.skills.runtime_validation:main"
        ):
            raise RuntimeError("runtime_validator_entry_point_invalid")
        if scripts.get("graph-os-certify-skills") != (
            "agent_utilities.deployment.skill_validation:main"
        ):
            raise RuntimeError("deployment_owner_entry_point_invalid")
        expected_assets = {
            "graph-os-generate-skill-runtime-profile": (
                "agent_utilities.deployment.skill_validation_assets:profile_main"
            ),
            "graph-os-generate-skill-certification": (
                "agent_utilities.deployment.skill_validation_assets:generator_main"
            ),
            "graph-os-skill-readiness": (
                "agent_utilities.deployment.skill_validation_assets:readiness_main"
            ),
            "graph-os-verify-skill-certification": (
                "agent_utilities.deployment.skill_validation_assets:verifier_main"
            ),
        }
        if any(scripts.get(name) != target for name, target in expected_assets.items()):
            raise RuntimeError("certification_asset_entry_point_invalid")
        _defaults, cases = load_matrix()
        if len(BUNDLED_SKILLS) != 10 or len(cases) != 20:
            raise RuntimeError("skill_campaign_cardinality_invalid")
        if not all(case.read_only for case in cases):
            raise RuntimeError("skill_campaign_mutation_invalid")
        if not SkillValidationDeployment.model_fields["kind"].is_required():
            raise RuntimeError("deployment_kind_contract_invalid")
        if not all(
            callable(item)
            for item in (
                derive_model_registry_proof,
                generate_runtime_profile,
                attest_installed_release_binding,
                generate_deployment,
                probe_readiness,
                prove_model_registry_runtime,
                run_deployment,
                verify_certification_documents,
                generate_assembly,
                assemble_manifest.assemble,
                check_compatibility.validate_skill_validation_deployment,
                check_compatibility.validate_skill_validation_lifecycle,
            )
        ):
            raise RuntimeError("certification_asset_surface_invalid")
        asset_source = (
            ROOT / "agent_utilities" / "deployment" / "skill_validation_assets.py"
        ).read_text(encoding="utf-8")
        if "shell=True" in asset_source or "os.getenv(" in asset_source:
            raise RuntimeError("certification_asset_boundary_invalid")
        lifecycle_source = (
            ROOT / "agent_utilities" / "deployment" / "skill_validation.py"
        ).read_text(encoding="utf-8")
        if (
            "shell=True" in lifecycle_source
            or "os.getenv(" in lifecycle_source
            or "_validate_evidence_destinations" not in lifecycle_source
            or "_process_counts" not in lifecycle_source
            or "_marked_engine_digest" not in lifecycle_source
            or "verify_release_bindings(deployment)" not in lifecycle_source
            or "attest_installed_release_binding(" not in lifecycle_source
            or "runtime_materials = load_runtime_materials(" not in lifecycle_source
            or "require_active_configuration=True" not in lifecycle_source
            or "EphemeralLoopbackOidcAuthority" not in lifecycle_source
        ):
            raise RuntimeError("certification_lifecycle_boundary_invalid")
        runtime_source = (
            ROOT / "agent_utilities" / "skills" / "runtime_validation.py"
        ).read_text(encoding="utf-8")
        if (
            "shell=True" in runtime_source
            or "os.getenv(" in runtime_source
            or "_SHELL_EXECUTABLES" not in runtime_source
            or "not executable.is_absolute()" not in runtime_source
        ):
            raise RuntimeError("certification_execution_boundary_invalid")
        skill_schemas = [schemas[name] for name in schema_names[:3]]
        if [
            schema["properties"]["apiVersion"]["const"] for schema in skill_schemas
        ] != [
            "graphos.io/v2",
            "graphos.io/v2",
            "graphos.io/v2",
        ]:
            raise RuntimeError("certification_schema_version_invalid")
        if [schema["properties"]["kind"]["const"] for schema in skill_schemas] != [
            "PrebundledSkillValidationEvidence",
            "SkillValidationDeployment",
            "SkillValidationLifecycleEvidence",
        ]:
            raise RuntimeError("certification_schema_kind_invalid")
        installed_release_fields = {
            "agentUtilitiesSha256",
            "agentUtilitiesFileCount",
            "distributionClosureSha256",
            "releasePythonSha256",
        }
        deployment_release_required = _nested(
            schemas["skill-validation-deployment.schema.json"],
            "properties",
            "release",
            "required",
        )
        lifecycle_release_required = _nested(
            schemas["skill-validation-deployment-evidence.schema.json"],
            "properties",
            "release",
            "required",
        )
        lifecycle_process_required = _nested(
            schemas["skill-validation-deployment-evidence.schema.json"],
            "properties",
            "processGate",
            "required",
        )
        deployment_required = schemas["skill-validation-deployment.schema.json"][
            "required"
        ]
        lifecycle_required = schemas[
            "skill-validation-deployment-evidence.schema.json"
        ]["required"]
        if (
            not isinstance(deployment_release_required, list)
            or not isinstance(lifecycle_release_required, list)
            or not isinstance(lifecycle_process_required, list)
            or not installed_release_fields <= set(deployment_release_required)
            or not installed_release_fields <= set(lifecycle_release_required)
            or "installedReleaseAttested" not in lifecycle_process_required
            or "terminalProcessCounts" not in lifecycle_process_required
            or "identityAuthority" not in deployment_required
            or "identityAuthority" not in lifecycle_required
            or "modelTransportProof" not in lifecycle_required
        ):
            raise RuntimeError("certification_installed_release_binding_invalid")
        required_catalogs = (
            _nested(
                schemas["release-assembly.schema.json"],
                "properties",
                "certifications",
                "required",
            ),
            _nested(
                schemas["release-manifest.schema.json"],
                "properties",
                "evidence",
                "properties",
                "certifications",
                "required",
            ),
            _nested(
                schemas["release-manifest.schema.json"],
                "properties",
                "certificationDigests",
                "required",
            ),
            _nested(
                schemas["operational-evidence.schema.json"],
                "properties",
                "release",
                "properties",
                "certificationDigests",
                "required",
            ),
        )
        if any(
            not isinstance(catalog, list) or set(catalog) != _CERTIFICATION_FIELDS
            for catalog in required_catalogs
        ):
            raise RuntimeError("certification_release_schema_binding_invalid")
        authorities = _nested(
            schemas["release-manifest.schema.json"],
            "$defs",
            "gateAuthority",
            "properties",
            "authority",
            "enum",
        )
        if (
            not isinstance(authorities, list)
            or "certification:skillValidationLifecycleEvidence" not in authorities
            or "certification:skillValidationDeployment" in authorities
            or check_compatibility._CERTIFICATION_DIGESTS != _CERTIFICATION_FIELDS
        ):
            raise RuntimeError("certification_release_authority_invalid")
    except Exception as exc:  # noqa: BLE001 - aggregate source-gate diagnostic
        print(f"skill certification source gate: FAIL ({type(exc).__name__})")
        return 1
    print(
        "skill certification source gate: PASS "
        "(10 skills, 20 cases, 3 skill schemas, 4 release schemas)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
