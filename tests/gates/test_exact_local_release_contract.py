"""Static gates for the current-only exact local GraphOS promoter."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import tomllib
import zipfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from scripts.release import check_release_wheel as wheel_contract

ROOT = Path(__file__).resolve().parents[2]
PROMOTER = ROOT / "scripts/release/promote_local_release.py"
WHEEL_CHECKER = ROOT / "scripts/release/check_release_wheel.py"
CANARY = ROOT / "agent_utilities/deployment/release_canary.py"
SPEC_SCHEMA = ROOT / "deploy/release/exact-local-release-spec.schema.json"
EVIDENCE_SCHEMA = ROOT / "deploy/release/exact-local-release-evidence.schema.json"
DOC = ROOT / "docs/release/exact-local-release.md"
PIPELINE = ROOT / ".github/workflows/pipeline.yml"

_SKILL_CERTIFICATION_NAMES = (
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
_SKILL_CERTIFICATION_ASSETS = {
    asset
    for skill in _SKILL_CERTIFICATION_NAMES
    for asset in (
        f"agent_utilities/skills/{skill}/SKILL.md",
        f"agent_utilities/skills/{skill}/agents/graph-os.yaml",
        f"agent_utilities/skills/{skill}/agents/openai.yaml",
    )
}
_SKILL_CERTIFICATION_WHEEL_MEMBERS = _SKILL_CERTIFICATION_ASSETS | {
    "agent_utilities/deployment/certification_oidc.py",
    "agent_utilities/deployment/skill_validation.py",
    "agent_utilities/deployment/skill_validation_assets.py",
    "agent_utilities/prompts/agent-utilities-expert.json",
    "agent_utilities/skills/__init__.py",
    "agent_utilities/skills/runtime_validation.py",
    "agent_utilities/skills/runtime_validation.yaml",
    "agent_utilities/skills/validation.py",
    "deploy/release/prebundled-skill-validation-evidence.schema.json",
    "deploy/release/skill-validation-deployment-evidence.schema.json",
    "deploy/release/skill-validation-deployment.schema.json",
}
_SKILL_CERTIFICATION_ENTRY_POINTS = {
    "agent-utilities-validate-skills",
    "graph-os-certify-skills",
    "graph-os-generate-skill-certification",
    "graph-os-generate-skill-runtime-profile",
    "graph-os-skill-readiness",
    "graph-os-verify-skill-certification",
}


def test_publish_pipeline_requires_private_reproducible_wheels() -> None:
    source = PIPELINE.read_text(encoding="utf-8")

    assert 'SOURCE_DATE_EPOCH: "0"' in source
    assert source.count("uv build --wheel --no-build-isolation") == 2
    assert "--out-dir dist-primary" in source
    assert "--out-dir dist-reproduction" in source
    assert source.count("scripts/release/check_release_wheel.py") == 2
    assert source.count("scripts/check_wheel_privacy.py") == 2
    assert "Smoke installed Langfuse privacy support outside the checkout" in source
    assert "import agent_utilities.observability.langfuse_trust" in source
    assert "import agent_utilities.security.persistence_privacy" in source
    assert "release wheel digest mismatch" in source
    assert "dist-primary/agent_utilities-*.whl dist/" in source
    assert "uv build --sdist" not in source
    assert "dist/*.tar.gz" not in source


def test_promoter_is_closed_offline_hash_locked_and_package_name_only() -> None:
    source = PROMOTER.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        '"--offline"',
        '"--no-index"',
        '"--find-links"',
        '"--require-hashes"',
        '"--no-deps"',
        '"--link-mode"',
        '"copy"',
        "direct-requirement-forbidden",
        "wheelhouse-not-closed",
        "installed-direct-url-record",
    ):
        assert required in source
    assert " @ file:" not in source
    assert "shell=True" not in source
    assert "os.system" not in source


def test_promoter_binds_native_record_process_and_atomic_rollback_gates() -> None:
    source = PROMOTER.read_text(encoding="utf-8")
    for required in (
        "record-content-mismatch",
        "unrecorded-site-package-file",
        "dependency-closure-not-minimal",
        "locked-extra-not-reachable",
        "engine-binary-digest-mismatch",
        "numeric-extension-digest-mismatch",
        "installed-symlink",
        "installed-hardlink",
        "installed-special-file",
        "installed-path-escape",
        "evidence-destination-must-be-new",
        "activation-journal",
        "python-toolchain-mismatch",
        "uv-toolchain-mismatch",
        "release-tree-writable",
        "running_graph_process_count",
        "graph-processes-running",
        "os.replace(",
        "_atomic_current_replace(root_fd, previous)",
        'evidence["activation"]["rollback"] = "completed"',
    ):
        assert required in source
    assert "shutil.rmtree" not in source
    assert "os.rmdir(release_root" not in source


def test_exact_spec_schema_has_no_extension_or_fallback_surface() -> None:
    schema = json.loads(SPEC_SCHEMA.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["kind"]["const"] == "ExactLocalRelease"
    packages = schema["properties"]["packages"]
    assert packages["additionalProperties"] is False
    assert packages["required"] == [
        "agent-utilities",
        "epistemic-graph",
        "langfuse-agent",
    ]
    native = schema["properties"]["nativeArtifacts"]
    assert native["required"] == [
        "epistemic-graph-server",
        "epistemic-graph-numeric",
    ]
    assert schema["properties"]["apiVersion"]["const"] == "graphos.io/v2"
    assert schema["properties"]["toolchain"]["required"] == ["python", "uv"]
    commands = schema["properties"]["commands"]
    assert commands["additionalProperties"] is False
    assert commands["required"] == ["canary", "doctor"]


def _valid_spec_document() -> dict:
    digest = "sha256:" + "1" * 64
    package = {
        "version": "1.2.3",
        "wheel": "package-1.2.3-py3-none-any.whl",
        "sha256": digest,
    }
    return {
        "apiVersion": "graphos.io/v2",
        "kind": "ExactLocalRelease",
        "releaseId": "release-test",
        "requirements": {"file": "release-requirements.txt", "sha256": digest},
        "packages": {
            "agent-utilities": dict(package),
            "epistemic-graph": dict(package),
            "langfuse-agent": dict(package),
        },
        "nativeArtifacts": {
            "epistemic-graph-server": digest,
            "epistemic-graph-numeric": digest,
        },
        "toolchain": {
            "python": {"version": "3.12.3", "sha256": digest},
            "uv": {"version": "0.11.7", "sha256": digest},
        },
        "commands": {
            "canary": {
                "entryPoint": "graph-os-release-canary",
                "arguments": ["--json"],
                "timeoutSeconds": 30,
            },
            "doctor": {
                "entryPoint": "agent-utilities-doctor",
                "arguments": [
                    "--json",
                    "--live",
                    "--only",
                    "engine_request_context",
                    "engine",
                    "langfuse",
                    "native_optimizer",
                    "skills",
                ],
                "timeoutSeconds": 120,
            },
        },
    }


def test_spec_schema_behavior_rejects_noncanonical_pep440_versions() -> None:
    schema = json.loads(SPEC_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    valid = _valid_spec_document()
    validator.validate(valid)

    for invalid_version in ("1.0 final", "01.0", "1.0+ABC-1"):
        invalid = copy.deepcopy(valid)
        invalid["packages"]["agent-utilities"]["version"] = invalid_version
        with pytest.raises(ValidationError):
            validator.validate(invalid)


def test_evidence_schema_is_path_free_and_aggregate_only() -> None:
    schema = json.loads(EVIDENCE_SCHEMA.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["privacySafe"]["const"] is True
    serialized = json.dumps(schema, sort_keys=True).casefold()
    for forbidden in (
        '"path"',
        '"hostname"',
        '"username"',
        '"endpoint"',
        '"argv"',
        '"stdout"',
        '"stderr"',
    ):
        assert forbidden not in serialized
    assert '"outputdigest"' in serialized
    assert '"artifactdigest"' in serialized
    promoted = schema["allOf"][0]["then"]["properties"]
    assert promoted["activation"]["properties"]["rollback"]["const"] == ("not-required")
    for role in ("venv", "install", "canary", "doctor"):
        assert promoted["commands"]["properties"][role]["$ref"].endswith(
            "passedCommandEvidence"
        )
    assert schema["properties"]["certificationArtifacts"]["oneOf"][1]["required"] == [
        "agentUtilitiesSha256",
        "agentUtilitiesFileCount",
        "distributionClosureSha256",
        "releasePythonSha256",
        "graphosSha256",
        "engineSha256",
    ]
    assert schema["properties"]["signature"]["$ref"].endswith("signature")
    closure = schema["properties"]["closure"]
    assert "distributionCount" in closure["required"]
    for removed in (
        "wheelCount",
        "lockedDistributionCount",
        "installedDistributionCount",
    ):
        assert removed not in serialized


def test_evidence_schema_behavior_has_no_unequal_distribution_count_state() -> None:
    schema = json.loads(EVIDENCE_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    digest = "sha256:" + "1" * 64
    raw_digest = "2" * 64
    evidence = {
        "apiVersion": "graphos.io/v2",
        "kind": "ExactLocalReleaseEvidence",
        "releaseId": "release-test",
        "status": "promoted",
        "specDigest": digest,
        "requirementsDigest": digest,
        "packages": {
            name: {"version": "1.2.3", "artifactDigest": digest}
            for name in ("agent-utilities", "epistemic-graph", "langfuse-agent")
        },
        "nativeArtifacts": {
            "epistemic-graph-server": digest,
            "epistemic-graph-numeric": digest,
        },
        "toolchain": {
            name: {"version": version, "artifactDigest": digest}
            for name, version in (("python", "3.12.3"), ("uv", "0.11.7"))
        },
        "certificationArtifacts": {
            "agentUtilitiesSha256": raw_digest,
            "agentUtilitiesFileCount": 10,
            "distributionClosureSha256": raw_digest,
            "releasePythonSha256": raw_digest,
            "graphosSha256": raw_digest,
            "engineSha256": raw_digest,
        },
        "closure": {
            "distributionCount": 3,
            "recordVerified": True,
            "directUrlRecordCount": 0,
            "symlinkCount": 0,
            "specialFileCount": 0,
            "nativeArtifactCount": 2,
            "dependencyEdgeCount": 1,
            "releaseTreeEntryCount": 1,
            "immutableAfterProof": True,
        },
        "processGate": {"beforePromotion": 0, "afterVerification": 0},
        "activation": {
            "method": "atomic-symlink-replace",
            "hadPreviousRelease": False,
            "rollback": "not-required",
        },
        "commands": {
            role: {"status": "passed", "exitCode": 0, "outputDigest": digest}
            for role in ("venv", "install", "canary", "doctor")
        },
        "errorCode": None,
        "privacySafe": True,
        "signature": {
            "algorithm": "ed25519",
            "keyId": "key:" + "3" * 64,
            "signature": "A" * 86,
            "subjectDigest": digest,
        },
    }
    validator.validate(evidence)

    zero = copy.deepcopy(evidence)
    zero["closure"]["distributionCount"] = 0
    with pytest.raises(ValidationError):
        validator.validate(zero)
    legacy = copy.deepcopy(evidence)
    legacy["closure"]["wheelCount"] = 3
    with pytest.raises(ValidationError):
        validator.validate(legacy)


def test_release_canary_is_installed_bounded_and_non_serving() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["graph-os-release-canary"] == (
        "agent_utilities.deployment.release_canary:main"
    )
    source = CANARY.read_text(encoding="utf-8")
    ast.parse(source)
    assert "mcp_server()" not in source
    assert "subprocess" not in source
    assert '"privacySafe": True' in source
    assert 'Path(sys.executable).with_name("epistemic-graph-server")' in source


def test_promoter_verifier_and_schemas_are_installed() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert scripts["assemble-exact-local-release"] == (
        "scripts.release.assemble_exact_local_release:main"
    )
    assert scripts["promote-local-graphos-release"] == (
        "scripts.release.promote_local_release:main"
    )
    assert scripts["materialize-component-wheelhouse"] == (
        "scripts.release.materialize_component_wheelhouse:main"
    )
    assert scripts["generate-oci-acquisition-attestation"] == (
        "scripts.release.generate_oci_acquisition_attestation:main"
    )
    assert scripts["generate-oci-vulnerability-scan-evidence"] == (
        "scripts.release.generate_oci_vulnerability_scan_evidence:main"
    )
    assert scripts["verify-local-graphos-release-evidence"] == (
        "scripts.release.promote_local_release:verify_main"
    )
    assert pyproject["tool"]["setuptools"]["package-data"]["deploy.release"] == [
        "*.schema.json",
        "*.catalog.json",
        "source-freeze-gates.json",
        "compatibility-matrix.yml",
        "certification-campaign.yml",
    ]
    included = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "scripts.release*" in included
    assert "deploy.release*" in included
    assert (ROOT / "scripts/release/__init__.py").is_file()
    assert (ROOT / "deploy/release/__init__.py").is_file()
    release_scripts = {
        name: target
        for name, target in scripts.items()
        if target.startswith(
            ("scripts.release.", "scripts.certification.", "scripts.scale.")
        )
    }
    assert release_scripts.items() <= wheel_contract._ENTRY_POINTS.items()
    assert {
        target.partition(":")[0].replace(".", "/") + ".py"
        for target in release_scripts.values()
    } <= wheel_contract._REQUIRED_MEMBERS
    checker = WHEEL_CHECKER.read_text(encoding="utf-8")
    ast.parse(checker)
    assert {
        "agent_utilities/deployment/certification_oidc.py",
        "agent_utilities/deployment/skill_validation.py",
        "agent_utilities/deployment/skill_validation_assets.py",
        "agent_utilities/skills/runtime_validation.py",
        "agent_utilities/skills/runtime_validation.yaml",
        "agent_utilities/skills/validation.py",
        "scripts/release/generate_oci_acquisition_attestation.py",
        "scripts/release/exact_artifact_closure.py",
        "scripts/release/exact_local_gates_manifest.py",
        "deploy/release/oci-scanner-attestation.schema.json",
        "deploy/release/oci-vulnerability-database-attestation.schema.json",
        "deploy/release/prebundled-skill-validation-evidence.schema.json",
        "deploy/release/skill-validation-deployment-evidence.schema.json",
        "deploy/release/skill-validation-deployment.schema.json",
    } <= wheel_contract._REQUIRED_MEMBERS
    assert wheel_contract._SCHEMA_ID_SUFFIXES[
        "deploy/release/oci-scanner-attestation.schema.json"
    ] == "oci-scanner-attestation-v1.json"
    assert wheel_contract._SCHEMA_ID_SUFFIXES[
        "deploy/release/oci-vulnerability-database-attestation.schema.json"
    ] == "oci-vulnerability-database-attestation-v1.json"
    assert wheel_contract._SCHEMA_ID_SUFFIXES[
        "deploy/release/prebundled-skill-validation-evidence.schema.json"
    ] == "prebundled-skill-validation-evidence-v2.json"
    assert wheel_contract._SCHEMA_ID_SUFFIXES[
        "deploy/release/skill-validation-deployment-evidence.schema.json"
    ] == "skill-validation-lifecycle-evidence-v2.json"
    assert wheel_contract._SCHEMA_ID_SUFFIXES[
        "deploy/release/skill-validation-deployment.schema.json"
    ] == "skill-validation-deployment-v2.json"
    assert wheel_contract._ENTRY_POINTS[
        "generate-oci-acquisition-attestation"
    ] == "scripts.release.generate_oci_acquisition_attestation:main"
    assert wheel_contract._ENTRY_POINTS[
        "generate-exact-local-gates-manifest"
    ] == "scripts.release.exact_local_gates_manifest:main"
    assert wheel_contract._ENTRY_POINTS[
        "bind-exact-local-release-evidence"
    ] == "scripts.release.exact_artifact_closure:main"
    assert {
        "agent-utilities-validate-skills": (
            "agent_utilities.skills.runtime_validation:main"
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
    }.items() <= wheel_contract._ENTRY_POINTS.items()
    for required in (
        "scripts/release/assemble_exact_local_release.py",
        "scripts/release/generate_oci_acquisition_attestation.py",
        "scripts/release/exact_artifact_closure.py",
        "scripts/release/exact_local_gates_manifest.py",
        "scripts/release/generate_oci_vulnerability_scan_evidence.py",
        "scripts/release/materialize_component_wheelhouse.py",
        "scripts/release/promote_local_release.py",
        "agent_utilities/deployment/certification_oidc.py",
        "agent_utilities/deployment/skill_validation.py",
        "agent_utilities/deployment/skill_validation_assets.py",
        "agent_utilities/skills/runtime_validation.py",
        "agent_utilities/skills/runtime_validation.yaml",
        "agent_utilities/skills/validation.py",
        "exact-local-release-evidence.schema.json",
        "oci-scanner-attestation.schema.json",
        "oci-scanner-attestation-v1.json",
        "oci-vulnerability-database-attestation.schema.json",
        "oci-vulnerability-database-attestation-v1.json",
        "oci-vulnerability-scan-evidence.schema.json",
        "oci-vulnerability-scan-evidence-v1.json",
        "prebundled-skill-validation-evidence.schema.json",
        "prebundled-skill-validation-evidence-v2.json",
        "skill-validation-deployment-evidence.schema.json",
        "skill-validation-lifecycle-evidence-v2.json",
        "skill-validation-deployment.schema.json",
        "skill-validation-deployment-v2.json",
        "assemble-exact-local-release",
        "generate-oci-acquisition-attestation",
        "generate-exact-local-gates-manifest",
        "bind-exact-local-release-evidence",
        "generate-oci-vulnerability-scan-evidence",
        "materialize-component-wheelhouse",
        "promote-local-graphos-release",
        "verify-local-graphos-release-evidence",
        "graph-os-certify-skills",
        "agent-utilities-validate-skills",
        "graph-os-generate-skill-certification",
        "graph-os-generate-skill-runtime-profile",
        "graph-os-skill-readiness",
        "graph-os-verify-skill-certification",
        "full-engine-requirement-invalid",
        'SpecifierSet(">=2.23.1,<3.0.0")',
    ):
        assert required in checker


def _synthetic_release_wheel(
    path: Path,
    *,
    omitted_member: str | None = None,
    omitted_entry_point: str | None = None,
    entry_point_overrides: dict[str, str] | None = None,
    extra_members: tuple[str, ...] = (),
    member_payloads: dict[str, bytes] | None = None,
) -> Path:
    members = wheel_contract._REQUIRED_MEMBERS - {omitted_member}
    entry_points = {
        name: target
        for name, target in wheel_contract._ENTRY_POINTS.items()
        if name != omitted_entry_point
    }
    entry_points.update(entry_point_overrides or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(members | set(extra_members)):
            payload = (member_payloads or {}).get(name)
            if payload is None and (
                name == wheel_contract._RELEASE_RESOURCE_CATALOG
                or name in wheel_contract._RELEASE_RESOURCE_PATHS
            ):
                payload = (ROOT / name).read_bytes()
            if payload is None:
                payload = b""
            archive.writestr(name, payload)
        archive.writestr(
            "agent_utilities-1.0.0.dist-info/METADATA",
            (
                "Metadata-Version: 2.4\n"
                "Name: agent-utilities\n"
                "Version: 1.0.0\n"
                "Requires-Dist: epistemic-graph[full]>=2.23.1,<3.0.0\n\n"
            ),
        )
        archive.writestr(
            "agent_utilities-1.0.0.dist-info/entry_points.txt",
            "[console_scripts]\n"
            + "".join(
                f"{name} = {target}\n" for name, target in sorted(entry_points.items())
            ),
        )
    return path


_PRODUCTION_CERTIFICATION_WHEEL_MEMBERS = {
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
    "scripts/scale/workload_contract.yml",
}
_PRODUCTION_CERTIFICATION_ENTRY_POINTS = {
    "graphos-certification-campaign",
    "graphos-certification-fault",
    "graphos-certification-load",
    "graphos-certification-metrics",
    "graphos-operational-evidence",
}
_EXACT_CLOSURE_WHEEL_MEMBERS = {
    "scripts/release/exact_artifact_closure.py",
    "scripts/release/exact_local_gates_manifest.py",
}
_EXACT_CLOSURE_ENTRY_POINTS = {
    "bind-exact-local-release-evidence",
    "generate-exact-local-gates-manifest",
}


def test_release_wheel_contract_requires_exact_skill_certification_surface(
    tmp_path: Path,
) -> None:
    assert wheel_contract._BUNDLED_SKILL_NAMES == _SKILL_CERTIFICATION_NAMES
    assert wheel_contract._BUNDLED_SKILL_ASSETS == _SKILL_CERTIFICATION_ASSETS
    assert _SKILL_CERTIFICATION_WHEEL_MEMBERS <= wheel_contract._REQUIRED_MEMBERS
    assert _SKILL_CERTIFICATION_ENTRY_POINTS <= wheel_contract._ENTRY_POINTS.keys()
    wheel_contract.check_wheel(_synthetic_release_wheel(tmp_path / "complete.whl"))

    for index, member in enumerate(sorted(_SKILL_CERTIFICATION_WHEEL_MEMBERS)):
        wheel = _synthetic_release_wheel(
            tmp_path / f"missing-member-{index}.whl",
            omitted_member=member,
        )
        with pytest.raises(
            wheel_contract.WheelContractError, match="release-surface-missing"
        ):
            wheel_contract.check_wheel(wheel)

    for index, entry_point in enumerate(
        sorted(_SKILL_CERTIFICATION_ENTRY_POINTS)
    ):
        wheel = _synthetic_release_wheel(
            tmp_path / f"missing-entry-point-{index}.whl",
            omitted_entry_point=entry_point,
        )
        with pytest.raises(
            wheel_contract.WheelContractError, match="entry-points-invalid"
        ):
            wheel_contract.check_wheel(wheel)

    stale_skill = _synthetic_release_wheel(
        tmp_path / "stale-skill.whl",
        extra_members=("agent_utilities/skills/retired-skill/SKILL.md",),
    )
    with pytest.raises(
        wheel_contract.WheelContractError, match="skill-surface-invalid"
    ):
        wheel_contract.check_wheel(stale_skill)


def test_release_wheel_contract_requires_installed_production_campaign_surface(
    tmp_path: Path,
) -> None:
    assert _PRODUCTION_CERTIFICATION_WHEEL_MEMBERS <= wheel_contract._REQUIRED_MEMBERS
    assert _PRODUCTION_CERTIFICATION_ENTRY_POINTS <= wheel_contract._ENTRY_POINTS.keys()

    for index, member in enumerate(sorted(_PRODUCTION_CERTIFICATION_WHEEL_MEMBERS)):
        wheel = _synthetic_release_wheel(
            tmp_path / f"missing-production-member-{index}.whl",
            omitted_member=member,
        )
        with pytest.raises(
            wheel_contract.WheelContractError, match="release-surface-missing"
        ):
            wheel_contract.check_wheel(wheel)

    for index, entry_point in enumerate(
        sorted(_PRODUCTION_CERTIFICATION_ENTRY_POINTS)
    ):
        wheel = _synthetic_release_wheel(
            tmp_path / f"missing-production-entry-{index}.whl",
            omitted_entry_point=entry_point,
        )
        with pytest.raises(
            wheel_contract.WheelContractError, match="entry-points-invalid"
        ):
            wheel_contract.check_wheel(wheel)


def test_release_wheel_contract_requires_installed_exact_closure_surface(
    tmp_path: Path,
) -> None:
    assert _EXACT_CLOSURE_WHEEL_MEMBERS <= wheel_contract._REQUIRED_MEMBERS
    assert _EXACT_CLOSURE_ENTRY_POINTS <= wheel_contract._ENTRY_POINTS.keys()

    for index, member in enumerate(sorted(_EXACT_CLOSURE_WHEEL_MEMBERS)):
        wheel = _synthetic_release_wheel(
            tmp_path / f"missing-closure-member-{index}.whl",
            omitted_member=member,
        )
        with pytest.raises(
            wheel_contract.WheelContractError, match="release-surface-missing"
        ):
            wheel_contract.check_wheel(wheel)

    for index, entry_point in enumerate(sorted(_EXACT_CLOSURE_ENTRY_POINTS)):
        missing = _synthetic_release_wheel(
            tmp_path / f"missing-closure-entry-{index}.whl",
            omitted_entry_point=entry_point,
        )
        with pytest.raises(
            wheel_contract.WheelContractError, match="entry-points-invalid"
        ):
            wheel_contract.check_wheel(missing)

        broken = _synthetic_release_wheel(
            tmp_path / f"broken-closure-entry-{index}.whl",
            entry_point_overrides={entry_point: "scripts.release.missing:main"},
        )
        with pytest.raises(
            wheel_contract.WheelContractError, match="entry-points-invalid"
        ):
            wheel_contract.check_wheel(broken)


def test_release_wheel_rejects_resource_or_catalog_schema_substitution(
    tmp_path: Path,
) -> None:
    schema_name = "deploy/release/release-configuration.schema.json"
    gutted_schema = json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://graphos.invalid/schemas/release-configuration-v1.json",
        }
    ).encode()
    wheel = _synthetic_release_wheel(
        tmp_path / "gutted-schema.whl",
        member_payloads={schema_name: gutted_schema},
    )
    with pytest.raises(
        wheel_contract.WheelContractError,
        match="release-resource-digest-mismatch",
    ):
        wheel_contract.check_wheel(wheel)

    catalog = json.loads(
        (ROOT / wheel_contract._RELEASE_RESOURCE_CATALOG).read_text()
    )
    for resource in catalog["resources"]:
        if resource["path"] == schema_name:
            resource["sha256"] = hashlib.sha256(gutted_schema).hexdigest()
    substituted_catalog = (
        json.dumps(catalog, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    wheel = _synthetic_release_wheel(
        tmp_path / "substituted-catalog.whl",
        member_payloads={
            schema_name: gutted_schema,
            wheel_contract._RELEASE_RESOURCE_CATALOG: substituted_catalog,
        },
    )
    with pytest.raises(
        wheel_contract.WheelContractError,
        match="release-resource-catalog-invalid",
    ):
        wheel_contract.check_wheel(wheel)


def test_operator_contract_is_in_strict_docs_navigation() -> None:
    document = DOC.read_text(encoding="utf-8")
    normalized_document = " ".join(document.split())
    navigation = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Exact Local GraphOS Releases: release/exact-local-release.md" in navigation
    for statement in (
        "It never deletes the previous release",
        "Root selection comes only from the package-name requirements lock",
        "atomically restores the previous `current` target",
        "contains no command arguments, filesystem paths, hostnames, usernames",
    ):
        assert statement in normalized_document
