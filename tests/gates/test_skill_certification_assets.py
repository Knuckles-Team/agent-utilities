"""Fail-closed gates for packaged skill-certification assets."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

import agent_utilities.deployment.skill_validation_assets as assets
import agent_utilities.skills.runtime_validation as runtime_validation
from agent_utilities.core.config import ChatModelConfig
from agent_utilities.deployment.skill_validation import (
    SkillValidationDeployment,
    _lifecycle_subject,
)
from agent_utilities.skills.runtime_validation import CaseResult


def _model(*, identity: str, level: str, base_url: str, referenced: bool = True):
    return ChatModelConfig.model_validate(
        {
            "id": identity,
            "provider": "openai",
            "intelligence_level": level,
            "base_url": base_url,
            "api_key_ref": "env://MODEL_KEY" if referenced else None,
        }
    )


def _models() -> list[ChatModelConfig]:
    return [
        _model(
            identity="synthetic-light",
            level="light",
            base_url="http://127.0.0.1:8001/v1",
        ),
        _model(
            identity="synthetic-normal",
            level="normal",
            base_url="https://10.0.0.10:8443/v1",
        ),
    ]


def _identity_authority() -> dict[str, object]:
    return {
        "mode": "ephemeral-https-loopback",
        "tokenTtlSeconds": 300,
        "tlsVerificationRequired": True,
        "lifecycleOwned": True,
        "renewableCredentialsRequired": True,
    }


def _deployment(
    *,
    configuration_digest: str,
    profile_digest: str,
    model_registry: dict[str, object],
) -> SkillValidationDeployment:
    return SkillValidationDeployment.model_validate(
        {
            "apiVersion": "graphos.io/v2",
            "kind": "SkillValidationDeployment",
            "identityAuthority": _identity_authority(),
            "release": {
                "id": "release-synthetic-v2",
                "specificationReference": "SYNTHETIC_RELEASE_SPEC",
                "specificationDigest": "sha256:" + "1" * 64,
                "promotionEvidenceReference": "SYNTHETIC_PROMOTION_EVIDENCE",
                "promotionEvidenceDigest": "sha256:" + "2" * 64,
                "agentUtilitiesSha256": "sha256:" + "8" * 64,
                "agentUtilitiesFileCount": 100,
                "distributionClosureSha256": "sha256:" + "9" * 64,
                "releasePythonSha256": "sha256:" + "a" * 64,
                "graphOsDigest": "sha256:" + "3" * 64,
                "engineDigest": "sha256:" + "4" * 64,
                "startCommandReference": "SYNTHETIC_START_COMMAND",
            },
            "runtime": {
                "configurationReference": "SYNTHETIC_RUNTIME_CONFIGURATION",
                "configurationDigest": configuration_digest,
                "profileReference": "SYNTHETIC_RUNTIME_PROFILE",
                "profileDigest": profile_digest,
                "endpointReference": "SYNTHETIC_GRAPHOS_ENDPOINT",
                "modelRegistry": model_registry,
            },
            "readiness": {
                "timeoutSeconds": 30,
                "pollIntervalMilliseconds": 100,
            },
            "validation": {
                "caseTimeoutSeconds": 30,
                "signerCommandReference": "SYNTHETIC_SIGNER_COMMAND",
                "verifierCommandReference": "SYNTHETIC_VERIFIER_COMMAND",
            },
            "shutdown": {"graceSeconds": 5},
        }
    )


def _passing_results() -> list[CaseResult]:
    _defaults, cases = runtime_validation.load_matrix()
    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        opaque = f"{index:064x}"
        results.append(
            CaseResult(
                case_id=case.case_id,
                skill=case.skill,
                mode=case.mode,
                model_class=case.model_class,
                model_selection="pass",
                skill_binding="pass",
                structural="pass",
                semantic="pass",
                delegation=("pass" if case.mode == "delegated" else "not-applicable"),
                trace="pass",
                parent_ingestion="pass",
                trace_linkage="run-evidence",
                selected_routes=case.expected_routes,
                run_ref="pref_run_" + opaque,
                trace_ref="pref_trace_" + opaque,
                model_ref="pref_model_" + opaque,
                skill_ref="pref_skill_" + opaque,
                skill_body_ref="pref_skill_body_" + opaque,
                trace_name="graph_run:pref_run_" + opaque,
                langfuse_match_count=1,
                parent_kg_readback_count=1,
            )
        )
    return results


def test_model_registry_proof_is_content_free_and_exact() -> None:
    proof = assets.derive_model_registry_proof(_models(), ["127.0.0.1", "10.0.0.10"])

    assert proof == {
        "digest": proof["digest"],
        "modelCount": 2,
        "lightCount": 1,
        "normalCount": 1,
        "localPrivateTransportOnly": True,
        "referenceBackedCredentialsOnly": True,
        "literalPrivateModelCount": 2,
        "privateDnsModelCount": 0,
        "runtimePrivateResolutionRequired": True,
    }
    rendered = json.dumps(proof, sort_keys=True)
    assert "synthetic-light" not in rendered
    assert "127.0.0.1" not in rendered
    assert "10.0.0.10" not in rendered


def test_model_registry_accepts_ipv6_unique_local_transport() -> None:
    models = [
        _model(
            identity="synthetic-light",
            level="light",
            base_url="https://[fd00::1]:8443/v1",
        ),
        _models()[1],
    ]

    proof = assets.derive_model_registry_proof(models, ["fd00::1", "10.0.0.10"])

    assert proof["localPrivateTransportOnly"] is True
    assert "fd00::1" not in json.dumps(proof, sort_keys=True)


def test_installed_release_binding_is_recomputed_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.release import promote_local_release as promoter

    proof = assets.derive_model_registry_proof(_models(), ["127.0.0.1", "10.0.0.10"])
    deployment = _deployment(
        configuration_digest="sha256:" + "5" * 64,
        profile_digest="sha256:" + "6" * 64,
        model_registry=proof,
    )
    start_executable = tmp_path / "release" / "runtime" / "bin" / "graph-os"
    start_executable.parent.mkdir(parents=True)
    start_executable.write_bytes(b"graph-os")
    certification = {
        "agentUtilitiesSha256": "8" * 64,
        "agentUtilitiesFileCount": 100,
        "distributionClosureSha256": "9" * 64,
        "releasePythonSha256": "a" * 64,
        "graphosSha256": "3" * 64,
        "engineSha256": "4" * 64,
    }
    promotion_evidence = {"certificationArtifacts": certification}
    monkeypatch.setattr(
        promoter, "attest_installed_release", lambda release_root: certification
    )

    binding = assets.attest_installed_release_binding(
        deployment,
        start_executable=start_executable,
        promotion_evidence=promotion_evidence,
    )

    assert binding == {
        "agentUtilitiesSha256": "sha256:" + "8" * 64,
        "agentUtilitiesFileCount": 100,
        "distributionClosureSha256": "sha256:" + "9" * 64,
        "releasePythonSha256": "sha256:" + "a" * 64,
        "graphOsDigest": "sha256:" + "3" * 64,
        "engineDigest": "sha256:" + "4" * 64,
    }

    mismatched = {**certification, "releasePythonSha256": "b" * 64}
    monkeypatch.setattr(
        promoter, "attest_installed_release", lambda release_root: mismatched
    )
    with pytest.raises(
        assets.CertificationAssetError,
        match="installed_release_attestation_mismatch",
    ):
        assets.attest_installed_release_binding(
            deployment,
            start_executable=start_executable,
            promotion_evidence=promotion_evidence,
        )


def test_engine_campaign_marker_survives_sanitized_child_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.knowledge_graph.core.graph_compute import (
        _engine_child_environment,
    )

    marker_name = "EPISTEMIC_GRAPH_SKILL_VALIDATION_INSTANCE"
    monkeypatch.setenv(marker_name, "opaque-campaign-marker")

    assert _engine_child_environment()[marker_name] == "opaque-campaign-marker"


@pytest.mark.parametrize(
    "command",
    [
        ["/bin/sh", "-c", "true"],
        ["python", "-c", "print('not executed')"],
    ],
)
def test_external_evidence_command_rejects_shells_and_relative_executables(
    command: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = "SYNTHETIC_EVIDENCE_COMMAND"
    monkeypatch.setenv(reference, json.dumps(command))

    with pytest.raises(RuntimeError, match="evidence_command_reference_invalid"):
        runtime_validation._external_command(reference)


def test_external_evidence_command_rejects_symlink_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = "SYNTHETIC_EVIDENCE_COMMAND"
    alias = tmp_path / "signature-adapter"
    alias.symlink_to(Path(sys.executable))
    monkeypatch.setenv(reference, json.dumps([str(alias)]))

    with pytest.raises(RuntimeError, match="evidence_command_reference_invalid"):
        runtime_validation._external_command(reference)


@pytest.mark.parametrize(
    ("models", "hosts"),
    [
        (
            [
                _model(
                    identity="synthetic-light",
                    level="light",
                    base_url="https://example.invalid/v1",
                ),
                _models()[1],
            ],
            ["10.0.0.10"],
        ),
        (
            [
                _model(
                    identity="synthetic-light",
                    level="light",
                    base_url="http://127.0.0.1:8001/v1",
                    referenced=False,
                ),
                _models()[1],
            ],
            ["127.0.0.1", "10.0.0.10"],
        ),
        (
            [_models()[0], _models()[0].model_copy(update={"id": "duplicate-tier"})],
            ["127.0.0.1"],
        ),
    ],
)
def test_model_registry_rejects_remote_unreferenced_and_ambiguous_models(
    models: list[ChatModelConfig], hosts: list[str]
) -> None:
    with pytest.raises(assets.CertificationAssetError):
        assets.derive_model_registry_proof(models, hosts)


def test_model_registry_proves_unique_private_dns_without_exposing_addresses() -> None:
    models = [
        _model(
            identity="synthetic-light",
            level="light",
            base_url="https://light.internal/v1",
        ),
        _model(
            identity="synthetic-normal",
            level="normal",
            base_url="https://normal.internal/v1",
        ),
    ]

    def resolver(host: str, _port: int | None, *, type: int):
        assert type > 0
        address = "10.0.0.11" if host == "light.internal" else "10.0.0.12"
        return [(2, 1, 6, "", (address, 443))]

    proof = assets.prove_model_registry_runtime(
        models,
        ["light.internal", "normal.internal"],
        resolver=resolver,
    )

    assert proof == {
        "modelCount": 2,
        "literalPrivateModelCount": 0,
        "privateDnsModelCount": 2,
        "privateDnsUniqueResolutionProven": True,
        "privateBoundaryProven": True,
        "dnsRebindingGuarded": True,
    }
    rendered = json.dumps(proof, sort_keys=True)
    assert "internal" not in rendered
    assert "10.0.0" not in rendered


@pytest.mark.parametrize(
    "answers",
    [
        ["93.184.216.34"],
        ["10.0.0.11", "10.0.0.12"],
    ],
)
def test_model_registry_rejects_public_or_ambiguous_private_dns(
    answers: list[str],
) -> None:
    models = [
        _model(
            identity="synthetic-light",
            level="light",
            base_url="https://model.internal/v1",
        ),
        _models()[1],
    ]

    def resolver(host: str, _port: int | None, *, type: int):
        assert type > 0
        selected = answers if host == "model.internal" else ["10.0.0.10"]
        return [(2, 1, 6, "", (address, 443)) for address in selected]

    with pytest.raises(
        assets.CertificationAssetError, match="runtime_model_private_dns_unproven"
    ):
        assets.prove_model_registry_runtime(
            models,
            ["model.internal", "10.0.0.10"],
            resolver=resolver,
        )


@pytest.mark.parametrize(
    ("base_url", "host"),
    [
        ("http://0.0.0.0:8001/v1", "0.0.0.0"),
        ("http://169.254.1.1:8001/v1", "169.254.1.1"),
        ("http://192.0.2.1:8001/v1", "192.0.2.1"),
        ("http://240.0.0.1:8001/v1", "240.0.0.1"),
        ("http://[::]:8001/v1", "::"),
        ("http://[fe80::1]:8001/v1", "fe80::1"),
        ("http://[2001:db8::1]:8001/v1", "2001:db8::1"),
    ],
)
def test_model_registry_rejects_special_nonprivate_addresses(
    base_url: str, host: str
) -> None:
    models = [
        _model(identity="synthetic-light", level="light", base_url=base_url),
        _models()[1],
    ]

    with pytest.raises(
        assets.CertificationAssetError, match="runtime_model_locality_unproven"
    ):
        assets.derive_model_registry_proof(models, [host, "10.0.0.10"])


def test_model_registry_rejects_invalid_port() -> None:
    models = [
        _model(
            identity="synthetic-light",
            level="light",
            base_url="http://127.0.0.1:99999/v1",
        ),
        _models()[1],
    ]

    with pytest.raises(
        assets.CertificationAssetError, match="runtime_model_transport_invalid"
    ):
        assets.derive_model_registry_proof(models, ["127.0.0.1", "10.0.0.10"])


def test_runtime_materials_are_recomputed_and_profile_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = tmp_path / "config.json"
    configuration.write_text(
        json.dumps(
            {
                "CHAT_MODELS": [model.model_dump() for model in _models()],
                "MODEL_HTTP_ALLOWED_PRIVATE_HOSTS": ["127.0.0.1", "10.0.0.10"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    configuration_digest = assets._digest(configuration.read_bytes())
    proof = assets.derive_model_registry_proof(_models(), ["127.0.0.1", "10.0.0.10"])
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "apiVersion": "graphos.io/v2",
                "kind": "SkillValidationRuntimeProfile",
                "configurationDigest": configuration_digest,
                "modelRegistryDigest": proof["digest"],
                "identityAuthority": _identity_authority(),
                "engineTopology": "local-autostart",
                "observability": "metadata-only",
                "sequential": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    deployment = _deployment(
        configuration_digest=configuration_digest,
        profile_digest=assets._digest(profile.read_bytes()),
        model_registry=proof,
    )
    monkeypatch.setenv("SYNTHETIC_RUNTIME_CONFIGURATION", str(configuration))
    monkeypatch.setenv("SYNTHETIC_RUNTIME_PROFILE", str(profile))

    assert assets.load_runtime_materials(
        deployment, require_active_configuration=False
    ) == {
        "modelRegistry": proof,
        "identityAuthority": _identity_authority(),
        "models": _models(),
        "modelPrivateHosts": ["127.0.0.1", "10.0.0.10"],
    }

    configuration.write_text("{}", encoding="utf-8")
    with pytest.raises(assets.CertificationAssetError):
        assets.load_runtime_materials(deployment, require_active_configuration=False)


def test_runtime_profile_generator_is_deterministic_and_reference_driven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configuration = tmp_path / "config.json"
    configuration.write_text(
        json.dumps(
            {
                "CHAT_MODELS": [model.model_dump() for model in _models()],
                "MODEL_HTTP_ALLOWED_PRIVATE_HOSTS": ["127.0.0.1", "10.0.0.10"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "profile.json"
    monkeypatch.setenv("SYNTHETIC_CONFIGURATION_REFERENCE", str(configuration))
    monkeypatch.setenv("SYNTHETIC_PROFILE_REFERENCE", str(profile))

    generated = assets.generate_runtime_profile(
        configuration_reference="SYNTHETIC_CONFIGURATION_REFERENCE",
        profile_reference="SYNTHETIC_PROFILE_REFERENCE",
    )
    first = profile.read_bytes()
    assert generated == {
        "apiVersion": "graphos.io/v2",
        "kind": "SkillValidationRuntimeProfile",
        "configurationDigest": assets._digest(configuration.read_bytes()),
        "modelRegistryDigest": assets.derive_model_registry_proof(
            _models(), ["127.0.0.1", "10.0.0.10"]
        )["digest"],
        "identityAuthority": _identity_authority(),
        "engineTopology": "local-autostart",
        "observability": "metadata-only",
        "sequential": True,
    }
    assert stat.S_IMODE(profile.stat().st_mode) == 0o600

    assert (
        assets.profile_main(
            [
                "--configuration-reference",
                "SYNTHETIC_CONFIGURATION_REFERENCE",
                "--profile-reference",
                "SYNTHETIC_PROFILE_REFERENCE",
            ]
        )
        == 0
    )
    assert profile.read_bytes() == first
    output = capsys.readouterr().out
    assert json.loads(output) == {"ok": True}
    assert str(tmp_path) not in output


def test_runtime_profile_generator_rejects_configuration_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = tmp_path / "config.json"
    configuration.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SYNTHETIC_SHARED_REFERENCE", str(configuration))

    with pytest.raises(
        assets.CertificationAssetError, match="runtime_profile_destination_invalid"
    ):
        assets.generate_runtime_profile(
            configuration_reference="SYNTHETIC_SHARED_REFERENCE",
            profile_reference="SYNTHETIC_SHARED_REFERENCE",
        )


def test_runtime_materials_reject_symlinked_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = tmp_path / "config.json"
    configuration.write_text(
        json.dumps(
            {
                "CHAT_MODELS": [model.model_dump() for model in _models()],
                "MODEL_HTTP_ALLOWED_PRIVATE_HOSTS": ["127.0.0.1", "10.0.0.10"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    configuration_digest = assets._digest(configuration.read_bytes())
    proof = assets.derive_model_registry_proof(_models(), ["127.0.0.1", "10.0.0.10"])
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "apiVersion": "graphos.io/v2",
                "kind": "SkillValidationRuntimeProfile",
                "configurationDigest": configuration_digest,
                "modelRegistryDigest": proof["digest"],
                "identityAuthority": _identity_authority(),
                "engineTopology": "local-autostart",
                "observability": "metadata-only",
                "sequential": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    alias = tmp_path / "config-alias.json"
    alias.symlink_to(configuration)
    deployment = _deployment(
        configuration_digest=configuration_digest,
        profile_digest=assets._digest(profile.read_bytes()),
        model_registry=proof,
    )
    monkeypatch.setenv("SYNTHETIC_RUNTIME_CONFIGURATION", str(alias))
    monkeypatch.setenv("SYNTHETIC_RUNTIME_PROFILE", str(profile))

    with pytest.raises(
        assets.CertificationAssetError, match="runtime_configuration_invalid"
    ):
        assets.load_runtime_materials(deployment, require_active_configuration=False)


@pytest.mark.asyncio
async def test_readiness_rejects_nonlocal_endpoint_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = assets.derive_model_registry_proof(_models(), ["127.0.0.1", "10.0.0.10"])
    deployment = _deployment(
        configuration_digest="sha256:" + "5" * 64,
        profile_digest="sha256:" + "6" * 64,
        model_registry=proof,
    )
    monkeypatch.setattr(
        assets,
        "load_runtime_materials",
        lambda _deployment, *, require_active_configuration: {
            "models": _models(),
            "modelPrivateHosts": ["127.0.0.1", "10.0.0.10"],
        },
    )
    monkeypatch.setenv("SYNTHETIC_GRAPHOS_ENDPOINT", "https://example.invalid/mcp")

    with pytest.raises(assets.CertificationAssetError):
        await assets.probe_readiness(deployment, request_timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0.0, float("nan"), float("inf"), 120.1])
async def test_readiness_rejects_invalid_timeout_before_material_resolution(
    timeout: float,
) -> None:
    proof = assets.derive_model_registry_proof(_models(), ["127.0.0.1", "10.0.0.10"])
    deployment = _deployment(
        configuration_digest="sha256:" + "5" * 64,
        profile_digest="sha256:" + "6" * 64,
        model_registry=proof,
    )

    with pytest.raises(
        assets.CertificationAssetError, match="readiness_timeout_invalid"
    ):
        await assets.probe_readiness(deployment, request_timeout=timeout)


def test_source_gate_binds_runtime_and_release_certification_surfaces(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import check_skill_validation_certification

    assert check_skill_validation_certification.main() == 0
    assert capsys.readouterr().out == (
        "skill certification source gate: PASS "
        "(10 skills, 20 cases, 3 skill schemas, 4 release schemas)\n"
    )


def test_independent_verifier_rejects_tampered_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsigned = {"apiVersion": "graphos.io/v2", "kind": "SyntheticEvidence"}
    subject_digest = runtime_validation._digest_bytes(
        runtime_validation._canonical_bytes(unsigned)
    )
    key_id = "key:" + "a" * 64

    def external(reference: str, payload: bytes) -> dict[str, object]:
        if reference == "SYNTHETIC_SIGNER_COMMAND":
            return {
                "algorithm": "ed25519",
                "keyId": key_id,
                "signature": "A" * 43,
                "subjectDigest": subject_digest,
            }
        return {"verified": True, "subjectDigest": subject_digest, "keyId": key_id}

    monkeypatch.setattr(runtime_validation, "_external_json", external)
    signed = runtime_validation.sign_and_verify_evidence(
        unsigned,
        signer_reference="SYNTHETIC_SIGNER_COMMAND",
        verifier_reference="SYNTHETIC_VERIFIER_COMMAND",
    )
    runtime_validation.verify_signed_evidence(
        signed, verifier_reference="SYNTHETIC_VERIFIER_COMMAND"
    )
    signed["kind"] = "TamperedEvidence"
    with pytest.raises(RuntimeError, match="evidence_signature_invalid"):
        runtime_validation.verify_signed_evidence(
            signed, verifier_reference="SYNTHETIC_VERIFIER_COMMAND"
        )


def test_standalone_verifier_cross_binds_running_engine_and_validation_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof = assets.derive_model_registry_proof(_models(), ["127.0.0.1", "10.0.0.10"])
    deployment = _deployment(
        configuration_digest="sha256:" + "5" * 64,
        profile_digest="sha256:" + "6" * 64,
        model_registry=proof,
    )
    key_id = "key:" + "a" * 64

    def external(_reference: str, payload: bytes) -> dict[str, object]:
        value = json.loads(payload)
        if "signature" not in value:
            subject_digest = runtime_validation._digest_bytes(
                runtime_validation._canonical_bytes(value)
            )
            return {
                "algorithm": "ed25519",
                "keyId": key_id,
                "signature": "A" * 43,
                "subjectDigest": subject_digest,
            }
        signature = value.pop("signature")
        subject_digest = runtime_validation._digest_bytes(
            runtime_validation._canonical_bytes(value)
        )
        return {
            "verified": True,
            "subjectDigest": subject_digest,
            "keyId": signature["keyId"],
        }

    monkeypatch.setattr(runtime_validation, "_external_json", external)
    monkeypatch.setattr(assets, "verify_release_bindings", lambda _deployment: {})
    monkeypatch.setattr(
        assets,
        "load_runtime_materials",
        lambda _deployment, *, require_active_configuration: proof,
    )
    monkeypatch.setattr(
        assets,
        "attest_installed_release_binding",
        lambda _deployment, *, start_executable, promotion_evidence: {},
    )
    start_executable = tmp_path / "graph-os"
    start_executable.write_text(
        f"#!{sys.executable}\nraise SystemExit(0)\n", encoding="utf-8"
    )
    start_executable.chmod(0o700)
    monkeypatch.setenv("SYNTHETIC_START_COMMAND", json.dumps([str(start_executable)]))
    validation_unsigned = runtime_validation.build_evidence(
        _passing_results(),
        generated_at="2030-01-01T00:00:00Z",
        release_id=deployment.release.id,
        release_specification_digest=deployment.release.specification_digest,
        promotion_evidence_digest=deployment.release.promotion_evidence_digest,
        graph_os_digest=deployment.release.graph_os_digest,
        engine_digest=deployment.release.engine_digest,
        runtime_config_digest=deployment.runtime.configuration_digest,
        runtime_profile_digest=deployment.runtime.profile_digest,
        model_registry_digest=deployment.runtime.model_registry.digest,
    )
    validation = runtime_validation.sign_and_verify_evidence(
        validation_unsigned,
        signer_reference="SYNTHETIC_SIGNER_COMMAND",
        verifier_reference="SYNTHETIC_VERIFIER_COMMAND",
    )
    validation_payload = runtime_validation.render_evidence(validation).encode("utf-8")
    lifecycle_unsigned = _lifecycle_subject(
        deployment,
        global_counts=(0, 1, 0),
        graph_os_counts=(0, 1, 0),
        engine_counts=(0, 1, 0),
        identity_authority_counts=(0, 1, 0),
        terminal_process_counts=(0, 0),
        identity_tls_verified=True,
        renewable_credentials_proven=True,
        identity_token_mint_count=3,
        model_transport_proof={
            "modelCount": 2,
            "literalPrivateModelCount": 2,
            "privateDnsModelCount": 0,
            "privateDnsUniqueResolutionProven": True,
            "privateBoundaryProven": True,
            "dnsRebindingGuarded": True,
        },
        engine_executable_digest=deployment.release.engine_digest,
        installed_release_attested=True,
        reaped=True,
        validator_exit_code=0,
        validation_evidence_digest=runtime_validation._digest_bytes(validation_payload),
        validation_case_count=20,
        error_code=None,
    )
    lifecycle = runtime_validation.sign_and_verify_evidence(
        lifecycle_unsigned,
        signer_reference="SYNTHETIC_SIGNER_COMMAND",
        verifier_reference="SYNTHETIC_VERIFIER_COMMAND",
    )
    deployment_path = tmp_path / "deployment.json"
    validation_path = tmp_path / "validation.json"
    lifecycle_path = tmp_path / "lifecycle.json"
    deployment_path.write_text(
        json.dumps(deployment.model_dump(by_alias=True), sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_bytes(validation_payload)
    lifecycle_path.write_text(
        runtime_validation.render_evidence(lifecycle), encoding="utf-8"
    )

    assets.verify_certification_documents(
        deployment_path=deployment_path,
        validation_evidence_path=validation_path,
        lifecycle_evidence_path=lifecycle_path,
    )

    lifecycle_unsigned["processGate"]["engineExecutableDigest"] = "sha256:" + "f" * 64
    tampered = runtime_validation.sign_and_verify_evidence(
        lifecycle_unsigned,
        signer_reference="SYNTHETIC_SIGNER_COMMAND",
        verifier_reference="SYNTHETIC_VERIFIER_COMMAND",
    )
    lifecycle_path.write_text(
        runtime_validation.render_evidence(tampered), encoding="utf-8"
    )
    with pytest.raises(
        assets.CertificationAssetError, match="lifecycle_engine_binding_mismatch"
    ):
        assets.verify_certification_documents(
            deployment_path=deployment_path,
            validation_evidence_path=validation_path,
            lifecycle_evidence_path=lifecycle_path,
        )

    lifecycle_unsigned["processGate"]["engineExecutableDigest"] = (
        deployment.release.engine_digest
    )
    lifecycle_unsigned["processGate"]["terminalProcessCounts"][
        "langfuseMcpChildren"
    ] = 1
    lifecycle_unsigned["processGate"]["reaped"] = False
    lifecycle_unsigned["result"] = "fail"
    lifecycle_unsigned["errorCode"] = "terminal_process_count_invalid"
    tampered = runtime_validation.sign_and_verify_evidence(
        lifecycle_unsigned,
        signer_reference="SYNTHETIC_SIGNER_COMMAND",
        verifier_reference="SYNTHETIC_VERIFIER_COMMAND",
    )
    lifecycle_path.write_text(
        runtime_validation.render_evidence(tampered), encoding="utf-8"
    )
    with pytest.raises(
        assets.CertificationAssetError,
        match="lifecycle_terminal_process_count_mismatch",
    ):
        assets.verify_certification_documents(
            deployment_path=deployment_path,
            validation_evidence_path=validation_path,
            lifecycle_evidence_path=lifecycle_path,
        )
