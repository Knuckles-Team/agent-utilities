"""Current-only lifecycle contract for exact bundled-skill certification."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from agent_utilities.deployment.skill_validation import (
    DeploymentError,
    SkillValidationDeployment,
    load_deployment,
    run_deployment,
)
from agent_utilities.skills.runtime_validation import (
    load_matrix,
    minimum_campaign_authority_ttl_seconds,
)


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o700)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _deployment(graph_os: Path) -> SkillValidationDeployment:
    return SkillValidationDeployment.model_validate(
        {
            "apiVersion": "graphos.io/v2",
            "kind": "SkillValidationDeployment",
            "identityAuthority": {
                "mode": "ephemeral-https-loopback",
                "tokenTtlSeconds": 300,
                "tlsVerificationRequired": True,
                "lifecycleOwned": True,
                "renewableCredentialsRequired": True,
            },
            "release": {
                "id": "release-certification-v1",
                "specificationReference": "SYNTHETIC_RELEASE_SPECIFICATION",
                "specificationDigest": "sha256:" + "1" * 64,
                "promotionEvidenceReference": "SYNTHETIC_PROMOTION_EVIDENCE",
                "promotionEvidenceDigest": "sha256:" + "2" * 64,
                "agentUtilitiesSha256": "sha256:" + "8" * 64,
                "agentUtilitiesFileCount": 100,
                "distributionClosureSha256": "sha256:" + "9" * 64,
                "releasePythonSha256": "sha256:" + "a" * 64,
                "graphOsDigest": _digest(graph_os),
                "engineDigest": "sha256:" + "4" * 64,
                "startCommandReference": "SYNTHETIC_GRAPHOS_COMMAND",
            },
            "runtime": {
                "configurationReference": "SYNTHETIC_CONFIGURATION_MATERIAL",
                "configurationDigest": "sha256:" + "5" * 64,
                "profileReference": "SYNTHETIC_PROFILE_REFERENCE",
                "profileDigest": "sha256:" + "6" * 64,
                "endpointReference": "SYNTHETIC_ENDPOINT_REFERENCE",
                "modelRegistry": {
                    "digest": "sha256:" + "7" * 64,
                    "modelCount": 2,
                    "lightCount": 1,
                    "normalCount": 1,
                    "localPrivateTransportOnly": True,
                    "referenceBackedCredentialsOnly": True,
                    "literalPrivateModelCount": 2,
                    "privateDnsModelCount": 0,
                    "runtimePrivateResolutionRequired": True,
                },
            },
            "readiness": {
                "timeoutSeconds": 5,
                "pollIntervalMilliseconds": 50,
            },
            "validation": {
                "caseTimeoutSeconds": 5,
                "signerCommandReference": "SYNTHETIC_SIGNER_COMMAND",
                "verifierCommandReference": "SYNTHETIC_VERIFIER_COMMAND",
            },
            "shutdown": {"graceSeconds": 2},
        }
    )


def _deployment_payload(graph_os: Path) -> dict[str, Any]:
    return _deployment(graph_os).model_dump(by_alias=True)


def test_identity_token_ttl_covers_current_default_campaign_boundary(
    tmp_path: Path,
) -> None:
    graph_os = tmp_path / "graph-os"
    _write_executable(graph_os, "raise SystemExit(0)\n")
    defaults, _cases = load_matrix()
    trace_timeout = int(defaults["trace_timeout_seconds"])
    required_ttl = minimum_campaign_authority_ttl_seconds(
        case_timeout=120,
        trace_timeout=trace_timeout,
        shutdown_grace=30,
    )
    assert required_ttl == 245

    for insufficient_ttl in (180, 214, required_ttl - 1):
        payload = _deployment_payload(graph_os)
        payload["validation"]["caseTimeoutSeconds"] = 120
        payload["shutdown"]["graceSeconds"] = 30
        payload["identityAuthority"]["tokenTtlSeconds"] = insufficient_ttl
        with pytest.raises(
            ValueError, match="identity_token_ttl_campaign_window_invalid"
        ):
            SkillValidationDeployment.model_validate(payload)

    boundary = _deployment_payload(graph_os)
    boundary["validation"]["caseTimeoutSeconds"] = 120
    boundary["shutdown"]["graceSeconds"] = 30
    boundary["identityAuthority"]["tokenTtlSeconds"] = required_ttl
    deployment = SkillValidationDeployment.model_validate(boundary)
    assert deployment.identity_authority.token_ttl_seconds == required_ttl


def test_identity_token_ttl_floor_is_computed_from_campaign_windows(
    tmp_path: Path,
) -> None:
    graph_os = tmp_path / "graph-os"
    _write_executable(graph_os, "raise SystemExit(0)\n")
    payload = _deployment_payload(graph_os)
    payload["validation"]["caseTimeoutSeconds"] = 1
    payload["shutdown"]["graceSeconds"] = 1
    payload["identityAuthority"]["tokenTtlSeconds"] = 180

    deployment = SkillValidationDeployment.model_validate(payload)

    assert deployment.identity_authority.token_ttl_seconds == 180


def test_deployment_schema_contains_only_references_and_digests(tmp_path) -> None:
    graph_os = tmp_path / "graph-os"
    _write_executable(graph_os, "import time\ntime.sleep(1)\n")
    value = _deployment(graph_os).model_dump(by_alias=True)
    root = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (root / "deploy/release/skill-validation-deployment.schema.json").read_text(
            encoding="utf-8"
        )
    )

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(value)
    serialized = json.dumps(value, sort_keys=True)
    assert "opaque-runtime-reference" not in serialized
    assert "profile:synthetic" not in serialized
    assert str(tmp_path) not in serialized


def test_deployment_document_rejects_symlink_material(tmp_path: Path) -> None:
    graph_os = tmp_path / "graph-os"
    _write_executable(graph_os, "import time\ntime.sleep(1)\n")
    target = tmp_path / "deployment.json"
    target.write_text(
        json.dumps(_deployment(graph_os).model_dump(by_alias=True)),
        encoding="utf-8",
    )
    alias = tmp_path / "deployment-alias.json"
    alias.symlink_to(target)

    with pytest.raises(DeploymentError, match="configuration_invalid"):
        load_deployment(alias)


@pytest.mark.parametrize(
    ("argv", "expected"),
    (
        (
            [sys.executable, "-m", "langfuse_agent.mcp_server"],
            "langfuse-mcp-child",
        ),
        (
            [sys.executable, "scripts/certification/loopback_oidc.py"],
            "loopback-oidc-fixture",
        ),
    ),
)
def test_auxiliary_process_classes_are_detected_without_persisting_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: str,
) -> None:
    import agent_utilities.deployment.skill_validation as lifecycle

    entry = tmp_path / "process"
    entry.mkdir()
    (entry / "cmdline").write_bytes(b"\x00".join(item.encode() for item in argv))
    monkeypatch.setattr(lifecycle.os, "readlink", lambda _path: sys.executable)

    assert lifecycle._process_kind(entry) == expected


def test_lifecycle_subject_requires_zero_terminal_auxiliary_processes(
    tmp_path: Path,
) -> None:
    import agent_utilities.deployment.skill_validation as lifecycle

    graph_os = tmp_path / "graph-os"
    _write_executable(graph_os, "raise SystemExit(0)\n")
    subject = lifecycle._lifecycle_subject(
        _deployment(graph_os),
        global_counts=(0, 1, 0),
        graph_os_counts=(0, 1, 0),
        engine_counts=(0, 1, 0),
        identity_authority_counts=(0, 1, 0),
        terminal_process_counts=(1, 0),
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
        engine_executable_digest="sha256:" + "4" * 64,
        installed_release_attested=True,
        reaped=False,
        validator_exit_code=0,
        validation_evidence_digest="sha256:" + "5" * 64,
        validation_case_count=20,
        error_code="terminal_process_count_invalid",
    )

    assert subject["result"] == "fail"
    assert subject["processGate"]["terminalProcessCounts"] == {
        "langfuseMcpChildren": 1,
        "loopbackOidcFixtures": 0,
    }
    assert str(tmp_path) not in json.dumps(subject, sort_keys=True)


def test_orchestrator_requires_three_fresh_distinct_sibling_outputs(
    tmp_path: Path,
) -> None:
    graph_os = tmp_path / "graph-os"
    _write_executable(graph_os, "import time\ntime.sleep(1)\n")
    deployment = _deployment(graph_os)
    report = tmp_path / "matrix.md"
    validation = tmp_path / "matrix.json"
    lifecycle = tmp_path / "lifecycle.json"

    with pytest.raises(DeploymentError, match="evidence_destinations_not_distinct"):
        run_deployment(
            deployment,
            deployment_path=tmp_path / "deployment.json",
            report_path=validation,
            validation_evidence_path=validation,
            lifecycle_evidence_path=lifecycle,
        )

    report.write_text("stale", encoding="utf-8")
    with pytest.raises(DeploymentError, match="evidence_destination_not_fresh"):
        run_deployment(
            deployment,
            deployment_path=tmp_path / "deployment.json",
            report_path=report,
            validation_evidence_path=validation,
            lifecycle_evidence_path=lifecycle,
        )


def test_orchestrator_proves_zero_one_zero_and_reaps_candidate(
    tmp_path, monkeypatch
) -> None:
    graph_os = tmp_path / "graph-os"
    validator = tmp_path / "agent-utilities-validate-skills"
    readiness = tmp_path / "graph-os-skill-readiness"
    signature_adapter = tmp_path / "signature-adapter"
    _write_executable(
        graph_os,
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: exit(0))\n"
        "while True: time.sleep(0.05)\n",
    )
    _write_executable(
        validator,
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "report = args[args.index('--report') + 1]\n"
        "evidence = args[args.index('--evidence') + 1]\n"
        "open(report, 'w', encoding='utf-8').write('# synthetic\\n')\n"
        "payload = {'cases': [{} for _ in range(20)], 'result': {'status': 'pass'}}\n"
        "open(evidence, 'w', encoding='utf-8').write(json.dumps(payload) + '\\n')\n",
    )
    _write_executable(readiness, "raise SystemExit(0)\n")
    _write_executable(
        signature_adapter,
        "import hashlib, json, sys\n"
        "value = json.load(sys.stdin)\n"
        "key = 'key:' + '4' * 64\n"
        "if 'signature' in value:\n"
        "    signature = value.pop('signature')\n"
        "    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()\n"
        "    result = {'verified': True, 'subjectDigest': 'sha256:' + hashlib.sha256(payload).hexdigest(), 'keyId': key}\n"
        "else:\n"
        "    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()\n"
        "    result = {'algorithm': 'ed25519', 'keyId': key, 'signature': 'A' * 43, 'subjectDigest': 'sha256:' + hashlib.sha256(payload).hexdigest()}\n"
        "json.dump(result, sys.stdout, sort_keys=True)\n",
    )
    monkeypatch.setenv("SYNTHETIC_GRAPHOS_COMMAND", json.dumps([str(graph_os)]))
    monkeypatch.setenv("SYNTHETIC_ENDPOINT_REFERENCE", "opaque-runtime-reference")
    monkeypatch.setenv("SYNTHETIC_SIGNER_COMMAND", json.dumps([str(signature_adapter)]))
    monkeypatch.setenv(
        "SYNTHETIC_VERIFIER_COMMAND", json.dumps([str(signature_adapter)])
    )
    import agent_utilities.deployment.skill_validation as lifecycle
    import agent_utilities.deployment.skill_validation_assets as assets

    monkeypatch.setattr(assets, "verify_release_bindings", lambda _deployment: {})
    monkeypatch.setattr(
        assets,
        "load_runtime_materials",
        lambda _deployment, *, require_active_configuration: {
            "models": [],
            "modelPrivateHosts": ["127.0.0.1"],
        },
    )
    monkeypatch.setattr(
        assets,
        "prove_model_registry_runtime",
        lambda _models, _hosts: {
            "modelCount": 2,
            "literalPrivateModelCount": 2,
            "privateDnsModelCount": 0,
            "privateDnsUniqueResolutionProven": True,
            "privateBoundaryProven": True,
            "dnsRebindingGuarded": True,
        },
    )
    monkeypatch.setattr(
        assets,
        "attest_installed_release_binding",
        lambda _deployment, *, start_executable, promotion_evidence: {},
    )
    monkeypatch.setattr(lifecycle, "_wait_until_ready", lambda *_args, **_kwargs: None)
    process_counts = iter(
        [
            lifecycle._ProcessCounts(0, 0, 0, 0, 0, ()),
            lifecycle._ProcessCounts(1, 1, 1, 1, 0, (Path("/proc/1"),)),
            # A stdio MCP child owns a separate process session.  Its exit can
            # become observable just after the GraphOS parent is reaped.
            lifecycle._ProcessCounts(0, 0, 0, 1, 0, ()),
            lifecycle._ProcessCounts(0, 0, 0, 0, 0, ()),
        ]
    )
    monkeypatch.setattr(
        lifecycle, "_process_counts", lambda _marker: next(process_counts)
    )
    monkeypatch.setattr(
        lifecycle,
        "_marked_engine_digest",
        lambda _marker: "sha256:" + "4" * 64,
    )
    monkeypatch.setattr(lifecycle, "_terminate_marked_engines", lambda *_args: None)

    class FakeAuthority:
        def __init__(self, *, token_ttl_seconds: int) -> None:
            assert token_ttl_seconds == 300
            self.running = False
            self.tls_verified = False
            self.token_mint_count = 3

        def start(self):
            self.running = True
            self.tls_verified = True
            return self

        def child_environment(self, environment, *, model_private_hosts):
            assert model_private_hosts == ["127.0.0.1"]
            return dict(environment)

        def prove_renewable(self) -> bool:
            return True

        def stop(self) -> None:
            self.running = False

    monkeypatch.setattr(lifecycle, "EphemeralLoopbackOidcAuthority", FakeAuthority)

    report = tmp_path / "matrix.md"
    validation_evidence = tmp_path / "matrix.json"
    lifecycle_evidence = tmp_path / "lifecycle.json"
    result = run_deployment(
        _deployment(graph_os),
        deployment_path=tmp_path / "deployment.json",
        report_path=report,
        validation_evidence_path=validation_evidence,
        lifecycle_evidence_path=lifecycle_evidence,
    )

    assert result == 0
    evidence = json.loads(lifecycle_evidence.read_text(encoding="utf-8"))
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "deploy/release/skill-validation-deployment-evidence.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(evidence)
    assert evidence["processGate"] == {
        "globalGraphOs": {"before": 0, "running": 1, "after": 0},
        "candidateGraphOs": {"before": 0, "running": 1, "after": 0},
        "candidateEngine": {"before": 0, "running": 1, "after": 0},
        "terminalProcessCounts": {
            "langfuseMcpChildren": 0,
            "loopbackOidcFixtures": 0,
        },
        "engineExecutableDigest": "sha256:" + "4" * 64,
        "installedReleaseAttested": True,
        "reaped": True,
    }
    assert evidence["identityAuthority"] == {
        "mode": "ephemeral-https-loopback",
        "lifecycleCounts": {"before": 0, "running": 1, "after": 0},
        "tlsVerified": True,
        "renewableCredentialsProven": True,
        "tokenMintCount": 3,
        "reaped": True,
    }
    assert evidence["result"] == "pass"
    serialized = json.dumps(evidence, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "opaque-runtime-reference" not in serialized
    assert "profile:synthetic" not in serialized
    assert not any(
        "graph-os"
        in (Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore"))
        and str(tmp_path)
        in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
        for pid in os.listdir("/proc")
        if pid.isdigit() and Path(f"/proc/{pid}/cmdline").exists()
    )
