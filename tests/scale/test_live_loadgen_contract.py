"""Bounded source fixtures for live-loadgen identity and artifact pins.

These tests deliberately use synthetic references and a ``.invalid`` endpoint.
They exercise only the fail-closed contract layer; they never contact an engine,
resolve a secret, or qualify a mock run as production certification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.certification.campaign import (
    CampaignError,
    _validate_live_load_command,
)
from scripts.scale.live_contract import (
    LiveRuntimeContract,
    LiveRuntimeContractError,
    mock_runtime,
)
from scripts.scale.loadgen_source_authority import (
    LoadgenSourceAuthorityError,
    _assert_workspace_registration,
    verify_tracked_manifest,
)
from scripts.scale.workload_contract import load_workload_contract

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "scale" / "live-loadgen-runtime.json"


def _environment() -> dict[str, str]:
    environment = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(environment, dict)
    contract = load_workload_contract()
    environment["LOADGEN_WORKLOAD_CONTRACT_DIGEST"] = contract.contract_digest
    return environment


def test_live_runtime_contract_binds_all_opaque_pins() -> None:
    contract = load_workload_contract()

    binding = LiveRuntimeContract.from_environment(
        contract,
        environment=_environment(),
    )

    report = binding.as_report()
    assert report["mode"] == "live"
    assert report["contract_digest"] == contract.contract_digest
    assert set(report) == {
        "mode",
        "release_digest",
        "topology_digest",
        "image_digest",
        "contract_digest",
        "identity_digest",
        "engine_authority_digest",
        "source_authority_digest",
    }


def test_live_runtime_contract_rejects_missing_authority_or_identity() -> None:
    contract = load_workload_contract()
    for field in (
        "GRAPH_SERVICE_ENDPOINTS",
        "KG_AUTH_TOKEN_REF",
        "ENGINE_TLS_PROFILE_REF",
        "LOADGEN_TENANT",
        "LOADGEN_PRINCIPAL",
        "LOADGEN_AUDIENCE",
        "LOADGEN_SOURCE_REPOSITORY",
        "LOADGEN_SOURCE_REVISION",
        "LOADGEN_SOURCE_MANIFEST_DIGEST",
        "LOADGEN_SOURCE_AUTHORITY_DIGEST",
    ):
        environment = _environment()
        environment.pop(field)
        with pytest.raises(LiveRuntimeContractError):
            LiveRuntimeContract.from_environment(contract, environment=environment)


def test_live_runtime_contract_rejects_ambiguous_engine_authority() -> None:
    contract = load_workload_contract()
    environment = _environment()
    environment["GRAPH_SERVICE_AUTH_SECRET"] = "synthetic-secret-reference"

    with pytest.raises(LiveRuntimeContractError):
        LiveRuntimeContract.from_environment(contract, environment=environment)


def test_live_runtime_contract_rejects_contract_or_argv_drift() -> None:
    contract = load_workload_contract()

    environment = _environment()
    environment["LOADGEN_WORKLOAD_CONTRACT_DIGEST"] = "sha256:" + "4" * 64
    with pytest.raises(LiveRuntimeContractError):
        LiveRuntimeContract.from_environment(contract, environment=environment)

    environment = _environment()
    with pytest.raises(LiveRuntimeContractError):
        LiveRuntimeContract.from_environment(
            contract,
            environment=environment,
            release_digest="sha256:" + "5" * 64,
        )

    environment = _environment()
    environment["LOADGEN_SOURCE_MANIFEST_DIGEST"] = "sha256:" + "6" * 64
    with pytest.raises(LiveRuntimeContractError):
        LiveRuntimeContract.from_environment(contract, environment=environment)


def test_mock_runtime_is_explicit_and_not_a_live_binding() -> None:
    runtime = mock_runtime(load_workload_contract())

    assert runtime["mode"] == "mock"
    assert set(runtime) == {"mode", "contract_digest"}


def test_production_campaign_binds_the_real_loadgen_entry_point() -> None:
    _validate_live_load_command(
        [
            "/opt/agent-utilities/bin/graphos-certification-load",
            "--engine",
            "live",
            "--scale",
            "1.0",
            "--duration-s",
            "{duration_seconds}",
            "--report-json",
            "{report_file}",
            "--release-digest",
            "{release_digest}",
        ]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"--engine": "mock"},
        {"--scale": "0.001"},
        {"--report-json": "./report.json"},
    ],
)
def test_production_campaign_rejects_mock_or_unbound_loadgen(mutation: dict[str, str]) -> None:
    command = [
        "/opt/agent-utilities/bin/graphos-certification-load",
        "--engine",
        "live",
        "--scale",
        "1.0",
        "--duration-s",
        "{duration_seconds}",
        "--report-json",
        "{report_file}",
        "--release-digest",
        "{release_digest}",
    ]
    for flag, value in mutation.items():
        command[command.index(flag) + 1] = value

    with pytest.raises(CampaignError):
        _validate_live_load_command(command)


def test_orphan_manifest_without_git_authority_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "compose.yml"
    workspace = tmp_path / "workspace.yml"
    workspace.write_text("services:\n  items: []\n", encoding="utf-8")
    payload = b"services:\n  loadgen:\n    image: example.invalid/loadgen@sha256:0\n"
    manifest.write_bytes(payload)

    with pytest.raises(
        LoadgenSourceAuthorityError,
        match="source_git_authority_missing",
    ):
        verify_tracked_manifest(
            manifest,
            repository="https://git.example.invalid/loadgen.git",
            revision="a" * 40,
            manifest_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
            workspace_manifest=workspace,
        )


def test_workspace_registry_must_name_the_loadgen_repository(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace.yml"
    workspace.write_text(
        "services:\n  items:\n    - name: other\n      url: https://git.example.invalid/other.git\n",
        encoding="utf-8",
    )

    with pytest.raises(
        LoadgenSourceAuthorityError,
        match="source_service_not_registered",
    ):
        _assert_workspace_registration(
            workspace,
            service_name="loadgen",
            repository="https://git.example.invalid/loadgen.git",
        )


def test_workspace_registry_ignores_non_service_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace.yml"
    workspace.write_text(
        "metadata:\n"
        "  source:\n"
        "    name: loadgen\n"
        "    url: https://git.example.invalid/loadgen.git\n"
        "services:\n"
        "  items: []\n",
        encoding="utf-8",
    )

    with pytest.raises(
        LoadgenSourceAuthorityError,
        match="source_service_not_registered",
    ):
        _assert_workspace_registration(
            workspace,
            service_name="loadgen",
            repository="https://git.example.invalid/loadgen.git",
        )
