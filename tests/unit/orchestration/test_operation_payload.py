"""RMDD-29 typed operation-payload contract and golden fixture tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_utilities.orchestration.operation_payload import (
    MAX_ARG_COUNT,
    MAX_ARTIFACT_PATTERN_COUNT,
    MAX_ENVIRONMENT_REFERENCE_COUNT,
    OPERATION_PAYLOAD_EXTENSION_KEY,
    OPERATION_PAYLOAD_VARIANTS,
    RepositoryBuildExecutionPayloadV1,
    canonical_payload_json,
    compose_operation_payload_extension_registry,
    operation_payload_from_mapping,
    operation_payload_variant,
    payload_digest,
)

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "rmdd_29_operation_payload.json"


def _raw_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _payload(**updates: object) -> RepositoryBuildExecutionPayloadV1:
    raw = _raw_fixture()
    raw.pop("payload_digest", None)
    raw.update(updates)
    return RepositoryBuildExecutionPayloadV1.model_validate(raw)


def test_golden_fixture_is_canonical_and_digest_stable() -> None:
    raw = _raw_fixture()
    payload = RepositoryBuildExecutionPayloadV1.model_validate(raw)

    assert payload.payload_digest == (
        "d07b8385b906c9fa74336cb30e711f330adc001d7e05a996e26c82e585330fac"
    )
    assert payload_digest(payload) == payload.payload_digest
    assert canonical_payload_json(payload).encode("utf-8") == canonical_payload_json(
        raw
    ).encode("utf-8")
    assert payload.model_dump(mode="json")["feature_set"] == ["default", "release"]
    assert payload.model_dump(mode="json")["artifact_patterns"] == [
        "target/release/*.sha256",
        "target/release/repository-manager",
    ]


@pytest.mark.parametrize(
    "field, value",
    [
        ("argv", ["cargo", "test"]),
        ("artifact_patterns", ["target/debug/output"]),
        ("config_digest", "f" * 64),
        ("toolchain_digest", "f" * 64),
        ("tree_sha", "f" * 40),
        ("generation_id", "generation-new"),
    ],
)
def test_every_execution_input_change_changes_payload_digest(
    field: str, value: object
) -> None:
    original = _payload()
    changed = _payload(**{field: value})
    assert changed.payload_digest != original.payload_digest


def test_semantically_unordered_sets_have_one_digest() -> None:
    original = _payload()
    raw = _raw_fixture()
    raw.pop("payload_digest", None)
    raw["feature_set"] = ["release", "default"]
    raw["environment_refs"] = ["RUSTFLAGS", "CARGO_HOME"]
    raw["artifact_patterns"] = [
        "target/release/repository-manager",
        "target/release/*.sha256",
    ]
    raw["cache_key_components"] = list(reversed(raw["cache_key_components"]))  # type: ignore[arg-type]
    reordered = RepositoryBuildExecutionPayloadV1.model_validate(raw)
    assert reordered.payload_digest == original.payload_digest


def test_payload_is_frozen_and_copy_recomputes_digest() -> None:
    payload = _payload()
    with pytest.raises(ValidationError):
        payload.argv = ("cargo", "clean")  # type: ignore[misc]
    changed = payload.model_copy(update={"argv": ("cargo", "test")})
    assert changed.argv == ("cargo", "test")
    assert changed.payload_digest != payload.payload_digest
    assert payload.argv == ("cargo", "build", "--locked")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda raw: raw.update({"kind": "repository.build-execution/v2"}),
        lambda raw: raw.update({"schema_version": "2"}),
        lambda raw: raw.update({"unknown": "field"}),
        lambda raw: raw.update({"payload_digest": "0" * 64}),
    ],
)
def test_unknown_version_variant_fields_and_tampered_digest_fail(
    mutator: object,
) -> None:
    raw = _raw_fixture()
    mutator(raw)  # type: ignore[operator]
    with pytest.raises((ValidationError, ValueError)):
        operation_payload_from_mapping(raw)


@pytest.mark.parametrize(
    "argv",
    [
        ["sh", "-c", "echo unsafe"],
        ["cargo", "build; rm -rf /"],
        ["cargo", "--token=secret-value"],
        ["cargo", "https://example.invalid/archive"],
    ],
)
def test_argv_is_not_a_shell_or_secret_envelope(argv: list[str]) -> None:
    with pytest.raises((ValidationError, ValueError)):
        _payload(argv=argv)


@pytest.mark.parametrize(
    "workdir",
    ["/tmp/build", "../outside", "a//b", "C:/build", "\\\\server\\share"],
)
def test_workdir_rejects_absolute_drive_unc_and_traversal(workdir: str) -> None:
    with pytest.raises((ValidationError, ValueError)):
        _payload(workdir=workdir)


@pytest.mark.parametrize(
    "patterns",
    [["../outside"], ["/tmp/output"], ["C:/output"], ["target//out"]],
)
def test_artifact_patterns_are_relative_closed_globs(patterns: list[str]) -> None:
    with pytest.raises((ValidationError, ValueError)):
        _payload(artifact_patterns=patterns)


@pytest.mark.parametrize("refs", [["TOKEN=raw-value"], ["https://secret.invalid"]])
def test_environment_refs_are_names_not_values_or_connections(refs: list[str]) -> None:
    with pytest.raises((ValidationError, ValueError)):
        _payload(environment_refs=refs)


def test_noncanonical_sha_digest_and_size_limits_fail_closed() -> None:
    for field, value in (
        ("base_sha", "A" * 40),
        ("tree_sha", "a" * 39),
        ("config_digest", "g" * 64),
    ):
        with pytest.raises((ValidationError, ValueError)):
            _payload(**{field: value})
    with pytest.raises((ValidationError, ValueError)):
        _payload(argv=["cargo"] * (MAX_ARG_COUNT + 1))
    with pytest.raises((ValidationError, ValueError)):
        _payload(
            artifact_patterns=[
                f"target/{index}" for index in range(MAX_ARTIFACT_PATTERN_COUNT + 1)
            ]
        )
    with pytest.raises((ValidationError, ValueError)):
        _payload(
            environment_refs=[
                f"ENV_{index}" for index in range(MAX_ENVIRONMENT_REFERENCE_COUNT + 1)
            ]
        )
    with pytest.raises((ValidationError, ValueError)):
        _payload(build_spec_name="x" * 257)


def test_closed_variant_registry_is_additive_to_resource_sibling() -> None:
    assert OPERATION_PAYLOAD_VARIANTS["repository.build-execution/v1"] is (
        RepositoryBuildExecutionPayloadV1
    )
    composed = compose_operation_payload_extension_registry(
        {"resource_reservation": {"schema_version": "1"}}
    )
    assert composed["resource_reservation"] == {"schema_version": "1"}
    assert composed[OPERATION_PAYLOAD_EXTENSION_KEY] == OPERATION_PAYLOAD_VARIANTS
    with pytest.raises(ValueError):
        compose_operation_payload_extension_registry(
            {OPERATION_PAYLOAD_EXTENSION_KEY: {"different": "variant"}}
        )
    with pytest.raises(ValueError):
        operation_payload_variant("repository.unknown/v1")


def test_copying_input_does_not_mutate_fixture_or_payload() -> None:
    raw = _raw_fixture()
    baseline = copy.deepcopy(raw)
    payload = operation_payload_from_mapping(raw)
    raw["argv"] = ["cargo", "clean"]
    assert payload.argv == ("cargo", "build", "--locked")
    assert baseline["argv"] == ["cargo", "build", "--locked"]
