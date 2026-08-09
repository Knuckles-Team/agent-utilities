"""RMDD-29 typed operation-payload contract and golden fixture tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from agent_utilities.orchestration.operation_payload import (
    MAX_ARG_COUNT,
    MAX_ARTIFACT_PATTERN_COUNT,
    MAX_ENVIRONMENT_REFERENCE_COUNT,
    OPERATION_PAYLOAD_EXTENSION_KEY,
    OPERATION_PAYLOAD_VARIANTS,
    RepositoryBuildExecutionPayloadV1,
    RepositoryOperationPayload,
    cache_key_digest_from_components,
    canonical_payload_json,
    compose_operation_payload_extension_registry,
    operation_payload_from_mapping,
    operation_payload_variant,
    payload_digest,
)

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "rmdd_29_operation_payload.json"
_PATH_FIXTURE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "rmdd_29_operation_payload_path_scoped.json"
)


def _raw_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _payload(**updates: object) -> RepositoryBuildExecutionPayloadV1:
    raw = _raw_fixture()
    raw.pop("payload_digest", None)
    raw.update(updates)
    raw_components = raw["cache_key_components"]
    assert isinstance(raw_components, list)
    components: dict[str, str] = {
        str(item["name"]): str(item["value"])
        for item in raw_components
        if isinstance(item, dict)
    }
    coupled = {
        "repository_id": "repo",
        "build_spec_name": "spec",
        "feature_set": "feature_set",
        "target_triple": "target_triple",
        "config_digest": "config_digest",
        "spec_digest": "spec_digest",
    }
    for field, component in coupled.items():
        if field in updates:
            components[component] = str(updates[field])
    if "tree_sha" in updates:
        components["tree_sha"] = str(updates["tree_sha"])
    if "generation_id" in updates:
        generation_id = updates["generation_id"]
        components["generation_id"] = str(generation_id) if generation_id else ""
        components["generation_digest"] = (
            hashlib.sha256(str(generation_id).encode()).hexdigest()
            if generation_id
            else ""
        )
    raw["cache_key_components"] = [
        {"name": name, "value": value} for name, value in components.items()
    ]
    if raw.get("cacheable"):
        raw["cache_key_digest"] = cache_key_digest_from_components(components)
    return RepositoryBuildExecutionPayloadV1.model_validate(raw)


def test_golden_fixture_is_canonical_and_digest_stable() -> None:
    raw = _raw_fixture()
    payload = RepositoryBuildExecutionPayloadV1.model_validate(raw)

    assert payload.payload_digest == (
        "f3ba70f467dfaca36863c0c0b7243a564dff49dd61e21299e0843ba7d014b5b2"
    )
    assert payload_digest(payload) == payload.payload_digest
    assert canonical_payload_json(payload).encode("utf-8") == canonical_payload_json(
        raw
    ).encode("utf-8")
    assert payload.model_dump(mode="json")["feature_set"] == "cargo build --locked"
    assert payload.model_dump(mode="json")["artifact_patterns"] == [
        "target/release/*.sha256",
        "target/release/repository-manager",
    ]


def test_cache_digest_does_not_authorize_a_mismatched_tree_component() -> None:
    raw = _raw_fixture()
    raw.pop("payload_digest", None)
    components = {
        str(item["name"]): str(item["value"])
        for item in raw["cache_key_components"]
        if isinstance(item, dict)
    }
    components["tree_sha"] = "f" * 40
    raw["cache_key_components"] = [
        {"name": name, "value": value} for name, value in components.items()
    ]
    raw["cache_key_digest"] = cache_key_digest_from_components(components)
    with pytest.raises((ValidationError, ValueError), match="tree"):
        RepositoryBuildExecutionPayloadV1.model_validate(raw)


def test_path_scoped_tree_identity_preserves_the_existing_32_hex_cache_component() -> (
    None
):
    raw = _raw_fixture()
    raw.pop("payload_digest", None)
    path_tree = "f" * 32
    components = {
        str(item["name"]): str(item["value"])
        for item in raw["cache_key_components"]
        if isinstance(item, dict)
    }
    components["tree_sha"] = path_tree
    raw["tree_sha"] = path_tree
    raw["cache_key_components"] = [
        {"name": name, "value": value} for name, value in components.items()
    ]
    raw["cache_key_digest"] = cache_key_digest_from_components(components)
    payload = RepositoryBuildExecutionPayloadV1.model_validate(raw)
    assert payload.tree_sha == path_tree
    assert payload.cache_key_components[-1].value == path_tree


def test_path_scoped_golden_fixture_matches_rm_contract() -> None:
    payload = RepositoryBuildExecutionPayloadV1.model_validate(
        json.loads(_PATH_FIXTURE.read_text(encoding="utf-8"))
    )
    assert len(payload.tree_sha) == 32
    assert payload.cache_key_digest == "v2:9c1ebe846484244a4b0afcadcac94dc4"
    assert payload.payload_digest == (
        "31566ae365e939ca01f9c8d248f71cf33967f83b1eabb0b132153694f8c727bd"
    )


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
    raw["environment_refs"] = ["RUSTFLAGS", "CARGO_HOME"]
    raw["artifact_patterns"] = [
        "target/release/repository-manager",
        "target/release/*.sha256",
    ]
    raw["cache_key_components"] = list(reversed(raw["cache_key_components"]))  # type: ignore[arg-type]
    reordered = RepositoryBuildExecutionPayloadV1.model_validate(raw)
    assert reordered.payload_digest == original.payload_digest


def test_type_adapter_exercises_the_closed_discriminated_payload() -> None:
    adapter = TypeAdapter(RepositoryOperationPayload)
    parsed = adapter.validate_python(_raw_fixture())
    assert isinstance(parsed, RepositoryBuildExecutionPayloadV1)
    with pytest.raises(ValidationError):
        adapter.validate_python({**_raw_fixture(), "kind": "repository.future/v1"})


def test_payload_is_frozen_and_copy_recomputes_digest() -> None:
    payload = _payload()
    with pytest.raises(ValidationError):
        payload.argv = ("cargo", "clean")  # type: ignore[misc]
    changed = payload.model_copy(update={"argv": ("cargo", "test")})
    assert changed.argv == ("cargo", "test")
    assert changed.payload_digest != payload.payload_digest
    assert payload.argv == ("cargo", "build", "--locked")


@pytest.mark.parametrize(
    "updates",
    [
        {"argv": ["sh", "-c", "echo unsafe"]},
        {"workdir": "../outside"},
        {"argv": ["cargo"] * 129},
        {"unknown": "field"},
        {"cacheable": False},
    ],
)
def test_copy_update_revalidates_all_security_and_contract_rules(
    updates: dict[str, object],
) -> None:
    with pytest.raises((ValidationError, ValueError)):
        _payload().model_copy(update=updates)


def test_existing_model_is_revalidated_at_the_boundary() -> None:
    payload = _payload()
    object.__setattr__(payload, "argv", ("sh", "-c", "echo unsafe"))
    with pytest.raises((ValidationError, ValueError)):
        operation_payload_from_mapping(payload)


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
        ("tree_sha", "a" * 33),
        ("base_sha", "a" * 32),
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


def test_cacheability_and_degraded_reason_are_explicit_and_consistent() -> None:
    raw = _raw_fixture()
    degraded_components: list[dict[str, str]] = [
        {"name": "key_version", "value": "v2"},
        {"name": "repo", "value": "repository-manager"},
        {"name": "spec", "value": "rmdd-10-build"},
    ]
    degraded_components.extend(
        {"name": name, "value": ""}
        for name in (
            "tree_sha",
            "feature_set",
            "toolchain_fingerprint",
            "target_triple",
            "config_digest",
            "spec_digest",
            "generation_id",
            "generation_digest",
        )
    )
    raw.update(
        {
            "cacheable": False,
            "cache_key_digest": None,
            "degraded_reason": "toolchain-unfingerprintable",
            "cache_key_components": degraded_components,
        }
    )
    raw.pop("payload_digest", None)
    degraded = RepositoryBuildExecutionPayloadV1.model_validate(raw)
    assert degraded.cache_key_digest is None
    for changes in (
        {"cacheable": True, "degraded_reason": "dirty-tree"},
        {"cacheable": False, "degraded_reason": ""},
        {"cacheable": False, "degraded_reason": "unknown"},
        {"cacheable": False, "cache_key_digest": "v2:295a07a0c39330b1b4460649b2d319af"},
    ):
        invalid = dict(raw)
        invalid.update(changes)
        invalid.pop("payload_digest", None)
        with pytest.raises((ValidationError, ValueError)):
            RepositoryBuildExecutionPayloadV1.model_validate(invalid)


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
