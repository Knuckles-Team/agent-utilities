"""Reference-only model authentication and header resolution."""

from __future__ import annotations

import pytest

from agent_utilities.core.model_runtime_auth import (
    ModelRuntimeAuthError,
    resolve_model_api_key,
    resolve_model_headers,
)


def test_resolves_api_key_reference_without_retaining_reference(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_API_KEY", "synthetic-runtime-material")

    assert (
        resolve_model_api_key(reference="env://TEST_MODEL_API_KEY")
        == "synthetic-runtime-material"
    )


def test_resolves_bounded_json_header_reference(monkeypatch):
    monkeypatch.setenv(
        "TEST_MODEL_HEADERS",
        '{"Authorization":"Bearer synthetic","X-Client":"client"}',
    )

    assert resolve_model_headers(reference="env://TEST_MODEL_HEADERS") == {
        "Authorization": "Bearer synthetic",
        "X-Client": "client",
    }


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"Host":"invalid"}',
        '{"X-Test":"line\\r\\ninjection"}',
        '{"X-Test":"first","x-test":"second"}',
        '{"X-Test":1}',
    ],
)
def test_rejects_invalid_resolved_header_material_without_details(
    monkeypatch,
    payload,
):
    monkeypatch.setenv("TEST_MODEL_HEADERS", payload)

    with pytest.raises(ModelRuntimeAuthError) as caught:
        resolve_model_headers(reference="env://TEST_MODEL_HEADERS")

    rendered = str(caught.value)
    assert payload not in rendered
    assert "TEST_MODEL_HEADERS" not in rendered


def test_rejects_ambiguous_sources_without_details(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_API_KEY", "synthetic-runtime-material")

    with pytest.raises(ModelRuntimeAuthError) as caught:
        resolve_model_api_key(
            value="direct-runtime-material",
            reference="env://TEST_MODEL_API_KEY",
        )

    rendered = str(caught.value)
    assert "synthetic-runtime-material" not in rendered
    assert "direct-runtime-material" not in rendered
    assert "TEST_MODEL_API_KEY" not in rendered


@pytest.mark.parametrize(
    "reference",
    ["env://not/a/name", "env://has-hyphen", "env://1STARTS_WITH_DIGIT"],
)
def test_rejects_non_shell_safe_environment_references(reference):
    with pytest.raises(ModelRuntimeAuthError, match="reference is invalid"):
        resolve_model_api_key(reference=reference)
