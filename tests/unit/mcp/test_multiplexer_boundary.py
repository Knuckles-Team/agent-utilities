from __future__ import annotations

import json

import pytest
from fastmcp.exceptions import ToolError

from agent_utilities.mcp.multiplexer import (
    MCPMultiplexer,
    _assert_bounded_delegated_value,
    _resolve_runtime_value,
    _runtime_materialized,
    _selected_child_provider_profile,
    attest_runtime_child_config,
)


def test_noncredential_field_cannot_expand_secret_environment_name() -> None:
    with pytest.raises(RuntimeError, match="credential field"):
        _resolve_runtime_value("${API_TOKEN}", sensitive=False)


def test_fleet_runtime_alias_prefers_direct_projection(monkeypatch) -> None:
    from agent_utilities.core.config import config

    monkeypatch.setattr(
        "agent_utilities.mcp.multiplexer.setting",
        lambda alias: (
            "direct-runtime-material" if alias == "CHILD_TOKEN_ALIAS" else None
        ),
    )
    monkeypatch.setattr(
        config,
        "mcp_fleet_secret_refs",
        {"CHILD_TOKEN_ALIAS": "secret://fleet/child/token"},
        raising=False,
    )
    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        lambda _reference: (_ for _ in ()).throw(AssertionError("fallback used")),
    )
    assert (
        _resolve_runtime_value("env://CHILD_TOKEN_ALIAS", sensitive=True)
        == "direct-runtime-material"
    )


def test_fleet_runtime_alias_uses_configured_ref_only_after_direct_miss(
    monkeypatch,
) -> None:
    from agent_utilities.core.config import config

    monkeypatch.setattr("agent_utilities.mcp.multiplexer.setting", lambda _alias: None)
    monkeypatch.setattr(
        config,
        "mcp_fleet_secret_refs",
        {"CHILD_TOKEN_ALIAS": "secret://fleet/child/token"},
        raising=False,
    )
    resolved_refs: list[str] = []

    def resolve(reference: str) -> str:
        resolved_refs.append(reference)
        return "mapped-runtime-material"

    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        resolve,
    )

    assert (
        _resolve_runtime_value("env://CHILD_TOKEN_ALIAS", sensitive=True)
        == "mapped-runtime-material"
    )
    assert resolved_refs == ["secret://fleet/child/token"]


def test_fleet_runtime_alias_fails_closed_without_direct_or_mapped_value(
    monkeypatch,
) -> None:
    from agent_utilities.core.config import config

    monkeypatch.setattr("agent_utilities.mcp.multiplexer.setting", lambda _alias: None)
    monkeypatch.setattr(config, "mcp_fleet_secret_refs", {}, raising=False)

    with pytest.raises(RuntimeError, match="unavailable"):
        _resolve_runtime_value("env://CHILD_TOKEN_ALIAS", sensitive=True)
    with pytest.raises(RuntimeError, match="invalid"):
        _resolve_runtime_value("env://lowercase_alias", sensitive=True)


def test_delegated_arguments_have_depth_and_size_boundaries() -> None:
    nested: object = "leaf"
    for _ in range(40):
        nested = {"next": nested}
    with pytest.raises(ToolError, match="structural boundary"):
        _assert_bounded_delegated_value(nested)
    with pytest.raises(ToolError, match="size boundary"):
        _assert_bounded_delegated_value({"value": "x" * (4 * 1024 * 1024 + 1)})


def test_catalog_symlink_is_not_executed(tmp_path) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"mcpServers": {"child": {"command": "child"}}}))
    link = tmp_path / "catalog.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    assert MCPMultiplexer(link).load_catalog() == {}


def test_catalog_rejects_inline_child_credentials(tmp_path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "child": {
                        "command": "child",
                        "env": {"API_TOKEN": "inline-secret"},
                    }
                }
            }
        )
    )

    assert MCPMultiplexer(catalog).load_catalog() == {}


def test_catalog_preserves_neutral_child_secret_alias(tmp_path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "child": {
                        "command": "child",
                        "env": {"API_TOKEN": "env://CHILD_TOKEN_ALIAS"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = MCPMultiplexer(catalog).load_catalog()

    assert loaded["child"]["env"] == {"API_TOKEN": "env://CHILD_TOKEN_ALIAS"}


def test_child_provider_profile_selection_is_strict() -> None:
    assert (
        _selected_child_provider_profile(
            {"provider_profile": "synthetic-provider"}, is_remote=False
        )
        == "synthetic-provider"
    )


@pytest.mark.parametrize(
    ("selection", "is_remote"),
    [
        ("INVALID", False),
        (" synthetic-provider", False),
        ("synthetic-provider", True),
    ],
)
def test_child_provider_profile_rejects_invalid_or_remote_selection(
    selection, is_remote
) -> None:
    with pytest.raises(RuntimeError, match="provider profile selection"):
        _selected_child_provider_profile(
            {"provider_profile": selection}, is_remote=is_remote
        )


def test_catalog_rejects_direct_provider_profile_environment(tmp_path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "child": {
                        "command": "child",
                        "env": {"AGENT_PROVIDER_PROFILE": "synthetic-provider"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert MCPMultiplexer(catalog).load_catalog() == {}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cfg: cfg.__setitem__("command", "different-child"),
        lambda cfg: cfg["args"].append("--different-mode"),
        lambda cfg: cfg.__setitem__("tls_profile_ref", "secret://trust/changed"),
        lambda cfg: cfg["allowed_private_hosts"].append("changed.invalid"),
        lambda cfg: cfg.__setitem__("timeout", 31.0),
        lambda cfg: cfg.__setitem__("provider_profile", "changed-provider"),
        lambda cfg: cfg.__setitem__("_graphos_parent_kg_ingestion", True),
    ],
)
def test_runtime_attestation_binds_complete_child_declaration(mutation) -> None:
    declaration = attest_runtime_child_config(
        {
            "command": "synthetic-child",
            "args": ["--mode", "read-only"],
            "env": {"API_TOKEN": "synthetic-runtime-material"},
            "tls_profile_ref": "secret://trust/profile",
            "allowed_private_hosts": ["service.invalid"],
            "timeout": 30.0,
            "_graphos_parent_kg_ingestion": False,
        }
    )

    assert _runtime_materialized(declaration) is True

    mutation(declaration)

    assert _runtime_materialized(declaration) is False


@pytest.mark.parametrize(
    "field, changed",
    [
        ("url", "https://changed.invalid/mcp"),
        ("transport", "sse"),
        ("headers", {"Authorization": "different-runtime-material"}),
    ],
)
def test_runtime_attestation_binds_remote_transport_destination(
    field: str, changed: object
) -> None:
    declaration = attest_runtime_child_config(
        {
            "url": "https://service.invalid/mcp",
            "transport": "streamable-http",
            "headers": {"Authorization": "synthetic-runtime-material"},
        }
    )

    assert _runtime_materialized(declaration) is True

    declaration[field] = changed

    assert _runtime_materialized(declaration) is False
