"""Server-side Prompts-over-MCP wiring (CONCEPT:AU-ECO.mcp.cross-process-prompt-harvest).

Covers ``_register_prompt_providers`` (called from ``create_mcp_server``
right after ``_register_skill_providers``): it must call ``mcp.add_resource``
once per ``*.json`` file under every directory
``resolve_prompt_provider_dirs`` resolves — the SAME discovery
``ingest_prompts_to_graph``'s fleet leg uses, not a re-implementation — and
must never crash server construction when a provider cannot be registered.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agent_utilities.mcp.server_factory import (
    _register_prompt_providers,
    create_mcp_server,
)


def test_register_prompt_providers_never_breaks_server_construction(caplog) -> None:
    fake_mcp = MagicMock(spec=[])  # no attributes at all, incl. no add_resource

    with caplog.at_level("DEBUG", logger="agent_utilities.mcp.server_factory"):
        _register_prompt_providers(fake_mcp)  # must not raise

    assert "prompt" in caplog.text.casefold()


def test_create_mcp_server_construction_succeeds() -> None:
    """``create_mcp_server`` builds a working server with prompt providers wired in."""
    args, mcp, _middlewares = create_mcp_server("Prompt Wiring Test", command_args=[])
    assert mcp is not None
    assert args is not None


def test_register_prompt_providers_registers_one_resource_per_json_file(
    monkeypatch, tmp_path: Path
) -> None:
    """One ``prompt://{provider}/{stem}`` resource per ``*.json`` file, using
    the SAME provider-dir discovery ``resolve_prompt_provider_dirs`` gives
    both the local boot ingest and the servicing-fleet-child leg."""
    provider_dir = tmp_path / "servicenow-api" / "prompts"
    provider_dir.mkdir(parents=True)
    (provider_dir / "incident-triage.json").write_text(
        '{"name": "incident-triage"}', "utf-8"
    )
    (provider_dir / "change-review.json").write_text(
        '{"name": "change-review"}', "utf-8"
    )
    # A leading-underscore file is a reserved/internal convention (mirrors
    # ``ingest_prompts_to_graph``'s ``pfile.name.startswith("_")`` skip) and
    # must not be exposed over the wire.
    (provider_dir / "_manifest.json").write_text("{}", "utf-8")

    monkeypatch.setattr(
        "agent_utilities.core.providers.resolve_prompt_provider_dirs",
        lambda: [("servicenow-api", provider_dir)],
    )

    fake_mcp = MagicMock()
    _register_prompt_providers(fake_mcp)

    assert fake_mcp.add_resource.call_count == 2
    registered_uris = {
        str(call.args[0].uri) for call in fake_mcp.add_resource.call_args_list
    }
    assert registered_uris == {
        "prompt://servicenow-api/incident-triage",
        "prompt://servicenow-api/change-review",
    }


def test_register_prompt_providers_one_bad_dir_does_not_sink_the_rest(
    monkeypatch, tmp_path: Path
) -> None:
    """A provider dir that fails to list must not prevent the others."""
    good_dir = tmp_path / "good-provider"
    good_dir.mkdir()
    (good_dir / "a.json").write_text('{"name": "a"}', "utf-8")

    class _BadPath:
        def glob(self, _pattern: str):
            raise OSError("permission denied")

    monkeypatch.setattr(
        "agent_utilities.core.providers.resolve_prompt_provider_dirs",
        lambda: [("bad-provider", _BadPath()), ("good-provider", good_dir)],
    )

    fake_mcp = MagicMock()
    _register_prompt_providers(fake_mcp)

    assert fake_mcp.add_resource.call_count == 1
