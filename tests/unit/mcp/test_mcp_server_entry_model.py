"""Tests for :class:`agent_utilities.models.mcp.MCPServerEntryModel`.

The typed CRUD shape for one ``mcp_config.json`` server entry -- mirrors the
same "exactly one of command/url" invariant
``MCPMultiplexer._open_one_session`` enforces at spawn time, so an invalid
entry is rejected at create/edit time rather than only at the next spawn.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_utilities.models.mcp import MCPServerEntryModel


def test_stdio_entry_is_valid():
    entry = MCPServerEntryModel(command="ansible-tower-mcp", args=["--foo"])
    assert entry.command == "ansible-tower-mcp"
    assert entry.url is None


def test_remote_entry_is_valid():
    entry = MCPServerEntryModel(url="https://egeria-mcp.example/mcp")
    assert entry.url == "https://egeria-mcp.example/mcp"
    assert entry.command is None


def test_rejects_neither_command_nor_url():
    with pytest.raises(ValidationError, match="Exactly one of"):
        MCPServerEntryModel()


def test_rejects_both_command_and_url():
    with pytest.raises(ValidationError, match="Exactly one of"):
        MCPServerEntryModel(command="graph-os", url="https://example/mcp")


def test_rejects_transport_without_url():
    with pytest.raises(ValidationError, match="requires 'url'"):
        MCPServerEntryModel(command="graph-os", transport="sse")


def test_schema_is_json_serializable():
    schema = MCPServerEntryModel.model_json_schema()
    assert "command" in schema["properties"]
    assert "url" in schema["properties"]
