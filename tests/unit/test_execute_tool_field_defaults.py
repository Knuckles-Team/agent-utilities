"""_execute_tool must resolve Field() defaults for omitted params.

Tool functions declare params as ``name: T = Field(default=...)``. FastMCP resolves
those defaults from the schema, but ``_execute_tool`` calls the raw function directly
(internal callers, the REST gateway, tests). Without resolution, an omitted param is
bound to the raw ``FieldInfo`` object, later blowing up downstream with
"'FieldInfo' object has no attribute 'replace'" / "not JSON serializable".
"""

from __future__ import annotations

import pytest
from pydantic import Field

from agent_utilities.mcp import kg_server


@pytest.mark.asyncio
@pytest.mark.concept("ORCH-1.37")
async def test_execute_tool_resolves_field_defaults(monkeypatch):
    seen: dict = {}

    async def fake_tool(
        cypher: str = Field(default=""),
        params: str = Field(default="{}"),
        limit: int = Field(default=50),
    ) -> str:
        seen["cypher"] = cypher
        seen["params"] = params
        seen["limit"] = limit
        return "ok"

    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "fake_tool", fake_tool)

    # Call with only `cypher` — the omitted params must arrive as their Field defaults,
    # NOT as raw FieldInfo objects.
    out = await kg_server._execute_tool("fake_tool", cypher="MATCH (n) RETURN n")
    assert out == "ok"
    assert seen["cypher"] == "MATCH (n) RETURN n"
    assert seen["params"] == "{}"
    assert seen["limit"] == 50
    # Critically, the resolved values are plain types, not FieldInfo.
    assert isinstance(seen["params"], str)
    assert isinstance(seen["limit"], int)
