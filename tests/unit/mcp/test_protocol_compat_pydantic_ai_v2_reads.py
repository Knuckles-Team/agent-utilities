"""Tests for the D-CDX-69 pydantic-ai MCP v2-attribute-read bridge.

The live ServiceNow delegation probe bound all 158 tools but emitted
`FastMCPDeprecationWarning` from installed `pydantic_ai/mcp.py`: it reads
`InitializeResult.serverInfo` and `Tool.inputSchema`/`Tool.outputSchema` instead
of the MCP SDK v2 `server_info`/`input_schema`/`output_schema` names, tripping
`fastmcp._compat`'s own warn-once camelCase compatibility shim. Those three
fields ARE covered by that shim (unlike the four gaps `install_mcp_v2_bridge`
closes elsewhere in this module), so the fix cannot be "add a bridging
property" — it has to stop `pydantic_ai.mcp.MCPToolset` from performing the
deprecated read at all, which is what
`protocol_compat._install_pydantic_ai_v2_read_bridge` does by replacing
`MCPToolset.__aenter__`/`.get_tools` with a version-pinned copy that reads the
v2 names directly.

Two things are proven here, with warnings treated as errors so a regression
cannot silently reappear:

1. The UNPATCHED camelCase read genuinely warns (so the second point isn't
   vacuously true because nothing in this SDK build warns at all any more).
2. The PATCHED `MCPToolset.__aenter__`/`.get_tools` read the correct values
   via the v2 attribute names and emit no warning doing so.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import warnings
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("pydantic_ai")
pytest.importorskip("mcp_types")
pytest.importorskip("fastmcp")

from agent_utilities.mcp import protocol_compat  # noqa: E402


def _skip_unless_patch_version_matches() -> None:
    installed = importlib.metadata.version("pydantic-ai-slim")
    if installed != protocol_compat._PATCHED_PYDANTIC_AI_VERSION:
        pytest.skip(
            "installed pydantic-ai-slim "
            f"{installed} != the D-CDX-69 patch pin "
            f"{protocol_compat._PATCHED_PYDANTIC_AI_VERSION}; the bridge "
            "intentionally no-ops outside its verified version (see "
            "_install_pydantic_ai_v2_read_bridge's docstring)."
        )


class _FakeSession:
    def __init__(self) -> None:
        self.set_logging_level = AsyncMock()


class _FakeClient:
    """Stands in for `fastmcp.client.Client`: a real (not Mock-dunder) async
    context manager so `AsyncExitStack.enter_async_context` works exactly as it
    does against the genuine client, without `unittest.mock`'s well-known
    magic-method-lookup pitfalls for `async with`."""

    def __init__(self, initialize_result: Any) -> None:
        self.initialize_result = initialize_result
        self.session = _FakeSession()

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _reset_camelcase_compat_shim() -> None:
    """Force a genuinely FRESH ``fastmcp._compat`` warn-once property for every
    field this test module reads (``Tool.inputSchema``/``.outputSchema``,
    ``InitializeResult.serverInfo``).

    ``fastmcp._compat.install()`` bridges each camelCase property with a
    warn-ONCE-per-(class, name) closure flag (``warned`` in ``_make_property``)
    -- process-lifetime state with no public reset, entirely independent of
    Python's own warnings-filter machinery (which ``warnings.catch_warnings``/
    ``simplefilter("always")`` only controls). Confirmed live (D-TIL-1): any
    OTHER test in the same pytest-xdist ``--dist loadfile`` worker that
    exercises a REAL, UNPATCHED ``pydantic_ai.mcp.MCPToolset.get_tools()``
    against a genuine ``mcp_types.Tool`` -- e.g.
    ``tests/integration/mcp/test_fastmcp_cross_version.py``'s live fastmcp-4
    round-trip, unpatched here because the installed ``pydantic-ai-slim``
    rarely matches the D-CDX-69 patch pin -- reads ``Tool.inputSchema`` for
    real and permanently latches that closure's ``warned`` flag, so this
    test's own read silently stops warning depending on run order.

    Merely flipping ``_compat._installed`` back to ``False`` and calling
    ``install()`` again is NOT enough to get a fresh closure once that has
    happened: ``install()``'s own "never shadow a real attribute" guard
    (``if camel in cls.__dict__: continue``) skips rebuilding any field
    already present on the class -- which it is, from the very first
    ``install()`` call at ``fastmcp`` import time. Deleting the installed
    attribute from the class first removes that guard's reason to skip, so
    the next ``install()`` really does rebuild a fresh, un-warned property --
    regardless of what ran before (or will run after) this test in the same
    worker process.
    """
    import mcp_types
    from fastmcp import _compat

    for camel in ("inputSchema", "outputSchema"):
        if camel in mcp_types.Tool.__dict__:
            delattr(mcp_types.Tool, camel)
    if "serverInfo" in mcp_types.InitializeResult.__dict__:
        delattr(mcp_types.InitializeResult, "serverInfo")
    _compat._installed = False
    _compat.install()


@pytest.fixture
def fresh_camelcase_compat_shim():
    """Yield with a genuinely fresh camelCase compat shim installed, and leave
    one behind at teardown too -- so this test can never be the leak for
    whatever else shares its xdist worker, matching the isolation contract
    ``tests/conftest.py``'s own autouse fixtures apply to other process-global
    registries (D-TIL-1)."""
    _reset_camelcase_compat_shim()
    try:
        yield
    finally:
        _reset_camelcase_compat_shim()


def test_camelcase_read_genuinely_warns_before_the_patch(
    fresh_camelcase_compat_shim,
) -> None:
    """Sanity check: proves the deprecation warning this item reports is real,
    not a false positive — so a passing patched-path test below is meaningful."""
    import mcp_types

    tool = mcp_types.Tool(name="t1", input_schema={"type": "object"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = tool.inputSchema
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "fastmcp._compat's camelCase shim did not warn; the D-CDX-69 premise no longer holds"
    )


def test_install_bridge_patches_toolset_methods() -> None:
    _skip_unless_patch_version_matches()
    import pydantic_ai.mcp as pam

    protocol_compat.install_mcp_v2_bridge()

    assert pam.MCPToolset.__aenter__.__name__ == "_v2_aenter"
    assert pam.MCPToolset.get_tools.__name__ == "_v2_get_tools"


def test_patched_aenter_reads_v2_server_info_without_warning() -> None:
    _skip_unless_patch_version_matches()
    import mcp_types
    import pydantic_ai.mcp as pam

    protocol_compat.install_mcp_v2_bridge()

    server_impl = mcp_types.Implementation(name="test-server", version="9.9")
    init_result = mcp_types.InitializeResult(
        protocol_version="2026-07-28",
        capabilities=mcp_types.ServerCapabilities(),
        server_info=server_impl,
        instructions="hello",
    )

    toolset = pam.MCPToolset.__new__(pam.MCPToolset)
    toolset._enter_lock = asyncio.Lock()
    toolset._running_count = 0
    toolset.client = _FakeClient(init_result)
    toolset.log_level = None
    toolset._exit_stack = None
    toolset._server_info = None
    toolset._server_capabilities = None
    toolset._instructions = None

    async def _run() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = await toolset.__aenter__()
        assert result is toolset
        assert toolset._server_info is server_impl
        assert toolset._instructions == "hello"
        assert toolset._running_count == 1

    asyncio.run(_run())


def test_patched_get_tools_reads_v2_input_output_schema_without_warning() -> None:
    _skip_unless_patch_version_matches()
    import mcp_types
    import pydantic_ai.mcp as pam

    protocol_compat.install_mcp_v2_bridge()

    fake_tool = mcp_types.Tool(
        name="do_thing",
        description="does a thing",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
    )

    toolset = pam.MCPToolset.__new__(pam.MCPToolset)
    toolset.max_retries = None
    toolset.include_return_schema = True
    toolset.list_tools = AsyncMock(return_value=[fake_tool])

    ctx = type("Ctx", (), {"max_retries": 1})()

    async def _run() -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return await toolset.get_tools(ctx)

    tools = asyncio.run(_run())

    assert set(tools) == {"do_thing"}
    tool_def = tools["do_thing"].tool_def
    assert tool_def.parameters_json_schema == fake_tool.input_schema
    assert tool_def.return_schema == fake_tool.output_schema


def test_bridge_skips_patching_outside_its_pinned_version(monkeypatch) -> None:
    """Off the pinned `pydantic-ai-slim` version, the bridge must warn once and
    leave `MCPToolset` untouched rather than silently apply a stale copied
    method body against a since-changed upstream implementation."""
    import pydantic_ai.mcp as pam

    protocol_compat._toolset_reads_patched = False
    real_version = importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "pydantic-ai-slim":
            return "0.0.0-not-the-pinned-version"
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    original_aenter = pam.MCPToolset.__aenter__

    with pytest.warns(RuntimeWarning, match="D-CDX-69"):
        protocol_compat._install_pydantic_ai_v2_read_bridge()

    assert pam.MCPToolset.__aenter__ is original_aenter
    assert protocol_compat._toolset_reads_patched is False
