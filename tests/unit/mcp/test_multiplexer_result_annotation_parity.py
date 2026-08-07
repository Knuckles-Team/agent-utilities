"""Regression tests for D-CDX-45: child MCP call-result forwarding must
preserve MCP semantic parity — per-content-block ``annotations`` (e.g.
``readOnlyHint``-style hints) and the result-level ``_meta`` — instead of
silently dropping them when converting a child's ``CallToolResult`` into the
host's ``ToolResult``.

Before the fix, ``_make_forwarder``'s ``_forward`` manually reconstructed
``ToolResult(content=..., structured_content=...)`` and never forwarded
``meta`` at all — a direct call to the child would carry its ``_meta``
faithfully, but the SAME call routed through graph-os/the multiplexer
silently dropped it, creating semantic drift between direct-provider and
forwarded calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import mcp.types
import pytest

from agent_utilities.mcp.multiplexer import (
    MCPMultiplexer,
    ToolResult,
    _make_forwarder,
    _tool_result_from_child,
)


def _child_result_with_annotations_and_meta() -> mcp.types.CallToolResult:
    return mcp.types.CallToolResult(
        content=[
            mcp.types.TextContent(
                type="text",
                text="hello",
                annotations=mcp.types.Annotations(
                    audience=["assistant"], priority=0.8
                ),
            )
        ],
        isError=False,
        _meta={"readOnlyHint": True, "child_trace_id": "abc123"},
    )


@pytest.mark.asyncio
async def test_forwarded_result_preserves_content_annotations(tmp_path) -> None:
    mux = MCPMultiplexer(tmp_path / "mcp_config.json")
    mux.call_proxied_tool = AsyncMock(
        return_value=_child_result_with_annotations_and_meta()
    )

    forwarded: ToolResult = await _make_forwarder(mux, "synthetic__tool")()

    assert len(forwarded.content) == 1
    block = forwarded.content[0]
    assert block.annotations is not None
    assert block.annotations.audience == ["assistant"]
    assert block.annotations.priority == 0.8


@pytest.mark.asyncio
async def test_forwarded_result_preserves_meta(tmp_path) -> None:
    """The wire ``_meta`` — including a readOnlyHint-style key — must survive
    the child -> multiplexer -> host conversion, not be silently dropped."""
    mux = MCPMultiplexer(tmp_path / "mcp_config.json")
    mux.call_proxied_tool = AsyncMock(
        return_value=_child_result_with_annotations_and_meta()
    )

    forwarded: ToolResult = await _make_forwarder(mux, "synthetic__tool")()

    assert forwarded.meta == {"readOnlyHint": True, "child_trace_id": "abc123"}


@pytest.mark.asyncio
async def test_direct_vs_forwarded_parity(tmp_path) -> None:
    """A direct wrap of the child's raw result and the multiplexer-forwarded
    result must agree on content, annotations, and meta — no semantic drift
    between calling a child directly and calling it through the fleet
    gateway."""
    mux = MCPMultiplexer(tmp_path / "mcp_config.json")
    raw_result = _child_result_with_annotations_and_meta()
    mux.call_proxied_tool = AsyncMock(return_value=raw_result)

    direct = ToolResult.from_mcp_result(raw_result)
    forwarded: ToolResult = await _make_forwarder(mux, "synthetic__tool")()

    assert forwarded.meta == direct.meta
    assert [c.annotations for c in forwarded.content] == [
        c.annotations for c in direct.content
    ]
    assert [getattr(c, "text", None) for c in forwarded.content] == [
        getattr(c, "text", None) for c in direct.content
    ]


@pytest.mark.asyncio
async def test_structured_content_still_forwarded(tmp_path) -> None:
    mux = MCPMultiplexer(tmp_path / "mcp_config.json")
    mux.call_proxied_tool = AsyncMock(
        return_value=mcp.types.CallToolResult(
            content=[mcp.types.TextContent(type="text", text="ok")],
            structuredContent={"key": "value"},
            isError=False,
        )
    )

    forwarded: ToolResult = await _make_forwarder(mux, "synthetic__tool")()
    assert forwarded.structured_content == {"key": "value"}


def test_annotation_change_after_reconnect_is_visible_in_forwarded_results() -> None:
    """A content-block annotation that CHANGES between child generations
    (e.g. a tool gains a readOnlyHint-equivalent after recovery) must be
    visible in the forwarded result of the NEW generation, not stuck on the
    stale value from before the reconnect."""
    before = mcp.types.CallToolResult(
        content=[
            mcp.types.TextContent(
                type="text",
                text="v1",
                annotations=mcp.types.Annotations(priority=0.2),
            )
        ],
        isError=False,
    )
    after = mcp.types.CallToolResult(
        content=[
            mcp.types.TextContent(
                type="text",
                text="v2",
                annotations=mcp.types.Annotations(priority=0.9),
            )
        ],
        isError=False,
    )

    forwarded_before = _tool_result_from_child(before)
    forwarded_after = _tool_result_from_child(after)

    assert forwarded_before.content[0].annotations.priority == 0.2
    assert forwarded_after.content[0].annotations.priority == 0.9


def test_fallback_still_forwards_meta_when_from_mcp_result_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a future FastMCP version drops ``ToolResult.from_mcp_result``, the
    degrade must still explicitly forward ``meta`` rather than silently
    reverting to the old lossy behavior."""
    from agent_utilities.mcp import multiplexer as mux_module

    monkeypatch.delattr(mux_module.ToolResult, "from_mcp_result", raising=True)

    result = _child_result_with_annotations_and_meta()
    forwarded = _tool_result_from_child(result)

    assert forwarded.meta == {"readOnlyHint": True, "child_trace_id": "abc123"}
    assert forwarded.content[0].annotations.audience == ["assistant"]
