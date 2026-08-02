"""Regression tests for D-CDX-51: the atomic child-schema rollback's
snapshot/swap of FastMCP's private ``LocalProvider._components`` registry
must be isolated behind an explicit shape-checked adapter that fails CLOSED
on an unrecognized layout BEFORE mutating anything — not a bare
``getattr(..., None)`` + ``isinstance(..., dict)`` check that would let a
future FastMCP version (secondary indexes, lifecycle hooks, ownership
metadata, a different container) silently corrupt host state.

The fleet is deliberately mixed-version (children on fastmcp 3.x, this
canonical tree on fastmcp >=4.0.0b1), so "a future FastMCP minor changes
this private shape" is not hypothetical here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

from agent_utilities.mcp.multiplexer import (
    MCPMultiplexer,
    UnsupportedLocalProviderLayout,
    _local_provider_component_snapshot,
    _swap_local_provider_components,
)


def _real_function_tool(name: str) -> FunctionTool:
    def _fn():
        return "ok"

    return FunctionTool(
        name=name,
        description="d",
        parameters={"type": "object", "properties": {}},
        fn=_fn,
    )


def test_snapshot_succeeds_against_the_real_fastmcp_provider() -> None:
    """The installed FastMCP 4.0.0b1's actual LocalProvider passes the shape
    check — this is the "known-good" counterpart to the fail-closed cases
    below, proving the check isn't so strict it rejects the real SDK."""
    host = FastMCP("shape-check-happy-path")
    host.add_tool(_real_function_tool("t1"))

    provider, snapshot = _local_provider_component_snapshot(host)

    assert provider is host._local_provider
    assert isinstance(snapshot, dict)
    assert len(snapshot) == 1
    # A defensive COPY, not the live dict.
    assert snapshot is not provider._components
    snapshot.clear()
    assert len(provider._components) == 1


def test_swap_replaces_the_live_registry() -> None:
    host = FastMCP("shape-check-swap")
    host.add_tool(_real_function_tool("t1"))
    provider, snapshot = _local_provider_component_snapshot(host)

    new_tool = _real_function_tool("t2")
    replacement = {new_tool.key: new_tool}
    _swap_local_provider_components(provider, replacement)

    assert provider._components == replacement


def test_swap_prefers_a_public_replace_api_if_present() -> None:
    """If a future FastMCP version adds a public atomic replacement method,
    the adapter must use it instead of touching the private attribute."""
    calls = []

    class _FakeProvider:
        def __init__(self):
            self._components = {"a": object()}

        def replace_components(self, components):
            calls.append(components)

    provider = _FakeProvider()
    new_components = {"b": object()}
    _swap_local_provider_components(provider, new_components)

    assert calls == [new_components]
    # The private attribute was NOT touched directly — the public API owns
    # the mutation, so the raw ``_components`` dict this fake provider set
    # up in ``__init__`` is exactly as it was, untouched by the adapter.
    assert list(provider._components.keys()) == ["a"]


@pytest.mark.parametrize(
    "bad_host",
    [
        SimpleNamespace(),  # no _local_provider at all
        SimpleNamespace(_local_provider=None),
        SimpleNamespace(_local_provider=SimpleNamespace()),  # no _components
        SimpleNamespace(_local_provider=SimpleNamespace(_components="not-a-dict")),
        SimpleNamespace(
            _local_provider=SimpleNamespace(_components={"ok": 1, 2: "bad-key-type"})
        ),
    ],
)
def test_snapshot_fails_closed_on_unrecognized_layouts(bad_host) -> None:
    with pytest.raises(UnsupportedLocalProviderLayout):
        _local_provider_component_snapshot(bad_host)


def test_snapshot_fails_closed_when_component_lacks_key_or_name() -> None:
    class _NoKeyOrName:
        pass

    bad_host = SimpleNamespace(
        _local_provider=SimpleNamespace(_components={"x": _NoKeyOrName()})
    )
    with pytest.raises(UnsupportedLocalProviderLayout):
        _local_provider_component_snapshot(bad_host)


@pytest.mark.asyncio
async def test_replace_exposed_forwarders_fails_closed_before_mutating(
    tmp_path,
) -> None:
    """When the host's provider has an unrecognized layout,
    ``_replace_exposed_forwarders`` must raise BEFORE touching anything —
    proven by the exposed set staying untouched."""
    from tests.unit.mcp.test_multiplexer_dynamic_gateway import (
        CNT,
        CNT_PREFIXED,
        CNT_TOOL,
        _fake_tool,
        _mux_with_children,
    )

    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})
    mounted = await mux.mount_child(CNT)
    assert [t.name for t in mounted] == [CNT_PREFIXED]

    class _BrokenHost:
        _local_provider = SimpleNamespace(_components="not-a-dict")

    mux._host_mcp = _BrokenHost()
    mux._exposed.add(CNT_PREFIXED)
    exposed_before = set(mux._exposed)

    old = {CNT_PREFIXED: _fake_tool(CNT_TOOL, "containers old")}
    new = {CNT_PREFIXED: _fake_tool(CNT_TOOL, "containers NEW schema")}

    with pytest.raises(UnsupportedLocalProviderLayout):
        mux._replace_exposed_forwarders(old, new)

    # Nothing was mutated — fail closed, not a partial/corrupted update.
    assert mux._exposed == exposed_before
