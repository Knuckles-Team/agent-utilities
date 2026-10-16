"""Host-provided bindings the harness receives from graph-os.

CONCEPT:AU-ORCH.adapter.transport-toolset-factory

The harness never serves or registers tools and never parses the fleet
catalog. When it runs inside graph-os, graph-os registers two providers at
startup:

* a **native toolset provider** that binds a caller-granted subset of graph-os's
  own served operations for one in-process delegation (no self-HTTP), and
* a **fleet catalog provider** that maps each configured fleet server to the
  public tool prefix graph-os serves it under and its client endpoint, and
* an **authority keepalive** that renews the caller's renewable session lease
  for the whole of one delegation.

Without a registered provider the harness fails closed (native toolsets) or
reports an empty fleet (catalog), never inventing a surface of its own.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NativeToolsetProvider:
    """graph-os's in-process binder for its own served operations."""

    available_tools: Callable[[], Iterable[str]]
    build_toolset: Callable[[list[str], str], Any]


@dataclass(frozen=True)
class FleetServer:
    """One fleet server as graph-os serves it: public prefix + client endpoint."""

    prefix: str
    url: str = ""


_NATIVE: NativeToolsetProvider | None = None
_FLEET: Callable[[], Mapping[str, FleetServer]] | None = None
_KEEPALIVE: Callable[[], contextlib.AbstractAsyncContextManager[None]] | None = None


def register_native_toolset_provider(provider: NativeToolsetProvider | None) -> None:
    """Install (or clear, with ``None``) the host's native toolset binder."""
    global _NATIVE
    _NATIVE = provider


def register_fleet_catalog_provider(
    provider: Callable[[], Mapping[str, FleetServer]] | None,
) -> None:
    """Install (or clear) the host's ``{server_name: FleetServer}`` view."""
    global _FLEET
    _FLEET = provider


def register_authority_keepalive(
    factory: Callable[[], contextlib.AbstractAsyncContextManager[None]] | None,
) -> None:
    """Install (or clear) the host's session-lease keepalive scope factory."""
    global _KEEPALIVE
    _KEEPALIVE = factory


@contextlib.asynccontextmanager
async def authority_keepalive_scope() -> AsyncIterator[None]:
    """Hold the host's lease keepalive open; a no-op without a host."""
    if _KEEPALIVE is None:
        yield
        return
    async with _KEEPALIVE():
        yield


def native_tool_names() -> frozenset[str]:
    """Names of the host's served operations available for native binding."""
    if _NATIVE is None:
        return frozenset()
    return frozenset(str(name) for name in _NATIVE.available_tools())


def build_native_toolset(tool_names: list[str], *, toolset_id: str) -> Any:
    """Bind ``tool_names`` from the host; fail closed when no host is attached."""
    if _NATIVE is None:
        raise RuntimeError(
            "native graph-os toolset requires the graph-os host to register a provider"
        )
    return _NATIVE.build_toolset(list(tool_names), toolset_id)


def fleet_catalog() -> dict[str, FleetServer]:
    """The host's ``{server_name: FleetServer}`` map (empty without a host)."""
    if _FLEET is None:
        return {}
    return {
        name.strip(): server
        for name, server in dict(_FLEET()).items()
        if isinstance(name, str) and name.strip() and isinstance(server, FleetServer)
    }


def clean_tool_name(prefix: str, server_name: str, original_tool_name: str) -> str:
    """The public name graph-os serves a fleet tool under.

    Strips redundant server/module prefixes from the raw tool name and keeps the
    namespaced result within the 44-character budget (so a client-prefixed name
    stays <= 64 characters).
    """
    if server_name.startswith("systems-manager-mcp-"):
        base_server = "systems-manager-mcp"
    elif server_name.startswith("container-manager-mcp-"):
        base_server = "container-manager-mcp"
    else:
        base_server = server_name
    clean_server = base_server.replace("-", "_").lower()
    strips = [
        f"{clean_server}_mcp_",
        f"{clean_server}_",
        f"{prefix}_mcp_",
        f"{prefix}_",
    ]
    if base_server.endswith("-mcp"):
        mod_server = base_server[:-4].replace("-", "_").lower()
        strips.extend((f"{mod_server}_mcp_", f"{mod_server}_"))
    cleaned = original_tool_name
    for strip in strips:
        if cleaned.startswith(strip):
            cleaned = cleaned[len(strip) :]
            break
    candidate = f"{prefix}__{cleaned}"
    if len(candidate) > 44:
        budget = 44 - len(prefix) - 2
        candidate = f"{prefix}__{cleaned[:budget].strip('_')}"
    return candidate
