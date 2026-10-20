"""Host-provided ports the harness receives from arbora.

CONCEPT:AU-ORCH.adapter.transport-toolset-factory

agent-utilities is the agent harness only. It never serves, registers or
connects to a tool surface itself and carries no MCP code: arbora owns the
MCP serving surface, every MCP client connection and every outbound
credential. When the harness runs inside arbora, arbora registers these
plain typed providers at startup:

* a **native toolset provider** that binds a caller-granted subset of arbora's
  own served operations for one in-process delegation (no self-HTTP);
* a **fleet catalog provider** that maps each configured fleet server to the
  public tool prefix arbora serves it under and its client endpoint;
* a **remote toolset provider** that builds the pydantic-ai toolset for one
  fleet endpoint or one client configuration file (arbora's MCP client);
* a **tool client factory** that opens a ``list_tools``/``call_tool`` client
  session for one endpoint (arbora's MCP client);
* an **outbound credential provider** (service-account auth, header, status,
  forced refresh) and a **delegated identity provider** (the caller's user
  token and on-behalf-of exchange);
* an **authority keepalive** that renews the caller's renewable session lease
  for the whole of one delegation;
* a **run decision provider** that answers, for one run request, the run plan:
  model, tool grant, plan steps, budgets, stop rules and harness settings (the
  decision itself comes from epistemic-graph through arbora);
* a **model provider** that turns a model id from that plan into a pydantic-ai
  model with arbora's credentials attached;
* a **run event sink** that receives every event of one run (arbora writes
  the trace into epistemic-graph);
* a **harness cycle provider** that drives one harness-evolution tick per
  task (propose, probe, compose, persist, reflect) through the served
  decision and graph ports.

Without a registered provider every binding port fails closed (raises) and
every read port reports the empty answer, never inventing a surface of its own.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agent_utilities.harness.plan import PlanItem, PlanState


class HostToolError(RuntimeError):
    """A tool call the host executed reported a tool-level error."""


class ToolClient(Protocol):
    """One open client session onto a tool endpoint, provided by the host."""

    async def list_tools(self) -> list[Any]: ...

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any: ...

    async def read_resource(self, uri: str) -> list[Any]: ...


@dataclass(frozen=True)
class NativeToolsetProvider:
    """arbora's in-process binder for its own served operations."""

    available_tools: Callable[[], Iterable[str]]
    build_toolset: Callable[[list[str], str], Any]


@dataclass(frozen=True)
class FleetServer:
    """One fleet server as arbora serves it: public prefix + client endpoint."""

    prefix: str
    url: str = ""


@dataclass(frozen=True)
class RemoteToolsetProvider:
    """arbora's builder for client toolsets the harness hands to an agent.

    ``build_url_toolset(url, toolset_id, timeout)`` returns one toolset bound to
    ``url`` with arbora's outbound credentials attached;
    ``load_config_toolsets(path)`` returns the toolsets a client configuration
    file declares.
    """

    build_url_toolset: Callable[[str, str | None, float], Any]
    load_config_toolsets: Callable[[Path], list[Any]]


@dataclass(frozen=True)
class OutboundCredentialProvider:
    """arbora's service-account credentials for outbound calls."""

    auth: Callable[[], Any]
    header: Callable[[], dict[str, str]]
    status: Callable[[], dict[str, object]]
    refresh: Callable[[], None]


@dataclass(frozen=True)
class DelegatedIdentityProvider:
    """arbora's caller identity: the user token and on-behalf-of exchange."""

    user_token: Callable[[], str | None]
    exchange_for_agent: Callable[[str], str | None]


ToolClientFactory = Callable[..., contextlib.AbstractAsyncContextManager[ToolClient]]

RunDecisionProvider = Callable[[Any], Any]
"""``(RunRequest) -> RunPlan`` (or an awaitable of it); see ``agent_utilities.graph``."""

ModelProvider = Callable[[str], Any]
"""``(model_id) -> pydantic-ai Model`` with the host's credentials attached."""

RunEventSink = Callable[[Mapping[str, Any]], None]
"""Receives one run event (a JSON-shaped mapping) in emission order."""

HarnessCycleProvider = Callable[..., Any]
"""``(request, plan, *, task_class, graph, config, root, mode) ->
CycleOutcome`` (or an awaitable of it): the host drives one
harness-evolution tick per task through the served decision and graph
ports; see ``agent_utilities.harness.harness_cycle``."""

_NATIVE: NativeToolsetProvider | None = None
_FLEET: Callable[[], Mapping[str, FleetServer]] | None = None
_KEEPALIVE: Callable[[], contextlib.AbstractAsyncContextManager[None]] | None = None
_REMOTE: RemoteToolsetProvider | None = None
_TOOL_CLIENT: ToolClientFactory | None = None
_CREDENTIALS: OutboundCredentialProvider | None = None
_IDENTITY: DelegatedIdentityProvider | None = None
_DECIDE: RunDecisionProvider | None = None
_MODELS: ModelProvider | None = None
_EVENTS: RunEventSink | None = None
_CYCLE: HarnessCycleProvider | None = None


def register_run_decision_provider(provider: RunDecisionProvider | None) -> None:
    """Install (or clear) the host's run-plan provider."""
    global _DECIDE
    _DECIDE = provider


def register_model_provider(provider: ModelProvider | None) -> None:
    """Install (or clear) the host's model-id resolver."""
    global _MODELS
    _MODELS = provider


def register_run_event_sink(sink: RunEventSink | None) -> None:
    """Install (or clear) the host's run event sink."""
    global _EVENTS
    _EVENTS = sink


def register_harness_cycle_provider(provider: HarnessCycleProvider | None) -> None:
    """Install (or clear) the host's harness-evolution cycle driver."""
    global _CYCLE
    _CYCLE = provider


def run_decision_provider() -> RunDecisionProvider | None:
    """The host's run-plan provider, or ``None`` (the harness then uses defaults)."""
    return _DECIDE


def harness_cycle_provider() -> HarnessCycleProvider | None:
    """The host's cycle driver, or ``None`` (the tick then fails closed)."""
    return _CYCLE


def resolve_model(model_id: str) -> Any:
    """The host-built model for ``model_id``.

    Without a host provider the id passes through unchanged, so pydantic-ai
    resolves it itself (``"provider:name"`` ids need that provider's extra).
    """
    return model_id if _MODELS is None else _MODELS(model_id)


def emit_run_event(event: Mapping[str, Any]) -> None:
    """Send one run event to the host sink; a no-op without a host."""
    if _EVENTS is not None:
        _EVENTS(event)


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


def register_remote_toolset_provider(provider: RemoteToolsetProvider | None) -> None:
    """Install (or clear) the host's client-toolset builder."""
    global _REMOTE
    _REMOTE = provider


def register_tool_client_factory(factory: ToolClientFactory | None) -> None:
    """Install (or clear) the host's ``open_tool_client`` factory."""
    global _TOOL_CLIENT
    _TOOL_CLIENT = factory


def register_outbound_credential_provider(
    provider: OutboundCredentialProvider | None,
) -> None:
    """Install (or clear) the host's outbound service-account credentials."""
    global _CREDENTIALS
    _CREDENTIALS = provider


def register_delegated_identity_provider(
    provider: DelegatedIdentityProvider | None,
) -> None:
    """Install (or clear) the host's caller-identity provider."""
    global _IDENTITY
    _IDENTITY = provider


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
            "native arbora toolset requires the arbora host to register a provider"
        )
    return _NATIVE.build_toolset(list(tool_names), toolset_id)


def build_url_toolset(
    url: str, *, toolset_id: str | None = None, timeout: float = 60.0
) -> Any:
    """The host-built client toolset for ``url``; fail closed without a host."""
    if _REMOTE is None:
        raise RuntimeError(
            "remote toolsets require the arbora host to register a provider"
        )
    return _REMOTE.build_url_toolset(str(url), toolset_id, float(timeout))


def load_config_toolsets(config_path: str | Path) -> list[Any]:
    """The toolsets a client configuration file declares; fail closed without a host."""
    if _REMOTE is None:
        raise RuntimeError(
            "configured toolsets require the arbora host to register a provider"
        )
    return list(_REMOTE.load_config_toolsets(Path(config_path)))


def open_tool_client(
    target: Any, *, timeout: float | None = None
) -> contextlib.AbstractAsyncContextManager[ToolClient]:
    """Open one host-provided tool client session; fail closed without a host."""
    if _TOOL_CLIENT is None:
        raise RuntimeError(
            "tool client sessions require the arbora host to register a factory"
        )
    return _TOOL_CLIENT(target, timeout=timeout)


def outbound_auth() -> Any:
    """The host's refresh-capable outbound auth (``None`` without a host)."""
    return None if _CREDENTIALS is None else _CREDENTIALS.auth()


def outbound_auth_header() -> dict[str, str]:
    """The host's current outbound ``Authorization`` header (empty without one)."""
    return {} if _CREDENTIALS is None else dict(_CREDENTIALS.header())


def outbound_auth_status() -> dict[str, object]:
    """The host's outbound-credential configuration status."""
    if _CREDENTIALS is None:
        return {
            "mode": "host-unbound",
            "ready": False,
            "problems": ["no arbora host credential provider is registered"],
            "warnings": [],
        }
    return dict(_CREDENTIALS.status())


def refresh_outbound_credentials() -> None:
    """Force the host to rotate its outbound bearer; fail closed without one."""
    if _CREDENTIALS is None:
        raise RuntimeError("outbound credential renewal requires the arbora host")
    _CREDENTIALS.refresh()


def current_user_token() -> str | None:
    """The caller's user token as the host resolved it (``None`` without one)."""
    return None if _IDENTITY is None else _IDENTITY.user_token()


def exchange_token_for_agent(label: str) -> str | None:
    """On-behalf-of exchange for ``label`` via the host (``None`` without one)."""
    if _IDENTITY is None:
        return None
    return _IDENTITY.exchange_for_agent(label)


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
    """The public name arbora serves a fleet tool under.

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


class ScaffoldPublisher(Protocol):
    """Host-owned publication of an actual run-local scaffold revision."""

    async def publish(self, revision: int, items: tuple[PlanItem, ...]) -> None: ...


ScaffoldPublicationFactory = Callable[
    [Any], contextlib.AbstractAsyncContextManager[ScaffoldPublisher]
]
_SCAFFOLD_PUBLICATION: ScaffoldPublicationFactory | None = None


def register_scaffold_publication_factory(
    factory: ScaffoldPublicationFactory | None,
) -> None:
    """Install the host binder; request metadata alone supplies no authority."""
    global _SCAFFOLD_PUBLICATION
    _SCAFFOLD_PUBLICATION = factory


class _ScopedPublisher:
    def __init__(self, publisher: ScaffoldPublisher) -> None:
        self.publisher = publisher
        self.active = True

    async def publish(self, revision: int, items: tuple[PlanItem, ...]) -> None:
        if not self.active:
            raise RuntimeError("scaffold publication scope is closed")
        try:
            await self.publisher.publish(revision, items)
        except Exception:
            raise RuntimeError("scaffold publication failed") from None


@contextlib.asynccontextmanager
async def scaffold_publication_scope(
    request: Any, state: PlanState | None
) -> AsyncIterator[ScaffoldPublisher | None]:
    """Publish actual state through a host binding, never a local read registry.

    The host owns verified identity, lease fencing, expiry and wire mapping.
    Missing providers make no availability claim. Failures may follow an external
    write; the accepted local revision is never rolled back.
    """
    factory = _SCAFFOLD_PUBLICATION
    if state is None or factory is None:
        yield None
        return
    try:
        scope = factory(request)
        publisher = _ScopedPublisher(await scope.__aenter__())
    except Exception:
        raise RuntimeError("scaffold publication binding failed") from None
    failure: BaseException | None = None
    try:
        await publisher.publish(*state.read_versioned())
        yield publisher
    except BaseException as exc:
        failure = exc
        raise
    finally:
        publisher.active = False
        try:
            await scope.__aexit__(
                type(failure) if failure is not None else None,
                failure,
                failure.__traceback__ if failure is not None else None,
            )
        except Exception:
            if failure is None:
                raise RuntimeError("scaffold publication close failed") from None
