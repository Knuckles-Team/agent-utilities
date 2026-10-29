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
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agent_utilities.graph.state import ToolStepResult
    from agent_utilities.harness.plan import PlanItem, PlanState


class HostToolError(RuntimeError):
    """A tool call the host executed reported a tool-level error."""


class ClientCapabilityTransport(Protocol):
    """One host-bound client session; the host owns grants and approval state."""

    async def call(self, method: str, params: Mapping[str, Any]) -> str:
        """Dispatch once or raise ClientCapabilityRefused; never infer approval."""
        ...


class ClientCapabilityRefused(RuntimeError):
    """Closed host refusal safe to return to a model without private details."""

    def __init__(self, reason: str) -> None:
        allowed = {
            "permission_required",
            "permission_denied",
            "client_disconnected",
            "capability_not_declared",
            "authority_unavailable",
        }
        self.reason = reason if reason in allowed else "authority_unavailable"
        super().__init__(self.reason)


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

ToolCaller = Callable[[str, str, Mapping[str, Any], str], Awaitable["ToolStepResult"]]
"""``(server, tool, arguments, task_id) -> ToolStepResult`` for one run."""

ModelProvider = Callable[[str], Any]
"""``(model_id) -> pydantic-ai Model`` with the host's credentials attached."""

RunEventSink = Callable[[Mapping[str, Any]], None]
"""Receives one run event (a JSON-shaped mapping) in emission order."""


class DurableRunEventPublisher(Protocol):
    """One host-owned durable writer bound to the admitted run."""

    async def publish(self, kind: str, content: Mapping[str, Any]) -> None:
        """Persist one validated public model/tool observation."""
        ...


DurableRunEventFactory = Callable[
    [Any], contextlib.AbstractAsyncContextManager[DurableRunEventPublisher | None]
]
"""Build one optional durable event publisher for a run request."""


@dataclass(frozen=True, slots=True)
class ActiveChildBinding:
    """Identity supplied by the host after child admission."""

    child_run_id: str
    parent_run_id: str
    admission_id: str
    lifecycle_revision: int


@dataclass(frozen=True, slots=True)
class ActiveChildMessage:
    """A bounded host-claimed message, not an AU admission decision."""

    binding: ActiveChildBinding
    message_id: str
    event_id: str
    content: str


class ActiveChildInputPort(Protocol):
    """Host-owned ordered admission, duplicate control and durable receipts.

    Receipt acknowledges insertion into this live run's volatile history, not
    model consumption. The host must retain uncertainty on a crash or failed
    receipt, and its context manager must settle undelivered inputs on exit.
    """

    binding: ActiveChildBinding

    async def take_ready(self) -> tuple[ActiveChildMessage, ...]:
        """Claim an ordered batch after revalidating admission and lifecycle.

        The host never returns a claimed message twice for this live run.
        """
        ...

    async def report_applied(self, message: ActiveChildMessage) -> None:
        """Durably acknowledge one message; failures leave host outcome uncertain."""
        ...


ActiveChildInputFactory = Callable[
    [Any], contextlib.AbstractAsyncContextManager[ActiveChildInputPort | None]
]


HarnessProbeProvider = Callable[..., Awaitable[Mapping[str, Any]]]
"""Host admission and execution of one materialized probe; not measurement proof."""

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
_TOOL_CALLER: ToolCaller | None = None
_MODELS: ModelProvider | None = None
_EVENTS: RunEventSink | None = None
_DURABLE_RUN_EVENTS: DurableRunEventFactory | None = None
_ACTIVE_CHILD_INPUT: ActiveChildInputFactory | None = None
_CYCLE: HarnessCycleProvider | None = None
_PROBE: HarnessProbeProvider | None = None


def register_run_decision_provider(provider: RunDecisionProvider | None) -> None:
    """Install (or clear) the host's run-plan provider."""
    global _DECIDE
    _DECIDE = provider


def register_tool_caller(caller: ToolCaller | None) -> None:
    """Install (or clear) the host's caller for admitted dependency tool steps."""
    global _TOOL_CALLER
    _TOOL_CALLER = caller


def register_model_provider(provider: ModelProvider | None) -> None:
    """Install (or clear) the host's model-id resolver."""
    global _MODELS
    _MODELS = provider


def register_run_event_sink(sink: RunEventSink | None) -> None:
    """Install (or clear) the host's run event sink."""
    global _EVENTS
    _EVENTS = sink


def register_durable_run_event_factory(
    factory: DurableRunEventFactory | None,
) -> None:
    """Install (or clear) the host binder for durable run-event writes.

    The request is only the host's binding key. AU does not derive authority,
    create a local event store, or open a transport from request metadata.
    """
    global _DURABLE_RUN_EVENTS
    _DURABLE_RUN_EVENTS = factory


def register_harness_probe_provider(provider: HarnessProbeProvider | None) -> None:
    """Install or clear the host-owned admitted probe executor."""
    global _PROBE
    _PROBE = provider


def harness_probe_provider() -> HarnessProbeProvider | None:
    """Return the host probe executor; absence cannot authorize execution."""
    return _PROBE


def register_harness_cycle_provider(provider: HarnessCycleProvider | None) -> None:
    """Install (or clear) the host's harness-evolution cycle driver."""
    global _CYCLE
    _CYCLE = provider


def run_decision_provider() -> RunDecisionProvider | None:
    """The host's run-plan provider, or ``None`` (the harness then uses defaults)."""
    return _DECIDE


def tool_caller() -> ToolCaller | None:
    """Return the host-bound dependency caller, or ``None`` without a host."""
    return _TOOL_CALLER


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


def register_active_child_input_factory(
    factory: ActiveChildInputFactory | None,
) -> None:
    """Install the host's admitted-child input binder."""
    global _ACTIVE_CHILD_INPUT
    _ACTIVE_CHILD_INPUT = factory


@contextlib.asynccontextmanager
async def active_child_input_scope(
    request: Any,
) -> AsyncIterator[ActiveChildInputPort | None]:
    """Open one optional host-owned child input binding."""
    factory = _ACTIVE_CHILD_INPUT
    if factory is None:
        yield None
        return
    async with factory(request) as port:
        yield port


@contextlib.asynccontextmanager
async def durable_run_event_scope(
    request: Any,
) -> AsyncIterator[DurableRunEventPublisher | None]:
    """Open the optional host-owned durable event port for one run."""
    factory = _DURABLE_RUN_EVENTS
    if factory is None:
        yield None
        return
    async with factory(request) as publisher:
        yield publisher


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
OptionalScaffoldPublicationFactory = Callable[
    [Any], contextlib.AbstractAsyncContextManager[ScaffoldPublisher | None]
]
_SCAFFOLD_PUBLICATION: OptionalScaffoldPublicationFactory | None = None


def register_scaffold_publication_factory(
    factory: ScaffoldPublicationFactory | None,
) -> None:
    """Install the host binder; request metadata alone supplies no authority."""
    global _SCAFFOLD_PUBLICATION
    _SCAFFOLD_PUBLICATION = factory


def register_optional_scaffold_publication_factory(
    factory: OptionalScaffoldPublicationFactory | None,
) -> None:
    """Install a host binder that may decline publication for an individual run.

    A declined binding leaves the local scaffold usable and makes no external
    availability claim. The host scope still closes on success or failure.
    """
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
        bound = await scope.__aenter__()
        publisher = _ScopedPublisher(bound) if bound is not None else None
    except Exception:
        raise RuntimeError("scaffold publication binding failed") from None
    failure: BaseException | None = None
    try:
        if publisher is not None:
            await publisher.publish(*state.read_versioned())
        yield publisher
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if publisher is not None:
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


RunHistoryProvider = Callable[[Any], Awaitable[str | None]]
_RUN_HISTORY: RunHistoryProvider | None = None


def register_run_history_provider(provider: RunHistoryProvider | None) -> None:
    """Bind the host-authorized prior-turn reader, never caller metadata."""
    global _RUN_HISTORY
    _RUN_HISTORY = provider


@dataclass(frozen=True, slots=True)
class CompletedChildBinding:
    """Host-authorized identity of a completed child's selected history."""

    child_id: str
    conversation_id: str
    predecessor_run_id: str
    history_revision: str


@dataclass(frozen=True, slots=True)
class RunCheckpointSelection:
    """One host-authorized read, not caller-supplied execution authority.

    The host binds run, caller and conversation to one consistent native read.
    Both fields are required: None means absent, not unavailable or unchecked.
    Completed history uses the existing version-one text envelope. Current-run
    content is opaque until the harness can restore it without replaying effects.
    """

    run_id: str
    current_run_checkpoint: str | None
    completed_history: str | None
    version: int = 1
    completed_child: CompletedChildBinding | None = None


RunCheckpointProvider = Callable[[Any], Awaitable[RunCheckpointSelection]]
_RUN_CHECKPOINTS: RunCheckpointProvider | None = None


def register_run_checkpoint_provider(provider: RunCheckpointProvider | None) -> None:
    """Bind a single consistent checkpoint selection read from the trusted host."""
    global _RUN_CHECKPOINTS
    _RUN_CHECKPOINTS = provider


async def _read_checkpoint_selection(
    request: Any, provider: RunCheckpointProvider
) -> str | None:
    try:
        selection = await provider(request)
    except Exception:
        raise RuntimeError("run checkpoint provider failed") from None
    if (
        not isinstance(selection, RunCheckpointSelection)
        or type(selection.version) is not int
        or selection.version != 1
        or not isinstance(selection.run_id, str)
        or not selection.run_id.strip()
        or selection.run_id != getattr(request, "run_id", None)
    ):
        raise ValueError("run checkpoint selection is invalid")
    # Presence is authoritative even for empty, malformed or in-flight content.
    # Never reinterpret it as prior history or silently start a replacement run.
    if selection.current_run_checkpoint is not None:
        raise RuntimeError("current-run checkpoint restore is unavailable")
    if selection.completed_history is not None and not isinstance(
        selection.completed_history, str
    ):
        raise ValueError("run checkpoint selection is invalid")
    if selection.completed_child is not None:
        from agent_utilities.harness.conversation_history import child_history_envelope

        return child_history_envelope(
            selection.completed_history,
            selection.completed_child,
            run_id=selection.run_id,
        )
    return selection.completed_history


async def read_run_history(request: Any) -> str | None:
    """Use one checkpoint read; legacy history is only for an unbound new port."""
    checkpoint_provider = _RUN_CHECKPOINTS
    if checkpoint_provider is not None:
        return await _read_checkpoint_selection(request, checkpoint_provider)
    provider = _RUN_HISTORY
    if provider is None:
        return None
    try:
        return await provider(request)
    except Exception:
        raise RuntimeError("conversation history provider failed") from None


async def read_recursive_run_history(request: Any | None) -> str | None:
    """Read one bound selection only for an explicitly identified recursive run."""
    checkpoint_provider = _RUN_CHECKPOINTS
    history_provider = _RUN_HISTORY
    if checkpoint_provider is None and history_provider is None:
        return None
    if request is None:
        raise ValueError("recursive history selection needs admitted lineage")
    if checkpoint_provider is not None:
        return await _read_checkpoint_selection(request, checkpoint_provider)
    assert history_provider is not None
    try:
        return await history_provider(request)
    except Exception:
        raise RuntimeError("conversation history provider failed") from None


class RunCheckpointPort(Protocol):
    """One host-owned checkpoint transaction for an admitted run.

    The harness only serializes and validates the bounded checkpoint payload.
    The host owns the durable snapshot, revision CAS and admission fencing.
    """

    admission_identity: str

    async def restore(self) -> str | None: ...

    async def save(self, content: str) -> None: ...


RunCheckpointFactory = Callable[
    [Any, Any], contextlib.AbstractAsyncContextManager[RunCheckpointPort | None]
]
_CHECKPOINT_FACTORY: RunCheckpointFactory | None = None


def register_run_checkpoint_factory(factory: RunCheckpointFactory | None) -> None:
    """Install (or clear) the host-owned native checkpoint binder."""
    global _CHECKPOINT_FACTORY
    _CHECKPOINT_FACTORY = factory


def run_checkpoint_factory_bound() -> bool:
    """Return whether a host checkpoint transaction is available for this process."""
    return _CHECKPOINT_FACTORY is not None


@contextlib.asynccontextmanager
async def run_checkpoint_scope(
    request: Any, plan: Any
) -> AsyncIterator[RunCheckpointPort | None]:
    """Open the optional host checkpoint transaction for one admitted run."""
    factory = _CHECKPOINT_FACTORY
    if factory is None:
        yield None
        return
    async with factory(request, plan) as checkpoint:
        yield checkpoint
