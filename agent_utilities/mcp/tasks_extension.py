"""Native GraphOS ``io.modelcontextprotocol/tasks`` extension, backed by WorkItem.

CONCEPT:AU-ECO.mcp.tasks-workitem-bridge

Serves the WorkItem-backed Tasks contract for a client connected directly to
GraphOS's own FastMCP 4 server, over whichever MCP protocol version that
client negotiates -- ``mcp>=2.0.0`` is now the default (see
``server_factory.py``, ``docs/architecture/fastmcp4-default.md``), and the
2026-07-28 stateless single-exchange transport it enables is served natively
by the installed SDK on this same server, not a separate process (BUG-069;
see ``docs/architecture/mcp-2026-protocol-surface.md``). This module used to
mirror an equivalent projection an isolated ``mcp_v2_gateway`` sidecar
carried over its own HTTP hop to a legacy GraphOS session; that sidecar has
been retired -- this is now the only implementation.

Deliberately does **not** mount ``fastmcp_tasks.extension.TasksExtension``:
that package's execution engine is hard-wired to a Docket/Redis backend
(``fastmcp_tasks.handlers``/``creation.py``, ``pydocket`` dependency) -- a
second, parallel job system this codebase's one ``WorkItem`` state machine
(``AU-P1-1``) already forbids duplicating (``AGENTS.md``: extend before you
add). This module only reuses the extension's *registration* mechanism
(``fastmcp.server.extensions.ServerExtension``/``MethodBinding``) -- never
its execution engine -- and backs every method with
``agent_utilities.orchestration.work_item`` directly, the exact same
authority ``graph_jobs`` (``agent_utilities/mcp/tools/job_tools.py``) uses.
The wire model field names below follow ``fastmcp_tasks.models`` (the
current SEP-2663 draft: flat task fields, ``inputRequests`` for
``input_required``) rather than the older, reduced revision the retired
sidecar had pinned to (its own ``TASKS_EXTENSION_REVISION``) for reasons
that no longer apply once there is only one implementation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field, field_validator

from agent_utilities.mcp.protocol_compat import mcp_protocol_exception

# MCP SDK v2 (which the `fastmcp>=4.0.0b1` floor pulls in) renamed
# `McpError` -> `MCPError`. The fleet is EXPLICITLY mixed-version: child images
# still ship fastmcp 3.x / SDK v1 while hostPath-mounting THIS working tree over
# their own site-packages, so a hard `from mcp.shared.exceptions import MCPError`
# raises ImportError at MODULE scope on every one of them (D-W2C2-3). That is the
# same class of break as the `fastmcp.server.extensions` guard below, one SDK
# down. `mcp_protocol_error()` binds whichever spelling the installed SDK
# exposes and raises loudly if neither does — it never degrades to a benign
# default, because a silently-unbound error type made `is_session_dead` return
# False for every exception once already. Errors this extension emits are built
# through the same resolver, preserving the installed SDK's wire exception.

# CONCEPT:AU-ECO.mcp.tasks-workitem-bridge -- the fleet is EXPLICITLY mixed
# fastmcp-version (D-SH-3: child images still ship fastmcp 3.4.4 while the
# canonical working tree, hostPath-mounted directly over each pod's
# site-packages, targets fastmcp>=4.0.0b1 -- see docs/architecture/
# fastmcp4-default.md). `fastmcp.server.extensions` (ServerExtension,
# MethodBinding, read_client_extension_settings) is a fastmcp-4-only module;
# importing it unguarded at module scope takes the WHOLE server down before
# anything else runs (D-W2C2-2: 58 fleet pods in CrashLoopBackOff on exactly
# this `ModuleNotFoundError`). Mirrors the same style already used for the
# MCP-SDK v1/v2 `McpError`/`MCPError` rename in `protocol_compat.py` and for
# optional middlewares in `server_factory._configure_middleware`: guard the
# import, degrade to a no-op capability, and log loudly so an operator can
# tell "Tasks extension unavailable on this image" from "server broken".
#
# ★ The guard covers `mcp_types` too. `mcp_types` is a fastmcp-4/SDK-v2-only
# distribution, so on a fastmcp-3 image `from mcp_types import ...` fails at
# module scope for exactly the same reason and with exactly the same blast
# radius. Guarding only `fastmcp.server.extensions` moved the crash three lines
# up rather than fixing it (observed live: 58 pods cleared the extensions
# import and then died on `mcp_types`). ONE guard, ONE failure mode, ONE flag —
# every fastmcp-4-only symbol this module needs is bound here or not at all.
if TYPE_CHECKING:
    from fastmcp.server.extensions import (
        MethodBinding,
        ServerExtension,
        read_client_extension_settings,
    )
    from mcp_types import RequestParams, Result
    from mcp_types.jsonrpc import MISSING_REQUIRED_CLIENT_CAPABILITY
    from mcp_types.version import MODERN_PROTOCOL_VERSIONS

    TASKS_EXTENSION_AVAILABLE = True
    _TASKS_EXTENSION_IMPORT_ERROR: ImportError | None = None
else:
    try:
        from fastmcp.server.extensions import (
            MethodBinding,
            ServerExtension,
            read_client_extension_settings,
        )
        from mcp_types import RequestParams, Result
        from mcp_types.jsonrpc import MISSING_REQUIRED_CLIENT_CAPABILITY
        from mcp_types.version import MODERN_PROTOCOL_VERSIONS
    except ImportError as _fastmcp_extensions_import_error:
        TASKS_EXTENSION_AVAILABLE = False
        _TASKS_EXTENSION_IMPORT_ERROR: ImportError | None = (
            _fastmcp_extensions_import_error
        )

        # Stand-ins for the `mcp_types` symbols. `RequestParams`/`Result` are only
        # ever used as pydantic base classes for the private `_GetTaskParams` /
        # `_GetTaskResult` shapes below, so a BaseModel keeps those class
        # definitions valid; the constants only need a value that never matches.
        from pydantic import BaseModel as _CompatBaseModel

        class RequestParams(_CompatBaseModel):
            """Stand-in for ``mcp_types.RequestParams`` (fastmcp<4)."""

        class Result(_CompatBaseModel):
            """Stand-in for ``mcp_types.Result`` (fastmcp<4)."""

        # -32002 is the SDK's own code for this condition; the value is inert here
        # because every path that reads it is gated behind TASKS_EXTENSION_AVAILABLE.
        MISSING_REQUIRED_CLIENT_CAPABILITY = -32002
        # Empty, so `_TASK_METHOD_VERSIONS` matches no protocol version at all —
        # the Tasks methods are correctly invisible on an image that cannot serve them.
        MODERN_PROTOCOL_VERSIONS: tuple[str, ...] = ()

        # Fallback stand-ins so `class WorkItemTasksExtension(ServerExtension)`
        # below still defines cleanly (it overrides `methods()` itself, so the
        # real base class's behavior is never needed on this path). Neither is
        # ever exercised for real: `server_factory.create_mcp_server` checks
        # `TASKS_EXTENSION_AVAILABLE` before calling `mcp.add_extension(...)`, so
        # `methods()` -- the only place `MethodBinding`/
        # `read_client_extension_settings` are used -- is never invoked. If some
        # other caller ever does instantiate the extension directly on a
        # fastmcp-3 image and reach that path, this raises the ORIGINAL
        # ModuleNotFoundError (chained, not swallowed) instead of a confusing
        # `NameError`.
        class ServerExtension:
            """Stand-in for ``fastmcp.server.extensions.ServerExtension`` (fastmcp<4)."""

            __slots__ = ()

        class MethodBinding:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise ModuleNotFoundError(
                    "fastmcp.server.extensions.MethodBinding requires fastmcp>=4.0.0b1"
                ) from _TASKS_EXTENSION_IMPORT_ERROR

        def read_client_extension_settings(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError(
                "fastmcp.server.extensions.read_client_extension_settings requires "
                "fastmcp>=4.0.0b1"
            ) from _TASKS_EXTENSION_IMPORT_ERROR
    else:
        TASKS_EXTENSION_AVAILABLE = True
        _TASKS_EXTENSION_IMPORT_ERROR = None

if TYPE_CHECKING:
    from mcp.server.context import ServerRequestContext

logger = logging.getLogger(__name__)

TASKS_EXTENSION_ID = "io.modelcontextprotocol/tasks"
# The revision is the wire contract revision used by this extension.  It is
# intentionally separate from the MCP protocol version: the protocol version
# gates method registration, while this value lets a multiplexer reject a
# route from a server that implements a different task projection.
TASKS_EXTENSION_REVISION = "2c1425d9a288b9b1f489430fe1e00bb392b47e48"
_TASK_DELEGATION_CHANNEL_ENV = "AGENT_UTILITIES_MCP_TASK_CHANNEL_SECRET"
_TASK_METHOD_VERSIONS = frozenset(MODERN_PROTOCOL_VERSIONS)


def _channel_proof(secret: str, token: str) -> str:
    """Bind one task proof to the private per-child-generation channel."""

    return hmac.new(
        secret.encode("utf-8"), token.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _delegation_payload(
    method: str,
    params: Mapping[str, Any],
    *,
    server: str,
    revision: str,
    caller: Mapping[str, Any],
) -> bytes:
    """Canonical bytes bound by the multiplexer service bearer.

    The task request body is hashed without its ``_meta`` envelope; the
    method, owning server, exact extension revision, and normalized caller are
    all covered by the resulting run-token binding. This prevents replaying a
    valid route for another task or owner while keeping secrets out of the MCP
    metadata itself.
    """

    body = {
        "caller": {
            "owner": str(caller.get("owner") or ""),
            "scopes": sorted(str(scope) for scope in caller.get("scopes", ())),
            "tenant": str(caller.get("tenant") or ""),
        },
        "method": method,
        "params": {str(key): value for key, value in params.items() if key != "_meta"},
        "revision": revision,
        "server": server,
    }
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _delegation_binding(
    method: str,
    params: Mapping[str, Any],
    *,
    server: str,
    revision: str,
    caller: Mapping[str, Any],
) -> str:
    return hashlib.sha256(
        _delegation_payload(
            method,
            params,
            server=server,
            revision=revision,
            caller=caller,
        )
    ).hexdigest()


def _mint_delegation_token(
    method: str,
    params: Mapping[str, Any],
    *,
    server: str,
    revision: str,
    caller: Mapping[str, Any],
) -> str:
    """Mint a short-lived, shared-secret run token for one task request."""

    from agent_utilities.security.run_token import mint_token

    binding = _delegation_binding(
        method,
        params,
        server=server,
        revision=revision,
        caller=caller,
    )
    return mint_token(
        f"mcp-task:{binding}",
        project=server,
        endpoints=(server,),
        operations=(method,),
        ttl_seconds=30.0,
        actor_id=str(caller.get("owner") or ""),
        tenant_id=str(caller.get("tenant") or ""),
    )


def _installed_fastmcp_version() -> str:
    """Best-effort ``fastmcp`` version string for the degrade-mode warning."""
    try:
        import importlib.metadata

        return f"fastmcp=={importlib.metadata.version('fastmcp')}"
    except Exception:  # noqa: BLE001 — purely cosmetic, never block the warning itself
        return "an unknown fastmcp version"


if not TASKS_EXTENSION_AVAILABLE:
    # Preserve and surface the cause (never a bare/silent fallback -- swallowed
    # ImportErrors have twice destroyed the diagnosis path on this program
    # today). `from None` is deliberately NOT used below: the original
    # ModuleNotFoundError is chained via `raise ... from` at the one call site
    # that actually needs to fail (`WorkItemTasksExtension()` is never
    # constructed on this path), and is logged here so it shows up even when
    # nothing ever tries to instantiate the extension.
    logger.warning(
        "tasks_extension: WorkItem Tasks extension (%s) unavailable on this "
        "image -- requires fastmcp>=4.0.0b1 (fastmcp.server.extensions), this "
        "server has %s. The server will start WITHOUT tasks/get, "
        "tasks/update, tasks/cancel; every other capability is unaffected. "
        "Root cause: %s",
        TASKS_EXTENSION_ID,
        _installed_fastmcp_version(),
        _TASKS_EXTENSION_IMPORT_ERROR,
    )

# WorkItem raw statuses that project onto the extension's "working" (or, with
# a live pending_input_request, "input_required") wire status. Mirrors
# mcp_v2_gateway.gateway.GraphOSV2Gateway._WORKING_RAW_STATUSES, but this
# module targets `graph_jobs`'s orchestrator-dispatch WorkItems, whose raw
# vocabulary never includes "queued"/"pending"/"executing" (those are
# ingest/legacy-bridge synonyms projected only by the gateway's downstream
# `graph_jobs(action="status")` JSON, not the raw WORK_ITEM_STATES tuple).
_WORKING_RAW_STATUSES = frozenset({"submitted", "ready", "leased", "running"})

# Native MCP task requests are control-plane messages, not a bulk payload
# channel.  Keep these limits aligned with the fleet's general tool-payload
# policy (64 KiB / 4,096 items / depth 24), while applying the tighter 512-byte
# identifier bound that is sufficient for every durable WorkItem namespace.
_MAX_TASK_ID_BYTES = 512
_MAX_TASK_INPUT_BYTES = 64 * 1024
_MAX_TASK_INPUT_ITEMS = 4_096
_MAX_TASK_INPUT_DEPTH = 24
_MAX_TASK_INPUT_STRING_BYTES = 16 * 1024


def _validate_task_input_bounds(value: Any) -> None:
    """Reject oversized, deep, cyclic, or non-JSON task input values."""

    items = 0
    seen: set[int] = set()
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_TASK_INPUT_DEPTH:
            raise ValueError("task input exceeds the maximum nesting depth")
        items += 1
        if items > _MAX_TASK_INPUT_ITEMS:
            raise ValueError("task input exceeds the maximum item count")

        if current is None or isinstance(current, bool):
            pass
        elif isinstance(current, int | float):
            if isinstance(current, float) and not math.isfinite(current):
                raise ValueError("task input contains a non-JSON value")
        elif isinstance(current, str):
            string_bytes = len(current.encode("utf-8"))
            if string_bytes > _MAX_TASK_INPUT_STRING_BYTES:
                raise ValueError("task input string exceeds the size limit")
        elif isinstance(current, dict):
            identity = id(current)
            if identity in seen:
                raise ValueError("task input contains a cycle")
            seen.add(identity)
            if len(current) > _MAX_TASK_INPUT_ITEMS - items:
                raise ValueError("task input exceeds the maximum item count")
            for key, child in current.items():
                if not isinstance(key, str):
                    raise ValueError("task input object keys must be strings")
                stack.append((child, depth + 1))
                stack.append((key, depth + 1))
        elif isinstance(current, list):
            identity = id(current)
            if identity in seen:
                raise ValueError("task input contains a cycle")
            seen.add(identity)
            if len(current) > _MAX_TASK_INPUT_ITEMS - items:
                raise ValueError("task input exceeds the maximum item count")
            stack.extend((child, depth + 1) for child in current)
        else:
            raise ValueError("task input contains a non-JSON value")

    try:
        encoded_bytes = len(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        raise ValueError("task input contains a non-JSON value") from None
    if encoded_bytes > _MAX_TASK_INPUT_BYTES:
        raise ValueError("task input exceeds the size limit")


def _validate_task_id(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("taskId must not be blank")
    if value != value.strip():
        raise ValueError("taskId must be trimmed")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError("taskId contains a control character")
    if len(value.encode("utf-8")) > _MAX_TASK_ID_BYTES:
        raise ValueError("taskId exceeds the size limit")
    return value


class _GetTaskParams(RequestParams):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(alias="taskId", max_length=_MAX_TASK_ID_BYTES)

    @field_validator("task_id")
    @classmethod
    def _bounded_task_id(cls, value: str) -> str:
        return _validate_task_id(value)


_CancelTaskParams = _GetTaskParams


class _UpdateTaskParams(RequestParams):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(alias="taskId", max_length=_MAX_TASK_ID_BYTES)
    input_responses: dict[str, Any] = Field(
        alias="inputResponses", max_length=_MAX_TASK_INPUT_ITEMS
    )

    @field_validator("task_id")
    @classmethod
    def _bounded_task_id(cls, value: str) -> str:
        return _validate_task_id(value)

    @field_validator("input_responses")
    @classmethod
    def _bounded_input_responses(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_task_input_bounds(value)
        return value


class _GetTaskResult(Result):
    """``GetTaskResult`` (SEP-2663): flat task fields plus exactly one of
    ``result``/``error``/``input_requests`` depending on ``status``."""

    model_config = ConfigDict(populate_by_name=True)

    result_type: str = Field(default="complete", serialization_alias="resultType")
    task_id: str = Field(serialization_alias="taskId")
    status: str
    created_at: str = Field(serialization_alias="createdAt")
    last_updated_at: str = Field(serialization_alias="lastUpdatedAt")
    ttl_ms: float | None = Field(default=None, serialization_alias="ttlMs")
    status_message: str | None = Field(
        default=None, serialization_alias="statusMessage"
    )
    poll_interval_ms: float | None = Field(
        default=1_000, serialization_alias="pollIntervalMs"
    )
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    input_requests: dict[str, Any] | None = Field(
        default=None, serialization_alias="inputRequests"
    )


class _AckResult(Result):
    """Acknowledgement shape for ``tasks/update``/``tasks/cancel``."""

    model_config = ConfigDict(populate_by_name=True)

    result_type: str = Field(default="complete", serialization_alias="resultType")
    task_id: str | None = Field(default=None, serialization_alias="taskId")
    status: str | None = None
    status_message: str | None = Field(
        default=None, serialization_alias="statusMessage"
    )
    created_at: str | None = Field(default=None, serialization_alias="createdAt")
    last_updated_at: str | None = Field(
        default=None, serialization_alias="lastUpdatedAt"
    )
    ttl_ms: float | None = Field(default=None, serialization_alias="ttlMs")
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


def _iso_timestamp(value: Any) -> str:
    if isinstance(value, bool):
        raise mcp_protocol_exception(-32603, "Task timestamp was invalid")
    if isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError):
            raise mcp_protocol_exception(-32603, "Task timestamp was invalid") from None
        return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if isinstance(value, datetime):
        parsed = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return (
            parsed.astimezone(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise mcp_protocol_exception(-32603, "Task timestamp was invalid") from None
        parsed = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        return (
            parsed.astimezone(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    raise mcp_protocol_exception(-32603, "Task timestamp was invalid")


class WorkItemTasksExtension(ServerExtension):
    """Backs ``tasks/get``/``tasks/update``/``tasks/cancel`` with native WorkItems.

    The native WorkItem state machine remains the only task authority.  Both
    graph orchestrator IDs and Repository Manager's durable ``rmjob`` /
    ``workitem:repository_manager`` IDs are projections of that one state,
    never a second queue.  Repository IDs are resolved through the
    tenant/owner-scoped adapter so a task poll, input update, or cancellation
    remains safe after a process restart or on a replica.

    ``server_id`` is included in response metadata and is used by the graph-os
    multiplexer to route follow-up task requests to the owning server.  The
    optional ``task_router`` is a narrow request forwarder; it does not own
    persistence, lifecycle, or execution.
    """

    identifier = TASKS_EXTENSION_ID

    def __init__(
        self,
        *,
        server_id: str | None = None,
        task_router: Any | None = None,
    ) -> None:
        self.server_id = (
            str(server_id).strip()
            if isinstance(server_id, str) and server_id.strip()
            else None
        )
        self._task_router = task_router

    def settings(self) -> dict[str, Any]:
        """Advertise the exact projection revision for capability routing."""

        return {"revision": TASKS_EXTENSION_REVISION}

    def set_task_router(self, router: Any | None) -> None:
        """Attach the owning graph-os multiplexer request router.

        FastMCP keeps extension instances on the host server.  The
        multiplexer is attached later, after the fleet catalog is loaded, so
        this setter avoids registering a second Tasks extension or task
        backend.  A router is deliberately duck-typed to keep this module
        usable by direct servers and lightweight tests.
        """

        self._task_router = router

    def methods(self) -> list[MethodBinding]:
        return [
            MethodBinding(
                method="tasks/get",
                params_type=_GetTaskParams,
                handler=self._handle_get,
                protocol_versions=_TASK_METHOD_VERSIONS,
            ),
            MethodBinding(
                method="tasks/update",
                params_type=_UpdateTaskParams,
                handler=self._handle_update,
                protocol_versions=_TASK_METHOD_VERSIONS,
            ),
            MethodBinding(
                method="tasks/cancel",
                params_type=_CancelTaskParams,
                handler=self._handle_cancel,
                protocol_versions=_TASK_METHOD_VERSIONS,
            ),
        ]

    def _require_tasks_capability(self, ctx: ServerRequestContext[Any, Any]) -> None:
        """SEP-2663: reject a task method the client did not opt into this request."""
        if read_client_extension_settings(ctx, TASKS_EXTENSION_ID) is None:
            raise mcp_protocol_exception(
                MISSING_REQUIRED_CLIENT_CAPABILITY,
                (
                    f"This request targets the tasks extension "
                    f"({TASKS_EXTENSION_ID}); the client did not declare it "
                    "for this request."
                ),
                {"requiredCapabilities": {"extensions": {TASKS_EXTENSION_ID: {}}}},
            )

    @staticmethod
    def _repository_task_id(task_id: str) -> bool:
        """Return whether an ID belongs to the RMDD repository namespace."""

        return task_id.startswith(("rmjob:", "workitem:repository_manager:"))

    @staticmethod
    def _actor_id(session: Any) -> str:
        actor = getattr(session, "actor", None)
        actor_id = str(getattr(actor, "actor_id", "") or "").strip()
        if not actor_id:
            raise mcp_protocol_exception(-32001, "Verified task owner is unavailable")
        return actor_id

    @classmethod
    def _caller_metadata(
        cls, session: Any, *, required_scope: str | None = None
    ) -> dict[str, Any]:
        """Return bounded identity metadata for a multiplexer hop.

        The values originate from the verified GraphSession, never request
        fields.  A child server still performs its own local authorization;
        this envelope is for route/audit continuity across a trusted
        multiplexer connection.
        """

        scopes = (
            (required_scope,)
            if required_scope is not None
            else getattr(session, "scopes", ())
        )
        return {
            "tenant": str(getattr(session, "tenant", "") or ""),
            "owner": cls._actor_id(session),
            "scopes": sorted(str(scope) for scope in scopes),
        }

    @staticmethod
    def _route_from_params(params: Any) -> dict[str, Any] | None:
        meta = getattr(params, "meta", None)
        if not isinstance(meta, Mapping):
            return None
        raw = meta.get(TASKS_EXTENSION_ID)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise mcp_protocol_exception(-32602, "Invalid Tasks route metadata")
        server = raw.get("server")
        revision = raw.get("revision")
        if (
            not isinstance(server, str)
            or not server.strip()
            or any(ord(char) < 0x20 for char in server)
        ):
            raise mcp_protocol_exception(-32602, "Invalid Tasks owning-server route")
        if revision != TASKS_EXTENSION_REVISION:
            raise mcp_protocol_exception(
                -32602,
                "Unsupported Tasks extension revision",
                {"expectedRevision": TASKS_EXTENSION_REVISION},
            )
        route = {"server": server.strip(), "revision": revision}
        # Preserve only the route target/revision.  Caller identity is
        # accepted below only as a delegated envelope emitted by the
        # multiplexer; a direct client cannot forge echoed authority metadata.
        caller = raw.get("caller")
        if isinstance(caller, Mapping):
            route["caller"] = dict(caller)
        delegation = raw.get("delegation")
        if delegation is not None:
            if not isinstance(delegation, Mapping):
                raise mcp_protocol_exception(-32001, "Invalid delegated task proof")
            token = delegation.get("token")
            if (
                delegation.get("issuer") != "mcp-multiplexer"
                or not isinstance(token, str)
                or not 32 <= len(token) <= 16_384
                or any(ord(character) < 0x20 for character in token)
            ):
                raise mcp_protocol_exception(-32001, "Invalid delegated task proof")
            route["delegation"] = {"issuer": "mcp-multiplexer", "token": token}
            channel = delegation.get("channel")
            if channel is not None:
                if (
                    not isinstance(channel, str)
                    or len(channel) != 64
                    or any(character not in "0123456789abcdef" for character in channel)
                ):
                    raise mcp_protocol_exception(-32001, "Invalid delegated task proof")
                route["delegation"]["channel"] = channel
        return route

    @staticmethod
    def _authorized_session(ctx: ServerRequestContext[Any, Any], scope: str) -> Any:
        """Resolve verified identity for extension requests (including HTTP).

        ``ActorContextMiddleware`` historically scoped tool calls only.  A
        native extension request is not a ``tools/call``, so use the ambient
        session when present and otherwise mint the same immutable authority
        from FastMCP's already-validated access token.  No token means fail
        closed; this never accepts tenant/owner from task params.
        """

        from agent_utilities.knowledge_graph.core.session import (
            SessionRequiredError,
            resolve_session,
        )

        try:
            return resolve_session(required_scope=scope)
        except SessionRequiredError:
            pass

        try:
            from fastmcp.server.dependencies import get_access_token

            token = get_access_token()
            claims = getattr(token, "claims", None) if token is not None else None
            if not isinstance(claims, Mapping) or not claims:
                raise PermissionError("Verified graph authority is required")
            from agent_utilities.security.request_identity import (
                actor_from_claims,
                mint_graph_session,
            )

            session = mint_graph_session(actor_from_claims(dict(claims)))
            session.require_scope(scope)
            return session
        except PermissionError:
            raise
        except Exception as exc:  # noqa: BLE001 — convert auth failures to a stable protocol error
            logger.warning(
                "tasks_extension: failed to resolve verified request identity (%s)",
                type(exc).__name__,
            )
            raise PermissionError("Verified graph authority is required") from None

    @staticmethod
    def _authorized_delegator(ctx: ServerRequestContext[Any, Any]) -> Any:
        """Resolve the verified service authority for a delegated task hop.

        The HMAC task proof carries the *end user's* bounded identity, not
        the identity of the MCP connection that delivered it.  A valid HMAC
        alone is therefore insufficient: a direct client that learns the
        shared secret must not be able to impersonate the multiplexer.  The
        inbound MCP bearer is verified by FastMCP before this method sees it;
        this method additionally binds issuer, audience, and an automated
        service principal with the explicit fleet-delegation capability.
        """

        # Local stdio children have no HTTP bearer context. Their parent
        # injects a random, per-connection-generation channel secret into the
        # child process environment; the request proof below must present the
        # corresponding MAC. This secret is never accepted from catalog JSON.
        channel_secret = os.environ.get(_TASK_DELEGATION_CHANNEL_ENV, "").strip()
        if 32 <= len(channel_secret) <= 512:
            from types import SimpleNamespace

            return SimpleNamespace(
                authenticated=True,
                issuer="stdio:mcp-multiplexer",
                audience="stdio-child-generation",
                service_principal="mcp-multiplexer",
                scopes=frozenset({"mcp:delegate"}),
                channel_secret=channel_secret,
            )
        try:
            from fastmcp.server.dependencies import get_access_token

            token = get_access_token()
        except Exception:
            token = None
        claims = getattr(token, "claims", None) if token is not None else None
        if not isinstance(claims, Mapping) or not claims:
            raise mcp_protocol_exception(
                -32001,
                "Authenticated multiplexer service authority is required",
            )
        try:
            from agent_utilities.core.config import config, setting
            from agent_utilities.security.identity import (
                base_capabilities,
                normalize_identity,
            )

            identity = normalize_identity(claims)
            expected_issuer = (
                getattr(config, "auth_jwt_issuer", None)
                or getattr(config, "mcp_jwt_issuer", None)
                or setting("FASTMCP_SERVER_AUTH_JWT_ISSUER", None)
                or setting("AUTH_JWT_ISSUER", None)
                or setting("MCP_JWT_ISSUER", None)
            )
            expected_audience = (
                getattr(config, "auth_jwt_audience", None)
                or getattr(config, "mcp_jwt_audience", None)
                or setting("FASTMCP_SERVER_AUTH_JWT_AUDIENCE", None)
                or setting("AUTH_JWT_AUDIENCE", None)
                or setting("MCP_JWT_AUDIENCE", None)
            )
            issuers = {
                value.strip()
                for value in str(expected_issuer or "").split(",")
                if value.strip()
            }
            audiences = {
                value.strip()
                for value in str(expected_audience or "").split(",")
                if value.strip()
            }
            issuer = str(claims.get("iss") or "").strip()
            raw_audience = claims.get("aud")
            if isinstance(raw_audience, str):
                token_audiences = (
                    {raw_audience.strip()} if raw_audience.strip() else set()
                )
            elif isinstance(raw_audience, (list, tuple, set, frozenset)):
                token_audiences = {
                    str(value).strip() for value in raw_audience if str(value).strip()
                }
            else:
                token_audiences = set()
            principal = str(
                getattr(token, "client_id", None)
                or claims.get("client_id")
                or claims.get("azp")
                or ""
            ).strip()
            capabilities = {
                str(scope).strip()
                for scope in (getattr(token, "scopes", None) or ())
                if str(scope).strip()
            }
            capabilities.update(
                str(scope).strip()
                for scope in base_capabilities(
                    identity,
                    getattr(config, "identity_group_capability_map", None),
                )
                if str(scope).strip()
            )
        except Exception:
            raise mcp_protocol_exception(
                -32001,
                "Authenticated multiplexer service authority is required",
            ) from None
        if (
            not issuers
            or issuer not in issuers
            or not audiences
            or not token_audiences.intersection(audiences)
            or not principal
            or not capabilities.intersection(
                {"mcp:delegate", "mcp:admin", "admin", "kg:admin"}
            )
        ):
            raise mcp_protocol_exception(
                -32001,
                "Authenticated multiplexer service authority is required",
            )
        from types import SimpleNamespace

        return SimpleNamespace(
            authenticated=True,
            issuer=issuer,
            audience=next(iter(token_audiences.intersection(audiences))),
            service_principal=principal,
            scopes=frozenset(capabilities),
            channel_secret=None,
        )

    @staticmethod
    def _require_delegator(authority: Any | None) -> None:
        """Defend the private delegated projection against bypass callers."""

        if (
            authority is None
            or not bool(getattr(authority, "authenticated", False))
            or not str(getattr(authority, "issuer", "") or "").strip()
            or not str(getattr(authority, "audience", "") or "").strip()
            or not str(getattr(authority, "service_principal", "") or "").strip()
            or not set(getattr(authority, "scopes", ())).intersection(
                {"mcp:delegate", "mcp:admin", "admin", "kg:admin"}
            )
        ):
            raise mcp_protocol_exception(
                -32001,
                "Authenticated multiplexer service authority is required",
            )

    def _response_meta(
        self, session: Any, route: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        server = self.server_id
        if not server:
            return None
        payload: dict[str, Any] = {
            "server": server,
            "revision": TASKS_EXTENSION_REVISION,
        }
        # Never echo a caller supplied route envelope.  Responses identify
        # the verified/delegated authority of the server that produced them.
        if session is not None:
            payload["caller"] = self._caller_metadata(session)
        return {TASKS_EXTENSION_ID: payload}

    def _validate_delegated_caller(
        self, route: Mapping[str, Any] | None, session: Any
    ) -> None:
        """Reject a route caller that is not the verified local authority.

        A route envelope is transport metadata, not a credential.  Until the
        multiplexer has a signed delegated-token handoff, the safe contract is
        exact equality with the child request's verified GraphSession.  This
        prevents a direct client from selecting its local server route and
        forging another tenant/owner in response metadata or adapter calls.
        """

        if not isinstance(route, Mapping) or "caller" not in route:
            return
        caller = route.get("caller")
        if not isinstance(caller, Mapping):
            raise mcp_protocol_exception(-32001, "Invalid delegated task identity")
        expected = self._caller_metadata(session)
        normalized = {
            "tenant": str(caller.get("tenant") or ""),
            "owner": str(caller.get("owner") or ""),
            "scopes": sorted(str(scope) for scope in caller.get("scopes", ())),
        }
        if normalized != expected:
            raise mcp_protocol_exception(
                -32001, "Delegated task identity does not match verified authority"
            )

    def _delegated_session(
        self,
        route: Mapping[str, Any] | None,
        *,
        method: str,
        params: Any,
        scope: str,
        service_authority: Any | None = None,
    ) -> Any | None:
        """Verify an authenticated multiplexer-on-behalf-of task envelope.

        The multiplexer mints a short-lived run token using the existing
        ``AGENT_UTILITIES_TOKEN_SECRET`` substrate.  The child validates the
        signature and expiry locally, then checks the token's endpoint,
        operation, tenant, owner, and canonical request binding. A direct
        end-user request cannot forge the delegated tenant/owner metadata.
        """

        if not isinstance(route, Mapping) or "delegation" not in route:
            return None
        self._require_delegator(service_authority)
        caller = route.get("caller")
        proof = route.get("delegation")
        if not isinstance(caller, Mapping) or not isinstance(proof, Mapping):
            raise mcp_protocol_exception(-32001, "Invalid delegated task identity")
        token = proof.get("token")
        if not isinstance(token, str) or not token:
            raise mcp_protocol_exception(
                -32001,
                "Authenticated task delegation is unavailable; use portable rm_jobs tools",
            )
        channel = proof.get("channel")
        channel_secret = str(
            getattr(service_authority, "channel_secret", "") or ""
        ).strip()
        if channel_secret:
            if not isinstance(channel, str) or not hmac.compare_digest(
                channel, _channel_proof(channel_secret, token)
            ):
                raise mcp_protocol_exception(-32001, "Invalid delegated task proof")
        elif channel is not None:
            # A channel MAC is meaningful only when the child has the private
            # per-generation secret; never accept it as an unsigned hint on a
            # remote bearer-authenticated connection.
            raise mcp_protocol_exception(-32001, "Invalid delegated task proof")
        try:
            from agent_utilities.security.run_token import validate_token

            decoded = validate_token(
                token,
                endpoint=str(self.server_id or ""),
                operation=method,
            )
        except Exception:
            raise mcp_protocol_exception(
                -32001, "Invalid or expired delegated task proof"
            ) from None
        binding = _delegation_binding(
            method,
            params.model_dump(mode="json", by_alias=True, exclude_none=True),
            server=str(self.server_id or ""),
            revision=TASKS_EXTENSION_REVISION,
            caller=caller,
        )
        expected = f"mcp-task:{binding}"
        if decoded.run_id != expected:
            raise mcp_protocol_exception(
                -32001, "Delegated task request binding is invalid"
            )
        owner = str(caller.get("owner") or "").strip()
        tenant = str(caller.get("tenant") or "").strip()
        scopes = frozenset(str(value) for value in caller.get("scopes", ()))
        if (
            not owner
            or not tenant
            or decoded.actor_id != owner
            or decoded.tenant_id != tenant
            or decoded.project != self.server_id
        ):
            raise mcp_protocol_exception(
                -32001, "Delegated task identity is incomplete"
            )
        accepted = {
            "kg:read": {"kg:read", "kg:write", "kg:admin"},
            "kg:write": {"kg:write", "kg:admin"},
        }.get(scope, {scope})
        if scopes.isdisjoint(accepted):
            raise mcp_protocol_exception(-32001, "Delegated task scope is insufficient")
        from types import SimpleNamespace

        # The delegated session is deliberately narrowed to this request's
        # effective operation.  Carrying the end user's full scope set into
        # the child would make response/audit metadata imply authority that
        # this hop did not need or authorize.
        return SimpleNamespace(
            tenant=tenant,
            scopes=frozenset({scope}),
            actor=SimpleNamespace(actor_id=owner),
        )

    def _authorized_request_session(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: Any,
        *,
        method: str,
        scope: str,
        route: Mapping[str, Any] | None,
    ) -> Any:
        service_authority = (
            self._authorized_delegator(ctx)
            if route is not None and "delegation" in route
            else None
        )
        delegated = self._delegated_session(
            route,
            method=method,
            params=params,
            scope=scope,
            service_authority=service_authority,
        )
        if delegated is not None:
            return delegated
        session = self._authorized_session(ctx, scope)
        self._validate_delegated_caller(route, session)
        return session

    async def _forward(
        self,
        method: str,
        params: Any,
        session: Any,
        route: Mapping[str, Any],
    ) -> Any:
        router = self._task_router
        if router is None or not callable(getattr(router, "forward_task_method", None)):
            raise mcp_protocol_exception(
                -32602,
                "Task belongs to another server and no owning-server route is available",
            )
        return await router.forward_task_method(
            method,
            params.model_dump(mode="json", by_alias=True, exclude_none=True),
            caller=self._caller_metadata(
                session,
                required_scope=("kg:read" if method == "tasks/get" else "kg:write"),
            ),
            route=route,
        )

    @staticmethod
    def _engine() -> Any:
        from agent_utilities.mcp import kg_server

        engine = kg_server._get_engine()
        if engine is None:
            raise mcp_protocol_exception(-32603, "IntelligenceGraphEngine not active.")
        return getattr(engine, "_work_item_engine", engine)

    @staticmethod
    def _run_trace(task_id: str) -> dict[str, Any] | None:
        """Best-effort ``:RunTrace`` lookup for ``task_id`` (D-25-4).

        ``None`` on ANY failure (no active engine, no correlated trace) --
        this is a best-effort enrichment layered on top of the WorkItem
        outcome that already answers a completed task authoritatively; a
        RunTrace lookup hiccup must never turn an otherwise-successful
        ``tasks/get`` read into an error.
        """
        try:
            from agent_utilities.mcp import kg_server
            from agent_utilities.orchestration.manager import Orchestrator

            engine = kg_server._get_engine()
            if engine is None:
                return None
            trace = Orchestrator(engine).get_run_trace(task_id)
        except Exception as exc:  # noqa: BLE001 — best-effort enrichment, see docstring
            logger.warning(
                "tasks_extension: RunTrace lookup for task %s failed: %s",
                task_id,
                exc,
            )
            return None
        # `get_run_trace`'s found-path always stamps `trace_id` -- the
        # unambiguous "this really is a RunTrace row" signal (its `status`
        # field is otherwise indistinguishable from a legitimate run outcome
        # value like "succeeded"/"failed").
        if "trace_id" not in trace:
            return None
        return trace

    async def _handle_get(
        self, ctx: ServerRequestContext[Any, Any], params: _GetTaskParams
    ) -> _GetTaskResult:
        self._require_tasks_capability(ctx)
        route = self._route_from_params(params)
        session = self._authorized_request_session(
            ctx, params, method="tasks/get", scope="kg:read", route=route
        )
        if route is not None and route["server"] != self.server_id:
            return await self._forward("tasks/get", params, session, route)
        return self._project(params.task_id, session=session, route=route)

    async def _handle_cancel(
        self, ctx: ServerRequestContext[Any, Any], params: _GetTaskParams
    ) -> _AckResult:
        self._require_tasks_capability(ctx)
        route = self._route_from_params(params)
        session = self._authorized_request_session(
            ctx, params, method="tasks/cancel", scope="kg:write", route=route
        )
        if route is not None and route["server"] != self.server_id:
            return await self._forward("tasks/cancel", params, session, route)

        if self._repository_task_id(params.task_id):
            return self._cancel_repository(params.task_id, session=session, route=route)

        from agent_utilities.orchestration import work_item as _wi

        item_id = _wi.orchestrator_work_item_id(params.task_id)
        item = _wi.get_work_item(self._engine(), item_id)
        if item is None or item.get("tenant") != session.tenant:
            raise mcp_protocol_exception(-32602, "Unknown task")
        if not _wi.cancel_work_item(self._engine(), item_id):
            raise mcp_protocol_exception(-32602, "Failed to cancel task")
        current = _wi.get_work_item(self._engine(), item_id)
        if current is None:
            raise mcp_protocol_exception(-32602, "Task disappeared after cancellation")
        return self._ack_from_generic_item(
            params.task_id, current, session=session, route=route
        )

    async def _handle_update(
        self, ctx: ServerRequestContext[Any, Any], params: _UpdateTaskParams
    ) -> _AckResult:
        self._require_tasks_capability(ctx)
        route = self._route_from_params(params)
        session = self._authorized_request_session(
            ctx, params, method="tasks/update", scope="kg:write", route=route
        )
        if route is not None and route["server"] != self.server_id:
            return await self._forward("tasks/update", params, session, route)

        if self._repository_task_id(params.task_id):
            return self._update_repository(
                params.task_id,
                params.input_responses,
                session=session,
                route=route,
            )

        from agent_utilities.orchestration import work_item as _wi

        item_id = _wi.orchestrator_work_item_id(params.task_id)
        item = _wi.get_work_item(self._engine(), item_id)
        if item is None or item.get("tenant") != session.tenant:
            raise mcp_protocol_exception(-32602, "Unknown task")
        if not _wi.submit_work_item_input(
            self._engine(),
            item_id,
            tenant=session.tenant,
            response=params.input_responses,
        ):
            raise mcp_protocol_exception(-32602, "Failed to submit task input")
        current = _wi.get_work_item(self._engine(), item_id)
        if current is None:
            raise mcp_protocol_exception(-32602, "Task disappeared after input update")
        return self._ack_from_generic_item(
            params.task_id, current, session=session, route=route
        )

    def _project(
        self,
        task_id: str,
        *,
        session: Any | None = None,
        route: Mapping[str, Any] | None = None,
    ) -> _GetTaskResult:
        if self._repository_task_id(task_id):
            if session is None:
                raise mcp_protocol_exception(
                    -32001, "Verified task owner is required for repository tasks"
                )
            return self._project_repository(task_id, session=session, route=route)

        from agent_utilities.orchestration import work_item as _wi

        item_id = _wi.orchestrator_work_item_id(task_id)
        item = _wi.get_work_item(self._engine(), item_id)
        if item is None:
            raise mcp_protocol_exception(-32602, "Unknown task")
        if session is not None and item.get("tenant") != session.tenant:
            raise mcp_protocol_exception(-32602, "Unknown task")
        raw_status = str(item.get("status") or "").lower()
        metadata = item.get("metadata")
        pending = (
            metadata.get("pending_input_request")
            if isinstance(metadata, dict)
            else None
        )
        if raw_status in _WORKING_RAW_STATUSES and isinstance(pending, dict):
            status = "input_required"
        elif raw_status in _WORKING_RAW_STATUSES:
            status = "working"
        elif raw_status == "succeeded":
            status = "completed"
        elif raw_status in {"failed", "dead_letter"}:
            status = "failed"
        elif raw_status == "cancelled":
            status = "cancelled"
        else:
            raise mcp_protocol_exception(-32603, "Unknown WorkItem status")
        result = _GetTaskResult(
            task_id=task_id,
            status=status,
            created_at=_iso_timestamp(item.get("created_at")),
            last_updated_at=_iso_timestamp(item.get("updated_at")),
            # Native WorkItems have no automatic record-expiration policy: they
            # are durable graph records (mirrors the gateway's own ttlMs: null).
            ttl_ms=None,
            meta=self._response_meta(session, route),
        )
        if status == "input_required":
            result.input_requests = {"request": pending}
        elif status == "completed":
            # D-25-4: `result_ref` is only ever an opaque completion marker
            # (e.g. "orchestrator:<job>:completed") -- never the real agent
            # output. `_execute_orchestrator_turn` pins the run's :RunTrace to
            # THIS SAME task_id (`run_id=envelope.job_id`), so the real output
            # is one more read away via the SAME `Orchestrator.get_run_trace`
            # `graph_jobs(action="status", job_id="trace:...")` already uses --
            # this module has direct engine access (unlike the isolated
            # gateway sidecar), so it calls it in-process rather than proxying
            # through another tool call.
            trace = self._run_trace(task_id)
            ref = item.get("result_ref")
            if trace is not None and trace.get("result_preview"):
                result.result = {
                    "resultPreview": trace["result_preview"],
                    "runId": trace.get("run_id") or task_id,
                }
            elif trace is not None and trace.get("error"):
                result.result = {
                    "error": trace["error"],
                    "runId": trace.get("run_id") or task_id,
                }
            elif ref is not None:
                result.result = {"resultRef": ref}
            else:
                result.result = {"status": "completed"}
        elif status == "failed":
            result.error = {"code": -32603, "message": "GraphOS WorkItem failed"}
        return result

    def _repository_view(self, task_id: str, session: Any) -> Any:
        """Load one repository view under the verified tenant and owner."""

        from agent_utilities.orchestration.repository_work_item import (
            get_repository_work_item,
        )

        try:
            view = get_repository_work_item(
                self._engine(),
                task_id,
                tenant=session.tenant,
                owner_id=self._actor_id(session),
            )
        except (TypeError, ValueError, PermissionError):
            view = None
        if view is None:
            # Deliberately collapse unknown, wrong-tenant, wrong-owner, and
            # malformed namespace IDs into one response so task existence is
            # not an oracle across tenants or actors.
            raise mcp_protocol_exception(-32602, "Unknown task")
        return view

    @staticmethod
    def _repository_raw_item(engine: Any, view: Any) -> Mapping[str, Any]:
        from agent_utilities.orchestration import work_item as _wi

        item = _wi.get_work_item(engine, view.work_item_id)
        if item is None:
            raise mcp_protocol_exception(
                -32603, "Repository WorkItem state is unavailable"
            )
        if (
            not isinstance(item, Mapping)
            or item.get("id") != view.work_item_id
            or not isinstance(item.get("status"), str)
            or not item.get("status", "").strip()
            or "created_at" not in item
            or "updated_at" not in item
            or not isinstance(item.get("metadata"), Mapping)
        ):
            raise mcp_protocol_exception(-32603, "Repository WorkItem state is corrupt")
        return item

    @staticmethod
    def _repository_status(raw_state: Any) -> str:
        state = str(raw_state or "").lower()
        if state in _WORKING_RAW_STATUSES:
            return "working"
        if state == "succeeded":
            return "completed"
        if state in {"failed", "dead-letter", "dead_letter"}:
            return "failed"
        if state == "cancelled":
            return "cancelled"
        raise mcp_protocol_exception(-32603, "Unknown repository WorkItem status")

    @staticmethod
    def _repository_domain_payload(view: Any) -> dict[str, Any]:
        from agent_utilities.orchestration.repository_work_item import (
            repository_result_from_view,
        )

        # The adapter is the authority for parsing domain reason/refusal
        # fields from the opaque error reference.  Keep the payload bounded to
        # that typed result, which contains only immutable correlations and
        # opaque artifact/result references (never command bodies or logs).
        payload = repository_result_from_view(view).model_dump(
            mode="json", exclude_none=True
        )
        # Keep the typed domain projection intact while exposing the stable
        # camelCase names MCP clients expect for the correlations most often
        # consumed by a result renderer.
        aliases = {
            "job_id": "jobId",
            "work_item_id": "workItemId",
            "request_id": "requestId",
            "repository_id": "repositoryId",
            "tenant_id": "tenantId",
            "result_ref": "resultRef",
            "error_ref": "errorRef",
            "failure_class": "failureClass",
            "refusal_code": "refusalCode",
        }
        for source, alias in aliases.items():
            if source in payload:
                payload[alias] = payload[source]
        return payload

    def _project_repository(
        self,
        task_id: str,
        *,
        session: Any,
        route: Mapping[str, Any] | None,
    ) -> _GetTaskResult:
        view = self._repository_view(task_id, session)
        raw = self._repository_raw_item(self._engine(), view)
        status = self._repository_status(view.state)
        pending = raw.get("metadata") if isinstance(raw, Mapping) else None
        pending_request = (
            pending.get("pending_input_request")
            if isinstance(pending, Mapping)
            else None
        )
        if status == "working" and isinstance(pending_request, Mapping):
            status = "input_required"
        created_at = raw.get("created_at") if isinstance(raw, Mapping) else None
        updated_at = raw.get("updated_at") if isinstance(raw, Mapping) else None
        result = _GetTaskResult(
            task_id=task_id,
            status=status,
            created_at=_iso_timestamp(created_at),
            last_updated_at=_iso_timestamp(updated_at),
            ttl_ms=None,
            meta=self._response_meta(session, route),
        )
        if status == "input_required":
            result.status_message = "Repository WorkItem is waiting for input"
            result.input_requests = {"request": dict(pending_request)}
        elif status == "completed":
            result.result = self._repository_domain_payload(view)
        elif status == "failed":
            domain = self._repository_domain_payload(view)
            result.error = {
                "code": -32603,
                "message": "Repository WorkItem failed",
                "failureClass": domain.get("failure_class"),
                "refusalCode": domain.get("refusal_code"),
                "errorRef": domain.get("error_ref"),
                "result": domain,
            }
        return result

    def _ack_from_view(
        self,
        task_id: str,
        view: Any,
        *,
        session: Any,
        route: Mapping[str, Any] | None,
        status_message: str | None = None,
    ) -> _AckResult:
        raw = self._repository_raw_item(self._engine(), view)
        status = self._repository_status(view.state)
        pending = raw.get("metadata") if isinstance(raw, Mapping) else None
        if (
            status == "working"
            and isinstance(pending, Mapping)
            and isinstance(pending.get("pending_input_request"), Mapping)
        ):
            status = "input_required"
        return _AckResult(
            task_id=task_id,
            status=status,
            status_message=status_message,
            created_at=_iso_timestamp(raw["created_at"]),
            last_updated_at=_iso_timestamp(raw["updated_at"]),
            ttl_ms=None,
            meta=self._response_meta(session, route),
        )

    def _cancel_repository(
        self,
        task_id: str,
        *,
        session: Any,
        route: Mapping[str, Any] | None,
    ) -> _AckResult:
        from agent_utilities.orchestration.repository_work_item import (
            cancel_repository_work_item,
        )

        # Read through the owner filter before invoking the adapter mutation.
        # The second read below makes a worker-vs-cancel race truthful: the
        # response reports the durable winner, rather than claiming cancelled
        # merely because our CAS was attempted.
        self._repository_view(task_id, session)
        cancelled = cancel_repository_work_item(
            self._engine(), task_id, tenant=session.tenant
        )
        after = self._repository_view(task_id, session)
        if after is None:  # pragma: no cover - defensive; _repository_view raises
            raise mcp_protocol_exception(-32602, "Unknown task")
        if not cancelled:
            state = self._repository_status(after.state)
            if state in {"completed", "failed", "cancelled"}:
                return self._ack_from_view(
                    task_id,
                    after,
                    session=session,
                    route=route,
                    status_message=(
                        "Cancellation lost a race with the durable terminal outcome"
                    ),
                )
            raise mcp_protocol_exception(-32602, "Failed to cancel task")
        if str(after.state) != "cancelled":
            return self._ack_from_view(
                task_id,
                after,
                session=session,
                route=route,
                status_message=(
                    "Cancellation raced with another durable WorkItem transition"
                ),
            )
        return self._ack_from_view(task_id, after, session=session, route=route)

    def _update_repository(
        self,
        task_id: str,
        input_responses: dict[str, Any],
        *,
        session: Any,
        route: Mapping[str, Any] | None,
    ) -> _AckResult:
        from agent_utilities.orchestration import work_item as _wi

        view = self._repository_view(task_id, session)
        updated = _wi.submit_work_item_input(
            self._engine(),
            view.work_item_id,
            tenant=session.tenant,
            response=input_responses,
        )
        after = self._repository_view(task_id, session)
        if not updated:
            state = self._repository_status(after.state)
            if state in {"completed", "failed", "cancelled"}:
                return self._ack_from_view(
                    task_id,
                    after,
                    session=session,
                    route=route,
                    status_message="Input update lost a race with the durable outcome",
                )
            raise mcp_protocol_exception(-32602, "Failed to submit task input")
        return self._ack_from_view(task_id, after, session=session, route=route)

    def _ack_from_generic_item(
        self,
        task_id: str,
        item: Mapping[str, Any],
        *,
        session: Any,
        route: Mapping[str, Any] | None,
    ) -> _AckResult:
        raw_status = str(item.get("status") or "").lower()
        if raw_status in _WORKING_RAW_STATUSES:
            status = "working"
        elif raw_status == "succeeded":
            status = "completed"
        elif raw_status in {"failed", "dead_letter"}:
            status = "failed"
        elif raw_status == "cancelled":
            status = "cancelled"
        else:
            raise mcp_protocol_exception(-32603, "Unknown WorkItem status")
        return _AckResult(
            task_id=task_id,
            status=status,
            created_at=_iso_timestamp(item.get("created_at")),
            last_updated_at=_iso_timestamp(item.get("updated_at")),
            ttl_ms=None,
            meta=self._response_meta(session, route),
        )
