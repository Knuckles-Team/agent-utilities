"""Verbose 1:1 MCP tool surface — one named tool per API-client method.

Fleet MCP servers expose a **condensed** action-routed surface: one tool per
domain (``servicenow_cmdb``) taking ``action`` + ``params_json`` and dispatching
to many client methods. That keeps the tool count small, but the model must
guess the right ``action`` string and the ``params_json`` shape.

This module provides the **verbose** alternative: one ``@mcp.tool`` per public
``api_client`` method — individually named (``servicenow_get_cmdb_instance``) and
carrying the method's own docstring as its description — so an LLM selects the
exact operation directly. It is the same client methods exposed one-to-one
instead of folded behind an ``action`` switch; nothing is reimplemented.

Selection is governed by one knob, :func:`tool_mode`
(``MCP_TOOL_MODE`` ∈ ``condensed`` | ``verbose`` | ``both``, default
``condensed``). Every verbose tool is tagged ``"verbose"`` (plus its domain) so
the existing ``DynamicVisibilityTransform`` (``--tools`` / tags / HTTP query)
can still slice the set per request.

Generation is by **runtime introspection of the client class** (no credentials
needed at build time — the live call binds the real client via ``Depends``).
Hand-written clients are ``**kwargs``, so each verbose tool takes a single
``params_json`` argument. Codegen'd connectors (e.g. onetrust) emit their own
fully-typed verbose modules from their OpenAPI manifest; this introspection path
is the universal fallback for everyone else.

CONCEPT:AU-ECO.mcp.tool-mode-standardization — MCP tool-mode standardization (verbose 1:1 surface)
"""

from __future__ import annotations

import inspect
import json
import keyword
import logging
import re
from typing import Any

from fastmcp import Context
from fastmcp.dependencies import Depends
from pydantic import Field

from agent_utilities.mcp.action_dispatch import DISCOVERY_ACTIONS, public_actions
from agent_utilities.mcp.concurrency import run_blocking
from agent_utilities.mcp.context_helpers import ctx_confirm_destructive

logger = logging.getLogger(__name__)

VALID_TOOL_MODES = ("condensed", "verbose", "both", "intent")
_DEFAULT_MODE = "condensed"

#: Tag stamped on EVERY condensed/verbose tool (any mode) so a deployment can
#: statically shrink the LLM-visible surface via the pre-existing
#: ``MCP_DISABLED_TAGS``/``DynamicVisibilityTransform`` knob (no new env var —
#: CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse) without touching REGISTERED_TOOLS/REST.
GRANULAR_TAG = "granular"

#: Tag stamped ONLY when ``MCP_TOOL_MODE=intent`` (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse):
#: marks a tool as held back from the DEFAULT session view. It stays fully
#: registered (REST + ``_execute_tool`` + REGISTERED_TOOLS unaffected) — the
#: fleet ``load_tools`` meta-tool reveals it per-session (see
#: :func:`agent_utilities.mcp.multiplexer.attach_fleet_loader`).
GATED_TAG = "gated"

#: OpenAPI/JSON-schema primitive -> Python type for synthesized typed signatures.
_PY_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

#: Method-name prefixes that mark a write that destroys data (used when a
#: manifest doesn't carry explicit ``http``/``destructive`` metadata).
_DESTRUCTIVE_PREFIXES = (
    "delete_",
    "remove_",
    "destroy_",
    "purge_",
    "drop_",
    "deactivate_",
)


#: Reserved parameter names a synthesized verbose signature already uses.
_RESERVED_PARAM_NAMES = frozenset({"client", "ctx", "self"})


def _provider_tools(mcp: Any) -> dict[str, Any]:
    """``{tool_name: tool_obj}`` from the FastMCP local provider (sync, best-effort).

    Used to diff which tools a ``register_<domain>_tools`` call added, so the central
    wiring can stamp the canonical domain tag and record the exact toggle. Returns
    ``{}`` if the provider internals aren't available (older/newer FastMCP) — callers
    then fall back to tag-based derivation.
    """
    provider = getattr(mcp, "_local_provider", None)
    components = getattr(provider, "_components", None)
    if not isinstance(components, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in components.items():
        if str(key).startswith("tool:") and getattr(value, "name", None):
            out[value.name] = value
    return out


def _is_typeable_param(param: dict) -> bool:
    """Whether a manifest param can become a typed function argument.

    A param name must be a real Python identifier, not a keyword, and not collide
    with the injected ``client``/``ctx``. Specs sometimes carry field names with
    colons/dots (e.g. SCIM ``urn:...`` keys) that fail this — such operations use
    the params_json fallback instead.
    """
    name = param.get("name")
    return (
        isinstance(name, str)
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and name not in _RESERVED_PARAM_NAMES
    )


def _is_destructive(method_name: str, op: dict | None) -> bool:
    """Whether an operation destroys data — gates a Context elicitation guard.

    Prefers explicit manifest metadata (``destructive`` flag, then an HTTP
    ``DELETE``); otherwise falls back to the method-name prefix.
    """
    if op:
        if op.get("destructive") is not None:
            return bool(op["destructive"])
        if str(op.get("http", "")).upper() == "DELETE":
            return True
    return method_name.startswith(_DESTRUCTIVE_PREFIXES)


def tool_mode() -> str:
    """Return the configured MCP tool surface: ``condensed``|``verbose``|``both``|``intent``.

    Reads ``MCP_TOOL_MODE`` through the shared config layer (so it is driven by
    the one XDG ``config.json``). Defaults to ``condensed`` — the small,
    action-routed surface — so existing deployments are unchanged. An
    unrecognized value falls back to ``condensed`` with a warning.

    ``intent`` (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse — Seam 8) is the
    "small/cheap-LLM" profile: the condensed action-routed tools still register
    (REST + ``_execute_tool`` + REGISTERED_TOOLS are unaffected — nothing is
    lost), but they are additionally tagged :data:`GATED_TAG` and held back from
    a session's default tool list; a handful of thin ``ask``/``find``/``write``/
    ``act``/``manage``/``why`` intent-verb tools (``mcp/tools/intent_tools.py``)
    become the default surface instead, resolving a natural-language intent to
    the right granular tool and dispatching through the SAME ``_execute_tool``
    core. ``load_tools`` (the fleet meta-tool) remains the escape hatch to any
    exact granular tool, gated or not.
    """
    # Imported lazily: config pulls in the whole settings stack, and tool_mode is
    # also called from module-load paths in the agents.
    from agent_utilities.core.config import setting

    mode = str(setting("MCP_TOOL_MODE", _DEFAULT_MODE)).strip().lower()
    if mode not in VALID_TOOL_MODES:
        logger.warning(
            "Unknown MCP_TOOL_MODE %r; expected one of %s. Falling back to %r.",
            mode,
            ", ".join(VALID_TOOL_MODES),
            _DEFAULT_MODE,
        )
        return _DEFAULT_MODE
    return mode


def _camel_to_snake(name: str) -> str:
    """``ChangeManagement`` -> ``change_management``."""
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def _tool_prefix(service: str) -> str:
    """Derive a tool-name prefix from a service name (``servicenow-api`` -> ``servicenow``)."""
    prefix = service.strip().lower()
    for suffix in ("-api", "-mcp", "-agent"):
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]
            break
    return re.sub(r"[^0-9a-z]+", "_", prefix).strip("_") or "tool"


def _defining_class(client_cls: type, name: str) -> type | None:
    """The class in ``client_cls``'s MRO that actually defines attribute ``name``."""
    for klass in client_cls.__mro__:
        if name in klass.__dict__:
            return klass
    return None


def _domain_methods(client_cls: type) -> dict[str, type]:
    """Public domain methods of ``client_cls`` mapped to their defining class.

    Excludes private names, anything inherited from ``object``, and base-infra
    methods (auth / pagination / retry live on a ``*Base`` class, never an API
    operation). The remaining methods are the real per-domain API operations.
    """
    methods: dict[str, type] = {}
    for name in dir(client_cls):
        if name.startswith("_"):
            continue
        attr = getattr(client_cls, name, None)
        if not callable(attr):
            continue
        owner = _defining_class(client_cls, name)
        if owner is None or owner is object:
            continue
        if owner.__name__.endswith("Base"):
            continue
        methods[name] = owner
    return methods


def _derive_domains(owners: dict[str, type]) -> dict[str, str]:
    """Map each method to a snake_case domain tag from its defining class name.

    Strips the longest common prefix shared by the domain classes (the service
    prefix, e.g. ``ServiceNowApi``) so ``ServiceNowApiCmdb`` -> ``cmdb`` and
    ``OneTrustAuditManagement`` -> ``audit_management``.
    """
    class_names = sorted({owner.__name__ for owner in owners.values()})
    common = ""
    if len(class_names) > 1:
        import os.path

        common = os.path.commonprefix(class_names)
        # Back off to a CamelCase boundary so the shared service prefix
        # (``ServiceNowApi``) is stripped without cutting a domain token in half
        # (``ServiceNowApiC`` -> ``ServiceNowApi``, not a stray ``mdb``).
        ref = class_names[0]
        k = len(common)
        while k > 0 and not ref[k : k + 1].isupper():
            k -= 1
        common = ref[:k]
    domains: dict[str, str] = {}
    for method, owner in owners.items():
        remainder = owner.__name__[len(common) :] if common else owner.__name__
        domain = _camel_to_snake(remainder) or _camel_to_snake(owner.__name__)
        domains[method] = domain
    return domains


def _build_params_json_tool(method_name: str, get_client: Any, *, destructive: bool):
    """A verbose tool that takes a single ``params_json`` blob (no typed params).

    Used for hand-written ``**kwargs`` clients where per-parameter metadata is not
    available — the universal introspection fallback. A live ``Context`` (injected
    only on served requests; ``None`` headless) gates destructive operations
    through an elicitation prompt.
    """

    async def _tool(
        params_json: str = Field(
            default="{}",
            description="JSON object of arguments for this operation.",
        ),
        client=Depends(get_client),
        ctx: Context | None = None,
    ) -> Any:
        kwargs = json.loads(params_json) if params_json else {}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if destructive and not await ctx_confirm_destructive(ctx, method_name):
            return {"cancelled": True, "operation": method_name}
        return await run_blocking(getattr(client, method_name), **kwargs)

    return _tool


def _build_typed_tool(
    method_name: str, params: list[dict], get_client: Any, *, destructive: bool
):
    """A verbose tool with a synthesized, fully-typed signature from a manifest.

    Each manifest param becomes a typed keyword argument with its description;
    required params are required in the schema, optional ones default to ``None``.
    All params dispatch by name to ``client.<method>(**kwargs)`` (the fleet clients
    route path/query/body internally), so only name/type/required/description are
    needed here. A live ``Context`` (hidden from the schema, ``None`` headless)
    gates destructive operations.
    """

    async def _tool(**call: Any) -> Any:
        client = call.pop("client")
        ctx = call.pop("ctx", None)
        kwargs = {k: v for k, v in call.items() if v is not None}
        if destructive and not await ctx_confirm_destructive(ctx, method_name):
            return {"cancelled": True, "operation": method_name}
        return await run_blocking(getattr(client, method_name), **kwargs)

    sig_params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for p in params:
        py_type = _PY_TYPES.get(str(p.get("type", "string")).lower(), str)
        description = p.get("description") or ""
        if p.get("required"):
            default = Field(description=description)
            annotation: Any = py_type
        else:
            default = Field(default=None, description=description)
            annotation = py_type | None
        sig_params.append(
            inspect.Parameter(
                p["name"],
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )
        annotations[p["name"]] = annotation
    sig_params.append(
        inspect.Parameter(
            "client", inspect.Parameter.KEYWORD_ONLY, default=Depends(get_client)
        )
    )
    # Context is injected on served requests, hidden from the input schema, and
    # None headless — declared with a plain ``None`` default (fastmcp 3.x form).
    sig_params.append(
        inspect.Parameter(
            "ctx",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=Context | None,
        )
    )
    _tool.__signature__ = inspect.Signature(sig_params)  # type: ignore[attr-defined]
    _tool.__annotations__ = {
        **annotations,
        "client": Any,
        "ctx": Context | None,
        "return": Any,
    }
    return _tool


def register_verbose_tools(
    mcp: Any,
    client_cls: type,
    get_client: Any,
    *,
    service: str,
    tool_prefix: str | None = None,
    domain_map: dict[str, str] | None = None,
    manifest: list[dict] | None = None,
) -> list[str]:
    """Register one verbose MCP tool per public domain method of ``client_cls``.

    Each tool is named ``<prefix>_<method>``, tagged ``{"verbose", <domain>}`` and
    dispatches to ``getattr(client, method)(**params)`` off the event loop via
    :func:`run_blocking`. The live client is bound per-call through
    ``Depends(get_client)`` — the class is introspected only for names, docs, and
    domain grouping, so no credentials are needed at registration time.

    **Fidelity is hybrid, decided per method:**

    - When ``manifest`` carries a normalized operation for the method *with*
      ``params``, a **fully-typed** tool is synthesized (one typed argument per
      parameter, with descriptions) — the rich tier sourced from an API spec
      (OpenAPI/Swagger/crawled-docs/PDF, normalized to the manifest).
    - Otherwise the method gets a **``params_json``** tool (the universal
      introspection fallback for ``**kwargs`` clients).

    Args:
        mcp: the FastMCP server to register on.
        client_cls: the API client class (e.g. ``servicenow_api.api_client.Api``).
        get_client: the FastMCP dependency that returns a live client instance.
        service: the service name, used to derive the tool-name prefix.
        tool_prefix: override the derived prefix.
        domain_map: optional ``{method: domain}`` override for tags.
        manifest: optional list of normalized operations
            ``{"method", "domain"?, "summary"?/"description"?, "params": [...]}``
            (e.g. ``<pkg>/api/_operation_manifest.py``'s ``OPERATIONS``). Drives the
            typed tier. Operations whose method is absent on the client are skipped.

    Returns:
        The list of registered tool names.
    """
    prefix = tool_prefix or _tool_prefix(service)
    owners = _domain_methods(client_cls)
    manifest_by_method = {
        op["method"]: op for op in (manifest or []) if op.get("method")
    }
    if not owners and not manifest_by_method:
        logger.warning(
            "register_verbose_tools: no domain methods found on %s for service %r",
            client_cls.__name__,
            service,
        )
        return []
    domains = _derive_domains(owners)
    registered: list[str] = []

    method_names = sorted(set(owners) | set(manifest_by_method))
    for method_name in method_names:
        op = manifest_by_method.get(method_name)
        # A manifest op for a method the client doesn't actually expose can't be
        # dispatched — skip it rather than register a tool that errors on call.
        if method_name not in owners and not hasattr(client_cls, method_name):
            logger.warning(
                "register_verbose_tools: manifest method %r not found on %s; skipping",
                method_name,
                client_cls.__name__,
            )
            continue

        domain = (
            (domain_map or {}).get(method_name)
            or (op or {}).get("domain")
            or domains.get(method_name)
            or "api"
        )
        # Avoid a doubled prefix when the client method is already named with the
        # service prefix (e.g. service ``postiz`` + method ``postiz_create_post``
        # would otherwise yield ``postiz_postiz_create_post``).
        if method_name == prefix or method_name.startswith(f"{prefix}_"):
            tool_name = method_name
        else:
            tool_name = f"{prefix}_{method_name}"
        params = (op or {}).get("params") or []
        destructive = _is_destructive(method_name, op)

        # Typed tier only when every param name is a usable Python identifier — some
        # specs carry body fields like ``urn:ietf:...:Group`` (SCIM) that cannot be a
        # function parameter; those operations fall back to the params_json tool so
        # the client still receives every field.
        if params and all(_is_typeable_param(p) for p in params):
            tool_fn = _build_typed_tool(
                method_name, params, get_client, destructive=destructive
            )
            doc = (
                (op or {}).get("summary") or (op or {}).get("description") or ""
            ).strip()
        else:
            tool_fn = _build_params_json_tool(
                method_name, get_client, destructive=destructive
            )
            method = getattr(client_cls, method_name, None)
            doc = (
                (op or {}).get("summary")
                or (op or {}).get("description")
                or (getattr(method, "__doc__", None) or "")
            ).strip()

        tool_fn.__name__ = tool_name
        tool_fn.__doc__ = doc or f"Invoke the {method_name} operation."
        mcp.tool(name=tool_name, tags={"verbose", domain, GRANULAR_TAG})(tool_fn)
        registered.append(tool_name)

    logger.debug(
        "Registered %d verbose tools for %s (prefix=%s, %d typed from manifest)",
        len(registered),
        service,
        prefix,
        sum(1 for m in method_names if (manifest_by_method.get(m) or {}).get("params")),
    )
    return registered


#: Shared ``register_*_tools``-shaped helpers excluded from module auto-discovery.
_SURFACE_HELPER_NAMES = frozenset({"register_verbose_tools", "register_tool_surface"})

#: The action-routing argument every condensed fleet tool exposes (its enum is the
#: list of operations the tool dispatches to). See ``mcp/action_dispatch.py``.
_ACTION_ARG = "action"

#: Attribute on the FastMCP server holding ``{tool_name: actions_provider}`` for
#: condensed tools whose ``action`` is a free-form ``str`` (populated at runtime via
#: ``public_actions(client)`` rather than a static ``Literal``). The provider is the
#: action list, a zero-arg callable returning it, or a client class to introspect.
_ACTION_PROVIDERS_ATTR = "_verbose_action_providers"

#: Discovery action names that are not real operations (they request the action
#: list) — never expanded into a verbose 1:1 tool.
_NON_OPERATION_ACTIONS = frozenset(DISCOVERY_ACTIONS)


def register_action_provider(mcp: Any, tool_name: str, actions: Any) -> None:
    """Register the dynamic action set of a free-form condensed tool for autowire.

    Action-routed connectors (atlassian, arr, …) declare ``action: str`` and obtain
    the valid action names at *runtime* from ``public_actions(client)`` — so the
    static-enum reader :func:`_action_enum` returns ``[]`` for them and
    :func:`autowire_verbose_from_condensed` would skip them. A connector (or the
    central surface) records its tool's action surface here so the auto-wire can
    still derive one verbose 1:1 tool per action without credentials.

    ``actions`` may be:

    - a concrete ``list[str]`` / iterable of action names;
    - a **zero-arg callable** returning that list (resolved at autowire time); or
    - a **client class** (``type``) — its public, callable, non-underscore methods
      are enumerated via :func:`public_actions` (credential-free: ``dir()`` on the
      *class* needs no live instance).

    Idempotent per ``(mcp, tool_name)`` — last write wins.

    CONCEPT:AU-ECO.mcp.verbose-auto-wire — verbose auto-wire enumerates dynamic (runtime) actions
    """
    providers: dict[str, Any] = getattr(mcp, _ACTION_PROVIDERS_ATTR, {})
    providers[tool_name] = actions
    setattr(mcp, _ACTION_PROVIDERS_ATTR, providers)


def _resolve_action_provider(provider: Any) -> list[str]:
    """Resolve a registered action provider to a sorted list of action names.

    Accepts the three forms documented on :func:`register_action_provider`. A
    client *class* is introspected with :func:`public_actions` (operating on the
    class object, so no live instance / credentials are needed). Anything that
    raises or yields no usable names resolves to ``[]`` (the tool is then skipped,
    keeping the auto-wire best-effort).
    """
    try:
        if isinstance(provider, type):
            # public_actions(dir()) works equally on a class or an instance.
            candidates: Any = public_actions(provider)
        elif callable(provider):
            candidates = provider()
        else:
            candidates = provider
        names = [str(a) for a in (candidates or []) if isinstance(a, str) and a]
    except Exception as exc:  # pragma: no cover - defensive per-provider
        logger.warning("verbose autowire: action provider failed: %s", exc)
        return []
    # Drop discovery keywords (list_actions/help/actions) and de-dup, stable-sorted.
    return sorted({n for n in names if n not in _NON_OPERATION_ACTIONS})


def _tool_action_names(tool: Any, providers: dict[str, Any]) -> list[str]:
    """Action names for a condensed tool: static enum first, else a dynamic provider.

    Returns the ``Literal`` enum when the tool declares one (the static path), else
    the resolved dynamic-action list from a registered provider (the runtime path),
    else ``[]`` (no ``action`` surface — skipped by the auto-wire).
    """
    enum = _action_enum(tool)
    if enum:
        return enum
    tool_name = getattr(tool, "name", None)
    provider = providers.get(tool_name) if isinstance(tool_name, str) else None
    if provider is not None:
        return _resolve_action_provider(provider)
    return []


def _action_enum(tool: Any) -> list[str]:
    """The ``action`` enum values of a condensed action-routed tool, or ``[]``.

    Reads the tool's published JSON-schema ``parameters`` (so it works for any
    tool, codegen'd or hand-written, without re-introspecting Python signatures).
    A condensed action-routed tool declares ``action`` as an enum (a ``Literal``
    surfaces as ``{"enum": [...]}``); a free-form ``action: str`` carries no enum
    and yields ``[]`` (those keep the client-introspection verbose path instead).
    """
    schema = getattr(tool, "parameters", None)
    if not isinstance(schema, dict):
        return []
    prop = (schema.get("properties") or {}).get(_ACTION_ARG)
    if not isinstance(prop, dict):
        return []
    enum = prop.get("enum")
    if isinstance(enum, list) and all(isinstance(v, str) for v in enum):
        return [v for v in enum if v]
    return []


def autowire_verbose_from_condensed(mcp: Any) -> list[str]:
    """Derive a verbose 1:1 surface from the already-registered condensed tools.

    The universal, **no-per-connector-edit** path: every connector registers its
    condensed action-routed tools (each ``<service>_<domain>`` tool takes an
    ``action`` + ``params_json`` and dispatches internally). For each such tool
    this registers one verbose tool ``<tool>__<action>`` that re-dispatches to the
    **same** condensed tool with ``action`` preset to that value, leaving every
    other argument (``params_json`` and any typed fields) as a passthrough.

    The action list is sourced two ways, in order:

    - **Static enum** — the tool declares ``action`` as a ``Literal`` (surfaces as a
      JSON-schema ``enum``); :func:`_action_enum` reads it directly.
    - **Dynamic / runtime** — the tool declares a free-form ``action: str`` and its
      actions come from ``public_actions(client)`` at call time (atlassian, arr,
      and most action-routed connectors). The valid names are obtained from an
      **action provider** registered via :func:`register_action_provider` (a list,
      a callable, or a client class introspected credential-free). Before this,
      such tools yielded zero verbose tools — the gap ECO-4.90 closes.

    It uses FastMCP's native ``Tool.from_tool`` transformation, so the verbose
    tool routes through the original tool's handler — preserving its ``Depends``
    client binding, ``Context`` injection, and result coercion with zero
    re-implementation. Each derived tool inherits the source tool's tags plus
    ``"verbose"`` so the existing ``DynamicVisibilityTransform`` can slice it.

    Tools with neither a static ``action`` enum nor a registered dynamic-action
    provider (no discoverable action surface at all) are skipped here —
    single-client connectors expose those one-to-one through
    :func:`register_verbose_tools` (client introspection) instead.

    Returns the list of derived verbose tool names.

    CONCEPT:AU-ECO.mcp.fleet-wide-verbose-auto — fleet-wide verbose auto-wire from condensed action enums
    CONCEPT:AU-ECO.mcp.verbose-auto-wire — verbose auto-wire enumerates dynamic (runtime) actions
    """
    try:
        from fastmcp.tools import Tool
        from fastmcp.tools.tool_transform import ArgTransform
    except Exception as exc:  # pragma: no cover - defensive (older FastMCP)
        logger.warning(
            "autowire_verbose_from_condensed: FastMCP transforms unavailable: %s", exc
        )
        return []

    source_tools = _provider_tools(mcp)
    providers: dict[str, Any] = getattr(mcp, _ACTION_PROVIDERS_ATTR, {})
    derived: list[str] = []
    for tool_name in sorted(source_tools):
        tool = source_tools[tool_name]
        # Don't re-expand an already-derived verbose tool (idempotent re-runs).
        if "__" in tool_name and getattr(tool, "tags", None) and "verbose" in tool.tags:
            continue
        actions = _tool_action_names(tool, providers)
        if not actions:
            continue
        src_tags = getattr(tool, "tags", None)
        base_tags = set(src_tags) if isinstance(src_tags, set) else set()
        for action in actions:
            verbose_name = f"{tool_name}__{action}"
            if verbose_name in source_tools:
                continue
            try:
                verbose_tool = Tool.from_tool(
                    tool,
                    name=verbose_name,
                    transform_args={
                        _ACTION_ARG: ArgTransform(default=action, hide=True)
                    },
                    tags=base_tags | {"verbose"},
                )
                mcp.add_tool(verbose_tool)
            except Exception as exc:  # pragma: no cover - defensive per-action
                logger.warning(
                    "autowire_verbose_from_condensed: could not derive %s: %s",
                    verbose_name,
                    exc,
                )
                continue
            derived.append(verbose_name)

    logger.debug(
        "autowire_verbose_from_condensed: derived %d verbose tools from %d condensed tools",
        len(derived),
        sum(1 for n in source_tools if _tool_action_names(source_tools[n], providers)),
    )
    return derived


def _condensed_entries(
    tool_registry: list | None,
    tools_module: Any,
    registrars: list | None,
) -> list[tuple[str, str, Any]]:
    """Normalize a condensed-tool declaration into ``[(tag, env_var, fn), ...]``.

    Three input shapes are accepted, in priority order:

    - ``tool_registry`` — an explicit ``[(tag, env_var, register_fn), ...]`` list
      (e.g. a codegen'd ``TOOL_REGISTRY``).
    - ``tools_module`` — a module/namespace to **auto-discover** every
      ``register_<tag>_tools`` callable; the per-domain toggle env var is derived
      as ``<TAG>TOOL`` (matching the fleet's existing names).
    - ``registrars`` — a list whose items are either a bare ``register_fn`` or a
      ``(tag, env_var, fn)`` tuple.
    """
    if tool_registry:
        return [tuple(e) for e in tool_registry]  # type: ignore[misc]

    if tools_module is not None:
        found: list[tuple[str, str, Any]] = []
        for name in sorted(vars(tools_module)):
            match = re.fullmatch(r"register_(.+)_tools", name)
            fn = getattr(tools_module, name)
            # Skip the shared helpers (they match the pattern but are not a
            # connector's domain registrars) and any non-callable.
            if not match or not callable(fn) or name in _SURFACE_HELPER_NAMES:
                continue
            tag = match.group(1)
            found.append((tag, f"{tag.upper()}TOOL", fn))
        return found

    entries: list[tuple[str, str, Any]] = []
    for item in registrars or []:
        if isinstance(item, tuple):
            entries.append(item)  # type: ignore[arg-type]
            continue
        name = getattr(item, "__name__", "")
        match = re.fullmatch(r"register_(.+)_tools", name)
        tag = match.group(1) if match else name
        entries.append((tag, f"{tag.upper()}TOOL", item))
    return entries


def gated_tool_names(mcp: Any) -> set[str]:
    """Tool names held back from the default session view by ``MCP_TOOL_MODE=intent``.

    Populated by :func:`register_tool_surface` as it stamps :data:`GATED_TAG`;
    empty in every other mode. The graph-os entry point
    (``mcp/kg_server.py::mcp_server``) reads this after building the server to
    seed the fleet multiplexer's local-tool gate
    (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse), so ``load_tools`` can reveal one of
    these exact granular tools for a session without mounting anything (they are
    already registered locally — just hidden by default).
    """
    return set(getattr(mcp, "_intent_gated_tools", ()) or ())


def register_tool_surface(
    mcp: Any,
    *,
    service: str,
    client_cls: type | None = None,
    get_client: Any = None,
    tool_registry: list | None = None,
    tools_module: Any = None,
    registrars: list | None = None,
    manifest: list[dict] | None = None,
    tool_prefix: str | None = None,
    verbose_targets: list[dict] | None = None,
    verbose_register: Any = None,
    autowire_condensed: bool = True,
    action_providers: dict[str, Any] | None = None,
) -> list[str]:
    """Register an agent's MCP tool surface per ``MCP_TOOL_MODE`` — the one place.

    This is the single fleet-wide entry point an agent's ``get_mcp_instance`` calls
    to wire its tools. It owns the whole selection so every connector inherits it
    with no per-agent branching:

    - ``condensed`` / ``both``: each discovered/declared ``register_<tag>_tools`` is
      gated by ``setting("<TAG>TOOL", True)`` (config.json-driven) and registered.
    - ``verbose`` / ``both``: :func:`register_verbose_tools` adds the 1:1 surface
      (typed from ``manifest`` where available, else ``params_json``).

    Declare the condensed registrars via exactly one of ``tool_registry`` /
    ``tools_module`` / ``registrars`` (see :func:`_condensed_entries`). Returns the
    list of registered condensed tags (mirrors the fleet's ``registered_tags``).

    **Verbose targets.** Single-client agents pass ``client_cls`` + ``get_client``
    (the common case). A **multi-client** agent (e.g. arr-mcp fronting Sonarr,
    Radarr, … — each its own client) instead passes ``verbose_targets``: a list of
    ``{"client_cls", "get_client", "tool_prefix"?, "service"?, "manifest"?}`` dicts,
    one per client; the verbose surface is built for each, all gated by the one
    ``MCP_TOOL_MODE`` here so the mode logic stays central.

    **Custom verbose builder.** An agent whose verbose surface isn't client-method
    based (e.g. graph-os, which dispatches actions into the API-gateway action core
    rather than a per-method client) passes ``verbose_register`` — a callable
    ``(mcp) -> None`` that registers its own 1:1 tools. It runs in verbose/both and
    counts as a verbose target for the empty-server guard.

    **Fleet-wide auto-wire (default ON).** In verbose/both mode, after the explicit
    verbose path runs, :func:`autowire_verbose_from_condensed` derives one verbose
    tool per ``action`` value of every condensed action-routed tool that just
    registered — so a connector exposing ONLY condensed tools (the common case)
    gets a 1:1 verbose surface with no per-connector edits. The action list comes
    from a static ``Literal`` enum where the tool has one; for free-form
    ``action: str`` tools (atlassian, arr, …) pass ``action_providers`` —
    ``{tool_name: actions}`` where ``actions`` is a list, a zero-arg callable, or a
    client class — and the auto-wire enumerates each tool's runtime actions
    (ECO-4.90). Pass ``autowire_condensed=False`` to opt a server out (e.g. one
    whose explicit targets already cover its whole surface). CONCEPT:AU-ECO.mcp.fleet-wide-verbose-auto.

    CONCEPT:AU-ECO.mcp.tool-mode-standardization — MCP tool-mode standardization (central surface wiring)
    """
    from agent_utilities.core.config import setting

    mode = tool_mode()
    registered_tags: list[str] = []

    targets = verbose_targets
    if targets is None and client_cls is not None and get_client is not None:
        targets = [
            {
                "client_cls": client_cls,
                "get_client": get_client,
                "tool_prefix": tool_prefix,
                "manifest": manifest,
            }
        ]
    has_verbose = bool(targets) or verbose_register is not None

    # Condensed registers in condensed/both/intent — and ALSO in verbose mode when
    # the agent has no verbose surface at all, so a condensed-only server is never
    # left empty by a deployment-wide MCP_TOOL_MODE=verbose meant for connectors.
    # ``intent`` registers the SAME condensed tools (REST/_execute_tool/
    # REGISTERED_TOOLS are unaffected — they are the backing surface the intent
    # verbs dispatch into) but additionally gates them from the default session
    # view (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse).
    if mode in ("condensed", "both", "intent") or (
        mode == "verbose" and not has_verbose
    ):
        toggles: dict[str, str] = getattr(mcp, "_condensed_tool_toggles", {})
        gated: set[str] = getattr(mcp, "_intent_gated_tools", set())
        for tag, env_var, register_fn in _condensed_entries(
            tool_registry, tools_module, registrars
        ):
            if not setting(env_var, True):
                continue
            before = set(_provider_tools(mcp))
            register_fn(mcp)
            # Tools this registrar just added belong to this domain: stamp the
            # canonical domain tag (so every condensed tool carries the tag that
            # matches its toggle — standardizing the ad-hoc per-author tags) and
            # record the exact tool->toggle-env map for docs/introspection.
            after = _provider_tools(mcp)
            for name in set(after) - before:
                tags_attr = getattr(after[name], "tags", None)
                if isinstance(tags_attr, set):
                    tags_attr.add(tag)
                    tags_attr.add(GRANULAR_TAG)
                    if mode == "intent":
                        tags_attr.add(GATED_TAG)
                        gated.add(name)
                toggles[name] = env_var
            registered_tags.append(tag)
        if toggles:
            mcp._condensed_tool_toggles = toggles
        if gated:
            mcp._intent_gated_tools = gated

    if mode in ("verbose", "both"):
        for target in targets or []:
            register_verbose_tools(
                mcp,
                target["client_cls"],
                target["get_client"],
                service=target.get("service", service),
                tool_prefix=target.get("tool_prefix"),
                manifest=target.get("manifest"),
            )
        if verbose_register is not None:
            verbose_register(mcp)
        # Universal fallback: expand every condensed action-routed tool's actions
        # into 1:1 verbose tools, so a connector exposing ONLY condensed tools still
        # gets a verbose surface with no per-connector wiring (ECO-4.89). Free-form
        # action tools enumerate via the registered action providers (ECO-4.90).
        if autowire_condensed:
            for tool_name, actions in (action_providers or {}).items():
                register_action_provider(mcp, tool_name, actions)
            autowire_verbose_from_condensed(mcp)

    return registered_tags
