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

CONCEPT:ECO-4.82 — MCP tool-mode standardization (verbose 1:1 surface)
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

from agent_utilities.mcp.concurrency import run_blocking
from agent_utilities.mcp.context_helpers import ctx_confirm_destructive

logger = logging.getLogger(__name__)

VALID_TOOL_MODES = ("condensed", "verbose", "both")
_DEFAULT_MODE = "condensed"

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
    """Return the configured MCP tool surface: ``condensed`` | ``verbose`` | ``both``.

    Reads ``MCP_TOOL_MODE`` through the shared config layer (so it is driven by
    the one XDG ``config.json``). Defaults to ``condensed`` — the small,
    action-routed surface — so existing deployments are unchanged. An
    unrecognized value falls back to ``condensed`` with a warning.
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
            doc = (op.get("summary") or op.get("description") or "").strip()
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
        mcp.tool(name=tool_name, tags={"verbose", domain})(tool_fn)
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

    CONCEPT:ECO-4.82 — MCP tool-mode standardization (central surface wiring)
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

    # Condensed registers in condensed/both — and ALSO in verbose mode when the
    # agent has no verbose surface at all, so a condensed-only server is never
    # left empty by a deployment-wide MCP_TOOL_MODE=verbose meant for connectors.
    if mode in ("condensed", "both") or (mode == "verbose" and not has_verbose):
        for tag, env_var, register_fn in _condensed_entries(
            tool_registry, tools_module, registrars
        ):
            if setting(env_var, True):
                register_fn(mcp)
                registered_tags.append(tag)

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

    return registered_tags
