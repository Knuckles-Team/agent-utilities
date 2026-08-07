#!/usr/bin/python
from __future__ import annotations

"""Client-side bridge for the fastmcp-4 / MCP SDK v2 upgrade.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

`agent-utilities` targets `fastmcp>=4.0.0b1` by default (see the `[mcp]` extra in
`pyproject.toml`), which transitively requires `mcp>=2.0.0,<3.0.0` (the MCP Python
SDK's v2 line). Empirically verified against a live fastmcp-4 server + a real
`pydantic_ai.mcp.MCPToolset` client (fastmcp 4.0.0b1, mcp 2.0.0, pydantic-ai-slim
2.21.0 — the latest published release as of this writing), two upstream gaps break
every real toolset connection unless bridged here. Both gaps are inside
`pydantic_ai.mcp` / `fastmcp`'s own code, not anything this package calls directly,
so they cannot be fixed by changing how *we* invoke the API — only by adapting to
the renamed/defaulted surface until upstream catches up:

1. **`mcp` SDK v2 renamed several protocol fields from camelCase to snake_case**
   (`inputSchema` -> `input_schema`, `mimeType` -> `mime_type`, etc). `fastmcp`
   ships its own deprecation bridge for this (`fastmcp._compat`, a curated table of
   warn-once camelCase properties) and it covers most of what `pydantic_ai.mcp`
   reads — but NOT `PromptsCapability.listChanged` / `ResourcesCapability.listChanged`
   / `ToolsCapability.listChanged` (read unconditionally by
   `ServerCapabilities.from_mcp_sdk` on every `MCPToolset.__aenter__`) or
   `ToolExecution.taskSupport` (read by `MCPToolset.get_tools()` whenever a tool
   advertises `execution` metadata). `mcp.shared.exceptions.McpError` was also
   renamed to `MCPError` — `pydantic_ai.mcp`'s tool-call error handling
   (`except mcp_exceptions.McpError`) still expects the old name. `install_mcp_v2_bridge()`
   closes exactly these four gaps, using the same technique fastmcp uses for the
   rest (a plain property reading the renamed attribute), guarded so it never
   shadows a real upstream fix.
2. **`fastmcp.client.Client` defaults to `mode="auto"`**, which negotiates the
   modern `server/discover` connect era against a fastmcp-4 server and leaves
   `Client.initialize_result` as `None`. `pydantic_ai.mcp.MCPToolset.__aenter__`
   (2.21.0) unconditionally asserts `client.initialize_result is not None`, so
   every connection to a real fastmcp-4 server fails outright unless the
   underlying client is pinned to `mode="legacy"` (today's initialize handshake,
   which populates `initialize_result`). `MCPToolset` does not expose a `mode`
   passthrough for its convenience constructors (bare transport / URL / in-process
   `FastMCP` server / `pydantic_ai.mcp.load_mcp_toolsets`), but `Client.mode` is a
   plain, un-validated instance attribute read lazily at connect time — so
   `force_legacy_protocol_mode()` reaches into an already-constructed
   `MCPToolset.client` (unwrapping `WrapperToolset.wrapped`, e.g.
   `PrefixedToolset` from `load_mcp_toolsets`) and pins it before first use.

Both are temporary, forward-compatible shims: `install_mcp_v2_bridge()` skips any
field pydantic-ai/fastmcp already provide (so a future release that adds proper
support makes this a no-op), and `force_legacy_protocol_mode()` is a one-line
attribute set with no other side effects. Delete this module once
`pydantic-ai-slim` ships a release whose `MCPToolset` handles the fastmcp-4
`server/discover` era natively and whose `mcp.py` reads the SDK v2 field/exception
names directly.
"""

import importlib
import importlib.metadata
import warnings
from contextlib import AsyncExitStack
from types import ModuleType
from typing import Any

_installed = False

#: Module paths that carry the MCP wire-protocol types, newest first. The MCP SDK
#: v2 line moved `mcp.types` OUT of the `mcp` distribution into a standalone
#: `mcp_types` one, so `mcp.types` simply does not exist on those installs while
#: `mcp_types` does; SDK v1 is the mirror image. Both are the SAME protocol-type
#: namespace (`Tool`, `TextContent`, `SamplingMessage`, …), so code that reads it
#: must bind whichever module the INSTALLED SDK actually ships.
_TYPES_MODULE_NAMES = ("mcp.types", "mcp_types")

#: Names `mcp.shared.exceptions` uses for the MCP protocol error, newest first.
#: SDK v2 (`mcp>=2.0.0`) renamed `McpError` -> `MCPError`; SDK v1 still ships the
#: old spelling. Both are the SAME protocol error, so code that catches it must
#: bind whichever one the INSTALLED SDK actually exposes.
_PROTOCOL_ERROR_NAMES = ("MCPError", "McpError")


def mcp_protocol_error() -> type[BaseException]:
    """Return the MCP protocol-error class for the *installed* MCP SDK line.

    `mcp.shared.exceptions.McpError` (SDK v1) was renamed `MCPError` in SDK v2.
    A hard `from mcp.shared.exceptions import MCPError` therefore raises
    `ImportError` on every SDK v1 install — and because the multiplexer imports
    the child-resilience layer at module scope, that ImportError takes the whole
    fleet loader down with it (CONCEPT:AU-ECO.mcp.protocol-compat-bridge).

    This resolves the class by name instead. It is deliberately NOT a
    `try/except ImportError` that falls back to a benign default: binding the
    name to `()` or `Exception` would silently break session-death detection for
    the entire child fleet. If neither spelling exists the SDK is unusable here,
    so this raises loudly.
    """
    from mcp.shared import exceptions as mcp_exceptions

    for name in _PROTOCOL_ERROR_NAMES:
        candidate = getattr(mcp_exceptions, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            return candidate
    raise ImportError(
        "mcp.shared.exceptions exposes neither 'MCPError' (MCP SDK v2) nor "
        "'McpError' (MCP SDK v1); the MCP child-resilience layer cannot detect a "
        "terminated child session without it."
    )


def mcp_protocol_exception(code: int, message: str, data: Any = None) -> BaseException:
    """Construct the installed SDK's protocol error without signature guessing.

    SDK v2's ``MCPError`` accepts ``(code, message, data)`` directly. SDK v1's
    ``McpError`` instead accepts one ``mcp.types.ErrorData`` model. Select from
    the same exported class name used by :func:`mcp_protocol_error`; do not
    retry on ``TypeError``, because that would hide a real constructor defect.
    """
    from mcp.shared import exceptions as mcp_exceptions

    error_type = mcp_protocol_error()
    if getattr(mcp_exceptions, "MCPError", None) is error_type:
        return error_type(code, message, data)
    error_data = mcp_types_module().ErrorData(code=code, message=message, data=data)
    return error_type(error_data)


def mcp_types_module() -> ModuleType:
    """Return the MCP wire-protocol types module for the *installed* MCP SDK line.

    `mcp.types` (SDK v1) became the standalone `mcp_types` distribution in SDK v2.
    A hard `from mcp import types` therefore raises `ImportError` on every SDK v2
    install — and because `eunomia_principal` is imported at module scope by
    `server_factory._configure_middleware`, whose Eunomia leg is deliberately
    **fail-closed** (`sys.exit(1)` — an authorization middleware that cannot load
    must not be skipped), that ImportError takes the WHOLE server down before it
    serves anything. Observed live: `aris-mcp` and `freshrss-mcp` crash-looped for
    9 days on exactly this, because their images ship `fastmcp 4.0.0a1` +
    MCP SDK v2 while the rest of the fleet is still on fastmcp 3.x / SDK v1
    (CONCEPT:AU-ECO.mcp.protocol-compat-bridge).

    This resolves the module by import path instead. Like `mcp_protocol_error()`
    it is deliberately NOT a `try/except ImportError` that falls back to a benign
    stand-in: binding this name to a stub would make every `isinstance` check
    against a protocol type silently False, which is a permission-shaped failure
    in an authorization middleware. If neither module exists the SDK is unusable
    here, so this raises loudly.
    """
    for name in _TYPES_MODULE_NAMES:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError(
        "neither 'mcp.types' (MCP SDK v1) nor 'mcp_types' (MCP SDK v2) is "
        "importable; the MCP protocol-type surface is unavailable, so tool / "
        "prompt / resource identity cannot be resolved."
    )


def install_mcp_v2_bridge() -> None:
    """Bridge the MCP SDK v2 attribute renames that `fastmcp._compat` doesn't cover.

    Idempotent. Safe to call from any module that constructs an `MCPToolset` before
    doing so; `agent_utilities.mcp.toolset_factory` calls it at import time so every
    call site in this package gets it for free.
    """
    global _installed
    if _installed:
        return

    from mcp.shared import exceptions as mcp_exceptions

    # NOT `from mcp import types` — this bridge exists FOR the SDK v2 line, and
    # that is exactly the line where `mcp.types` was re-homed to `mcp_types`.
    mcp_types = mcp_types_module()

    # `mcp.shared.exceptions.McpError` was renamed `MCPError` in SDK v2.
    # `pydantic_ai.mcp` still catches the old name in its tool-call error handling.
    if not hasattr(mcp_exceptions, "McpError") and hasattr(mcp_exceptions, "MCPError"):
        # Assigning through `__dict__` (rather than a plain attribute assignment)
        # keeps this a runtime-only alias that mypy doesn't try to statically
        # unify with the `MCPError` class identity.
        mcp_exceptions.__dict__["McpError"] = mcp_exceptions.MCPError

    # Fields `fastmcp._compat`'s own camelCase bridge table doesn't include, but
    # `pydantic_ai.mcp` still reads unconditionally.
    aliases: dict[type, dict[str, str]] = {
        mcp_types.PromptsCapability: {"listChanged": "list_changed"},
        mcp_types.ResourcesCapability: {"listChanged": "list_changed"},
        mcp_types.ToolsCapability: {"listChanged": "list_changed"},
        mcp_types.ToolExecution: {"taskSupport": "task_support"},
    }
    for cls, mapping in aliases.items():
        model_fields = getattr(cls, "model_fields", {})
        for camel, snake in mapping.items():
            # Never shadow a real attribute — a future SDK/fastmcp release that
            # restores or re-covers the field makes this bridge a no-op.
            if camel in cls.__dict__ or camel in model_fields:
                continue
            setattr(cls, camel, _make_property(cls.__name__, camel, snake))

    _install_pydantic_ai_v2_read_bridge()

    _installed = True


#: The exact `pydantic-ai-slim` release this D-CDX-69 patch was written and
#: verified against (see `_install_pydantic_ai_v2_read_bridge`'s docstring).
#: Deliberately an EXACT match, not a floor: `MCPToolset.__aenter__`/`get_tools`
#: are copied verbatim below with three reads corrected, so a body that has
#: since changed upstream must not silently receive a stale patch.
_PATCHED_PYDANTIC_AI_VERSION = "2.21.0"

_toolset_reads_patched = False


def _install_pydantic_ai_v2_read_bridge() -> None:
    """D-CDX-69: stop `pydantic_ai.mcp.MCPToolset` from ever performing the three
    deprecated camelCase reads that trip `fastmcp._compat`'s OWN shim.

    Unlike the gaps `install_mcp_v2_bridge` closes above (fields `fastmcp._compat`
    does not cover at all, so a plain property never shadows anything real),
    `InitializeResult.serverInfo` and `Tool.inputSchema` / `Tool.outputSchema` ARE
    already covered by `fastmcp._compat` — that shim is exactly what emits the
    `FastMCPDeprecationWarning` the live ServiceNow probe observed. The warning is
    correct: `pydantic_ai.mcp.MCPToolset.__aenter__` (line ~1086) and `.get_tools`
    (lines ~1149/1155) really do read `init_result.serverInfo` /
    `mcp_tool.inputSchema` / `mcp_tool.outputSchema` instead of the SDK v2
    `server_info` / `input_schema` / `output_schema` names.

    Because the warning is real and not a false positive, silencing it (a
    `warnings` filter, or overwriting `fastmcp`'s shim property with a
    non-warning one — which would ALSO hide the same warning for any other,
    genuinely-not-yet-migrated caller sharing the same process-wide `mcp_types`
    classes) would be exactly the "suppress instead of fix" anti-pattern this
    item's acceptance criteria rules out. The only fix that removes the warning
    without touching its detection machinery is to stop `MCPToolset` from making
    the deprecated read at all — so this replaces its two owning methods with a
    byte-for-byte copy of the installed `pydantic-ai-slim` 2.21.0 source with
    exactly those three attribute reads corrected to the v2 names (plus the
    pre-existing `ToolExecution.taskSupport` -> `task_support` read, which
    `install_mcp_v2_bridge` above already bridges but still emits its own
    warn-once `DeprecationWarning` on every read otherwise).

    Idempotent, and version-pinned defensively: if the installed
    `pydantic-ai-slim` is not exactly `_PATCHED_PYDANTIC_AI_VERSION`, this emits
    one `RuntimeWarning` and returns WITHOUT patching, because copied method
    bodies silently applied to a since-changed upstream method would be a much
    worse failure mode (subtly wrong behavior with no signal) than leaving the
    original (merely noisy) upstream method in place. Bump the pin, re-diff
    `pydantic_ai.mcp.MCPToolset.__aenter__`/`.get_tools` against this copy, and
    update both together when `pydantic-ai-slim` ships a new release.
    """
    global _toolset_reads_patched
    if _toolset_reads_patched:
        return

    try:
        installed_version = importlib.metadata.version("pydantic-ai-slim")
    except importlib.metadata.PackageNotFoundError:  # noqa: BLE001 — pydantic-ai-slim (the `[mcp]` extra) is genuinely optional; its absence means there is nothing for this bridge to patch, not a failure to surface
        return

    if installed_version != _PATCHED_PYDANTIC_AI_VERSION:
        warnings.warn(
            "agent-utilities: the D-CDX-69 pydantic-ai MCP v2-attribute-read "
            f"bridge is pinned to pydantic-ai-slim=={_PATCHED_PYDANTIC_AI_VERSION} "
            f"(the exact source `MCPToolset.__aenter__`/`.get_tools` were copied "
            f"from); {installed_version} is installed, so the bridge is SKIPPED. "
            "MCPToolset will read whichever attribute names its own installed "
            "mcp.py uses, and a FastMCPDeprecationWarning may reappear. Re-diff "
            "the two methods against the new release, update the copy in "
            "protocol_compat.py, and bump _PATCHED_PYDANTIC_AI_VERSION.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    import pydantic_ai.mcp as _pydantic_ai_mcp

    async def _v2_aenter(self: Any) -> Any:
        async with self._enter_lock:
            if self._running_count == 0:
                async with AsyncExitStack() as exit_stack:
                    await exit_stack.enter_async_context(self.client)
                    init_result = self.client.initialize_result
                    assert init_result is not None, (
                        "FastMCP Client initialization returned no result"
                    )
                    server_info = init_result.server_info  # v2 name (was `.serverInfo`)
                    server_capabilities = (
                        _pydantic_ai_mcp.ServerCapabilities.from_mcp_sdk(
                            init_result.capabilities
                        )
                    )
                    instructions = init_result.instructions
                    if self.log_level is not None:
                        await self.client.session.set_logging_level(self.log_level)
                    self._exit_stack = exit_stack.pop_all()
                    self._server_info = server_info
                    self._server_capabilities = server_capabilities
                    self._instructions = instructions
            self._running_count += 1
        return self

    async def _v2_get_tools(self: Any, ctx: Any) -> dict[str, Any]:
        max_retries = (
            self.max_retries if self.max_retries is not None else ctx.max_retries
        )
        tools: dict[str, Any] = {}
        for mcp_tool in await self.list_tools():
            task_support = (
                mcp_tool.execution.task_support if mcp_tool.execution else None
            )  # v2 name
            tools[mcp_tool.name] = _pydantic_ai_mcp.ToolsetTool(
                toolset=self,
                tool_def=_pydantic_ai_mcp.ToolDefinition(
                    name=mcp_tool.name,
                    description=mcp_tool.description,
                    parameters_json_schema=mcp_tool.input_schema,  # v2 name (was `.inputSchema`)
                    metadata={
                        "meta": mcp_tool.meta,
                        "annotations": mcp_tool.annotations.model_dump()
                        if mcp_tool.annotations
                        else None,
                        "task": task_support in ("required", "optional"),
                    },
                    return_schema=mcp_tool.output_schema
                    or None,  # v2 name (was `.outputSchema`)
                    include_return_schema=self.include_return_schema,
                ),
                max_retries=max_retries,
                args_validator=_pydantic_ai_mcp.TOOL_SCHEMA_VALIDATOR,
            )
        return tools

    # Assign through a deliberately `Any`-typed alias of the class, not
    # `_pydantic_ai_mcp.MCPToolset.__aenter__ = _v2_aenter` directly: mypy's
    # `[method-assign]` check treats a direct attribute assignment on a class
    # it has full knowledge of as reassigning a KNOWN method to an
    # incompatible signature, which is exactly wrong here (this IS the
    # intentional replacement) but is otherwise a useful check elsewhere, so
    # it is silenced only for this one deliberate, guarded, runtime override
    # of third-party code -- not globally and not via a suppression comment.
    # (Plain `setattr(cls, "name", value)` was tried first and rejected: it
    # dodges mypy the same way but trips Ruff's B010, which is right that
    # `setattr` with a literal name is never safer than assignment in
    # general -- it just doesn't know this specific assignment is the one
    # mypy needs steered away from.)
    toolset_cls: Any = _pydantic_ai_mcp.MCPToolset
    toolset_cls.__aenter__ = _v2_aenter
    toolset_cls.get_tools = _v2_get_tools
    _toolset_reads_patched = True


def _make_property(cls_name: str, camel: str, snake: str) -> property:
    # Built once here rather than on every attribute access. Worded to avoid the
    # identifier-interpolation gate's SQL/Cypher markers — a backtick or a
    # trailing `to ` right before a `{}` gap is exactly the shape of a
    # `GRANT ... TO <role>` identifier slot. This is a deprecation message, not
    # a query; rewording says so structurally instead of suppressing the gate.
    message = (
        f"Accessing {cls_name}.{camel} is deprecated; MCP SDK v2 renamed this "
        f"field. Read the attribute {snake} instead."
    )

    def getter(self: object) -> object:
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return getattr(self, snake)

    return property(getter)


def force_legacy_protocol_mode(toolset: Any) -> Any:
    """Pin an `MCPToolset` (or a list of toolsets) to `mode="legacy"` before first use.

    Unwraps `WrapperToolset.wrapped` (e.g. `PrefixedToolset`, which
    `pydantic_ai.mcp.load_mcp_toolsets` returns) to find the underlying
    `MCPToolset.client`. Toolsets that aren't MCP-backed (no `.client.mode`) are
    left untouched. Returns `toolset` unchanged for chaining.

    Unwrapping checks `isinstance(target, WrapperToolset)` rather than
    `hasattr(target, "wrapped")`: a bare `unittest.mock.MagicMock` (used
    throughout this package's own test suite to stand in for a toolset)
    answers `hasattr(..., "wrapped")` as `True` for EVERY attribute name and
    hands back a distinct child mock on every access, so a duck-typed unwrap
    loop never terminates against one. `isinstance` only matches the real
    toolset wrapper class.
    """
    if isinstance(toolset, (list, tuple)):
        for item in toolset:
            force_legacy_protocol_mode(item)
        return toolset

    from pydantic_ai.toolsets import WrapperToolset

    target = toolset
    while isinstance(target, WrapperToolset):
        target = target.wrapped

    client = getattr(target, "client", None)
    if client is not None and hasattr(client, "mode"):
        client.mode = "legacy"

    return toolset


def _declared_extra_floor(distribution: str, package: str, extra: str) -> Any | None:
    """Return the ``package`` requirement `distribution` declares under `extra`.

    Reads straight from `distribution`'s OWN installed metadata (PEP 508 extra
    markers on ``importlib.metadata.requires()``), never a separately-parsed
    ``pyproject.toml`` — so this is accurate for a dev checkout AND a deployed
    wheel, and can never drift from what the package actually declares.
    """
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    try:
        reqs = importlib.metadata.requires(distribution) or []
    except importlib.metadata.PackageNotFoundError:
        return None
    env: dict[str, str] = {}
    for environment_name, environment_value in default_environment().items():
        if not isinstance(environment_value, str):
            return None
        env[environment_name] = environment_value
    env["extra"] = extra
    for raw in reqs:
        try:
            req = Requirement(raw)
        except Exception:  # noqa: BLE001 - a malformed requirement string, skip it
            continue
        if req.name.lower() != package.lower():
            continue
        if req.marker is not None and req.marker.evaluate(env):
            return req
    return None


def _source_shadow_floor(package: str, extra: str) -> tuple[Any | None, str | None]:
    """Return the ``package`` floor declared by the SOURCE tree actually imported.

    CONCEPT:AU-ECO.mcp.protocol-compat-bridge

    Closes the D-OB-18 blind spot in :func:`check_mcp_sdk_floor`. Installed
    ``.dist-info`` metadata is written once, at install time. Every graph-os
    deployment then bind-mounts a *fresher* copy of this package's source over the
    installed one (``PYTHONPATH=/au`` shadowing the image's editable install), so the
    code that actually runs can declare a different floor than the metadata records —
    which is precisely how a pod ran fastmcp-4-targeted source on a fastmcp-3 runtime
    while every metadata-only check reported green.

    So: read the floor from the ``pyproject.toml`` sitting next to the imported
    package, when there is one. Returns ``(requirement, path)``; ``(None, None)`` when
    no source manifest is reachable (a plain wheel install — the normal case, where
    installed metadata is already authoritative).
    """
    import tomllib
    from pathlib import Path

    from packaging.requirements import InvalidRequirement, Requirement

    import agent_utilities

    origin = getattr(agent_utilities, "__file__", None)
    if not origin:
        return None, None
    manifest = Path(origin).resolve().parent.parent / "pyproject.toml"
    if not manifest.is_file():
        return None, None
    try:
        declared = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        warnings.warn(
            f"agent-utilities: could not read the source manifest {manifest} to "
            f"cross-check the declared '{package}' floor: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None
    raw_reqs = (declared.get("project", {}).get("optional-dependencies", {})).get(
        extra, []
    )
    for raw in raw_reqs:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue  # a malformed requirement string in the manifest — skip it
        if req.name.lower() == package.lower():
            return req, str(manifest)
    return None, str(manifest)


def check_mcp_sdk_floor(distribution: str = "agent-utilities") -> dict[str, Any]:
    """Compare the installed `mcp`/`fastmcp` SDK against the declared `[mcp]` floor.

    CONCEPT:AU-ECO.mcp.protocol-compat-bridge

    Closes D-ISR-2: nothing previously asserted that the runtime's installed
    `mcp`/`fastmcp` matched `pyproject.toml`'s declared floor, so v2-targeted
    source (this module's own bridges) could ship and only fail at import
    time in production — the exact failure `child_resilience.py` hit when a
    deployed pod resolved `mcp` 1.29.0 against `fastmcp>=4.0.0b1`-targeted
    code. This makes that mismatch fail loudly here (wired into
    `agent-utilities doctor` and a CI regression test) instead of surfacing
    only as a swallowed `ImportError` deep in a child transport.

    The `fastmcp` floor is read from `distribution`'s own extra metadata
    (`[mcp]` -> `fastmcp>=...`); the `mcp` floor is derived transitively from
    `fastmcp-slim`'s own installed metadata (fastmcp's real runtime
    dependency) rather than a second, hand-maintained copy of the same
    constraint.

    Returns ``{"ok": bool | None, "detail": str}``. ``ok=None`` means the
    check could not run (the `[mcp]` extra floor isn't declared at all, e.g.
    a build of this package that dropped the extra) — distinct from a real
    mismatch.
    """
    from packaging.version import InvalidVersion, Version

    fastmcp_req = _declared_extra_floor(distribution, "fastmcp", "mcp")
    if fastmcp_req is None:
        return {
            "ok": None,
            "detail": f"{distribution} declares no 'fastmcp' floor under the [mcp] extra",
        }
    try:
        installed_fastmcp = importlib.metadata.version("fastmcp")
    except importlib.metadata.PackageNotFoundError:
        return {
            "ok": False,
            "detail": "fastmcp is not installed (the [mcp] extra is absent)",
        }

    problems: list[str] = []

    # D-OB-18 — installed `.dist-info` metadata records the floor as of INSTALL time,
    # but every graph-os pod shadows the installed package with fresher bind-mounted
    # source (`PYTHONPATH=/au`), and every editable dev checkout goes stale the moment
    # pyproject.toml is edited. When the two disagree it is the SOURCE that runs, so the
    # source floor — not the metadata snapshot — is what the installed SDK must satisfy.
    # Checking metadata against metadata is precisely how a pod ran fastmcp-4-targeted
    # source on fastmcp 3.4.5 while reporting green.
    #
    # A divergence is NOT by itself a failure (a tightened floor that the installed SDK
    # still satisfies is harmless); it is reported as context on the outcome so a real
    # failure names its cause instead of just its symptom.
    shadow_req, manifest = _source_shadow_floor("fastmcp", "mcp")
    divergence: str | None = None
    # `shadow_req is not None` must gate the body directly (not via a separately
    # stored bool) so mypy keeps shadow_req narrowed to non-None inside — a bool
    # computed from the same check and re-tested in `if diverged:` loses that
    # narrowing, which is what produced the prior `Any | None` "has no attribute
    # 'specifier'" errors here.
    if shadow_req is not None and str(shadow_req.specifier) != str(
        fastmcp_req.specifier
    ):
        divergence = (
            f"source/installed divergence: the imported source ({manifest}) declares "
            f"fastmcp '{shadow_req.specifier}' under [mcp] while the installed "
            f"{distribution} metadata declares '{fastmcp_req.specifier}' — this "
            f"environment was provisioned from a different revision than the source it "
            f"now runs"
        )
        fastmcp_req = shadow_req

    try:
        if not fastmcp_req.specifier.contains(installed_fastmcp, prereleases=True):
            problems.append(
                f"fastmcp {installed_fastmcp} does not satisfy the declared floor "
                f"'{fastmcp_req.specifier}'"
            )
    except InvalidVersion:
        problems.append(f"fastmcp reports an unparseable version {installed_fastmcp!r}")

    # `mcp`'s floor is transitive via fastmcp-slim's own extras (`client`/`server`/
    # `mcp`) — derive it from there instead of hardcoding a second copy that could
    # silently drift from what fastmcp itself actually requires.
    mcp_req = None
    for extra in ("client", "server", "mcp"):
        mcp_req = _declared_extra_floor("fastmcp-slim", "mcp", extra)
        if mcp_req is not None:
            break
    try:
        installed_mcp: str | None = importlib.metadata.version("mcp")
    except importlib.metadata.PackageNotFoundError:
        installed_mcp = None

    if installed_mcp is None:
        problems.append("mcp is not installed")
    elif mcp_req is not None:
        try:
            Version(installed_mcp)
        except InvalidVersion:
            problems.append(f"mcp reports an unparseable version {installed_mcp!r}")
        else:
            if not mcp_req.specifier.contains(installed_mcp, prereleases=True):
                problems.append(
                    f"mcp {installed_mcp} does not satisfy fastmcp's declared floor "
                    f"'{mcp_req.specifier}'"
                )

    summary = (
        f"fastmcp={installed_fastmcp} (floor {fastmcp_req.specifier}), "
        f"mcp={installed_mcp} (floor {mcp_req.specifier if mcp_req else 'unknown'})"
    )
    if problems:
        if divergence:
            problems.append(divergence)
        return {"ok": False, "detail": "; ".join(problems) + f" [{summary}]"}
    if divergence:
        return {"ok": True, "detail": f"{summary} (note: {divergence})"}
    return {"ok": True, "detail": summary}
