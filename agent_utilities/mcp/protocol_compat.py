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

import importlib.metadata
import warnings
from typing import Any

_installed = False

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


def install_mcp_v2_bridge() -> None:
    """Bridge the MCP SDK v2 attribute renames that `fastmcp._compat` doesn't cover.

    Idempotent. Safe to call from any module that constructs an `MCPToolset` before
    doing so; `agent_utilities.mcp.toolset_factory` calls it at import time so every
    call site in this package gets it for free.
    """
    global _installed
    if _installed:
        return

    from mcp import types as mcp_types
    from mcp.shared import exceptions as mcp_exceptions

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

    _installed = True


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
    env = {**default_environment(), "extra": extra}
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
    diverged = shadow_req is not None and str(shadow_req.specifier) != str(
        fastmcp_req.specifier
    )
    if diverged:
        divergence = (
            f"source/installed divergence: the imported source ({manifest}) declares "
            f"fastmcp '{shadow_req.specifier}' under [mcp] while the installed "
            f"{distribution} metadata declares '{fastmcp_req.specifier}' — this "
            f"environment was provisioned from a different revision than the source it "
            f"now runs"
        )
        fastmcp_req = shadow_req
    else:
        divergence = None

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
