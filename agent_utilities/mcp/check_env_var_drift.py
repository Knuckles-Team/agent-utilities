"""Detect env-var / config drift in an agent package — the code is the source of truth.

CONCEPT:AU-OS.config.env-var-drift-guard — Env-var drift guard.

A connector documents its env vars in several places: ``.env.example``, every
``mcp_config*.json`` ``env`` block, ``docker/*compose*.yml`` ``environment:``, and the
README tables. The **only** authority for *which vars actually exist* is the code that
reads them. This module computes the code-read set and diffs the declared sets against it,
emitting drift findings:

- ``DEAD`` — a var declared in config/docs that **no code reads** (e.g. a scaffolder's
  ``*_TOKEN``, or per-endpoint ``*_TOOL`` toggles the framework never honours). Remove it.
- ``UNDOCUMENTED`` — a var the code reads that is **missing from** ``.env.example``. Add it.
- ``MISSING_TOOL_MODE`` — a launch-style ``mcp_config.json`` ``env`` block (or README
  example) with **no** ``MCP_TOOL_MODE`` (users can't discover the surface). Add it.
- ``MALFORMED_VALUE`` — a config ``env`` value with a whitespace-padded substitution like
  ``"${ VAR:-True }"``. Use ``"${VAR:-True}"`` (no spaces inside the braces).
- ``AGENT_VAR_IN_MCP`` — an agent-runtime var (``AGENT_DESCRIPTION``, ``MCP_URL``, a
  ``*_ENABLE`` companion suite …) sitting in an **MCP-server** config; it launches the
  agent, never the server. Move it to the agent config.
- ``STALE_EXAMPLE`` — a README ``mcp_config`` example ``env`` key that isn't in the
  code-read surface (a leftover scaffold placeholder). Regenerate the examples.

The code-read set (DEAD-suppression) =
  ``setting("VAR", …)`` reads in the package
  ∪ derived tool toggles: ``register_<tag>_tools`` → ``<TAG>TOOL``
  ∪ compose ``image:``/``command:``/``entrypoint:``/``args:`` ``${VAR}`` substitutions —
    a genuine read docker compose performs, not just ``environment:`` blocks
  ∪ dynamic/derived reads a function composes at runtime from a literal argument, declared
    via the CONCEPT:AU-OS.config.dynamic-env-family ``dynamic_env_prefix_arg`` /
    ``dynamic_env_suffixes`` attribute convention (see ``transport_security.py``)
  ∪ the package's own ``scripts/*.py`` — real first-party dev/CI tooling with a genuine
    read, but excluded from the UNDOCUMENTED-eligible set below (see ``_script_reads``)
  ∪ the inherited agent-utilities surface (``readme_env_vars.INHERITED_ENV`` + framework extras)
  ∪ ``setting("VAR", …)`` reads in agent-utilities core.

The documentable/UNDOCUMENTED-eligible set is narrower: package reads + toggles + compose
substitutions + dynamic-family reads — NOT ``scripts/`` and NOT the inherited/framework
surface, so a dev-only gate script's own config never forces an entry into a package's
public ``.env.example``.

Usage (from an agent repo root)::

    python -m agent_utilities.mcp.check_env_var_drift            # human report
    python -m agent_utilities.mcp.check_env_var_drift --check    # exit 1 on drift (pre-commit)
    python -m agent_utilities.mcp.check_env_var_drift --json     # machine-readable findings

Wire ``--check`` as a pre-commit hook (the agent-package-builder scaffold does).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeGuard

# Direct script execution puts ``agent_utilities/mcp`` first on ``sys.path``.
# Force this checkout's repository root ahead of any globally installed copy so
# the guard and its mutually importing env-source helper share one implementation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_utilities.mcp.env_policy import is_agent_only
from agent_utilities.mcp.readme_env_vars import INHERITED_ENV, parse_env_example

# Env reads the code performs. ``setting(...)`` is the sanctioned accessor, but bare
# ``os.getenv`` / ``os.environ.get`` / ``os.environ[...]`` (and ``from os import …`` forms)
# are also real reads — count them so a live var is never mis-flagged DEAD. A subscript
# ``os.environ["X"] = ...`` is a WRITE (cross-process signaling, not config to document),
# so the subscript branch excludes an assignment via negative lookahead. The third
# alternative is the OTHER sanctioned accessor (AGENTS.md "Configuration discipline"): a
# typed ``AgentConfig`` field declared ``Field(..., alias="VAR")`` — pydantic reads the env
# var by that alias at parse time, so it is a real read even though no ``setting()``/
# ``getenv`` call appears at the use site. Requiring an upper-case-first alias value keeps
# this from matching unrelated (lower-case) pydantic/SQL/Cypher aliases (e.g. ``alias="r"``).
# The fourth alternative is a one-level-indirect dynamic read: every writeback sink
# declares ``enable_flag = "<SINK>_ENABLE_WRITE"`` (a class attribute), and the shared
# runner reads it via ``setting(sink.enable_flag, False)`` — the literal var name never
# appears next to a ``setting(``/``getenv(`` call, only next to its declaration. These are
# matched via AST inspection (see :func:`_env_names_from_tree`) rather than a line-oriented
# regular expression so multiline calls resolve correctly and lookalike snippets embedded in
# string literals are excluded; ``_ENV_NAME`` validates each literal candidate name.
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
# ``register_<tag>_tools`` — a condensed registrar; toggle env var is ``<TAG>TOOL``.
# The leading lookbehind keeps a PRIVATE helper out of the tag set. A name such as
# ``_register_<tag>_tools`` contains the public spelling as a substring, and ``_``
# is a word character so ``\b`` alone does not separate them. Without the
# lookbehind, an extraction that creates such a helper mints a phantom
# ``<TAG>TOOL`` toggle that no runtime code reads -- the runtime discovers
# registrars from the CALLER's module namespace, never by text scan. That is how
# ``METATOOL`` (from multiplexer's private meta-tool helper) came to be
# documented in .env.example despite nothing ever reading it.
#
# NOTE: spell no public ``register_*_tools`` literal in this file -- this scanner
# reads every .py including itself, so an example in a comment mints a phantom.
_REGISTRAR = re.compile(r"(?<![0-9A-Za-z_])register_([a-z][a-z0-9_]*?)_tools\b")
# A ``- "KEY=value"`` or ``KEY: value`` line inside a compose ``environment:`` list/map.
_COMPOSE_ENV = re.compile(r"""^\s*-?\s*["']?([A-Z][A-Z0-9_]*)["']?\s*[:=]""")
# A ``${...}`` shell substitution — inner text is inspected for stray whitespace.
_SUBST = re.compile(r"\$\{([^}]*)\}")
# A fenced ```json block (README mcp_config examples).
_JSON_FENCE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)

# README markers/anchors that bound the mcp_config example region (see readme_mcp_examples).
README_START = "<!-- MCP-CONFIG-EXAMPLES:START -->"
README_END = "<!-- MCP-CONFIG-EXAMPLES:END -->"
README_HEADING = "## MCP Configuration Examples"
README_ADDL = "<!-- BEGIN GENERATED: additional-deployment-options -->"

# Framework vars read inside agent-utilities (create_agent / gateway / telemetry / connector
# base) on behalf of every connector — legitimately documentable, never "dead".
FRAMEWORK_EXTRA: frozenset[str] = frozenset(
    {
        "AUTH_TYPE",
        "AGENT_PROVIDER_PROFILE",
        "AGENT_DESCRIPTION",
        "AGENT_SYSTEM_PROMPT",
        "DEFAULT_AGENT_NAME",
        "ENABLE_OTEL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        # Parent-resolved OTLP headers are an internal, process-lifetime value.
        # Operators configure only OTEL_EXPORTER_OTLP_HEADERS_REF.
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_HEADERS_REF",
        "OTEL_EXPORTER_OTLP_PUBLIC_KEY_REF",
        "OTEL_EXPORTER_OTLP_SECRET_KEY_REF",
        "OTEL_TLS_PROFILE",
        "OTEL_TLS_PROFILE_REF",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "PROVIDER",
        "MODEL_ID",
        "ENABLE_WEB_UI",
        "MCP_URL",
    }
)
# Connector-standard suffixes read by the agent-utilities connector base, not the package.
_SAFE_SUFFIXES: tuple[str, ...] = ()
# Generic process / library runtime vars legitimately set in a launch env but never read
# via ``setting()`` — not app config, so not "dead".
RUNTIME_ALLOWLIST: frozenset[str] = frozenset(
    {
        "TERM",
        "NO_COLOR",
        "FORCE_COLOR",
        "FASTMCP_LOG_LEVEL",
        "PYTHONUNBUFFERED",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "LOG_LEVEL",
        "TZ",
        # OS / desktop / shell conventions the code reads for interop (build a subprocess
        # env, detect a display server, resolve an XDG override) — never app config to set
        # in .env.example; the OS or shell sets these, not a deployer of this package.
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "PWD",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "TMUX",
        "VIRTUAL_ENV",
        "APPDATA",
        "LOCALAPPDATA",
        "SYSTEMROOT",
        "PROGRAMFILES",
        # R-07: Windows' own identity vars, read by unified_install.py's
        # _secure_mkdir() to build a `DOMAIN\user` icacls principal when
        # expressing a POSIX 0o700 mkdir's owner-only intent via an ACL
        # instead (Windows has no permission bits) -- the OS sets these,
        # never a deployer via .env.example, same as USER/HOME above.
        "USERDOMAIN",
        "USERNAME",
        "XDG_CONFIG_HOME",
        "XDG_STATE_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "CI",
        "PYTEST_CURRENT_TEST",
        "container",
        # A well-known third-party ecosystem convention (HuggingFace's own cache-dir var,
        # read for interop when pointing an embedding model at a shared cache) — not an
        # agent-utilities-defined config surface.
        "HF_HOME",
    }
)
# Library-owned runtime vars identified by prefix (consumed by the library, not our code).
_RUNTIME_PREFIXES: tuple[str, ...] = ("FASTMCP_",)
# Synthetic constants that exist ONLY to exercise the config-loading machinery itself in a
# unit test (``tests/unit/core/test_setting_accessor.py`` probing ``setting()`` type
# inference, ``test_load_config.py``/``test_base_utilities.py`` (configuration-source
# tests probing dotenv/config.json and explicit process/XDG loading), and a stale
# probe in ``test_kg_autorouting.py``) — never real app config a deployer
# would set, so they must not be pushed into ``.env.example``.
TEST_FIXTURE_VARS: frozenset[str] = frozenset(
    {
        "AU_T_STR",
        "AU_T_INT",
        "AU_T_FLOAT",
        "AU_T_BOOL",
        "AU_T_LIST",
        "MY_VERBOSE_TEST_KEY",
        "PUSH_ENV_VAR_MARKER",
        "LIGHTWEIGHT_MODEL",
        # a fake writeback sink's enable_flag, defined only in
        # tests/unit/knowledge_graph/enrichment/test_writeback_approval.py to exercise the
        # approval flow — no real "TestHomeSink" sink or system-of-record exists.
        "TESTHS_ENABLE_WRITE",
    }
)
# Files that document the env-read patterns above USING those exact placeholder names as
# prose (a ``setting`` call spelled out with a "VAR" placeholder, a getenv-style lookup
# spelled out with an "X" placeholder), not a real read — skip them so the scanner doesn't
# mistake its own documentation for a live var literally named "VAR"/"X".
_SELF_DOC_FILES: frozenset[str] = frozenset(
    {"check_env_var_drift.py", "check_no_env_sprawl.py"}
)
# docker/*.compose.yml recipes for bundled THIRD-PARTY infra images (see
# ``_compose_env_keys``) — their ``environment:`` block belongs to that image, not to
# agent-utilities' own code-read surface.
_THIRD_PARTY_COMPOSE_FILES: frozenset[str] = frozenset(
    {
        "docker-compose.kafka.yml",
        "kafka-kraft.compose.yml",
        "egeria.compose.yml",
        "jena_fuseki.compose.yml",
        "neo4j.compose.yml",
        "paradedb.compose.yml",
        "pg-age.compose.yml",
        "pg-age-full.compose.yml",
        "falkordb.compose.yml",
    }
)


_HOST_SUFFIXES = ("_BASE_URL", "_URL", "_HOST")
_WALK_SKIP_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)
# Directories whose ``*.py`` is prose/fixture code, not a real runtime read surface:
# ``docs``/``examples``/``reports`` are documentation snippets and generated output,
# ``test``/``tests`` are unit-test fixtures (see TEST_FIXTURE_VARS and the
# ``_derive_toggle_vars`` tests-skip above for the analogous reasoning there). ``scripts`` IS
# still excluded from this *main* scan — see ``_script_reads`` below for why its reads are
# handled separately rather than simply un-excluding it here.
_NON_RUNTIME_SOURCE_DIRS = frozenset(
    {
        "docs",
        "examples",
        "reports",
        "scripts",
        "test",
        "tests",
    }
)


def _walk_files(root: Path, *, suffix: str) -> list[Path]:
    """Walk source files without entering dependency/cache trees.

    Pruning at ``os.walk`` directory boundaries is essential: filtering paths
    produced by ``Path.rglob`` still traverses entire virtual environments and
    can exhaust memory or file handles on large WSL-mounted environments.
    """
    found: list[Path] = []
    for directory, names, files in os.walk(root):
        names[:] = sorted(
            name
            for name in names
            if name not in _WALK_SKIP_DIRS and not name.startswith(".")
        )
        base = Path(directory)
        found.extend(base / name for name in files if name.endswith(suffix))
    return sorted(found)


def _stem(var: str) -> str:
    """The service/domain stem of a var, suffixes stripped (for alias/rename matching)."""
    return re.sub(
        r"(_BASE_URL|_URL|_HOST|_TOKEN|_API_KEY|_KEY|_SECRET|TOOL)$",
        "",
        var,
    )


def _is_host_var(var: str) -> bool:
    return var.endswith(_HOST_SUFFIXES)


def _literal_env_name(node: ast.AST | None) -> str | None:
    """Return an uppercase environment name represented by a literal AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if _ENV_NAME.fullmatch(node.value) else None
    return None


def _is_environ(node: ast.AST) -> bool:
    """Return whether *node* denotes ``os.environ`` or an imported ``environ``."""
    return (
        isinstance(node, ast.Name)
        and node.id == "environ"
        or isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr == "environ"
    )


def _is_os_module(node: ast.AST) -> bool:
    """Return whether *node* denotes the imported ``os`` module."""
    if isinstance(node, ast.Name) and node.id == "os":
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "__import__"
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "os"
    )


def _is_setting_reader_name(func: ast.expr) -> bool:
    """Return whether *func* is a bare call to a sanctioned setting reader
    (``setting``/``_setting``/``getenv``) — the direct-call half of
    :func:`_is_env_reader_call`, reused below to recognize a passthrough
    helper's own internal read."""
    return isinstance(func, ast.Name) and func.id in {"_setting", "setting", "getenv"}


def _comprehension_reads_setting_over(node: ast.AST, param_names: set[str]) -> bool:
    """True when *node* is a single-generator comprehension iterating one of
    *param_names* whose element calls a setting reader with that generator's
    own loop variable — the shape of ``any(setting(key) for key in keys)``."""
    if not isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
        return False
    if len(node.generators) != 1:
        return False
    comp = node.generators[0]
    if comp.ifs or not (
        isinstance(comp.iter, ast.Name) and comp.iter.id in param_names
    ):
        return False
    if not isinstance(comp.target, ast.Name):
        return False
    loop_var = comp.target.id
    return any(
        isinstance(inner, ast.Call)
        and _is_setting_reader_name(inner.func)
        and inner.args
        and isinstance(inner.args[0], ast.Name)
        and inner.args[0].id == loop_var
        for inner in ast.walk(node.elt)
    )


def _collect_setting_passthrough_helpers(tree: ast.AST) -> set[str]:
    """Same-module indirection, one level further removed than
    :func:`_collect_alias_literals`'s variable alias: a local helper like

        def _any_setting(*keys: str) -> bool:
            return any(setting(key) for key in keys)

    forwards each of ITS OWN CALLERS' literal string arguments into
    ``setting()`` — a genuine read for every literal passed at a call site
    (``_any_setting("GITLAB_TOKEN", "GITLAB_API_TOKEN")``), even though
    neither literal ever appears next to a ``setting(``/``getenv(`` call
    itself (see ``agent_utilities/knowledge_graph/core/hydration.py``'s
    ``_any_setting``/``_all_settings``, which drove 15 ``.env.example``
    vars to a false DEAD finding before this helper existed). Matches any
    function whose vararg or plain parameters are looped over inside a
    setting-reading comprehension, not just these two specific names, so a
    future helper shaped the same way is recognized without another edit
    here."""
    helpers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        param_names: set[str] = set()
        if node.args.vararg is not None:
            param_names.add(node.args.vararg.arg)
        param_names.update(a.arg for a in node.args.args)
        param_names.update(a.arg for a in node.args.kwonlyargs)
        if not param_names:
            continue
        if any(
            _comprehension_reads_setting_over(inner, param_names)
            for inner in ast.walk(node)
        ):
            helpers.add(node.name)
    return helpers


def _passthrough_helper_call_literals(node: ast.Call, helpers: set[str]) -> list[str]:
    """Literal env-name arguments passed to a same-module setting-passthrough
    helper (see :func:`_collect_setting_passthrough_helpers`) at one call
    site."""
    if not (isinstance(node.func, ast.Name) and node.func.id in helpers):
        return []
    return [literal for arg in node.args if (literal := _literal_env_name(arg))]


_DirectSettingForwarder = tuple[str, int | None]


def _forwarded_call_argument(
    call: ast.Call, forwarders: dict[str, _DirectSettingForwarder]
) -> ast.expr | None:
    """Return the expression a reader/helper receives as its env-name argument."""
    if _is_env_reader_call(call):
        return call.args[0] if call.args else None
    if not isinstance(call.func, ast.Name):
        return None
    callee = forwarders.get(call.func.id)
    if callee is None:
        return None
    parameter, position = callee
    keyword_value = next(
        (keyword.value for keyword in call.keywords if keyword.arg == parameter), None
    )
    if keyword_value is not None:
        return keyword_value
    if position is None or len(call.args) <= position:
        return None
    return call.args[position]


def _function_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[list[ast.arg], set[str]]:
    """Return ordered positional arguments and every named parameter."""
    positional = [*node.args.posonlyargs, *node.args.args]
    parameters = {argument.arg for argument in positional}
    parameters.update(argument.arg for argument in node.args.kwonlyargs)
    return positional, parameters


def _forwarded_parameter_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    forwarders: dict[str, _DirectSettingForwarder],
) -> str | None:
    """Find the function parameter forwarded to a known configuration reader."""
    _, parameters = _function_parameters(node)
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        argument = _forwarded_call_argument(inner, forwarders)
        if isinstance(argument, ast.Name) and argument.id in parameters:
            return argument.id
    return None


def _forwarder_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef, parameter: str
) -> _DirectSettingForwarder:
    """Describe how a caller supplies ``parameter`` to ``node``."""
    positional, _ = _function_parameters(node)
    position = next(
        (
            index
            for index, argument in enumerate(positional)
            if argument.arg == parameter
        ),
        None,
    )
    return parameter, position


def _discover_forwarder_pass(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    forwarders: dict[str, _DirectSettingForwarder],
) -> bool:
    """Discover one dependency layer of direct setting forwarders."""
    discovered: dict[str, _DirectSettingForwarder] = {}
    for node in functions:
        if node.name in forwarders:
            continue
        parameter = _forwarded_parameter_name(node, forwarders)
        if parameter is not None:
            discovered[node.name] = _forwarder_signature(node, parameter)
    forwarders.update(discovered)
    return bool(discovered)


def _collect_direct_setting_forwarders(
    tree: ast.AST,
) -> dict[str, _DirectSettingForwarder]:
    """Find local helpers that pass one parameter directly to ``setting()``.

    Typed configuration loaders commonly accept an explicit mapping for tests and
    fall back to the process configuration otherwise::

        def _configured_value(env, name, default=None):
            if env is not None:
                return env.get(name, default)
            return setting(name, default)

    The literal environment name therefore lives at the helper's call site, not at
    the eventual reader. Record the forwarded parameter and its positional index so
    only that argument is credited as a read; unrelated uppercase literals passed to
    the same helper remain ignored.
    """
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    forwarders: dict[str, _DirectSettingForwarder] = {}
    # At most one new forwarding layer can be discovered per pass. Repeating once
    # per function reaches the fixpoint without embedding the analysis in a deeply
    # branched while-loop.
    for _ in functions:
        if not _discover_forwarder_pass(functions, forwarders):
            break
    return forwarders


def _direct_forwarder_call_literal(
    node: ast.Call, forwarders: dict[str, _DirectSettingForwarder]
) -> str | None:
    """Resolve the env-name literal passed to one direct forwarding helper."""
    if not isinstance(node.func, ast.Name):
        return None
    forwarder = forwarders.get(node.func.id)
    if forwarder is None:
        return None
    parameter, position = forwarder
    for keyword in node.keywords:
        if keyword.arg == parameter:
            return _literal_env_name(keyword.value)
    if position is not None and len(node.args) > position:
        return _literal_env_name(node.args[position])
    return None


def _direct_forwarder_literals(
    tree: ast.AST, forwarders: dict[str, _DirectSettingForwarder]
) -> set[str]:
    """Collect env-name literals from calls to known forwarding helpers."""
    return {
        literal
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        if (literal := _direct_forwarder_call_literal(node, forwarders))
    }


def _collect_alias_literals(tree: ast.AST) -> dict[str, str]:
    """Same-module one-level indirection: ``_LOG_LEVEL_ENV = "MCP_V2_GATEWAY_LOG_LEVEL"``
    followed by ``os.environ.get(_LOG_LEVEL_ENV)``. The var name never appears
    literally next to a reader call, so a literal-argument-only scan reported it as
    DEAD ("declared in compose, read by nothing") even though it is read on every
    start-up — a FALSE finding no edit to the compose file can satisfy. Only names
    that are BOTH bound to an env-name literal here AND passed to a reader below are
    counted, so this stays a read detector and never degenerates into "any uppercase
    string constant is an env var"."""
    alias_literals: dict[str, str] = {}
    for node in ast.walk(tree):
        alias_value: ast.expr | None
        if isinstance(node, ast.Assign):
            alias_targets: list[ast.expr] = list(node.targets)
            alias_value = node.value
        elif isinstance(node, ast.AnnAssign):
            alias_targets = [node.target]
            alias_value = node.value
        else:
            continue
        literal = _literal_env_name(alias_value)
        if not literal:
            continue
        for target in alias_targets:
            if isinstance(target, ast.Name):
                alias_literals[target.id] = literal
            elif isinstance(target, ast.Attribute):
                alias_literals[target.attr] = literal
    return alias_literals


def _is_env_reader_call(node: ast.Call) -> bool:
    direct_reader = isinstance(node.func, ast.Name) and node.func.id in {
        "_setting",
        "setting",
        "getenv",
    }
    os_getenv = (
        isinstance(node.func, ast.Attribute)
        and _is_os_module(node.func.value)
        and node.func.attr == "getenv"
    )
    environ_get = (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and _is_environ(node.func.value)
    )
    return direct_reader or os_getenv or environ_get


def _is_field_call(node: ast.Call) -> bool:
    return (isinstance(node.func, ast.Name) and node.func.id == "Field") or (
        isinstance(node.func, ast.Attribute) and node.func.attr == "Field"
    )


def _field_alias_names(node: ast.Call) -> list[str]:
    names = []
    for keyword in node.keywords:
        if keyword.arg == "alias":
            name = _literal_env_name(keyword.value)
            if name:
                names.append(name)
    return names


def _handle_call_node(node: ast.Call, env_name_arg: Any, found: set[str]) -> None:
    if _is_env_reader_call(node) and node.args:
        name = env_name_arg(node.args[0])
        if name:
            found.add(name)
    if _is_field_call(node):
        found.update(_field_alias_names(node))


def _is_environ_subscript_read(node: ast.AST) -> TypeGuard[ast.Subscript]:
    """``os.environ[...]`` read. A ``TypeGuard`` so the caller may reach
    ``node.slice`` — a plain ``bool`` narrows nothing, which is what left the
    caller reading ``.slice`` off a bare ``ast.AST``."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and _is_environ(node.value)
    )


def _flag_assign_targets_and_value(
    node: ast.AST,
) -> tuple[list[ast.expr] | None, ast.expr | None]:
    if isinstance(node, ast.Assign):
        return node.targets, node.value
    if isinstance(node, ast.AnnAssign):
        return [node.target], node.value
    return None, None


def _is_enable_flag_target(target: ast.expr) -> bool:
    return (isinstance(target, ast.Name) and target.id == "enable_flag") or (
        isinstance(target, ast.Attribute) and target.attr == "enable_flag"
    )


def _handle_enable_flag_assign(node: ast.AST, found: set[str]) -> None:
    targets, value = _flag_assign_targets_and_value(node)
    if targets is None:
        return
    if any(_is_enable_flag_target(target) for target in targets):
        name = _literal_env_name(value)
        if name:
            found.add(name)


def _scan_env_reads_and_flags(
    tree: ast.AST, env_name_arg: Any, passthrough_helpers: set[str] | None = None
) -> set[str]:
    found: set[str] = set()
    helpers = passthrough_helpers or set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            _handle_call_node(node, env_name_arg, found)
            found.update(_passthrough_helper_call_literals(node, helpers))
        if _is_environ_subscript_read(node):
            name = env_name_arg(node.slice)
            if name:
                found.add(name)
        _handle_enable_flag_assign(node, found)
    return found


def _env_names_from_tree(tree: ast.AST) -> set[str]:
    """Collect literal runtime configuration reads from one parsed module.

    AST inspection handles multiline calls and excludes lookalike snippets embedded
    in strings. It also distinguishes ``os.environ[...]`` reads from assignment
    writes, which a line-oriented regular expression cannot do reliably.
    """
    alias_literals = _collect_alias_literals(tree)
    passthrough_helpers = _collect_setting_passthrough_helpers(tree)
    direct_forwarders = _collect_direct_setting_forwarders(tree)

    def _env_name_arg(node: ast.AST | None) -> str | None:
        """Resolve a reader argument: a literal, or a same-module alias to one."""
        literal = _literal_env_name(node)
        if literal:
            return literal
        if isinstance(node, ast.Name):
            return alias_literals.get(node.id)
        if isinstance(node, ast.Attribute):
            return alias_literals.get(node.attr)
        return None

    found = _scan_env_reads_and_flags(tree, _env_name_arg, passthrough_helpers)
    found.update(_direct_forwarder_literals(tree, direct_forwarders))
    return found


def _scan_setting_calls(root: Path) -> set[str]:
    """Every env-var literal read (``setting`` or bare ``os.getenv``/``os.environ``) in
    ``*.py`` under ``root``. Skips assignment writes and reads nested inside a string
    literal (codegen templates)."""
    found: set[str] = set()
    for py in _walk_files(root, suffix=".py"):
        try:
            relative_parts = py.relative_to(root).parts
        except ValueError:
            relative_parts = py.parts
        if any(part in _NON_RUNTIME_SOURCE_DIRS for part in relative_parts[:-1]):
            continue
        if py.name in _SELF_DOC_FILES:
            continue
        if py.name in _SELF_DOC_FILES:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        found.update(_env_names_from_tree(tree))
    found.discard("")
    return found


def _script_reads(root: Path) -> set[str]:
    """``setting()``/``getenv()`` literals read inside the package's own ``scripts/*.py`` —
    real first-party code the checker must not be blind to (e.g.
    ``scripts/validate_falkordb.py`` genuinely reads ``FALKORDB_URI``/``GRAPHDB_PASSWORD``,
    so declaring either in ``.env.example`` must not be reported DEAD).

    Deliberately kept SEPARATE from ``pkg_reads`` rather than folded into the main
    ``_scan_setting_calls`` surface: ``scripts/`` in this fleet is scaffolded, maintainer-only
    dev/CI tooling (validation harnesses, local gate runners) that reads its config with a
    hardcoded fallback default at the call site — e.g.
    an ``A2A_URL`` environment read defaulting to ``http://127.0.0.1:9016/a2a/`` in
    ``scripts/validate_a2a_agent.py``, or ``AGENT_UTILITIES_ROOT`` in the identical
    ``scripts/run_agent_utilities_gate.py`` shared byte-for-byte across 60+ packages in this
    fleet. Folding those reads into the documentable/UNDOCUMENTED surface too (tried first)
    demanded every package's *public* ``.env.example`` document dev-only gate-script knobs
    that a deployer of the package never sets — an UNDOCUMENTED false positive identical
    across the whole fleet, observed directly while fixing this blind spot. So ``script_reads``
    feeds only DEAD-suppression (``code_read``), never ``documentable``.

    Trade-off — this narrower fix still MISSES the inverse case: a var read ONLY by a script,
    genuinely undeclared anywhere, stays silently undocumented (no UNDOCUMENTED finding) even
    though a human running that script would need to discover the var by reading its source.
    That is judged the lesser risk versus forcing scaffolded dev tooling into every package's
    public config contract.
    """
    scripts_dir = root / "scripts"
    if not scripts_dir.is_dir():
        return set()
    return _scan_setting_calls(scripts_dir)


def _derive_toggle_vars(root: Path) -> set[str]:
    """``register_<tag>_tools`` → ``<TAG>TOOL`` (the framework's auto-derived toggle name).

    Scans only production code, not ``tests/`` — unit tests routinely define a throwaway
    ``def register_<fake_tag>_tools(mcp): pass`` test double to exercise the tool-surface
    dispatch mechanism generically (e.g. ``tests/unit/mcp/test_verbose_tools.py``,
    ``tests/mcp/test_check_env_var_drift.py``), which is not a real registrar and would
    otherwise mint a phantom ``<FAKE_TAG>TOOL`` var with no real toggle behind it.
    """
    tags: set[str] = set()
    for py in _walk_files(root, suffix=".py"):
        if "tests" in py.parts or "test" in py.parts:
            continue
        try:
            tags.update(_REGISTRAR.findall(py.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    # The shared surface helpers match the pattern but are not domain registrars.
    tags -= {"verbose", "tool_surface"}
    return {f"{t.upper()}TOOL" for t in tags}


@lru_cache(maxsize=1)
def _agent_utilities_reads() -> frozenset[str]:
    """``setting(...)`` literals read inside the installed agent-utilities core."""
    import agent_utilities

    au_root = Path(agent_utilities.__file__).resolve().parent
    return frozenset(_scan_setting_calls(au_root))


# CONCEPT:AU-OS.config.dynamic-env-family — a general, INSPECTABLE escape hatch for env
# names a function composes at runtime from a caller-supplied literal (e.g.
# ``agent_utilities.core.transport_security.resolve_tls_profile(service="mealie")``
# composes ``MEALIE_TLS_PROFILE``/``MEALIE_TLS_PROFILE_REF`` from ``service`` at runtime —
# no literal var name ever appears next to a ``setting()``/``getenv()`` call site for a
# static scan to see). Rather than hardcoding "mealie" (or any other service) into this
# checker, the COMPOSING function declares its own family via two attributes set right
# after its ``def`` (see ``transport_security.py`` for the canonical example):
#   func.dynamic_env_prefix_arg  -- the parameter name whose literal string value (given
#                                    positionally or by keyword at a call site) seeds the
#                                    prefix
#   func.dynamic_env_suffixes    -- the "_SUFFIX" family appended to
#                                    ``upper(re.sub(r"[^A-Za-z0-9]", "_", prefix)).strip("_")``
# This scanner discovers any function, anywhere in agent-utilities OR the package under
# check, that publishes this pair — a new dynamic-name family requires only publishing the
# declaration next to the function, never a checker change.
_DYNAMIC_ENV_PREFIX_ATTR = "dynamic_env_prefix_arg"
_DYNAMIC_ENV_SUFFIXES_ATTR = "dynamic_env_suffixes"
# func_name -> (prefix_arg_name, prefix_arg_position | None, suffix family)
_DynamicFamily = tuple[str, "int | None", tuple[str, ...]]


def _tuple_assign_targets_and_value(
    node: ast.AST,
) -> tuple[list[ast.expr] | None, ast.expr | None]:
    if isinstance(node, ast.Assign):
        return list(node.targets), node.value
    if isinstance(node, ast.AnnAssign):
        return ([node.target] if node.target else []), node.value
    return None, None


def _string_tuple_literal(value: ast.expr | None) -> list[str] | None:
    """All-string-literal elements of a Tuple/List AST node, or None if the
    node is not a Tuple/List or any element is not a literal string (all-or-
    nothing, matching the original break-on-first-miss)."""
    if not isinstance(value, (ast.Tuple, ast.List)):
        return None
    strings: list[str] = []
    for elt in value.elts:
        literal = _literal_str(elt)
        if literal is None:
            return None
        strings.append(literal)
    return strings or None


def _module_level_string_tuples(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    """``NAME = ("A", "B")``/``NAME: T = (...)`` module-level string-tuple constants, so a
    ``dynamic_env_suffixes`` declaration may reference a shared named constant instead of
    repeating the literal tuple at every function that shares one family."""
    out: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        targets, value = _tuple_assign_targets_and_value(node)
        if targets is None:
            continue
        strings = _string_tuple_literal(value)
        if strings is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                out[target.id] = tuple(strings)
    return out


def _function_param_order(tree: ast.AST) -> dict[str, list[str]]:
    """func name -> ordered positional-or-keyword parameter names (one module's functions)."""
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [a.arg for a in node.args.posonlyargs] + [
                a.arg for a in node.args.args
            ]
            out[node.name] = names
    return out


def _literal_str(node: ast.AST | None) -> str | None:
    """A plain string-constant value (no ``_ENV_NAME`` upper-case restriction — a dynamic
    family's seed literal, e.g. a lower-case service name like ``"mealie"``, is not itself
    an env-var name)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dynamic_prefix(literal: str) -> str:
    """The same sanitizer ``transport_security._service_prefix`` applies: upper-case,
    non-alphanumeric -> "_", strip leading/trailing "_"."""
    normalized = "".join(
        ch if ch.isalnum() else "_" for ch in str(literal or "").strip().upper()
    ).strip("_")
    return normalized


def _record_dynamic_suffixes(
    func_name: str,
    value: ast.expr,
    named_tuples: dict[str, tuple[str, ...]],
    suffix_sets: dict[str, tuple[str, ...]],
) -> None:
    if isinstance(value, ast.Name) and value.id in named_tuples:
        suffix_sets[func_name] = named_tuples[value.id]
    elif isinstance(value, (ast.Tuple, ast.List)):
        strings = [s for e in value.elts if (s := _literal_str(e)) is not None]
        if strings:
            suffix_sets[func_name] = tuple(strings)


def _record_dynamic_family_assign(
    target: ast.expr,
    value: ast.expr,
    named_tuples: dict[str, tuple[str, ...]],
    prefix_args: dict[str, str],
    suffix_sets: dict[str, tuple[str, ...]],
) -> None:
    if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)):
        return
    func_name = target.value.id
    if target.attr == _DYNAMIC_ENV_PREFIX_ATTR:
        literal = _literal_str(value)
        if literal:
            prefix_args[func_name] = literal
    elif target.attr == _DYNAMIC_ENV_SUFFIXES_ATTR:
        _record_dynamic_suffixes(func_name, value, named_tuples, suffix_sets)


def _dynamic_prefix_and_suffix_assigns(
    tree: ast.AST, named_tuples: dict[str, tuple[str, ...]]
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    prefix_args: dict[str, str] = {}
    suffix_sets: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            _record_dynamic_family_assign(
                target, node.value, named_tuples, prefix_args, suffix_sets
            )
    return prefix_args, suffix_sets


def _dynamic_families_from_assigns(
    prefix_args: dict[str, str],
    suffix_sets: dict[str, tuple[str, ...]],
    param_order: dict[str, list[str]],
) -> dict[str, _DynamicFamily]:
    families: dict[str, _DynamicFamily] = {}
    for func_name, prefix_arg in prefix_args.items():
        suffixes = suffix_sets.get(func_name)
        if not suffixes:
            continue
        params = param_order.get(func_name, [])
        position = params.index(prefix_arg) if prefix_arg in params else None
        families[func_name] = (prefix_arg, position, suffixes)
    return families


def _scan_dynamic_families(root: Path) -> dict[str, _DynamicFamily]:
    """Discover every ``func.dynamic_env_prefix_arg``/``func.dynamic_env_suffixes``
    declaration under *root* (see CONCEPT:AU-OS.config.dynamic-env-family above)."""
    families: dict[str, _DynamicFamily] = {}
    for py in _walk_files(root, suffix=".py"):
        if _should_skip_env_scan_file(root, py):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        named_tuples = _module_level_string_tuples(tree)
        param_order = _function_param_order(tree)
        prefix_args, suffix_sets = _dynamic_prefix_and_suffix_assigns(
            tree, named_tuples
        )
        families.update(
            _dynamic_families_from_assigns(prefix_args, suffix_sets, param_order)
        )
    return families


@lru_cache(maxsize=1)
def _agent_utilities_dynamic_families() -> dict[str, _DynamicFamily]:
    """Dynamic-env-family declarations published inside the installed agent-utilities
    core (cached — parsing the whole package on every ``analyze()`` call is wasteful)."""
    import agent_utilities

    au_root = Path(agent_utilities.__file__).resolve().parent
    return _scan_dynamic_families(au_root)


def _should_skip_env_scan_file(root: Path, py: Path) -> bool:
    """Shared file-skip guard for the AST-based env scanners
    (`_scan_dynamic_families`, `_scan_dynamic_family_reads`): a file under a
    non-runtime source dir (tests/, docs/, ...) or a self-documenting file
    (this module itself) is never a real call site."""
    try:
        relative_parts = py.relative_to(root).parts
    except ValueError:
        relative_parts = py.parts
    if any(part in _NON_RUNTIME_SOURCE_DIRS for part in relative_parts[:-1]):
        return True
    return py.name in _SELF_DOC_FILES


def _matched_dynamic_family(
    node: ast.Call, families: dict[str, _DynamicFamily]
) -> _DynamicFamily | None:
    """The (prefix_arg, position, suffixes) family this call node targets, or None."""
    func_name: str | None = None
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        func_name = node.func.attr
    return families.get(func_name) if func_name else None


def _resolve_dynamic_prefix_literal(
    node: ast.Call, prefix_arg: str, position: int | None
) -> str | None:
    literal: str | None = None
    for keyword in node.keywords:
        if keyword.arg == prefix_arg:
            literal = _literal_str(keyword.value)
            break
    if literal is None and position is not None and len(node.args) > position:
        literal = _literal_str(node.args[position])
    return literal


def _dynamic_family_reads_for_call(
    node: ast.Call, families: dict[str, _DynamicFamily]
) -> set[str]:
    """Concrete var names implied by ONE call node, or an empty set."""
    family = _matched_dynamic_family(node, families)
    if family is None:
        return set()
    prefix_arg, position, suffixes = family
    literal = _resolve_dynamic_prefix_literal(node, prefix_arg, position)
    if not literal:
        return set()
    prefix = _dynamic_prefix(literal)
    if not prefix:
        return set()
    return {f"{prefix}_{suffix}" for suffix in suffixes}


def _scan_dynamic_family_reads(
    root: Path, families: dict[str, _DynamicFamily]
) -> set[str]:
    """Every concrete var name implied by a call to a declared dynamic-family function
    with a literal prefix argument, e.g. ``resolve_configured_tls_profile("mealie")`` ->
    ``{"MEALIE_TLS_PROFILE", "MEALIE_TLS_PROFILE_REF"}``. A non-literal (runtime-computed)
    prefix argument cannot be resolved statically and is silently skipped — same limit as
    every other static scan in this module."""
    if not families:
        return set()
    found: set[str] = set()
    for py in _walk_files(root, suffix=".py"):
        if _should_skip_env_scan_file(root, py):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            found.update(_dynamic_family_reads_for_call(node, families))
    return found


def _is_scaffold_mcp_config(cfg: Path) -> bool:
    """agent_utilities/data/mcp_config.json is a packaged SCAFFOLDING template
    (a generic "example-mcp" server) shipped for other packages/tools to copy
    the schema from — not a launch config for agent-utilities' own servers, so
    its placeholder env keys (e.g. a bare "API_KEY") are never a real read here."""
    return "data" in cfg.parts and cfg.name == "mcp_config.json"


def _env_blocks_from_mcp_config(cfg: Path) -> list[tuple[Path, dict[str, str]]]:
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    blocks: list[tuple[Path, dict[str, str]]] = []
    for server in (data.get("mcpServers") or {}).values():
        env = server.get("env")
        if isinstance(env, dict):
            blocks.append((cfg, env))
    return blocks


def _dedupe_env_blocks(
    blocks: list[tuple[Path, dict[str, str]]],
) -> list[tuple[Path, dict[str, str]]]:
    # de-dup by resolved path while preserving the per-server granularity
    seen: set[tuple[str, frozenset[str]]] = set()
    uniq: list[tuple[Path, dict[str, str]]] = []
    for path, env in blocks:
        key = (str(path.resolve()), frozenset(env))
        if key not in seen:
            seen.add(key)
            uniq.append((path, env))
    return uniq


def _mcp_config_env_blocks(root: Path) -> list[tuple[Path, dict[str, str]]]:
    """Every ``mcpServers.<name>.env`` block across ``mcp_config*.json`` files."""
    blocks: list[tuple[Path, dict[str, str]]] = []
    for cfg in _walk_files(root, suffix=".json"):
        if not cfg.name.startswith("mcp_config"):
            continue
        if _is_scaffold_mcp_config(cfg):
            continue
        blocks.extend(_env_blocks_from_mcp_config(cfg))
    return _dedupe_env_blocks(blocks)


def _readme_example_region(text: str) -> str:
    """The README slice that holds the mcp_config examples (markers preferred, else the
    ``## MCP Configuration Examples`` heading up to the additional-deployment marker)."""
    if README_START in text and README_END in text:
        return text[text.index(README_START) : text.index(README_END)]
    if README_HEADING in text:
        start = text.index(README_HEADING)
        end = (
            text.index(README_ADDL)
            if README_ADDL in text and text.index(README_ADDL) > start
            else len(text)
        )
        return text[start:end]
    return ""


def _readme_example_env_blocks(root: Path) -> list[dict[str, str]]:
    """Every ``mcpServers.<name>.env`` dict in the README's fenced ```json examples."""
    readme = root / "README.md"
    if not readme.exists():
        return []
    region = _readme_example_region(readme.read_text(encoding="utf-8"))
    blocks: list[dict[str, str]] = []
    for m in _JSON_FENCE.finditer(region):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        for server in (data.get("mcpServers") or {}).values():
            env = server.get("env")
            if isinstance(env, dict):
                blocks.append(env)
    return blocks


def _compose_block_should_close(stripped: str, indent: int, ref_indent: int) -> bool:
    """A YAML-indentation heuristic shared by both compose scanners: does this
    line close the currently-open indented block (an ``environment:`` block
    for `_compose_env_keys`, an ``image:``/``command:``/``entrypoint:``/
    ``args:`` block for `_compose_subst_reads`)? A blank line never closes a
    block; a ``-`` list-item line never does either (it continues the block
    at the SAME nominal indent as its parent key)."""
    return bool(stripped) and indent <= ref_indent and not stripped.startswith("-")


def _compose_candidate_files(root: Path) -> list[Path]:
    return [*root.glob("*compose*.y*ml"), *root.glob("docker/*compose*.y*ml")]


def _env_keys_from_compose_file(comp: Path) -> set[str]:
    try:
        lines = comp.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    keys: set[str] = set()
    in_env = False
    env_indent = 0
    for raw in lines:
        stripped = raw.strip()
        if re.match(r"^environment\s*:", stripped):
            in_env = True
            env_indent = len(raw) - len(raw.lstrip())
            continue
        if not in_env:
            continue
        indent = len(raw) - len(raw.lstrip())
        if _compose_block_should_close(stripped, indent, env_indent):
            in_env = False
            continue
        m = _COMPOSE_ENV.match(raw)
        if m:
            keys.add(m.group(1))
    return keys


def _compose_env_keys(root: Path) -> dict[str, set[str]]:
    """Env keys referenced in each ``*compose*.yml`` ``environment:`` section.

    Skips ``_THIRD_PARTY_COMPOSE_FILES`` — recipe files that stand up a bundled
    THIRD-PARTY infra image (a database/broker/platform: postgres, neo4j, kafka,
    falkordb, egeria-platform, jena-fuseki) rather than agent-utilities' own service.
    Their ``environment:`` block configures that upstream image's own entrypoint
    script (e.g. the official postgres image reads ``POSTGRES_USER``/``_PASSWORD``/
    ``_DB`` to bootstrap itself) — it is never a code-read surface for THIS package,
    so cross-checking it against our ``setting()``/``Field(alias=...)`` surface would
    only ever produce false "DEAD" findings.
    """
    out: dict[str, set[str]] = {}
    for comp in _compose_candidate_files(root):
        if comp.name in _THIRD_PARTY_COMPOSE_FILES:
            continue
        keys = _env_keys_from_compose_file(comp)
        if keys:
            out[comp.name] = keys
    return out


# Compose keys whose scalar (``image: ${VAR}``) or list (``command:\n  - "${VAR}"``) value
# may hold a ``${VAR...}`` substitution that docker compose itself resolves from the
# deployer's shell/``.env`` at ``docker compose up`` time — a genuine read of that var, just
# not a Python ``setting()``/``getenv()`` call. Previously only ``environment:`` was scanned
# (see ``_compose_env_keys`` above), so ``image:
# ${MEDIA_DOWNLOADER_MCP_IMAGE:?set-...-to-image@sha256-digest}`` (pinning the image to an
# operator-supplied digest — CONCEPT:AU-OS.deployment.mutable-image-refs) was invisible and
# its var reported DEAD even though every ``docker compose up`` genuinely requires it.
# Deliberately scoped to these four keys (not e.g. ``labels:``/``volumes:``/``ports:``/
# resource-limit scalars like ``mem_limit:``/``cpus:``) — the task that surfaced this blind
# spot named exactly these; widening further is a known, bounded limitation, not a promise
# this scan makes.
_COMPOSE_SUBST_KEYS = frozenset({"image", "command", "entrypoint", "args"})
_COMPOSE_KEY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")


def _subst_var_names(text: str) -> set[str]:
    """Every ``${VAR...}`` substitution's leading identifier in *text* (handles the
    ``${VAR}``/``${VAR:-default}``/``${VAR:?message}``/``${VAR-default}``/``${VAR?message}``
    forms; a malformed/whitespace-padded one is still a real var reference here even though
    it is separately flagged ``MALFORMED_VALUE`` for mcp_config env blocks)."""
    names: set[str] = set()
    for m in _SUBST.finditer(text):
        inner = m.group(1).strip()
        name_m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", inner)
        if name_m and _ENV_NAME.fullmatch(name_m.group(1)):
            names.add(name_m.group(1))
    return names


def _subst_reads_from_compose_file(comp: Path) -> set[str]:
    try:
        lines = comp.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    found: set[str] = set()
    in_block = False
    block_indent = 0
    for raw in lines:
        stripped = raw.strip()
        key_match = _COMPOSE_KEY_LINE.match(stripped)
        if key_match and key_match.group(1) in _COMPOSE_SUBST_KEYS:
            in_block = True
            block_indent = len(raw) - len(raw.lstrip())
            inline_value = key_match.group(2)
            if inline_value:
                found.update(_subst_var_names(inline_value))
            continue
        if not in_block:
            continue
        indent = len(raw) - len(raw.lstrip())
        if _compose_block_should_close(stripped, indent, block_indent):
            in_block = False
            continue
        if stripped:
            found.update(_subst_var_names(raw))
    return found


def _compose_subst_reads(root: Path) -> set[str]:
    """Every ``${VAR...}`` substitution inside a compose ``image:``/``command:``/
    ``entrypoint:``/``args:`` value across ``*compose*.yml`` files (see
    ``_COMPOSE_SUBST_KEYS`` above). Skips ``_THIRD_PARTY_COMPOSE_FILES`` for the same
    reason ``_compose_env_keys`` does — a bundled third-party image's own launch
    configuration is not this package's code-read surface."""
    found: set[str] = set()
    for comp in _compose_candidate_files(root):
        if comp.name in _THIRD_PARTY_COMPOSE_FILES:
            continue
        found.update(_subst_reads_from_compose_file(comp))
    return found


def _is_framework_known(var: str, code_read: set[str]) -> bool:
    return (
        var in code_read
        or var in INHERITED_ENV
        or var in FRAMEWORK_EXTRA
        or var in RUNTIME_ALLOWLIST
        or var in TEST_FIXTURE_VARS
        or var.startswith(_RUNTIME_PREFIXES)
        or any(var.endswith(suf) for suf in _SAFE_SUFFIXES)
    )


def _compute_code_read_set(root: Path) -> tuple[set[str], set[str], set[str]]:
    """(pkg_reads, toggles, code_read) -- the package's own code-read surface,
    derived toggles, and the full code-read set (including framework/script
    reads) that DEAD-suppression checks against."""
    pkg_reads = _scan_setting_calls(root)
    toggles = _derive_toggle_vars(root)
    import agent_utilities

    package_root = Path(agent_utilities.__file__).resolve().parent
    resolved_root = root.resolve()
    framework_reads = (
        set()
        if package_root == resolved_root
        or resolved_root in package_root.parents
        or package_root in resolved_root.parents
        else set(_agent_utilities_reads())
    )

    # Dynamic/derived reads (CONCEPT:AU-OS.config.dynamic-env-family) — families may be
    # declared either in agent-utilities itself (the common case: a shared framework helper
    # like ``resolve_tls_profile``) or in the package's own code, so both are scanned; call
    # sites are only ever meaningful inside the package under check.
    dynamic_families = dict(_agent_utilities_dynamic_families())
    dynamic_families.update(_scan_dynamic_families(root))
    dynamic_reads = _scan_dynamic_family_reads(root, dynamic_families)

    # Compose ``image:``/``command:``/``entrypoint:``/``args:`` ``${VAR}`` substitutions —
    # a genuine read docker compose itself performs, not just ``environment:`` (see
    # ``_compose_subst_reads`` above). Folded into ``pkg_reads`` (not a separate framework-
    # only set) because the same "this package's own read surface" reasoning as any other
    # ``setting()``/``getenv()`` literal applies: it can suppress a false DEAD *and* raise a
    # genuine UNDOCUMENTED if the var is missing from ``.env.example``.
    pkg_reads = pkg_reads | _compose_subst_reads(root) | dynamic_reads

    # scripts/ reads (see ``_script_reads`` above) suppress a false DEAD but deliberately do
    # NOT widen ``pkg_reads``/``documentable`` — kept out of the var that feeds UNDOCUMENTED.
    script_reads = _script_reads(root)

    code_read = pkg_reads | toggles | framework_reads | script_reads
    return pkg_reads, toggles, code_read


def _resolve_declared_env(root: Path) -> set[str]:
    env_example = root / ".env.example"
    if not env_example.exists():
        return set()
    return {r[0] for r in parse_env_example(env_example.read_text(encoding="utf-8"))}


def _build_declared_sources(
    declared_env: set[str],
    mcp_blocks: list[tuple[Path, dict[str, str]]],
    compose: dict[str, set[str]],
    root: Path,
) -> dict[str, set[str]]:
    declared_sources: dict[str, set[str]] = {}
    for var in declared_env:
        declared_sources.setdefault(var, set()).add(".env.example")
    for path, env in mcp_blocks:
        for var in env:
            declared_sources.setdefault(var, set()).add(_rel(path, root))
    for name, keys in compose.items():
        for var in keys:
            declared_sources.setdefault(var, set()).add(f"docker/{name}")
    return declared_sources


def _dead_findings(
    declared_sources: dict[str, set[str]], code_read: set[str]
) -> list[dict]:
    findings = []
    for var, sources in sorted(declared_sources.items()):
        # placeholder template keys like <YOUR_X> never appear as A-Z names; skip secrets keys read.
        if _is_framework_known(var, code_read):
            continue
        # a strong "rename" hint: a sibling the code DOES read with the same stem
        hint = _rename_hint(var, code_read)
        findings.append(
            {
                "type": "DEAD",
                "var": var,
                "sources": sorted(sources),
                "hint": hint,
            }
        )
    return findings


def _should_skip_undocumented(var: str, declared_host_stems: set[str]) -> bool:
    # skip framework-inherited vars (shown in the inherited table)
    if var in INHERITED_ENV or var in FRAMEWORK_EXTRA:
        return True
    # skip system/OS/runtime vars the code reads for interop (e.g. PATH passthrough,
    # an XDG override) — same allowlist that already suppresses these from DEAD; it
    # must apply here too, or a runtime var flip-flops between DEAD and UNDOCUMENTED
    # instead of being silently accepted like the framework vars above.
    if var in RUNTIME_ALLOWLIST or var.startswith(_RUNTIME_PREFIXES):
        return True
    # skip synthetic config-machinery test fixtures (never real deployable config)
    if var in TEST_FIXTURE_VARS:
        return True
    # Skip a legacy/upstream child-process host alias whose canonical AU host
    # sibling is already documented (e.g. legacy LANGFUSE_HOST when the Langfuse
    # MCP adapter emits the child SDK's BASE_URL and LANGFUSE_BASE_URL is in
    # .env.example) — but never suppress a credential/toggle just because a host
    # of the same stem exists.
    return _is_host_var(var) and _stem(var) in declared_host_stems


def _undocumented_findings(
    pkg_reads: set[str], toggles: set[str], declared_env: set[str]
) -> list[dict]:
    # UNDOCUMENTED — code reads it but it's absent from .env.example (additive, safe).
    documentable = (pkg_reads | toggles) - declared_env
    declared_host_stems = {_stem(v) for v in declared_env if _is_host_var(v)}
    findings = []
    for var in sorted(documentable):
        if _should_skip_undocumented(var, declared_host_stems):
            continue
        findings.append(
            {"type": "UNDOCUMENTED", "var": var, "sources": ["(code)"], "hint": ""}
        )
    return findings


def _missing_tool_mode_findings(
    mcp_blocks: list[tuple[Path, dict[str, str]]], root: Path
) -> list[dict]:
    # MISSING_TOOL_MODE — a launch-style mcp_config env block without MCP_TOOL_MODE.
    findings = []
    for path, env in mcp_blocks:
        if "MCP_TOOL_MODE" not in env:
            findings.append(
                {
                    "type": "MISSING_TOOL_MODE",
                    "var": "MCP_TOOL_MODE",
                    "sources": [_rel(path, root)],
                    "hint": 'add "MCP_TOOL_MODE": "intent" to the env block',
                }
            )
    return findings


def _malformed_value_findings(
    mcp_blocks: list[tuple[Path, dict[str, str]]], root: Path
) -> list[dict]:
    # MALFORMED_VALUE — a whitespace-padded substitution like "${ VAR:-True }".
    findings = []
    for path, env in mcp_blocks:
        for var, value in env.items():
            if any(
                m.group(1) != m.group(1).strip() for m in _SUBST.finditer(str(value))
            ):
                findings.append(
                    {
                        "type": "MALFORMED_VALUE",
                        "var": var,
                        "sources": [_rel(path, root)],
                        "hint": 'use "${VAR:-default}" (no spaces inside the braces)',
                    }
                )
    return findings


def _agent_var_in_mcp_findings(
    mcp_blocks: list[tuple[Path, dict[str, str]]], root: Path, is_agent_only: Any
) -> list[dict]:
    # AGENT_VAR_IN_MCP — an agent-runtime var in an MCP-server config env block.
    findings = []
    for path, env in mcp_blocks:
        for var in sorted(env):
            if is_agent_only(var):
                findings.append(
                    {
                        "type": "AGENT_VAR_IN_MCP",
                        "var": var,
                        "sources": [_rel(path, root)],
                        "hint": "agent-only — move to the agent config (not the MCP server)",
                    }
                )
    return findings


def _readme_allowed_vars(
    pkg_reads: set[str], toggles: set[str], is_agent_only: Any
) -> set[str]:
    return {
        variable
        for variable in pkg_reads | toggles
        if variable not in INHERITED_ENV
        and variable not in FRAMEWORK_EXTRA
        and variable not in RUNTIME_ALLOWLIST
        and not variable.startswith(_RUNTIME_PREFIXES)
        and not any(variable.endswith(suffix) for suffix in _SAFE_SUFFIXES)
        and not is_agent_only(variable)
    } | {"MCP_TOOL_MODE", "TRANSPORT", "HOST", "PORT"}


def _readme_findings(root: Path, allowed: set[str], is_agent_only: Any) -> list[dict]:
    # README mcp_config examples — STALE_EXAMPLE + missing MCP_TOOL_MODE.
    findings = []
    for env in _readme_example_env_blocks(root):
        if "MCP_TOOL_MODE" not in env:
            findings.append(
                {
                    "type": "MISSING_TOOL_MODE",
                    "var": "MCP_TOOL_MODE",
                    "sources": ["README.md (example)"],
                    "hint": 'add "MCP_TOOL_MODE": "intent" to the example env block',
                }
            )
        for var in sorted(env):
            if var not in allowed:
                findings.append(
                    {
                        "type": "STALE_EXAMPLE",
                        "var": var,
                        "sources": ["README.md (example)"],
                        "hint": (
                            "agent-only — belongs in the agent config"
                            if is_agent_only(var)
                            else "not a code-read var — regenerate the examples"
                        ),
                    }
                )
    return findings


def analyze(root: Path) -> dict:
    """Compute the code-read set and diff the declared sets against it."""
    pkg_reads, toggles, code_read = _compute_code_read_set(root)

    declared_env = _resolve_declared_env(root)
    mcp_blocks = _mcp_config_env_blocks(root)
    compose = _compose_env_keys(root)
    declared_sources = _build_declared_sources(declared_env, mcp_blocks, compose, root)

    findings: list[dict] = []
    findings += _dead_findings(declared_sources, code_read)
    findings += _undocumented_findings(pkg_reads, toggles, declared_env)
    findings += _missing_tool_mode_findings(mcp_blocks, root)
    findings += _malformed_value_findings(mcp_blocks, root)
    findings += _agent_var_in_mcp_findings(mcp_blocks, root, is_agent_only)

    allowed = _readme_allowed_vars(pkg_reads, toggles, is_agent_only)
    findings += _readme_findings(root, allowed, is_agent_only)

    return {
        "package": root.resolve().name,
        "code_read_count": len(code_read),
        "findings": findings,
        "drift": len(findings),
    }


def _rename_hint(dead_var: str, code_read: set[str]) -> str:
    """If a read var shares this var's stem, the dead var is likely a rename of it."""
    stem = re.sub(r"(_URL|_BASE_URL|_HOST|_TOKEN|_KEY|TOOL)$", "", dead_var)
    if len(stem) < 3:
        return ""
    for read in code_read:
        if read != dead_var and read.startswith(stem):
            return f"likely rename of `{read}`"
    return ""


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _format(report: dict) -> str:
    lines = [f"env-var drift — {report['package']}: {report['drift']} finding(s)"]
    if not report["findings"]:
        lines.append("  ✓ clean (config matches the code-read env surface)")
        return "\n".join(lines)
    for f in report["findings"]:
        loc = ", ".join(f["sources"])
        hint = f"  → {f['hint']}" if f["hint"] else ""
        lines.append(f"  [{f['type']}] {f['var']}  ({loc}){hint}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect env-var/config drift.")
    parser.add_argument("root", nargs="?", default=".", help="Package root")
    parser.add_argument("--check", action="store_true", help="Exit 1 on any drift")
    parser.add_argument("--json", action="store_true", help="Emit JSON findings")
    args = parser.parse_args(argv)

    report = analyze(Path(args.root))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format(report))

    if args.check and report["drift"]:
        print(
            "\nenv-var drift detected — config/docs disagree with the code-read "
            "surface. Fix the sources above (the code is the source of truth).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
