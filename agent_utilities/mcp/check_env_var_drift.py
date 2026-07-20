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

The code-read set =
  ``setting("VAR", …)`` reads in the package
  ∪ derived tool toggles: ``register_<tag>_tools`` → ``<TAG>TOOL``
  ∪ the inherited agent-utilities surface (``readme_env_vars.INHERITED_ENV`` + framework extras)
  ∪ ``setting("VAR", …)`` reads in agent-utilities core.

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

# Direct script execution puts ``agent_utilities/mcp`` first on ``sys.path``.
# Force this checkout's repository root ahead of any globally installed copy so
# the guard and its mutually importing env-source helper share one implementation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_utilities.mcp.readme_env_vars import INHERITED_ENV, parse_env_example

_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
# ``register_<tag>_tools`` — a condensed registrar; toggle env var is ``<TAG>TOOL``.
_REGISTRAR = re.compile(r"register_([a-z][a-z0-9_]*?)_tools\b")
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
# inference, configuration-source tests probing explicit process/XDG loading, and a
# stale probe in ``test_kg_autorouting.py``) — never real app config a deployer
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


def _env_names_from_tree(tree: ast.AST) -> set[str]:
    """Collect literal runtime configuration reads from one parsed module.

    AST inspection handles multiline calls and excludes lookalike snippets embedded
    in strings. It also distinguishes ``os.environ[...]`` reads from assignment
    writes, which a line-oriented regular expression cannot do reliably.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
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
            if (direct_reader or os_getenv or environ_get) and node.args:
                name = _literal_env_name(node.args[0])
                if name:
                    found.add(name)

            field_call = (
                isinstance(node.func, ast.Name)
                and node.func.id == "Field"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "Field"
            )
            if field_call:
                for keyword in node.keywords:
                    if keyword.arg == "alias":
                        name = _literal_env_name(keyword.value)
                        if name:
                            found.add(name)

        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and _is_environ(node.value)
        ):
            name = _literal_env_name(node.slice)
            if name:
                found.add(name)

        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == "enable_flag"
            or isinstance(target, ast.Attribute)
            and target.attr == "enable_flag"
            for target in targets
        ):
            name = _literal_env_name(value)
            if name:
                found.add(name)
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
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        found.update(_env_names_from_tree(tree))
    found.discard("")
    return found


def _derive_toggle_vars(root: Path) -> set[str]:
    """``register_<tag>_tools`` → ``<TAG>TOOL`` (the framework's auto-derived toggle name).

    Scans only production code, not ``tests/`` — unit tests routinely define a throwaway
    ``def register_<fake_tag>_tools(mcp): pass`` test double to exercise the tool-surface
    dispatch mechanism generically (e.g. ``tests/test_verbose_tools.py``,
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


def _mcp_config_env_blocks(root: Path) -> list[tuple[Path, dict[str, str]]]:
    """Every ``mcpServers.<name>.env`` block across ``mcp_config*.json`` files."""
    blocks: list[tuple[Path, dict[str, str]]] = []
    for cfg in _walk_files(root, suffix=".json"):
        if not cfg.name.startswith("mcp_config"):
            continue
        if "data" in cfg.parts and cfg.name == "mcp_config.json":
            # agent_utilities/data/mcp_config.json is a packaged SCAFFOLDING template
            # (a generic "example-mcp" server) shipped for other packages/tools to copy
            # the schema from — not a launch config for agent-utilities' own servers, so
            # its placeholder env keys (e.g. a bare "API_KEY") are never a real read here.
            continue
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for server in (data.get("mcpServers") or {}).values():
            env = server.get("env")
            if isinstance(env, dict):
                blocks.append((cfg, env))
    # de-dup by resolved path while preserving the per-server granularity
    seen: set[tuple[str, frozenset[str]]] = set()
    uniq: list[tuple[Path, dict[str, str]]] = []
    for path, env in blocks:
        key = (str(path.resolve()), frozenset(env))
        if key not in seen:
            seen.add(key)
            uniq.append((path, env))
    return uniq


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
    candidates = [
        *root.glob("*compose*.y*ml"),
        *root.glob("docker/*compose*.y*ml"),
    ]
    for comp in candidates:
        if comp.name in _THIRD_PARTY_COMPOSE_FILES:
            continue
        try:
            lines = comp.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        keys: set[str] = set()
        in_env = False
        env_indent = 0
        for raw in lines:
            stripped = raw.strip()
            if re.match(r"^environment\s*:", stripped):
                in_env = True
                env_indent = len(raw) - len(raw.lstrip())
                continue
            if in_env:
                indent = len(raw) - len(raw.lstrip())
                if stripped and indent <= env_indent and not stripped.startswith("-"):
                    in_env = False
                    continue
                m = _COMPOSE_ENV.match(raw)
                if m:
                    keys.add(m.group(1))
        if keys:
            out[comp.name] = keys
    return out


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


def analyze(root: Path) -> dict:
    """Compute the code-read set and diff the declared sets against it."""
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
    code_read = pkg_reads | toggles | framework_reads

    env_example = root / ".env.example"
    declared_env = (
        {r[0] for r in parse_env_example(env_example.read_text(encoding="utf-8"))}
        if env_example.exists()
        else set()
    )
    mcp_blocks = _mcp_config_env_blocks(root)
    compose = _compose_env_keys(root)

    findings: list[dict] = []

    # DEAD — declared somewhere but read by nothing and not a framework var.
    declared_sources: dict[str, set[str]] = {}
    for var in declared_env:
        declared_sources.setdefault(var, set()).add(".env.example")
    for path, env in mcp_blocks:
        for var in env:
            declared_sources.setdefault(var, set()).add(_rel(path, root))
    for name, keys in compose.items():
        for var in keys:
            declared_sources.setdefault(var, set()).add(f"docker/{name}")

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

    # UNDOCUMENTED — code reads it but it's absent from .env.example (additive, safe).
    documentable = (pkg_reads | toggles) - declared_env
    declared_host_stems = {_stem(v) for v in declared_env if _is_host_var(v)}
    for var in sorted(documentable):
        # skip framework-inherited vars (shown in the inherited table)
        if var in INHERITED_ENV or var in FRAMEWORK_EXTRA:
            continue
        # skip system/OS/runtime vars the code reads for interop (e.g. PATH passthrough,
        # an XDG override) — same allowlist that already suppresses these from DEAD; it
        # must apply here too, or a runtime var flip-flops between DEAD and UNDOCUMENTED
        # instead of being silently accepted like the framework vars above.
        if var in RUNTIME_ALLOWLIST or var.startswith(_RUNTIME_PREFIXES):
            continue
        # skip synthetic config-machinery test fixtures (never real deployable config)
        if var in TEST_FIXTURE_VARS:
            continue
        # Skip an upstream child-process host variable whose canonical AU host
        # sibling is documented (for example the Langfuse MCP adapter emits the
        # child SDK's BASE_URL from LANGFUSE_HOST). Never suppress a credential
        # or toggle merely because a host with the same stem exists.
        if _is_host_var(var) and _stem(var) in declared_host_stems:
            continue
        findings.append(
            {"type": "UNDOCUMENTED", "var": var, "sources": ["(code)"], "hint": ""}
        )

    # MISSING_TOOL_MODE — a launch-style mcp_config env block without MCP_TOOL_MODE.
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

    # env_sources imports this module, so defer the import to call time (no import cycle).
    from agent_utilities.mcp.env_sources import is_agent_only

    # MALFORMED_VALUE — a whitespace-padded substitution like "${ VAR:-True }".
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

    # AGENT_VAR_IN_MCP — an agent-runtime var in an MCP-server config env block.
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

    # README mcp_config examples — STALE_EXAMPLE + missing MCP_TOOL_MODE.
    allowed = {
        variable
        for variable in pkg_reads | toggles
        if variable not in INHERITED_ENV
        and variable not in FRAMEWORK_EXTRA
        and variable not in RUNTIME_ALLOWLIST
        and not variable.startswith(_RUNTIME_PREFIXES)
        and not any(variable.endswith(suffix) for suffix in _SAFE_SUFFIXES)
        and not is_agent_only(variable)
    } | {"MCP_TOOL_MODE", "TRANSPORT", "HOST", "PORT"}
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
