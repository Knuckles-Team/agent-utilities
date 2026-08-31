"""Focused contract tests for Atlas's governed source catalogue."""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core import source_catalog
from agent_utilities.knowledge_graph.orchestration import engine_federation


class _Registry:
    def status(self):
        return {
            "connections": [
                {
                    "name": "pg-main",
                    "backend_type": "postgresql",
                    "connected": True,
                },
                {
                    "name": "puppygraph-prod",
                    "backend_type": "opencypher",
                    "connected": True,
                    "probed": True,
                },
                {
                    "name": "neo4j-prod",
                    "backend_type": "neo4j",
                    "connected": True,
                },
            ]
        }

    def export_specs(self):
        return [
            {
                "name": "pg-main",
                "backend_type": "postgresql",
                "connection_profile_ref": "vault://profiles/postgres",
                "tls_profile_ref": "vault://tls/postgres",
                # These values must never be projected by the catalogue.
                "host": "db.internal.example",
                "password": "do-not-project",
            },
            {
                "name": "puppygraph-prod",
                "backend_type": "opencypher",
                "connection_profile_ref": "secret://profiles/puppygraph",
            },
        ]


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _is_sensitive_key(value: str) -> bool:
    """Match sensitive field names without substring false positives.

    ``supported`` contains the substring ``port`` but is a harmless catalogue
    field.  Normalize the public camelCase projection first, then compare
    complete identifier components so this assertion protects the boundary
    without rejecting ordinary words.
    """

    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", value).casefold()
    parts = set(re.split(r"[_-]+", normalized))
    return bool(
        parts & {"endpoint", "dsn", "uri", "host", "port", "user", "password", "secret"}
    )


def test_catalog_projects_registries_with_truthful_provider_states(monkeypatch):
    monkeypatch.setattr(
        source_catalog,
        "_source_connector_types",
        lambda: ("database", "reader"),
    )
    config = SimpleNamespace(
        external_graph_connectors=[
            {
                "name": "puppygraph-prod",
                "source_alias": "puppygraph-prod",
                "backend": "opencypher",
                "connection_profile_ref": "vault://profiles/puppygraph",
            }
        ],
        provider_configs={
            "postgresql": {
                "enabled": True,
                "endpoint_ref": "env://PG_ENDPOINT",
                "tls_profile_ref": "vault://tls/postgres",
            }
        },
    )

    payload = source_catalog.build_source_catalog(config=config, registry=_Registry())
    providers = {item["provider"]: item for item in payload["sources"]}

    assert providers["postgresql"]["availability"] == "available"
    assert providers["postgresql"]["available"] is True
    assert providers["postgresql"]["connectionProfileRef"] == (
        "vault://profiles/postgres"
    )
    assert providers["puppygraph"]["availability"] == "available"
    assert providers["puppygraph"]["queryMode"] == "cypher"
    assert providers["neo4j"]["availability"] == "available"
    assert providers["neo4j"]["connectionNames"] == ["neo4j-prod"]
    assert providers["spark"]["sync"]["supported"] is False
    assert "compute-only" in providers["spark"]["sync"]["reason"]
    assert providers["teradata"]["availability"] == "unsupported"
    assert providers["teradata"]["available"] is False
    assert payload["querySurfaces"] == [
        {
            "id": "natural_language",
            "available": True,
            "tool": "graph_ask",
            "reason": "Graph-OS natural-language planning is exposed by graph_ask/nl_query",
        },
        {
            "id": "uql",
            "available": True,
            "tool": "engine_query",
            "action": "uql",
            "reason": "the engine_query uql action is the governed cross-modal query seam",
        },
    ]

    assert not any(_is_sensitive_key(key) for key in _walk_keys(payload))
    assert "db.internal.example" not in str(payload)
    assert "do-not-project" not in str(payload)


def test_catalog_requires_composition_registry(monkeypatch):
    """The projection must not reach upward to discover a process registry."""
    monkeypatch.setattr(source_catalog, "_source_connector_types", lambda: ())
    config = SimpleNamespace(
        external_graph_connectors=[],
        provider_configs={
            "postgresql": {
                "enabled": True,
                "endpoint_ref": "env://PG_ENDPOINT",
            }
        },
    )

    payload = source_catalog.build_source_catalog(config=config)
    postgresql = next(
        item for item in payload["sources"] if item["provider"] == "postgresql"
    )

    assert postgresql["availability"] == "unavailable"
    assert postgresql["available"] is False
    assert "connection registry" in postgresql["reason"]
    assert postgresql["connectionNames"] == ["postgresql"]


_MCP_MODULE = "agent_utilities.mcp"
_DYNAMIC_IMPORTS = frozenset({"__import__", "import_module"})


def _is_mcp_module(target: str) -> bool:
    return target == _MCP_MODULE or target.startswith(f"{_MCP_MODULE}.")


def _resolve_internal_module(
    target: str, *, required: bool = False
) -> tuple[str | None, str | None]:
    """Resolve an AU module, returning explicit discovery failures.

    ``importlib.util.find_spec`` applies Python's package/module resolution,
    unlike matching the spelling of an ``ImportFrom.module`` field.  Restrict
    resolution to this package so optional third-party imports do not make the
    architecture test depend on the full extras environment.

    A ``from package import name`` candidate may be an ordinary attribute, so
    a missing optional submodule is not itself unresolved.  The imported base
    module and every forbidden MCP candidate are required, however: a finder
    failure must not turn a known boundary edge into a clean result.
    """

    if target != "agent_utilities" and not target.startswith("agent_utilities."):
        return None, None
    must_resolve = required or _is_mcp_module(target)
    try:
        if importlib.util.find_spec(target) is not None:
            return target, None
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        if must_resolve:
            return None, f"{target!r} module discovery failed: {type(exc).__name__}"
        return None, None
    if must_resolve:
        return None, f"{target!r} module discovery returned no spec"
    return None, None


def _import_from_targets(
    node: ast.Import | ast.ImportFrom, *, package: str
) -> tuple[tuple[str | None, bool, str | None], ...]:
    """Resolve absolute and relative imports to real AU module candidates."""

    if isinstance(node, ast.Import):
        return tuple((alias.name, True, None) for alias in node.names)

    if node.level:
        raw = "." * node.level + (node.module or "")
        try:
            base = importlib.util.resolve_name(raw, package)
        except ImportError:
            return ((None, True, "relative import cannot be resolved"),)
    elif node.module:
        base = node.module
    else:
        return ()

    # ``from package import name`` imports ``package`` and, when ``name`` is
    # a submodule, also loads ``package.name``.  Keep both candidates and let
    # module resolution discard ordinary attributes.
    return (
        (base, True, None),
        *((f"{base}.{alias.name}", False, None) for alias in node.names),
    )


def _binding_names(target: ast.expr) -> set[str]:
    """Return simple names bound by an assignment target."""

    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.List, ast.Tuple)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_binding_names(item))
        return names
    return set()


def _contains_dynamic_reference(value: ast.expr, names: set[str]) -> bool:
    """Detect a direct dynamic-import callable reference in an expression."""

    return any(
        (isinstance(node, ast.Name) and node.id in names)
        or (isinstance(node, ast.Attribute) and node.attr in _DYNAMIC_IMPORTS)
        for node in ast.walk(value)
    )


def _assignment_bindings(
    tree: ast.AST,
) -> tuple[tuple[set[str], ast.expr], ...]:
    """Collect simple assignment names and their values."""

    bindings: list[tuple[set[str], ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            bindings.extend(
                (_binding_names(target), node.value) for target in node.targets
            )
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bindings.append((_binding_names(node.target), node.value))
    return tuple(bindings)


def _expand_dynamic_aliases(
    assignments: tuple[tuple[set[str], ast.expr], ...], names: set[str]
) -> frozenset[str]:
    """Close simple assignment aliases over the known dynamic callables."""

    while (
        aliases := {
            name
            for bound, value in assignments
            if _contains_dynamic_reference(value, names)
            for name in bound
        }
        - names
    ):
        names.update(aliases)
    return frozenset(names)


def _dynamic_import_names(tree: ast.AST) -> frozenset[str]:
    """Collect direct imports and simple assignment aliases for dynamic loads."""

    imported = {"importlib": "import_module", "builtins": "__import__"}
    names = set(_DYNAMIC_IMPORTS) | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and not node.level
        for alias in node.names
        if alias.name == imported.get(node.module or "")
    }

    # Resolve direct assignment chains (``loader = importlib.import_module``
    # and ``other = loader``) to a fixed point.  This is deliberately bounded
    # to simple names; unknown callable indirection remains fail-closed below.
    return _expand_dynamic_aliases(_assignment_bindings(tree), names)


def _dynamic_import_target(
    node: ast.Call, *, package: str, dynamic_names: frozenset[str]
) -> tuple[tuple[str | None, bool, str | None], ...]:
    """Resolve a literal dynamic import, or report an unresolved target.

    A variable-driven ``import_module``/``__import__`` call is an unknown
    edge, not evidence of no edge.  The caller keeps that distinction and
    fails the architecture assertion closed.
    """

    function_name = getattr(node.func, "id", getattr(node.func, "attr", None))
    if function_name not in dynamic_names | _DYNAMIC_IMPORTS:
        return ()

    argument = next(iter(node.args), None)
    if not isinstance(argument, ast.Constant):
        return ((None, True, "dynamic import target is not a string literal"),)
    target = argument.value
    if not isinstance(target, str) or not target:
        return ((None, True, "dynamic import target is not a module name"),)
    if target.startswith("."):
        package_kw = next((kw for kw in node.keywords if kw.arg == "package"), None)
        parent = getattr(getattr(package_kw, "value", None), "value", None)
        if not isinstance(parent, str):
            return ((None, True, "relative dynamic import has no literal package"),)
        try:
            target = importlib.util.resolve_name(target, parent)
        except ImportError:
            return ((None, True, "relative dynamic import cannot be resolved"),)
    return ((target, True, None),)


def _resolved_import_graph(
    path: Path, *, module_name: str
) -> tuple[set[tuple[str, str]], tuple[str, ...]]:
    """Build resolved AU import edges, including deferred imports.

    ``ast.walk`` intentionally visits imports nested in functions and
    ``TYPE_CHECKING`` blocks.  The eager-cycle checker is a separate gate; this
    assertion protects the stronger KG-to-MCP boundary at every import phase.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module_name.rpartition(".")[0]
    dynamic_names = _dynamic_import_names(tree)
    edges: set[tuple[str, str]] = set()
    unresolved: list[str] = []
    handlers = {
        ast.Import: lambda node: _import_from_targets(node, package=package),
        ast.ImportFrom: lambda node: _import_from_targets(node, package=package),
        ast.Call: lambda node: _dynamic_import_target(
            node, package=package, dynamic_names=dynamic_names
        ),
    }
    for node in ast.walk(tree):
        handler = handlers.get(type(node))
        if handler is None:
            continue
        for target, required, reason in handler(node):
            if reason is not None:
                unresolved.append(f"{path}:line {node.lineno}: {reason}")
                continue
            assert target is not None
            resolved, discovery_error = _resolve_internal_module(
                target, required=required
            )
            if discovery_error is not None:
                unresolved.append(f"{path}:line {node.lineno}: {discovery_error}")
            elif resolved is not None:
                edges.add((module_name, resolved))
    return edges, tuple(unresolved)


def test_catalog_and_federation_have_no_mcp_import_edge():
    """KG layers consume MCP-owned capabilities only through composition."""
    edges: set[tuple[str, str]] = set()
    unresolved: list[str] = []
    for module in (source_catalog, engine_federation):
        module_edges, module_unresolved = _resolved_import_graph(
            Path(module.__file__), module_name=module.__name__
        )
        edges.update(module_edges)
        unresolved.extend(module_unresolved)

    assert not unresolved, (
        "unresolved import targets cannot be treated as a clean boundary: "
        + ", ".join(unresolved)
    )
    forbidden = sorted(
        (source, target)
        for source, target in edges
        if target == "agent_utilities.mcp" or target.startswith("agent_utilities.mcp.")
    )
    assert not forbidden, f"resolved KG-to-MCP import edges: {forbidden}"


def test_import_graph_tracks_dynamic_aliases_and_unknown_targets(tmp_path):
    path = tmp_path / "dynamic_aliases.py"
    path.write_text(
        "from builtins import __import__ as builtin_loader\n"
        "from importlib import import_module as module_loader\n"
        'literal = module_loader("agent_utilities.mcp")\n'
        "def load_unknown(target):\n"
        "    return builtin_loader(target)\n"
        "def load_from_assignment(target):\n"
        "    loader = importlib.import_module\n"
        "    return loader(target)\n"
        "def load_builtin_from_assignment(target):\n"
        "    loader = __import__\n"
        "    return loader(target)\n",
        encoding="utf-8",
    )

    edges, unresolved = _resolved_import_graph(
        path,
        module_name="agent_utilities.knowledge_graph.orchestration.synthetic",
    )

    assert (
        "agent_utilities.knowledge_graph.orchestration.synthetic",
        _MCP_MODULE,
    ) in edges
    assert len(unresolved) == 3
    assert all("not a string literal" in item for item in unresolved)


def test_import_graph_accepts_literal_external_dynamic_alias(tmp_path):
    path = tmp_path / "known_good_dynamic_alias.py"
    path.write_text(
        'from importlib import import_module as module_loader\nmodule_loader("json")\n',
        encoding="utf-8",
    )

    edges, unresolved = _resolved_import_graph(
        path,
        module_name="agent_utilities.knowledge_graph.orchestration.synthetic",
    )

    assert not edges
    assert not unresolved


@pytest.mark.parametrize("discovery_failure", ["none", "error"])
def test_import_graph_reports_internal_discovery_failure(
    tmp_path, monkeypatch, discovery_failure
):
    path = tmp_path / "unavailable_module.py"
    path.write_text("from agent_utilities.mcp import kg_server\n", encoding="utf-8")
    if discovery_failure == "none":
        monkeypatch.setattr(importlib.util, "find_spec", lambda _target: None)
    else:

        def _raise(_target):
            raise ImportError("module finder unavailable")

        monkeypatch.setattr(importlib.util, "find_spec", _raise)

    edges, unresolved = _resolved_import_graph(
        path,
        module_name="agent_utilities.knowledge_graph.orchestration.synthetic",
    )

    assert not edges
    assert unresolved
    expected = (
        "module discovery returned no spec"
        if discovery_failure == "none"
        else "module discovery failed: ImportError"
    )
    assert all(expected in item for item in unresolved)


def test_generic_opencypher_requires_a_probe(monkeypatch):
    monkeypatch.setattr(source_catalog, "_source_connector_types", lambda: ())
    config = SimpleNamespace(
        external_graph_connectors=[
            {
                "name": "graph-read",
                "source_alias": "graph-read",
                "backend": "opencypher",
                "connection_profile_ref": "vault://profiles/graph",
            }
        ],
        provider_configs={},
    )

    payload = source_catalog.build_source_catalog(
        config=config,
        registry=SimpleNamespace(
            status=lambda: {"connections": []}, export_specs=lambda: []
        ),
    )
    opencypher = next(
        item for item in payload["sources"] if item["provider"] == "opencypher"
    )
    assert opencypher["availability"] == "unverified"
    assert opencypher["available"] is False
    assert "probe" in opencypher["reason"]


def test_provider_runtime_profile_projects_refs_without_resolving_values(monkeypatch):
    monkeypatch.setattr(source_catalog, "_source_connector_types", lambda: ())
    config = SimpleNamespace(
        external_graph_connectors=[],
        provider_configs={
            "postgres-primary": {
                "enabled": True,
                "endpoint_ref": "env://PG_ENDPOINT",
                "credential_refs": {"password": "secret://profiles/postgres"},
                "tls_profile_ref": "vault://tls/postgres",
                # A literal endpoint would be invalid AgentConfig, and must
                # never be copied if a permissive fixture supplies one.
                "endpoint": "postgresql://db.internal.example:5432/app",
            }
        },
    )

    payload = source_catalog.build_source_catalog(
        config=config,
        registry=SimpleNamespace(
            status=lambda: {"connections": []}, export_specs=lambda: []
        ),
    )
    postgresql = next(
        item for item in payload["sources"] if item["provider"] == "postgresql"
    )

    assert postgresql["availability"] == "configured"
    assert "connectionProfileRef" not in postgresql
    assert postgresql["authProfileRef"] == "secret://profiles/postgres"
    assert postgresql["tlsProfileRef"] == "vault://tls/postgres"
    assert "db.internal.example" not in str(postgresql)


def test_sync_preview_is_single_source_and_non_executable():
    preview = source_catalog.normalize_source_sync_preview(
        source="database",
        mode="full",
        ids_json='["row-1"]',
        connection="pg-main",
        graph="tenant-main",
    )

    assert preview == {
        "schema_version": "atlas-source-sync-preview.v1",
        "source": "database",
        "mode": "full",
        "ids": ["row-1"],
        "connection": "pg-main",
        "graph": "tenant-main",
        "entrypoint": "source_sync",
        "wouldExecute": False,
        "reason": "preview only; execute the normalized request through source_sync",
    }

    with pytest.raises(ValueError, match="exactly one source"):
        source_catalog.normalize_source_sync_preview(source="all")
    with pytest.raises(ValueError, match="mode"):
        source_catalog.normalize_source_sync_preview(source="database", mode="write")
    with pytest.raises(ValueError, match="neutral aliases"):
        source_catalog.normalize_source_sync_preview(
            source="database", connection="vault://profiles/postgres"
        )
