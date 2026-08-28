"""Connector Ontology Manifest discovery + compile-before-sync gate (D17).

Wires the C5 Connector Ontology Manifest (:mod:`connector_manifest`,
:mod:`manifest_compiler`, :mod:`ontology_integrity`) into the live
:func:`agent_utilities.knowledge_graph.core.source_sync.sync_source` entrypoint:
before a source's data is pulled, its owned ``connector_manifest.yml`` is required,
compiled, release-signature checked, and its ``provenance.integrity.hash``
re-verified. The boundary fails closed on a missing or hand-edited manifest, a
missing/changed installed preset provider, or a changed server/tool contract.

CONCEPT:AU-KG.ontology.connector-manifest-gate — this module is the ``source_sync``
wiring leg (D17); the CLI sweep gate lives in ``scripts/check_connector_manifests.py``
and shares the same :func:`check_manifest_bytes`/compile path so both surfaces agree.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "NATIVE_FINGERPRINT_FORMAT",
    "SOURCE_TO_CONNECTOR_PACKAGE",
    "native_activation_contract",
    "native_activation_fingerprint",
    "native_activation_fingerprints",
    "native_activation_fingerprint_modules",
    "MANDATORY_NAMED_CONNECTOR_SOURCES",
    "INTERNAL_INTROSPECTION_SOURCES",
    "INTERNAL_MANIFEST_EXEMPT_SOURCES",
    "mandatory_connector_packages",
    "bundled_provider_contract",
    "resolve_agents_root",
    "resolve_connector_package",
    "find_connector_manifest",
    "check_manifest_bytes",
    "undeclared_mutating_tools",
    "precheck_source",
    "manifest_required",
    "required_connector_sources",
]

# Curated aliases for the (common) cases where a ``_DELTA_HANDLERS``/``PACKAGE_PRESETS``
# source key doesn't literally match its ``agents/<pkg>`` directory name — e.g. two
# delta handlers (jira/confluence) share ONE connector package (atlassian-agent), and
# several handlers use the domain/vendor name rather than the repo directory name.
# Deliberately small and hand-verified; unknown sources fall through to the generic
# suffix-guessing in :func:`resolve_connector_package` rather than growing this table
# with a guess.
SOURCE_TO_CONNECTOR_PACKAGE: dict[str, str] = {
    "ard": "native-source-connectors",
    "arxiv": "native-source-connectors",
    "database": "native-source-connectors",
    "external_graph": "native-source-connectors",
    "filesystem": "native-source-connectors",
    "git_markdown": "native-source-connectors",
    "graphql_document": "native-source-connectors",
    "reader": "native-source-connectors",
    "rest": "native-source-connectors",
    "rss": "native-source-connectors",
    "web": "native-source-connectors",
    "leanix": "leanix-agent",
    "archivebox": "archivebox-api",
    "gitlab": "gitlab-api",
    "freshrss": "freshrss-agent",
    "jira": "atlassian-agent",
    "confluence": "atlassian-agent",
    "plane": "plane-agent",
    "dockerhub": "dockerhub-api",
    "langfuse": "langfuse-agent",
    "technitium": "technitium-dns-mcp",
    "tunnel_manager": "tunnel-manager",
    "uptime_kuma": "uptime-kuma-agent",
    "home_assistant": "home-assistant-agent",
    "twenty": "twenty-mcp",
    "audiobookshelf": "audiobookshelf-mcp",
    "firefly_iii": "firefly-iii-mcp",
    "paperless_ngx": "paperless-ngx-mcp",
    "gramps": "gramps-mcp",
    "camunda": "camunda-mcp",
    "aris": "aris-mcp",
    "egeria": "egeria-mcp",
    "servicenow": "servicenow-api",
}

# Suffix variants tried (in this order) when a source has no curated alias and no
# exact-name directory — mirrors the fleet's own ``*-mcp``/``*-agent``/``*-api``
# naming convention (CONCEPT:AU-KG.ontology.capability-node-aliases-lexical).
_GUESS_SUFFIXES: tuple[str, ...] = ("-mcp", "-agent", "-api", "-manager")

# Native activation surfaces that are not document-source connector classes.
# The direct property-graph importer writes ChangeEnvelopes itself, so pretending it
# implements ``source_connector_v1`` would create a false compatibility surface. Its
# real module and current-only interface are pinned in the same signed native bundle.
_NATIVE_DIRECT_ACTIVATION_CONTRACTS: dict[str, tuple[str, str]] = {
    "external_graph": (
        "external_graph_ingestion_v1",
        "agent_utilities.knowledge_graph.ingestion.external_graph",
    ),
}

# The native manifest cannot rely on a hash of the activation module alone: some
# security-critical collaborators are injected by the caller and therefore do not
# appear in that module's import graph.  These roots join the activation module
# before the deterministic local-import closure is calculated.  The resulting
# digest remains installation-path independent and does not import the dependencies.
_NATIVE_CRITICAL_FINGERPRINT_ROOTS: dict[str, tuple[str, ...]] = {
    "external_graph": (
        "agent_utilities.core.config",
        "agent_utilities.knowledge_graph.core.connection_registry",
        "agent_utilities.knowledge_graph.ingestion.change_envelope",
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest",
        "agent_utilities.knowledge_graph.ingestion.external_graph_schema",
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate",
        "agent_utilities.security.persistence_privacy",
        "agent_utilities.security.secrets_client",
    ),
    "graphql_document": (
        "agent_utilities.core.config",
        "agent_utilities.core.http_client",
        "agent_utilities.core.transport_security",
        "agent_utilities.knowledge_graph.ingestion.change_envelope",
        "agent_utilities.protocols.source_connectors.http_safety",
        "agent_utilities.security.persistence_privacy",
        "agent_utilities.security.secrets_client",
    ),
}

NATIVE_FINGERPRINT_FORMAT = "agent-utilities-local-module-closure-v1"
_LOCAL_PACKAGE = "agent_utilities"
_MAX_FINGERPRINT_MODULES = 4_096


def _local_package_root() -> Path:
    return Path(__file__).resolve().parents[2]


class _ImportVisitor(ast.NodeVisitor):
    """Find import statements while optionally excluding lazy function bodies."""

    def __init__(self, *, include_function_bodies: bool) -> None:
        self.include_function_bodies = include_function_bodies
        self.imports: list[ast.Import | ast.ImportFrom] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.imports.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.imports.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        if self.include_function_bodies:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        if self.include_function_bodies:
            self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        if self.include_function_bodies:
            self.generic_visit(node)


def _absolute_import_module(
    node: ast.ImportFrom,
    *,
    current_module: str,
    current_is_package: bool,
) -> str:
    if not node.level:
        return str(node.module or "")
    current_package = (
        current_module if current_is_package else current_module.rpartition(".")[0]
    )
    parts = current_package.split(".") if current_package else []
    trim = node.level - 1
    if trim >= len(parts):
        return ""
    base = parts[: len(parts) - trim]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _import_node_targets(node: ast.Import, package_name: str) -> set[str]:
    return {
        alias.name
        for alias in node.names
        if alias.name == package_name or alias.name.startswith(f"{package_name}.")
    }


def _import_from_targets(
    node: ast.ImportFrom,
    *,
    current_module: str,
    current_path: Path,
    package_name: str,
    resolve_module: Callable[[str], Path | None],
) -> set[str]:
    base = _absolute_import_module(
        node,
        current_module=current_module,
        current_is_package=current_path.name == "__init__.py",
    )
    if base != package_name and not base.startswith(f"{package_name}."):
        return set()
    targets = {base}
    base_path = resolve_module(base)
    if base_path is not None and base_path.name == "__init__.py":
        # ".".join(...) rather than an f-string: a Python dotted import
        # path derived from static AST analysis of the SCANNED module's
        # own source (never a query) — this two-part dotted shape is
        # otherwise indistinguishable from a schema-qualified table cast
        # at the AST level.
        targets.update(
            ".".join((base, alias.name)) for alias in node.names if alias.name != "*"
        )
    return targets


def _local_import_targets(
    tree: ast.AST,
    *,
    current_module: str,
    current_path: Path,
    package_name: str,
    resolve_module: Callable[[str], Path | None],
    include_function_bodies: bool,
) -> set[str]:
    visitor = _ImportVisitor(
        include_function_bodies=(
            include_function_bodies and current_path.name != "__init__.py"
        )
    )
    visitor.visit(tree)
    targets: set[str] = set()
    for node in visitor.imports:
        if isinstance(node, ast.Import):
            targets.update(_import_node_targets(node, package_name))
            continue
        targets.update(
            _import_from_targets(
                node,
                current_module=current_module,
                current_path=current_path,
                package_name=package_name,
                resolve_module=resolve_module,
            )
        )
    return targets


def _package_ancestors(module_name: str, *, package_name: str) -> tuple[str, ...]:
    parts = module_name.split(".")
    return tuple(
        ".".join(parts[:index])
        for index in range(1, len(parts))
        if ".".join(parts[:index]) == package_name
        or ".".join(parts[:index]).startswith(f"{package_name}.")
    )


def _resolve_local_module(
    module_name: str,
    *,
    package_name: str,
    package_root: Path,
    resolved_package_root: Path,
    path_cache: dict[str, Path | None],
) -> Path | None:
    if module_name in path_cache:
        return path_cache[module_name]
    if module_name == package_name:
        relative_parts: tuple[str, ...] = ()
    elif module_name.startswith(f"{package_name}."):
        relative_parts = tuple(module_name[len(package_name) + 1 :].split("."))
    else:
        path_cache[module_name] = None
        return None
    module_path = package_root.joinpath(*relative_parts).with_suffix(".py")
    package_path = package_root.joinpath(*relative_parts, "__init__.py")
    module_exists = module_path.is_file()
    package_exists = package_path.is_file()
    if module_exists and package_exists:
        raise RuntimeError(f"native module {module_name!r} is ambiguous")
    path = package_path if package_exists else module_path if module_exists else None
    if path is not None:
        try:
            path.resolve().relative_to(resolved_package_root)
        except ValueError as exc:
            raise RuntimeError(
                f"native module {module_name!r} escapes its package root"
            ) from exc
    path_cache[module_name] = path
    return path


def _cached_module_source(
    module_name: str,
    path: Path,
    source_cache: dict[str, tuple[bytes, ast.Module]],
) -> tuple[bytes, ast.Module]:
    cached = source_cache.get(module_name)
    if cached is not None:
        return cached
    try:
        source = path.read_bytes()
        tree = ast.parse(source, filename=module_name)
    except (OSError, SyntaxError, UnicodeError, ValueError) as exc:
        raise RuntimeError(
            f"native module {module_name!r} cannot be fingerprinted"
        ) from exc
    source_cache[module_name] = (source, tree)
    return source, tree


@dataclass
class _ClosureState:
    """Mutable BFS state for one local-module import-closure walk."""

    pending: deque[str]
    visited: set[str]
    missing: set[str]
    module_hashes: dict[str, str]
    required: set[str]


def _visit_closure_module(
    module_name: str,
    state: _ClosureState,
    *,
    resolve_module: Callable[[str], Path | None],
    source_cache: dict[str, tuple[bytes, ast.Module]],
    package_name: str,
    deep_roots: set[str],
) -> None:
    path = resolve_module(module_name)
    if path is None:
        if module_name in state.required:
            raise RuntimeError(
                f"native fingerprint root {module_name!r} is unavailable"
            )
        state.missing.add(module_name)
        return
    source, tree = _cached_module_source(module_name, path, source_cache)
    state.module_hashes[module_name] = hashlib.sha256(source).hexdigest()
    state.pending.extend(_package_ancestors(module_name, package_name=package_name))
    state.pending.extend(
        sorted(
            _local_import_targets(
                tree,
                current_module=module_name,
                current_path=path,
                package_name=package_name,
                resolve_module=resolve_module,
                include_function_bodies=module_name in deep_roots,
            )
        )
    )


def _local_module_closure_fingerprint(
    root_modules: tuple[str, ...],
    *,
    package_name: str,
    package_root: Path,
    deep_import_roots: frozenset[str] | None = None,
    module_path_cache: dict[str, Path | None] | None = None,
    module_source_cache: dict[str, tuple[bytes, ast.Module]] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Hash a deterministic local-Python import closure.

    File-system paths never enter the canonical payload. Missing package-local
    imports are recorded explicitly, so adding a formerly absent optional module
    also invalidates the signed fingerprint.
    """

    roots = tuple(sorted(set(root_modules)))
    if not roots:
        raise RuntimeError("native fingerprint roots are empty")
    resolved_package_root = package_root.resolve()
    deep_roots = set(deep_import_roots or roots)
    path_cache = module_path_cache if module_path_cache is not None else {}
    source_cache = module_source_cache if module_source_cache is not None else {}

    def _resolve_module(module_name: str) -> Path | None:
        return _resolve_local_module(
            module_name,
            package_name=package_name,
            package_root=package_root,
            resolved_package_root=resolved_package_root,
            path_cache=path_cache,
        )

    state = _ClosureState(
        pending=deque(roots),
        visited=set(),
        missing=set(),
        module_hashes={},
        required=set(roots),
    )

    while state.pending:
        module_name = state.pending.popleft()
        if module_name in state.visited:
            continue
        state.visited.add(module_name)
        if len(state.visited) > _MAX_FINGERPRINT_MODULES:
            raise RuntimeError("native module dependency closure is too large")
        _visit_closure_module(
            module_name,
            state,
            resolve_module=_resolve_module,
            source_cache=source_cache,
            package_name=package_name,
            deep_roots=deep_roots,
        )

    payload = {
        "format": NATIVE_FINGERPRINT_FORMAT,
        "roots": list(roots),
        "modules": dict(sorted(state.module_hashes.items())),
        "missing_local_imports": sorted(state.missing),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), tuple(sorted(state.module_hashes))


def _native_fingerprint_roots(
    source_type: str, *, activation_module: str
) -> tuple[str, ...]:
    normalized = str(source_type or "").strip().lower()
    return tuple(
        sorted(
            {
                activation_module,
                *_NATIVE_CRITICAL_FINGERPRINT_ROOTS.get(normalized, ()),
            }
        )
    )


def _native_activation_fingerprint_evidence(
    source_type: str,
    *,
    module_path_cache: dict[str, Path | None] | None = None,
    module_source_cache: dict[str, tuple[bytes, ast.Module]] | None = None,
) -> tuple[str, tuple[str, ...]]:
    normalized = str(source_type or "").strip().lower()
    contract = native_activation_contract(source_type)
    if contract is None:
        raise RuntimeError(f"native source {normalized!r} is unavailable")
    _interface, activation_module = contract
    return _local_module_closure_fingerprint(
        _native_fingerprint_roots(
            normalized,
            activation_module=activation_module,
        ),
        package_name=_LOCAL_PACKAGE,
        package_root=_local_package_root(),
        deep_import_roots=frozenset({activation_module}),
        module_path_cache=module_path_cache,
        module_source_cache=module_source_cache,
    )


def native_activation_fingerprint(source_type: str) -> str:
    """Fingerprint the source's activation code and local dependency closure."""

    digest, _modules = _native_activation_fingerprint_evidence(source_type)
    return digest


def native_activation_fingerprints(source_types: tuple[str, ...]) -> dict[str, str]:
    """Fingerprint an inventory against one deterministic source-code snapshot."""

    path_cache: dict[str, Path | None] = {}
    source_cache: dict[str, tuple[bytes, ast.Module]] = {}
    fingerprints: dict[str, str] = {}
    for source_type in sorted(set(source_types)):
        digest, _modules = _native_activation_fingerprint_evidence(
            source_type,
            module_path_cache=path_cache,
            module_source_cache=source_cache,
        )
        fingerprints[source_type] = digest
    return fingerprints


def native_activation_fingerprint_modules(source_type: str) -> tuple[str, ...]:
    """Return the auditable, path-free module inventory bound by the fingerprint."""

    _digest, modules = _native_activation_fingerprint_evidence(source_type)
    return modules


def native_activation_contract(source_type: str) -> tuple[str, str] | None:
    """Return ``(interface, implementation_module)`` for a native source."""

    normalized = str(source_type or "").strip().lower()
    direct = _NATIVE_DIRECT_ACTIVATION_CONTRACTS.get(normalized)
    if direct is not None:
        return direct
    from ...protocols.source_connectors.registry import get_connector_class

    connector_class = get_connector_class(normalized)
    if connector_class is None:
        return None
    return "source_connector_v1", connector_class.__module__


# Source aliases for connector packages whose source identifiers differ from
# their package names. Jira and Confluence, for example, share one connector
# package. The signed bundle makes every packaged connector mandatory, and
# operators cannot weaken or selectively extend this boundary. Each entry is
# whatever string :func:`precheck_source`/:func:`manifest_required` will actually
# receive as ``source`` — the ``sync_source`` source key where one is registered
# (``jira``/``confluence``/``gitlab``/``servicenow``/``leanix``/``langfuse``/
# ``tunnel_manager``), else the ``agents/<pkg>`` directory name itself (which
# :func:`resolve_connector_package` resolves via its own-name exact-match branch).
#
# L27 CLOSED (AU-P1-5, CONCEPT:AU-KG.ingest.envelope-atomic-transaction): the 5 that
# used to have no ``source_sync`` call site at all (``microsoft-agent``,
# ``container-manager-mcp``, ``documentdb-mcp``, ``repository-manager``,
# ``systems-manager``, ``vector-mcp``) now each have a live, dispatchable
# ``_DELTA_HANDLERS`` entry (``source_sync._sync_ops_mcp_connector`` + its 6 thin
# wrappers) — envelope-native from day one. All named connectors below are now
# gated on a LIVE ``sync_source`` code path, not just a name in this set.
MANDATORY_NAMED_CONNECTOR_SOURCES: frozenset[str] = frozenset(
    {
        # atlassian-agent — two source_sync source keys share one connector package
        "jira",
        "confluence",
        # gitlab-api
        "gitlab",
        # servicenow-api
        "servicenow",
        # leanix-agent
        "leanix",
        # langfuse-agent
        "langfuse",
        # tunnel-manager
        "tunnel_manager",
        # microsoft-agent — Graph API (email/Teams/SharePoint) minimal snapshot pull
        "microsoft-agent",
        # container-manager-mcp, documentdb-mcp, repository-manager, systems-manager,
        # vector-mcp — action/ops MCP connectors; each a minimal snapshot-pull
        # handler (source_sync._sync_ops_mcp_connector), gated by their own
        # agents/<pkg> directory name.
        "container-manager-mcp",
        "documentdb-mcp",
        "repository-manager",
        "systems-manager",
        "vector-mcp",
    }
)


# D17 gate exemption — INTERNAL INFRASTRUCTURE INTROSPECTION, not external supply
# chain (ingestion-hydration-program.md D17 gate fix).
#
# The compile-before-sync gate (:func:`precheck_source`/:func:`manifest_required`)
# exists to defend one specific threat model: an EXTERNAL, third-party data source
# silently drifting from the tool/field contract that was actually reviewed and
# signed off (AU-P0-4 fail-closed connector permissions). That model presumes an
# external maintainer, an external release cadence, and a trust boundary crossed
# at ingest time — which is exactly why every genuine connector package must carry
# a signed ``connector_manifest.yml`` before ``sync_source`` will pull its data.
#
# ``fleet`` / ``fleet_connectors`` are not that. They are ``source_sync``'s OWN
# capability-elevation probes (``_sync_fleet`` / ``_sync_fleet_connectors``,
# ``source_sync.py``): in-process code THIS repo ships, reading tool schemas off
# MCP servers THIS deployment already runs, so the resulting ``:Tool``/
# ``:MCPServer`` nodes describe infrastructure WE control. There is no external
# party to sign a manifest on behalf of, no upstream release to drift from, and no
# external data payload crossing a trust boundary — only our own already-trusted
# process introspecting our own already-trusted fleet. Requiring a signed manifest
# here would not add a security boundary; it would just make internal
# introspection permanently unsyncable, since no one owns an "external" release to
# certify.
#
# ``manifest_required()`` was widened to universal (commit 274d4c37, "every
# non-empty source is governed") to close the inverse gap — an unknown EXTERNAL
# connector being less restricted than an onboarded one. That widening did not
# consider internal introspection sources at all; it caught them as a side effect,
# not a considered decision. This set is the explicit, by-name carve-out for that:
# deliberately small and hand-verified (like :data:`SOURCE_TO_CONNECTOR_PACKAGE`
# above), NEVER pattern/prefix-matched, so nothing else silently rides along. Any
# other internal introspection source must be added here explicitly, with the same
# scrutiny, not inferred.
INTERNAL_INTROSPECTION_SOURCES: frozenset[str] = frozenset(
    {
        # source_sync._sync_fleet — probes the deployed MCP fleet's live tool
        # schemas into :Tool/:MCPServer nodes (KG-2.9 fleet capability elevation).
        "fleet",
        # source_sync._sync_fleet_connectors — the per-package connector-object
        # capability sweep alongside the fleet tool-schema probe above.
        "fleet_connectors",
    }
)

# Internal, in-process sources that never cross an external connector trust
# boundary. ``package_install`` only re-drives agent-utilities-owned ingestion
# primitives over already-installed, locally trusted packages; it has no remote
# provider, schema preset, or external payload for a connector manifest to pin.
INTERNAL_MANIFEST_EXEMPT_SOURCES: frozenset[str] = (
    INTERNAL_INTROSPECTION_SOURCES | frozenset({"package_install"})
)


def resolve_agents_root() -> Path:
    """The ``agent-packages/agents`` fleet root (``AGENTS_ROOT`` override, else
    ``WORKSPACE_PATH/agent-packages/agents``, else the dev-checkout-relative default)."""
    from ...core.config import setting

    override = (setting("AGENTS_ROOT", default="") or "").strip()
    if override:
        return Path(override)
    ws = (setting("WORKSPACE_PATH", default="") or "").strip()
    if ws:
        return Path(ws) / "agent-packages" / "agents"
    return Path(__file__).resolve().parents[3].parent / "agents"


def resolve_connector_package(
    source: str, *, agents_root: Path | None = None
) -> str | None:
    """The ``agents/<pkg>`` directory name for a ``source_sync`` source key, or ``None``.

    Tries, in order: the curated :data:`SOURCE_TO_CONNECTOR_PACKAGE` alias, an exact
    directory-name match, ``source`` with underscores→hyphens, then each of
    :data:`_GUESS_SUFFIXES` appended. Returns ``None`` (never a guess written back)
    when nothing on disk matches — the caller treats that as "not onboarded yet".
    """
    root = agents_root if agents_root is not None else resolve_agents_root()
    norm = (source or "").strip().lower()
    if not norm:
        return None

    alias = SOURCE_TO_CONNECTOR_PACKAGE.get(norm)
    if alias and (root / alias).is_dir():
        return alias

    candidates = [norm, norm.replace("_", "-")]
    candidates += [
        f"{c}{suf}" for c in (norm, norm.replace("_", "-")) for suf in _GUESS_SUFFIXES
    ]
    for cand in candidates:
        if (root / cand).is_dir():
            return cand
    return None


def bundled_manifests_root() -> Path:
    """The in-repo staging root for manifests bundled WITH agent-utilities itself
    (``agent_utilities/knowledge_graph/ontology/connector_manifests/<pkg>/``).

    AU-P1-6: the fleet's live ``agents/<pkg>`` checkouts are a *separate* set of
    repos this package doesn't own/write — this bundled copy is the pinned,
    committed-in-agent-utilities fallback that makes the complete signed fleet
    (including :data:`MANDATORY_NAMED_CONNECTOR_SOURCES`) resolvable even when
    :func:`resolve_agents_root` (the live fleet checkout) isn't present, e.g. a
    standalone agent-utilities checkout or CI runner. A live fleet checkout, if
    present, always wins (checked first in :func:`find_connector_manifest`) —
    this is the pinned floor, not an override.
    """
    return Path(__file__).resolve().parent / "connector_manifests"


def mandatory_connector_packages() -> frozenset[str]:
    """Every connector package shipped in the signed fleet bundle.

    The package list is derived from bundled artifacts instead of a hand-kept
    shortlist, so onboarding a provider and its signed manifest automatically
    makes its source presets fail closed in standalone wheels and GraphOS.
    """

    root = bundled_manifests_root()
    return frozenset(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "connector_manifest.yml").is_file()
    )


def _load_bundled_manifest(path: Path, normalized: str) -> tuple[Any, Any]:
    import yaml

    from .connector_manifest import ConnectorManifest

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = ConnectorManifest.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - path-free fail-closed boundary
        raise ValueError("bundled provider manifest is invalid") from exc
    if manifest.connector.casefold() != normalized:
        raise ValueError("bundled provider identity differs from its directory")
    return manifest, raw


def _validate_bundled_sync(
    sync: Any, presets: dict[str, dict[str, Any]], fingerprints: dict[str, str]
) -> tuple[str, str, str]:
    preset = str(sync.preset or "")
    tool = str(sync.tool or "")
    digest = str(sync.tool_schema_sha256 or "").strip().lower()
    if not preset or not tool or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("bundled provider sync contract is incomplete")
    if preset in presets:
        raise ValueError("bundled provider declares a duplicate preset")
    existing = fingerprints.get(tool)
    if existing is not None and existing != digest:
        raise ValueError("bundled provider declares conflicting tool fingerprints")
    return preset, tool, digest


def _bundled_provider_presets(
    manifest: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    presets: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    for sync in manifest.sync:
        preset, tool, digest = _validate_bundled_sync(sync, presets, fingerprints)
        presets[preset] = dict(sync.raw)
        fingerprints[tool] = digest
    return presets, fingerprints


def bundled_provider_contract(
    provider: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]] | None:
    """Return one release-pinned remote provider's preset/schema contract.

    GraphOS reaches the connector fleet over MCP; the connector distributions are
    therefore normally *not* installed in the GraphOS image.  The bundled signed
    manifest is the deployment-independent provider snapshot for that topology.
    Its complete-manifest signature and ``ontology.lock`` pin cover every raw
    preset and exact live-tool schema digest.  An installed provider remains
    authoritative and is checked separately by ``mcp_tool``; this fallback exists
    only when no local distribution owns the provider.

    ``None`` means no bundled provider exists.  A present but invalid bundle raises
    ``ValueError`` so callers fail closed rather than silently using a central
    compatibility preset.
    """

    normalized = (provider or "").strip().casefold()
    if not normalized:
        return None
    path = bundled_manifests_root() / normalized / "connector_manifest.yml"
    if not path.is_file():
        return None

    manifest, raw = _load_bundled_manifest(path, normalized)
    # Same split as `precheck_source` above: the property this fallback needs is
    # "this bundle is exactly what the release ledger recorded", which the pin
    # answers without a key. In-repo manifests carry no signature any more.
    pin_violations = _attestation_violations(
        manifest,
        label=_manifest_label(path),
        raw=raw if isinstance(raw, dict) else None,
    )
    if pin_violations:
        raise ValueError("bundled provider manifest is not release-pinned")

    return _bundled_provider_presets(manifest)


def find_connector_manifest(
    source: str, *, agents_root: Path | None = None
) -> Path | None:
    """The ``connector_manifest.yml`` path for ``source``'s connector package, if any.

    Checks the live fleet root first (``agents_root``/:func:`resolve_agents_root`),
    then falls back to :func:`bundled_manifests_root` (AU-P1-6) so the complete
    mandatory fleet resolves even without a sibling ``agent-packages`` checkout.
    """
    pkg = resolve_connector_package(source, agents_root=agents_root)
    if pkg is not None:
        root = agents_root if agents_root is not None else resolve_agents_root()
        path = root / pkg / "connector_manifest.yml"
        if path.exists():
            return path

    bundled_pkg = resolve_connector_package(
        source, agents_root=bundled_manifests_root()
    )
    if bundled_pkg is not None:
        bundled_path = bundled_manifests_root() / bundled_pkg / "connector_manifest.yml"
        if bundled_path.exists():
            return bundled_path

    return None


def check_manifest_bytes(
    path: Path,
    *,
    require_signature: bool = False,
    require_release_pin: bool = False,
    require_provider: bool = False,
    require_declared_actions: bool = False,
    agents_root: Path | None = None,
) -> list[str]:
    """Compile + integrity-check one manifest file; returns violations (empty = OK).

    ``agents_root`` (CA-32) is used ONLY by ``require_declared_actions``'s tool
    discovery (:func:`undeclared_mutating_tools`) — ``path`` itself may be a
    bundled in-repo manifest copy with no source tree next to it, so the
    package's real source root cannot be inferred from ``path`` alone.
    ``None`` (default) resolves via :func:`resolve_agents_root`.

    ``require_declared_actions`` (CA-32/DEC-CA-07, off by default) additionally
    runs :func:`undeclared_mutating_tools` against the manifest's own
    ``agents/<pkg>`` package and fails closed if any of its tools carry an
    explicit mutating signal (a ``{"mutating"}`` tag or a ``destructiveHint``/
    ``readOnlyHint`` annotation) with no matching ``actions[].id``. Off by
    default because it is a NEW invariant no shipped package has been swept
    for yet (CA-32 lane Non-goals): 8 of 72 packages already tag a tool
    mutating (a real, live ``annotations={"readOnlyHint": False, ...}``, not a
    hypothetical signal — CA-32-W01 fleet audit, re-measured against the live
    fleet) without declaring it in ``actions:``, so turning this on
    unconditionally today would correctly, but out-of-lane-scope, fail those
    8 (``audio-transcriber``, ``container-manager-mcp``, ``lakekeeper-mcp``,
    ``microsoft-agent``, ``opensearch-mcp``, ``spark-mcp``,
    ``systems-manager``, ``tunnel-manager``). The static scan only recognizes
    ``@mcp.tool(...)`` decorator syntax, not a ``mcp.tool(...)(func)``
    call-registration pattern (one further, known false-negative — e.g.
    ``genius-agent`` registers a mutating tool that way and is invisible to
    this scan); this rule is a best-effort signal, not an exhaustive one.
    Callers that want it enforced (CA-40..46's own CI, which starts clean, or
    a future fleet-wide sweep of the rest) opt in explicitly.

    Shares the exact compile/hash path :mod:`scripts.check_connector_manifests` uses
    (kept in sync deliberately — this is the ``source_sync``-side twin of that CLI
    gate, CONCEPT:AU-KG.ontology.connector-manifest-gate).

    ``require_signature`` and ``require_release_pin`` are two DIFFERENT questions
    that used to be answered by one code path:

    * ``require_signature`` — "did a release authority sign this, and does that
      signature match the pin?" Meaningful only for an artifact that LEAVES this
      repository (``release_signer_for_publication``). In-repo artifacts are no
      longer signed: git already supplies integrity and authorship for anything
      committed here.
    * ``require_release_pin`` — "is this manifest byte-for-byte the one the
      release ledger records?" Needs NO key, and is the check that actually
      catches a stale or hand-edited bundle. Crucially it covers the WHOLE
      document, including the ``sync`` preset/tool-schema block that
      ``provenance.integrity.hash`` (an ontology-graph hash) does not reach.

    Splitting them is what lets the runtime ingestion gate keep full-document
    tamper detection after in-repo signing was removed. Requiring a signature
    there instead would refuse every connector in the fleet, since every bundled
    manifest is now ``UNSIGNED-PREVIEW``.
    """
    return _check_manifest_bytes(
        path,
        require_signature=require_signature,
        require_release_pin=require_release_pin,
        require_provider=require_provider,
        require_declared_actions=require_declared_actions,
        agents_root=agents_root,
    )


def _manifest_label(path: Path) -> str:
    """Stable artifact label that never exposes a host-local checkout path."""
    return f"{path.parent.name}/connector_manifest.yml"


def _manifest_lock_entry(manifest: Any) -> dict[str, Any]:
    from . import ontology_integrity

    lock = ontology_integrity.load_lock(
        Path(__file__).resolve().parent.parent / "ontology.lock"
    )
    return dict(lock.get(f"agents/{manifest.connector}/connector_manifest.yml") or {})


def _release_pin_violations(
    manifest: Any, *, label: str, raw: dict[str, Any] | None = None
) -> list[str] | None:
    """Is this manifest byte-for-byte the one ``ontology.lock`` records?

    The keyless half of what :func:`_signature_violations` used to do in one
    step. It hashes the SAME canonical pre-image the release signature covered
    (:func:`ontology_integrity.canonical_manifest_hash` over the literal parsed
    document — see that function's note on why ``raw`` is preferred over a
    re-dumped model) and compares it with the ``manifest_hash`` the release
    ledger pins.

    This is the check that keeps the runtime ingestion gate honest now that
    in-repo manifests are unsigned. ``provenance.integrity.hash`` covers only
    the compiled ontology graph, so renaming a ``sync`` entry's ``tool`` — a
    change that redirects live ingestion — moves no ontology triple and passes
    that hash unchanged. It DOES move ``manifest_hash``. Losing this check is
    what would have made the unsigning refactor a real reduction in coverage;
    keeping it means only *provenance* was dropped, not *integrity*.

    Returns ``None`` when the release ledger has no entry for this connector at
    all — that is "this manifest is not one of ours", a different answer from
    "it is ours and it does not match", and :func:`_attestation_violations` is
    what decides whether the absence is acceptable.
    """
    from . import ontology_integrity

    pin = _manifest_lock_entry(manifest)
    pinned_hash = pin.get("manifest_hash")
    if not pinned_hash:
        return None
    manifest_hash = ontology_integrity.canonical_manifest_hash(
        raw if raw is not None else manifest
    )
    if pinned_hash != manifest_hash:
        return [
            f"[release-pin] {label}: complete manifest content differs from its "
            "release pin — the manifest was edited after the ledger was written, "
            "or the ledger is stale. Regenerate via "
            "scripts/update_ontology_lock.py."
        ]
    return []


def _attestation_violations(
    manifest: Any, *, label: str, raw: dict[str, Any] | None = None
) -> list[str]:
    """A manifest must be attested by AT LEAST ONE mechanism before activation.

    Two mechanisms exist and they cover different deployments:

    * an Ed25519 release SIGNATURE, for a provider distribution that reaches
      this deployment from outside the repository, and
    * the ``ontology.lock`` release PIN, for the 68 connector manifests this
      package bundles — which are unsigned by design since in-repo signing was
      removed (git already supplies integrity and authorship in-tree).

    Whichever the manifest actually carries is the one enforced, and carrying
    NEITHER is refused. There is no downgrade path: stripping a signature falls
    through to a pin the editor cannot forge (it lives in this package, not in
    the manifest), and forging a signature fails the trusted-signer check.

    Requiring a signature unconditionally here — which is what this path did
    before the split — refused every connector in the fleet the moment in-repo
    signing was removed, because every bundled manifest became
    ``UNSIGNED-PREVIEW``.
    """
    provenance = manifest.provenance
    if provenance.signer and provenance.signature:
        return _signature_violations(manifest, label=label, raw=raw)
    pin_violations = _release_pin_violations(manifest, label=label, raw=raw)
    if pin_violations is None:
        return [
            f"[attestation] {label}: manifest carries no release signature and "
            "ontology.lock records no manifest_hash for it, so nothing attests "
            "its contents. Either sign it for publication or record it in the "
            "release ledger (scripts/update_ontology_lock.py)."
        ]
    return pin_violations


def _pin_mismatch_violations(
    pin: dict[str, Any], manifest_hash: str, provenance: Any, label: str
) -> list[str]:
    if not pin.get("manifest_hash"):
        return []
    violations: list[str] = []
    if pin.get("manifest_hash") != manifest_hash:
        violations.append(
            f"[signature] {label}: complete manifest content differs from its release pin"
        )
    if pin.get("signer") != provenance.signer:
        violations.append(f"[signature] {label}: signer differs from its release pin")
    if pin.get("signature") != provenance.signature:
        violations.append(
            f"[signature] {label}: signature differs from its release pin"
        )
    if pin.get("signature_algorithm") != provenance.signature_algorithm:
        violations.append(
            f"[signature] {label}: algorithm differs from its release pin"
        )
    if pin.get("signing_public_key") != provenance.signing_public_key:
        violations.append(
            f"[signature] {label}: public key differs from its release pin"
        )
    return violations


def _signature_violations(
    manifest: Any, *, label: str, raw: dict[str, Any] | None = None
) -> list[str]:
    """Verify the full manifest against Ed25519 evidence and its release pin.

    Hashes ``raw`` (the literal YAML-parsed document) when the caller has it,
    rather than re-dumping the validated ``manifest`` model. A field added to
    :class:`ProvenanceSpec` (or any nested model) after a manifest was generated
    and signed is absent from that manifest's own YAML; ``model_validate`` fills
    it with the schema default, and ``model.model_dump()`` then writes that
    default back out, injecting a byte the original signer never saw into the
    hash pre-image -- a false "signature invalid" verdict for every already-
    signed manifest in the fleet the day such a field is added (measured:
    ``dependency_lock_digest``, GOC-16/BUG-234, broke all 68). Hashing the raw
    document instead is exact by construction: at generation time the signed
    hash is computed from a fully-populated model whose ``model_dump()`` output
    *is* what gets written to the YAML file, so the file's literal parsed
    content is always byte-identical to what was signed, for both an old
    manifest predating a field and a new one carrying it. A tampered document
    still changes the raw dict and still changes the hash -- this only removes
    the *reload* round-trip's opportunity to inject un-signed defaults; it does
    not widen what content is covered by the signature.
    """
    from . import ontology_integrity

    provenance = manifest.provenance
    if not provenance.signer or not provenance.signature:
        return [f"[signature] {label}: manifest is unsigned"]
    if provenance.signer not in ontology_integrity.DEFAULT_TRUSTED_SIGNERS:
        return [f"[signature] {label}: signer {provenance.signer!r} is not trusted"]

    manifest_hash = ontology_integrity.canonical_manifest_hash(
        raw if raw is not None else manifest
    )
    pin = _manifest_lock_entry(manifest)
    pinned_key = pin.get("signing_public_key")
    trusted_keys = (
        (str(pinned_key),)
        if isinstance(pinned_key, str) and pinned_key
        else ontology_integrity.release_trusted_public_keys()
    )
    if not ontology_integrity.verify_release_signature(
        manifest_hash,
        provenance.signature,
        signer_id=provenance.signer,
        algorithm=provenance.signature_algorithm,
        public_key=provenance.signing_public_key,
        trusted_public_keys=trusted_keys,
    ):
        return [
            f"[signature] {label}: Ed25519 release signature or trusted public-key "
            "pin is invalid"
        ]
    return _pin_mismatch_violations(pin, manifest_hash, provenance, label)


def _native_fingerprint_sidecar_pins(
    path: Path, label: str
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        sidecar = json.loads(
            (path.parent / "tool_schema_fingerprints.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None, [
            f"[tool-schema] {label}: native connector fingerprint sidecar is invalid"
        ]
    if (
        not isinstance(sidecar, dict)
        or sidecar.get("format") != NATIVE_FINGERPRINT_FORMAT
    ):
        return None, [
            f"[tool-schema] {label}: native connector fingerprint format is invalid"
        ]
    pins = sidecar.get("sources", {})
    if not isinstance(pins, dict) or not pins:
        return None, [
            f"[tool-schema] {label}: native connector fingerprint map is empty"
        ]
    return pins, []


def _native_sync_violations(
    sync: Any,
    *,
    label: str,
    pins: dict[str, Any],
    path_cache: dict[str, Path | None],
    source_cache: dict[str, tuple[bytes, ast.Module]],
) -> list[str]:
    source_type = str(sync.tool or "")
    contract = native_activation_contract(source_type)
    if contract is None:
        return [f"[provider] {label}: native source {source_type!r} is unavailable"]
    interface, _module_name = contract
    violations: list[str] = []
    expected_raw = {
        "source_type": source_type,
        "interface": interface,
    }
    if sync.server != "agent-utilities" or sync.raw != expected_raw:
        violations.append(
            f"[tool-schema] {label}: native preset {sync.preset!r} differs "
            "from its signed activation contract"
        )
    try:
        actual, _modules = _native_activation_fingerprint_evidence(
            source_type,
            module_path_cache=path_cache,
            module_source_cache=source_cache,
        )
    except (OSError, RuntimeError, SyntaxError, UnicodeError):
        violations.append(
            f"[provider] {label}: native source {source_type!r} code is unavailable"
        )
        return violations
    pinned = str(pins.get(source_type) or "")
    if not pinned or pinned != actual or sync.tool_schema_sha256 != pinned:
        violations.append(
            f"[tool-schema] {label}: native source {source_type!r} differs "
            "from its signed code fingerprint"
        )
    return violations


def _native_registry_coverage_violation(
    manifest: Any, declared: set[str], label: str
) -> str | None:
    # BUG-161 — a THIRD, independent cross-check against the LIVE registry, not
    # the sidecar. The manifest and its `tool_schema_fingerprints.json` sidecar
    # are written together by the SAME generator run
    # (`scripts/generate_native_connector_manifest.py`), so a source added to
    # :data:`SOURCE_TO_CONNECTOR_PACKAGE` (and thus actually registered/activated
    # in code) without re-running the generator leaves the manifest and sidecar
    # in lockstep with EACH OTHER while both silently drift from what the code
    # actually registers — the `declared != set(pins)` check above cannot see
    # this because it only ever compares the manifest to its own sidecar. This
    # needs no signing key (pure code/manifest comparison, like the fingerprint
    # check above), so it fires identically in an unsigned sandbox and in real
    # CI/release.
    live_registered = {
        source
        for source, package in SOURCE_TO_CONNECTOR_PACKAGE.items()
        if package == manifest.connector
    }
    missing = sorted(live_registered - declared)
    extra = sorted(declared - live_registered)
    if not missing and not extra:
        return None
    detail = []
    if missing:
        detail.append(f"missing={missing}")
    if extra:
        detail.append(f"extra={extra}")
    return (
        f"[coverage] {label}: native manifest and live registry inventory "
        f"differ ({', '.join(detail)})"
    )


def _native_provider_violations(manifest: Any, *, path: Path, label: str) -> list[str]:
    """Verify the in-package connector registry against its signed source pins."""

    pins, sidecar_violations = _native_fingerprint_sidecar_pins(path, label)
    if pins is None:
        return sidecar_violations

    violations: list[str] = []
    path_cache: dict[str, Path | None] = {}
    source_cache: dict[str, tuple[bytes, ast.Module]] = {}
    for sync in manifest.sync:
        violations.extend(
            _native_sync_violations(
                sync,
                label=label,
                pins=pins,
                path_cache=path_cache,
                source_cache=source_cache,
            )
        )
    declared = {str(sync.tool or "") for sync in manifest.sync}
    if declared != set(pins):
        violations.append(
            f"[tool-schema] {label}: native manifest and fingerprint inventory differ"
        )
    coverage_violation = _native_registry_coverage_violation(manifest, declared, label)
    if coverage_violation is not None:
        violations.append(coverage_violation)
    return violations


def _provider_schema_context(
    manifest: Any, label: str
) -> tuple[dict[str, Any] | None, dict[str, str] | None, list[str]]:
    from ...protocols.source_connectors.connectors.mcp_tool import (
        McpToolSourceError,
        provider_tool_presets,
        provider_tool_schema_fingerprints,
    )

    try:
        provider_presets = provider_tool_presets(manifest.connector)
    except McpToolSourceError as exc:
        return None, None, [f"[provider] {label}: {exc}"]
    if provider_presets is None:
        return (
            None,
            None,
            [
                f"[provider] {label}: source preset provider {manifest.connector!r} "
                "is not installed or cannot be resolved"
            ],
        )
    if not manifest.sync:
        return (
            None,
            None,
            [f"[tool-schema] {label}: mandatory connector declares no sync presets"],
        )

    try:
        live_schema_pins = provider_tool_schema_fingerprints(manifest.connector)
    except McpToolSourceError as exc:
        return None, None, [f"[tool-schema] {label}: {exc}"]
    if not live_schema_pins:
        return (
            None,
            None,
            [
                f"[tool-schema] {label}: connector-owned tool_schema_fingerprints.json "
                "is missing or empty"
            ],
        )
    return provider_presets, live_schema_pins, []


def _provider_sync_violations(
    sync: Any,
    *,
    label: str,
    provider_presets: dict[str, Any],
    live_schema_pins: dict[str, str],
) -> list[str]:
    actual = provider_presets.get(sync.preset)
    if actual is None:
        return [f"[tool-schema] {label}: provider is missing preset {sync.preset!r}"]
    violations: list[str] = []
    expected = dict(sync.raw)
    if json.dumps(actual, sort_keys=True, separators=(",", ":")) != json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ):
        violations.append(
            f"[tool-schema] {label}: provider preset {sync.preset!r} "
            "differs from the signed manifest"
        )
    if not sync.server or not sync.tool:
        violations.append(
            f"[tool-schema] {label}: preset {sync.preset!r} has no server/tool"
        )
        return violations
    pinned = live_schema_pins.get(sync.tool)
    if not pinned:
        violations.append(
            f"[tool-schema] {label}: connector sidecar has no fingerprint for "
            f"tool {sync.tool!r}"
        )
    if not sync.tool_schema_sha256:
        violations.append(
            f"[tool-schema] {label}: signed preset {sync.preset!r} has no "
            "tool_schema_sha256"
        )
    elif pinned and pinned != sync.tool_schema_sha256:
        violations.append(
            f"[tool-schema] {label}: connector sidecar fingerprint for "
            f"{sync.tool!r} differs from the signed manifest"
        )
    return violations


def _provider_violations(manifest: Any, *, path: Path, label: str) -> list[str]:
    """Require the installed provider to match every manifest sync preset exactly."""
    if manifest.connector == "native-source-connectors":
        return _native_provider_violations(manifest, path=path, label=label)
    provider_presets, live_schema_pins, violations = _provider_schema_context(
        manifest, label
    )
    if provider_presets is None or live_schema_pins is None:
        return violations
    for sync in manifest.sync:
        violations.extend(
            _provider_sync_violations(
                sync,
                label=label,
                provider_presets=provider_presets,
                live_schema_pins=live_schema_pins,
            )
        )
    return violations


# ── CA-32 / DEC-CA-07: undeclared-mutating-tool gate rule ──────────────────
#
# 2026-08-26 fleet audit (CA-32-W01, re-measured against the live fleet
# post-rebase): most ``agents/<pkg>`` tool registrations only pass a
# domain-grouping ``tags={"documents"}``-style set, but the signal this rule
# looks for is REAL and already live in the fleet — e.g.
# ``audio-transcriber``'s ``transcribe_audio``/``transcribe_media`` and seven
# other packages already pass a standard MCP ``annotations={"readOnlyHint":
# False, ...}``/``{"destructiveHint": True}`` (8/72 packages, 0 using a
# ``{"mutating"}`` tag, as measured: ``audio-transcriber``,
# ``container-manager-mcp``, ``lakekeeper-mcp``, ``microsoft-agent``,
# ``opensearch-mcp``, ``spark-mcp``, ``systems-manager``, ``tunnel-manager``).
# None of those 8 packages' ``connector_manifest.yml`` declares those tools
# under ``actions:`` today (the block holds only the generic a2a-capability
# pair, CA-32-W01) — a real, correct violation this rule WOULD raise the
# moment it runs, not a false positive. It stays fail-OPEN-by-default (see
# ``require_declared_actions`` below) precisely because retroactively fixing
# those 8 (or the other 64 packages' undeclared CRUD actions) is a fleet-wide
# sweep out of THIS lane's scope (CA-32 lane file Non-goals: "Editing any
# package's manifest") — CA-40..46 opt new packages into it from day one
# instead (DEC-CA-07 "Consequences"). The scan is decorator-only (see the
# ``require_declared_actions`` docstring above); a package that registers
# tools via ``mcp.tool(...)(func)`` call syntax instead of ``@mcp.tool(...)``
# is a further, known false-negative (e.g. ``genius-agent``).
#
# Static/AST only, deliberately mirroring how ``scripts/gen_mcp_fleet_registry.
# discover()`` already reads a sibling connector's source text without
# importing it: connector packages are separate repos/venvs this package does
# not install.
_MUTATING_TOOL_TAG = "mutating"


def _mcp_tool_decorator_call(node: ast.expr) -> ast.Call | None:
    """``node`` itself, narrowed to :class:`ast.Call`, when it is a decorator
    call shaped like ``<anything>.tool(...)`` — else ``None``.

    Returns the narrowed node (not a bare bool) so callers can access
    ``.keywords`` on the result without re-asserting ``isinstance`` — mypy
    cannot narrow a caller's variable from a helper's boolean return alone.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if (isinstance(func, ast.Attribute) and func.attr == "tool") or (
        isinstance(func, ast.Name) and func.id == "tool"
    ):
        return node
    return None


def _literal_set_contains(node: ast.expr | None, target: str) -> bool:
    if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return False
    return any(
        isinstance(elt, ast.Constant) and elt.value == target for elt in node.elts
    )


def _bool_constant(node: ast.expr | None, *, expected: bool) -> bool:
    return isinstance(node, ast.Constant) and node.value is expected


def _dict_literal_pairs(node: ast.Dict) -> list[tuple[str, ast.expr]]:
    return [
        (key.value, value)
        for key, value in zip(node.keys, node.values, strict=False)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]


def _call_keyword_pairs(node: ast.Call) -> list[tuple[str, ast.expr]]:
    return [(kw.arg, kw.value) for kw in node.keywords if kw.arg]


def _annotation_pairs(node: ast.expr | None) -> list[tuple[str, ast.expr]]:
    if isinstance(node, ast.Dict):
        return _dict_literal_pairs(node)
    if isinstance(node, ast.Call):
        return _call_keyword_pairs(node)
    return []


def _annotations_mark_mutating(node: ast.expr | None) -> bool:
    """``annotations={"destructiveHint": True}`` / ``{"readOnlyHint": False}``,
    as either a dict literal or a ``ToolAnnotations(...)`` call — both are
    valid ``fastmcp`` ``@mcp.tool(annotations=...)`` shapes."""
    for name, value in _annotation_pairs(node):
        if name == "destructiveHint" and _bool_constant(value, expected=True):
            return True
        if name == "readOnlyHint" and _bool_constant(value, expected=False):
            return True
    return False


def _is_explicitly_mutating_tool(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    for decorator in node.decorator_list:
        call = _mcp_tool_decorator_call(decorator)
        if call is None:
            continue
        for kw in call.keywords:
            if kw.arg == "tags" and _literal_set_contains(kw.value, _MUTATING_TOOL_TAG):
                return True
            if kw.arg == "annotations" and _annotations_mark_mutating(kw.value):
                return True
    return False


_TOOL_SCAN_EXCLUDED_DIR_NAMES = frozenset(
    {".venv", "venv", "tests", "test", "build", "dist", "__pycache__", ".git"}
)


def _mutating_tool_names_in_file(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()
    return {
        candidate.name
        for candidate in ast.walk(tree)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _is_explicitly_mutating_tool(candidate)
    }


def undeclared_mutating_tools(
    pkg: str,
    *,
    agents_root: Path | None = None,
    manifest: Any = None,
) -> list[str]:
    """Names of ``pkg``'s own explicitly-mutating-tagged MCP tools that are
    NOT declared in ``manifest.actions[].id`` (DEC-CA-07).

    Static AST scan of every ``*.py`` file under ``agents_root/pkg`` (never a
    live import — see the module comment above this function). Returns ``[]``
    when the package directory or the manifest is absent, or when nothing in
    it is explicitly tagged/annotated mutating — the fail-open-on-no-signal
    half of the contract.
    """
    root = agents_root if agents_root is not None else resolve_agents_root()
    pkg_root = root / pkg
    if not pkg_root.is_dir():
        return []
    declared = {str(action.id) for action in (getattr(manifest, "actions", None) or ())}
    mutating: set[str] = set()
    for path in sorted(pkg_root.rglob("*.py")):
        if _TOOL_SCAN_EXCLUDED_DIR_NAMES.intersection(path.parts):
            continue
        mutating.update(_mutating_tool_names_in_file(path))
    return sorted(mutating - declared)


def _undeclared_action_violations(
    manifest: Any, *, path: Path, label: str, agents_root: Path | None = None
) -> list[str]:
    """Note: ``path`` is deliberately NOT used to infer the source tree here.

    ``path`` may be a live ``agents/<pkg>/connector_manifest.yml`` OR a
    bundled in-repo copy under :func:`bundled_manifests_root` (no source, YAML
    only) — ``path.parent.parent`` is only correct for the former. Tool
    discovery always needs the package's real source, so this defers to
    :func:`undeclared_mutating_tools`'s own :func:`resolve_agents_root`
    default (or an explicit override) keyed on ``manifest.connector`` instead.
    """
    undeclared = undeclared_mutating_tools(
        manifest.connector, agents_root=agents_root, manifest=manifest
    )
    if not undeclared:
        return []
    names = ", ".join(repr(name) for name in undeclared)
    return [
        f"[actions] {label}: tool(s) {names} are registered with an explicit "
        "mutating signal (tags={'mutating'} or annotations={'destructiveHint': "
        "True}/{'readOnlyHint': False}) but not declared in actions[] — add "
        "an ActionSpec with a matching id (DEC-CA-07)."
    ]


def _load_and_validate_manifest(path: Path, label: str) -> tuple[Any, Any, list[str]]:
    import yaml

    from .connector_manifest import ConnectorManifest

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = ConnectorManifest.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        return (
            None,
            None,
            [
                f"[schema] {label}: does not parse/validate as a ConnectorManifest "
                f"({type(exc).__name__})"
            ],
        )
    return manifest, data, []


def _compiled_manifest_graph(manifest: Any, label: str) -> tuple[Any, list[str]]:
    from .manifest_compiler import compile_manifest, export_manifest_ttl

    try:
        spec = compile_manifest(manifest)
        ttl = export_manifest_ttl(spec, source=manifest.resolved_ontology_source)
        import rdflib

        g = rdflib.Graph()
        g.parse(data=ttl, format="turtle")
    except ImportError:
        # rdflib is deliberately excluded from the lean `serving` plane
        # (see KG-2.242) and lives only in the `[owl]` extra — but THIS gate
        # (the mandatory D17 compile-before-sync check for the signed connector
        # fleet) has no engine-native fallback and
        # genuinely needs it to parse/hash the compiled ontology. Degrade to one
        # clear, actionable line instead of a bare ModuleNotFoundError bubbling
        # up as "manifest does not compile cleanly".
        return None, [
            f"[dependency] {label}: the connector-manifest compile-before-sync "
            "gate needs rdflib to parse/hash the compiled ontology, and it is "
            "not installed on this deployment — install the 'owl' extra "
            "(pip install 'agent-utilities[owl]', or add it to this service's "
            "image) to enable manifest-gated sync for this source."
        ]
    except Exception as exc:  # noqa: BLE001
        return None, [
            f"[compile] {label}: manifest does not compile cleanly "
            f"({type(exc).__name__})"
        ]
    return g, []


def _attestation_gate_violations(
    manifest: Any,
    *,
    label: str,
    raw: Any,
    require_signature: bool,
    require_release_pin: bool,
) -> list[str]:
    normalized_raw = raw if isinstance(raw, dict) else None
    if require_signature:
        return _signature_violations(manifest, label=label, raw=normalized_raw)
    if require_release_pin:
        return _attestation_violations(manifest, label=label, raw=normalized_raw)
    return []


def _check_manifest_bytes(
    path: Path,
    *,
    require_signature: bool = False,
    require_release_pin: bool = False,
    require_provider: bool = False,
    require_declared_actions: bool = False,
    agents_root: Path | None = None,
) -> list[str]:
    """Implementation shared by runtime and direct hash-only callers."""
    from . import ontology_integrity

    label = _manifest_label(path)
    manifest, data, schema_violations = _load_and_validate_manifest(path, label)
    if manifest is None:
        return schema_violations

    g, compile_violations = _compiled_manifest_graph(manifest, label)
    if g is None:
        return compile_violations

    violations: list[str] = []
    digest, triple_count = ontology_integrity.canonical_hash(g)
    if digest != manifest.provenance.integrity.hash:
        violations.append(
            f"[integrity] {label}: recomputed hash {digest} (n={triple_count}) != "
            f"provenance.integrity.hash {manifest.provenance.integrity.hash} — "
            "the manifest was hand-edited after signing, or is stale. Regenerate via "
            "scripts/generate_connector_manifests.py."
        )
    violations.extend(
        _attestation_gate_violations(
            manifest,
            label=label,
            raw=data,
            require_signature=require_signature,
            require_release_pin=require_release_pin,
        )
    )
    if require_provider:
        violations.extend(_provider_violations(manifest, path=path, label=label))
    if require_declared_actions:
        violations.extend(
            _undeclared_action_violations(
                manifest, path=path, label=label, agents_root=agents_root
            )
        )
    return violations


def required_connector_sources() -> set[str]:
    """Certified source inventory used for reporting and bundle coverage.

    Enforcement itself is universal: :func:`manifest_required` governs every
    non-empty source, including sources not yet present in this inventory.
    """
    return set(MANDATORY_NAMED_CONNECTOR_SOURCES) | set(mandatory_connector_packages())


def manifest_required(source: str) -> bool:
    """Return whether an external source must carry a certified manifest.

    Every non-empty EXTERNAL source is governed.  The former allowlist/pass-through
    model made an unknown connector less restricted than an onboarded one, which is
    the inverse of a fail-closed supply-chain boundary.  The enterprise source
    catalog remains useful for inventory reporting, but it no longer controls
    enforcement.

    :data:`INTERNAL_MANIFEST_EXEMPT_SOURCES` is the explicit carve-out for
    internal infrastructure self-introspection and the in-process package
    reconciliation orchestrator. Neither crosses a third-party connector trust
    boundary, so the external-supply-chain threat model this gate defends does
    not apply. Every other source, including unknown/unbundled ones, remains
    governed exactly as before.
    """
    normalized = (source or "").strip().lower()
    if not normalized:
        return False
    return normalized not in INTERNAL_MANIFEST_EXEMPT_SOURCES


def precheck_source(source: str, *, agents_root: Path | None = None) -> dict[str, Any]:
    """The ``sync_source`` compile-before-sync gate (D17).

    A source with no discoverable ``connector_manifest.yml`` is rejected.  There
    is no runtime pass-through for an unowned connector; development fixtures must
    inject a signed test bundle or stub this gate explicitly.  When a manifest is
    found, returns
    ``{"checked": True, "ok": bool, "connector": str, "manifest_path": str,
    "violations": [...]}`` — fail-closed: ``ok=False`` means the caller MUST
    refuse to sync rather than pull data through a manifest that doesn't match
    what was actually reviewed/signed.

    CONCEPT:AU-P0-4 fail-closed connector permissions — missing, unsigned,
    providerless, or schema-drifted bundles all produce the same refusal contract.

    :data:`INTERNAL_MANIFEST_EXEMPT_SOURCES` short-circuit here to a clean pass:
    they are owned, in-process operations over already-trusted local state, not
    an external supply chain, so there is no external manifest to present.
    """
    normalized = (source or "").strip().lower()
    if normalized in INTERNAL_MANIFEST_EXEMPT_SOURCES:
        return {
            "checked": True,
            "ok": True,
            "connector": normalized,
            "manifest_path": None,
            "violations": [],
            "exempt_reason": (
                "internal-introspection"
                if normalized in INTERNAL_INTROSPECTION_SOURCES
                else "internal-orchestration"
            ),
        }

    path = find_connector_manifest(source, agents_root=agents_root)
    if path is None:
        return {
            "checked": True,
            "ok": False,
            "connector": resolve_connector_package(source, agents_root=agents_root),
            "manifest_path": None,
            "violations": [
                f"[missing] no connector_manifest.yml found for governed "
                f"source {source!r} — "
                "generate and certify one before activating this source."
            ],
        }

    # Runtime ingestion gate. Requires the release PIN, not a signature: every
    # bundled in-repo manifest is `UNSIGNED-PREVIEW` since in-repo signing was
    # removed, so `require_signature=True` here refused every connector in the
    # fleet. The pin still covers the complete document, including the `sync`
    # presets and tool-schema digests this path is about to act on.
    violations = _check_manifest_bytes(
        path,
        require_release_pin=True,
        require_provider=True,
    )
    return {
        "checked": True,
        "ok": not violations,
        "connector": path.parent.name,
        "manifest_path": _manifest_label(path),
        "violations": violations,
    }
