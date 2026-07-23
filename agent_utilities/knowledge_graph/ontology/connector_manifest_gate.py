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
from collections import deque
from collections.abc import Callable
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
    "mandatory_connector_packages",
    "resolve_agents_root",
    "resolve_connector_package",
    "find_connector_manifest",
    "check_manifest_bytes",
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
    "database": "native-source-connectors",
    "external_graph": "native-source-connectors",
    "filesystem": "native-source-connectors",
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
            targets.update(
                alias.name
                for alias in node.names
                if alias.name == package_name
                or alias.name.startswith(f"{package_name}.")
            )
            continue

        base = _absolute_import_module(
            node,
            current_module=current_module,
            current_is_package=current_path.name == "__init__.py",
        )
        if base != package_name and not base.startswith(f"{package_name}."):
            continue
        targets.add(base)
        base_path = resolve_module(base)
        if base_path is not None and base_path.name == "__init__.py":
            targets.update(
                f"{base}.{alias.name}" for alias in node.names if alias.name != "*"
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
    pending = deque(roots)
    required = set(roots)
    resolved_package_root = package_root.resolve()
    deep_roots = set(deep_import_roots or roots)
    path_cache = module_path_cache if module_path_cache is not None else {}
    source_cache = module_source_cache if module_source_cache is not None else {}

    def _resolve_module(module_name: str) -> Path | None:
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
        path = (
            package_path if package_exists else module_path if module_exists else None
        )
        if path is not None:
            try:
                path.resolve().relative_to(resolved_package_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"native module {module_name!r} escapes its package root"
                ) from exc
        path_cache[module_name] = path
        return path

    visited: set[str] = set()
    missing: set[str] = set()
    module_hashes: dict[str, str] = {}

    while pending:
        module_name = pending.popleft()
        if module_name in visited:
            continue
        visited.add(module_name)
        if len(visited) > _MAX_FINGERPRINT_MODULES:
            raise RuntimeError("native module dependency closure is too large")
        path = _resolve_module(module_name)
        if path is None:
            if module_name in required:
                raise RuntimeError(
                    f"native fingerprint root {module_name!r} is unavailable"
                )
            missing.add(module_name)
            continue
        try:
            cached_source = source_cache.get(module_name)
            if cached_source is None:
                source = path.read_bytes()
                tree = ast.parse(source, filename=module_name)
                source_cache[module_name] = (source, tree)
            else:
                source, tree = cached_source
        except (OSError, SyntaxError, UnicodeError, ValueError) as exc:
            raise RuntimeError(
                f"native module {module_name!r} cannot be fingerprinted"
            ) from exc
        module_hashes[module_name] = hashlib.sha256(source).hexdigest()
        pending.extend(_package_ancestors(module_name, package_name=package_name))
        pending.extend(
            sorted(
                _local_import_targets(
                    tree,
                    current_module=module_name,
                    current_path=path,
                    package_name=package_name,
                    resolve_module=_resolve_module,
                    include_function_bodies=module_name in deep_roots,
                )
            )
        )

    payload = {
        "format": NATIVE_FINGERPRINT_FORMAT,
        "roots": list(roots),
        "modules": dict(sorted(module_hashes.items())),
        "missing_local_imports": sorted(missing),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), tuple(sorted(module_hashes))


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
    require_provider: bool = False,
) -> list[str]:
    """Compile + integrity-check one manifest file; returns violations (empty = OK).

    Shares the exact compile/hash path :mod:`scripts.check_connector_manifests` uses
    (kept in sync deliberately — this is the ``source_sync``-side twin of that CLI
    gate, CONCEPT:AU-KG.ontology.connector-manifest-gate).
    """
    return _check_manifest_bytes(
        path,
        require_signature=require_signature,
        require_provider=require_provider,
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


def _signature_violations(manifest: Any, *, label: str) -> list[str]:
    """Verify the full manifest against Ed25519 evidence and its release pin."""
    from . import ontology_integrity

    provenance = manifest.provenance
    if not provenance.signer or not provenance.signature:
        return [f"[signature] {label}: manifest is unsigned"]
    if provenance.signer not in ontology_integrity.DEFAULT_TRUSTED_SIGNERS:
        return [f"[signature] {label}: signer {provenance.signer!r} is not trusted"]

    manifest_hash = ontology_integrity.canonical_manifest_hash(manifest)
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
    if pin.get("manifest_hash"):
        violations: list[str] = []
        if pin.get("manifest_hash") != manifest_hash:
            violations.append(
                f"[signature] {label}: complete manifest content differs from its release pin"
            )
        if pin.get("signer") != provenance.signer:
            violations.append(
                f"[signature] {label}: signer differs from its release pin"
            )
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
    return []


def _native_provider_violations(manifest: Any, *, path: Path, label: str) -> list[str]:
    """Verify the in-package connector registry against its signed source pins."""

    try:
        sidecar = json.loads(
            (path.parent / "tool_schema_fingerprints.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return [
            f"[tool-schema] {label}: native connector fingerprint sidecar is invalid"
        ]
    if (
        not isinstance(sidecar, dict)
        or sidecar.get("format") != NATIVE_FINGERPRINT_FORMAT
    ):
        return [
            f"[tool-schema] {label}: native connector fingerprint format is invalid"
        ]
    pins = sidecar.get("sources", {})
    if not isinstance(pins, dict) or not pins:
        return [f"[tool-schema] {label}: native connector fingerprint map is empty"]

    violations: list[str] = []
    path_cache: dict[str, Path | None] = {}
    source_cache: dict[str, tuple[bytes, ast.Module]] = {}
    for sync in manifest.sync:
        source_type = str(sync.tool or "")
        contract = native_activation_contract(source_type)
        if contract is None:
            violations.append(
                f"[provider] {label}: native source {source_type!r} is unavailable"
            )
            continue
        interface, _module_name = contract
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
            continue
        pinned = str(pins.get(source_type) or "")
        if not pinned or pinned != actual or sync.tool_schema_sha256 != pinned:
            violations.append(
                f"[tool-schema] {label}: native source {source_type!r} differs "
                "from its signed code fingerprint"
            )
    declared = {str(sync.tool or "") for sync in manifest.sync}
    if declared != set(pins):
        violations.append(
            f"[tool-schema] {label}: native manifest and fingerprint inventory differ"
        )
    return violations


def _provider_violations(manifest: Any, *, path: Path, label: str) -> list[str]:
    """Require the installed provider to match every manifest sync preset exactly."""
    if manifest.connector == "native-source-connectors":
        return _native_provider_violations(manifest, path=path, label=label)
    from ...protocols.source_connectors.connectors.mcp_tool import (
        McpToolSourceError,
        provider_tool_presets,
        provider_tool_schema_fingerprints,
    )

    try:
        provider_presets = provider_tool_presets(manifest.connector)
    except McpToolSourceError as exc:
        return [f"[provider] {label}: {exc}"]
    if provider_presets is None:
        return [
            f"[provider] {label}: source preset provider {manifest.connector!r} "
            "is not installed or cannot be resolved"
        ]
    if not manifest.sync:
        return [f"[tool-schema] {label}: mandatory connector declares no sync presets"]

    try:
        live_schema_pins = provider_tool_schema_fingerprints(manifest.connector)
    except McpToolSourceError as exc:
        return [f"[tool-schema] {label}: {exc}"]
    if not live_schema_pins:
        return [
            f"[tool-schema] {label}: connector-owned tool_schema_fingerprints.json "
            "is missing or empty"
        ]

    violations: list[str] = []
    for sync in manifest.sync:
        actual = provider_presets.get(sync.preset)
        if actual is None:
            violations.append(
                f"[tool-schema] {label}: provider is missing preset {sync.preset!r}"
            )
            continue
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
            continue
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


def _check_manifest_bytes(
    path: Path,
    *,
    require_signature: bool = False,
    require_provider: bool = False,
) -> list[str]:
    """Implementation shared by runtime and direct hash-only callers."""
    import yaml

    from . import ontology_integrity
    from .connector_manifest import ConnectorManifest
    from .manifest_compiler import compile_manifest, export_manifest_ttl

    violations: list[str] = []
    label = _manifest_label(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = ConnectorManifest.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        return [
            f"[schema] {label}: does not parse/validate as a ConnectorManifest "
            f"({type(exc).__name__})"
        ]

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
        return [
            f"[dependency] {label}: the connector-manifest compile-before-sync "
            "gate needs rdflib to parse/hash the compiled ontology, and it is "
            "not installed on this deployment — install the 'owl' extra "
            "(pip install 'agent-utilities[owl]', or add it to this service's "
            "image) to enable manifest-gated sync for this source."
        ]
    except Exception as exc:  # noqa: BLE001
        return [
            f"[compile] {label}: manifest does not compile cleanly "
            f"({type(exc).__name__})"
        ]

    digest, triple_count = ontology_integrity.canonical_hash(g)
    if digest != manifest.provenance.integrity.hash:
        violations.append(
            f"[integrity] {label}: recomputed hash {digest} (n={triple_count}) != "
            f"provenance.integrity.hash {manifest.provenance.integrity.hash} — "
            "the manifest was hand-edited after signing, or is stale. Regenerate via "
            "scripts/generate_connector_manifests.py."
        )
    if require_signature:
        violations.extend(_signature_violations(manifest, label=label))
    if require_provider:
        violations.extend(_provider_violations(manifest, path=path, label=label))
    return violations


def required_connector_sources() -> set[str]:
    """Certified source inventory used for reporting and bundle coverage.

    Enforcement itself is universal: :func:`manifest_required` governs every
    non-empty source, including sources not yet present in this inventory.
    """
    return set(MANDATORY_NAMED_CONNECTOR_SOURCES) | set(mandatory_connector_packages())


def manifest_required(source: str) -> bool:
    """Return whether an external source must carry a certified manifest.

    Every non-empty source is governed.  The former allowlist/pass-through model
    made an unknown connector less restricted than an onboarded one, which is the
    inverse of a fail-closed supply-chain boundary.  The enterprise source catalog
    remains useful for inventory reporting, but it no longer controls enforcement.
    """
    return bool((source or "").strip())


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
    """
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

    violations = _check_manifest_bytes(
        path,
        require_signature=True,
        require_provider=True,
    )
    return {
        "checked": True,
        "ok": not violations,
        "connector": path.parent.name,
        "manifest_path": _manifest_label(path),
        "violations": violations,
    }
