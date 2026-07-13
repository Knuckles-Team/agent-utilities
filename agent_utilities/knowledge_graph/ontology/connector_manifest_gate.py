"""Connector Ontology Manifest discovery + compile-before-sync gate (D17).

Wires the C5 Connector Ontology Manifest (:mod:`connector_manifest`,
:mod:`manifest_compiler`, :mod:`ontology_integrity`) into the live
:func:`agent_utilities.knowledge_graph.core.source_sync.sync_source` entrypoint:
before a source's data is pulled, its ``agents/<pkg>/connector_manifest.yml`` (if
one exists) is compiled and its ``provenance.integrity.hash`` re-verified —
**fail-closed** on a hand-edited/tampered manifest, a silent **pass-through** (no
manifest yet, or the source has no ``agents/*`` connector package) so this never
blocks a source that hasn't been onboarded to C5 yet.

CONCEPT:AU-KG.ontology.connector-manifest-gate — this module is the ``source_sync``
wiring leg (D17); the CLI sweep gate lives in ``scripts/check_connector_manifests.py``
and shares the same :func:`check_manifest_bytes`/compile path so both surfaces agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "SOURCE_TO_CONNECTOR_PACKAGE",
    "MANDATORY_NAMED_CONNECTOR_SOURCES",
    "resolve_agents_root",
    "resolve_connector_package",
    "find_connector_manifest",
    "check_manifest_bytes",
    "precheck_source",
    "manifest_required",
    "enterprise_required_sources",
]

# Curated aliases for the (common) cases where a ``_DELTA_HANDLERS``/``PACKAGE_PRESETS``
# source key doesn't literally match its ``agents/<pkg>`` directory name — e.g. two
# delta handlers (jira/confluence) share ONE connector package (atlassian-agent), and
# several handlers use the domain/vendor name rather than the repo directory name.
# Deliberately small and hand-verified; unknown sources fall through to the generic
# suffix-guessing in :func:`resolve_connector_package` rather than growing this table
# with a guess.
SOURCE_TO_CONNECTOR_PACKAGE: dict[str, str] = {
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

# AU-P1-6: the 12 high-value named connectors whose ``connector_manifest.yml`` is
# now MANDATORY — baked into :func:`enterprise_required_sources` unconditionally
# (unlike the opt-in ``CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE`` env var, an operator
# does NOT need to name these for the fail-closed policy to apply). Each entry is
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
# ``_DELTA_HANDLERS`` entry (``source_sync._sync_ops_mcp_connector`` + its 5 thin
# wrappers) — envelope-native from day one. All 12 named connectors below are now
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
    ws = (setting("WORKSPACE_PATH", default="/home/apps/workspace") or "").strip()
    return Path(ws) / "agent-packages" / "agents"


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
    committed-in-agent-utilities fallback that makes the 12
    :data:`MANDATORY_NAMED_CONNECTOR_SOURCES` manifests resolvable even when
    :func:`resolve_agents_root` (the live fleet checkout) isn't present, e.g. a
    standalone agent-utilities checkout or CI runner. A live fleet checkout, if
    present, always wins (checked first in :func:`find_connector_manifest`) —
    this is the pinned floor, not an override.
    """
    return Path(__file__).resolve().parent / "connector_manifests"


def find_connector_manifest(
    source: str, *, agents_root: Path | None = None
) -> Path | None:
    """The ``connector_manifest.yml`` path for ``source``'s connector package, if any.

    Checks the live fleet root first (``agents_root``/:func:`resolve_agents_root`),
    then falls back to :func:`bundled_manifests_root` (AU-P1-6) — so the 12
    mandatory named connectors resolve even without the sibling ``agent-packages``
    checkout present.
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


def check_manifest_bytes(path: Path) -> list[str]:
    """Compile + integrity-check one manifest file; returns a violation list (empty = OK).

    Shares the exact compile/hash path :mod:`scripts.check_connector_manifests` uses
    (kept in sync deliberately — this is the ``source_sync``-side twin of that CLI
    gate, CONCEPT:AU-KG.ontology.connector-manifest-gate), minus the signature-stub
    notice (source_sync fails closed on a hash mismatch alone; the signer allowlist
    is a defense-in-depth layer checked separately once X6 signer infra is wired).
    """
    import yaml

    from . import ontology_integrity
    from .connector_manifest import ConnectorManifest
    from .manifest_compiler import compile_manifest, export_manifest_ttl

    violations: list[str] = []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = ConnectorManifest.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        return [
            f"[schema] {path}: does not parse/validate as a ConnectorManifest: {exc}"
        ]

    try:
        spec = compile_manifest(manifest)
        ttl = export_manifest_ttl(spec, source=manifest.resolved_ontology_source)
        import rdflib

        g = rdflib.Graph()
        g.parse(data=ttl, format="turtle")
    except ImportError as exc:
        # rdflib is deliberately excluded from the lean `serving` plane
        # (see KG-2.242) and lives only in the `[owl]` extra — but THIS gate
        # (the mandatory D17 compile-before-sync check for the 12 named
        # AU-P1-6 connectors, leanix included) has no engine-native fallback and
        # genuinely needs it to parse/hash the compiled ontology. Degrade to one
        # clear, actionable line instead of a bare ModuleNotFoundError bubbling
        # up as "manifest does not compile cleanly".
        return [
            f"[dependency] {path}: the connector-manifest compile-before-sync "
            "gate needs rdflib to parse/hash the compiled ontology, and it is "
            "not installed on this deployment — install the 'owl' extra "
            "(pip install 'agent-utilities[owl]', or add it to this service's "
            f"image) to enable manifest-gated sync for this source. ({exc})"
        ]
    except Exception as exc:  # noqa: BLE001
        return [f"[compile] {path}: manifest does not compile cleanly: {exc}"]

    digest, triple_count = ontology_integrity.canonical_hash(g)
    if digest != manifest.provenance.integrity.hash:
        violations.append(
            f"[integrity] {path}: recomputed hash {digest} (n={triple_count}) != "
            f"provenance.integrity.hash {manifest.provenance.integrity.hash} — "
            "the manifest was hand-edited after signing, or is stale. Regenerate via "
            "scripts/generate_connector_manifests.py."
        )
    return violations


def enterprise_required_sources() -> set[str]:
    """Source keys whose sync MUST refuse without a compiled+verified manifest.

    ``MANDATORY_NAMED_CONNECTOR_SOURCES`` (AU-P1-6) is baked in **unconditionally**
    — the 12 high-value named connectors (atlassian/jira+confluence, gitlab,
    servicenow, microsoft, leanix, langfuse, tunnel-manager, container-manager,
    documentdb, repository-manager, systems-manager, vector) always require a
    passing manifest, with no operator opt-in needed.

    On top of that baseline, ``CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE`` — a
    comma-separated list of additional ``sync_source`` source keys (e.g.
    ``"twenty,egeria"``), lower-cased and stripped — remains an explicit,
    per-deployment **opt-in** (CONCEPT:AU-P0-4 fail-closed connector
    permissions): an operator activating any OTHER connector for enterprise use
    names it here so its sync provably refuses without manifest coverage,
    WITHOUT retroactively blocking the ~40 fleet sources that have no
    ``connector_manifest.yml`` yet and were never meant to require one for
    dev/local use. Empty env var -> only the AU-P1-6 mandatory 12 are required.
    """
    from ...core.config import setting

    raw = setting("CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE", default="") or ""
    opted_in = {s.strip().lower() for s in str(raw).split(",") if s.strip()}
    return set(MANDATORY_NAMED_CONNECTOR_SOURCES) | opted_in


def manifest_required(source: str) -> bool:
    """True when ``source`` was opted into the enterprise require-manifest policy."""
    return (source or "").strip().lower() in enterprise_required_sources()


def precheck_source(source: str, *, agents_root: Path | None = None) -> dict[str, Any]:
    """The ``sync_source`` compile-before-sync gate (D17).

    Returns ``{"checked": False, ...}`` when the source has no discoverable
    ``connector_manifest.yml`` yet AND was not opted into the enterprise
    require-manifest policy (:func:`manifest_required`) — a silent pass-through
    (most sources aren't onboarded to C5 yet, and this must never regress an
    existing dev/local sync). When a manifest IS found, returns
    ``{"checked": True, "ok": bool, "connector": str, "manifest_path": str,
    "violations": [...]}`` — fail-closed: ``ok=False`` means the caller MUST
    refuse to sync rather than pull data through a manifest that doesn't match
    what was actually reviewed/signed.

    CONCEPT:AU-P0-4 fail-closed connector permissions — when a source IS opted
    into :func:`manifest_required` (an explicit enterprise-activation allowlist,
    ``CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE``) but no manifest is found on disk,
    this ALSO returns a fail-closed ``{"checked": True, "ok": False, ...}`` so
    ``sync_source`` refuses the sync the exact same way it refuses a
    tampered/hand-edited manifest — "unknown" never silently means "allowed" for
    a source an operator has explicitly designated as enterprise-gated.
    """
    path = find_connector_manifest(source, agents_root=agents_root)
    if path is None:
        if manifest_required(source):
            return {
                "checked": True,
                "ok": False,
                "connector": resolve_connector_package(source, agents_root=agents_root),
                "manifest_path": None,
                "violations": [
                    f"[missing] no connector_manifest.yml found for enterprise-gated "
                    f"source {source!r} (CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE) — "
                    "generate one via scripts/generate_connector_manifests.py before "
                    "activating this source."
                ],
            }
        return {"checked": False, "reason": "no connector_manifest.yml for this source"}

    violations = check_manifest_bytes(path)
    return {
        "checked": True,
        "ok": not violations,
        "connector": path.parent.name,
        "manifest_path": str(path),
        "violations": violations,
    }
