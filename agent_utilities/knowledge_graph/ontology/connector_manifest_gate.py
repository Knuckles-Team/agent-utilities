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
    "resolve_agents_root",
    "resolve_connector_package",
    "find_connector_manifest",
    "check_manifest_bytes",
    "precheck_source",
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


def resolve_agents_root() -> Path:
    """The ``agent-packages/agents`` fleet root (``AGENTS_ROOT`` override, else
    ``WORKSPACE_PATH/agent-packages/agents``, else the dev-checkout-relative default)."""
    from ...core.config import setting

    override = (setting("AGENTS_ROOT", default="") or "").strip()
    if override:
        return Path(override)
    ws = (setting("WORKSPACE_PATH", default="/home/apps/workspace") or "").strip()
    return Path(ws) / "agent-packages" / "agents"


def resolve_connector_package(source: str, *, agents_root: Path | None = None) -> str | None:
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
    candidates += [f"{c}{suf}" for c in (norm, norm.replace("_", "-")) for suf in _GUESS_SUFFIXES]
    for cand in candidates:
        if (root / cand).is_dir():
            return cand
    return None


def find_connector_manifest(
    source: str, *, agents_root: Path | None = None
) -> Path | None:
    """The ``connector_manifest.yml`` path for ``source``'s connector package, if any."""
    pkg = resolve_connector_package(source, agents_root=agents_root)
    if pkg is None:
        return None
    root = agents_root if agents_root is not None else resolve_agents_root()
    path = root / pkg / "connector_manifest.yml"
    return path if path.exists() else None


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
        return [f"[schema] {path}: does not parse/validate as a ConnectorManifest: {exc}"]

    try:
        spec = compile_manifest(manifest)
        ttl = export_manifest_ttl(spec, source=manifest.resolved_ontology_source)
        import rdflib

        g = rdflib.Graph()
        g.parse(data=ttl, format="turtle")
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


def precheck_source(source: str, *, agents_root: Path | None = None) -> dict[str, Any]:
    """The ``sync_source`` compile-before-sync gate (D17).

    Returns ``{"checked": False, ...}`` when the source has no discoverable
    ``connector_manifest.yml`` yet (a silent pass-through — most sources aren't
    onboarded to C5 yet, and this must never regress an existing sync). When a
    manifest IS found, returns ``{"checked": True, "ok": bool, "connector": str,
    "manifest_path": str, "violations": [...]}`` — fail-closed: ``ok=False`` means
    the caller MUST refuse to sync rather than pull data through a manifest that
    doesn't match what was actually reviewed/signed.
    """
    path = find_connector_manifest(source, agents_root=agents_root)
    if path is None:
        return {"checked": False, "reason": "no connector_manifest.yml for this source"}

    violations = check_manifest_bytes(path)
    return {
        "checked": True,
        "ok": not violations,
        "connector": path.parent.name,
        "manifest_path": str(path),
        "violations": violations,
    }
