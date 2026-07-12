"""Package-install -> KG auto-extension consumer (CONCEPT:AU-KG.ingest.package-install-autoingest).

Closes the "auto-extend the KG when a package is installed" loop. The
universal-installer (``universal-skills``' ``universal-installer`` skill)
already materializes a newly-installed/updated package's skills + prompts +
ontology into the unified XDG tree
(:func:`agent_utilities.core.unified_install.install_unified` --
``$XDG_DATA_HOME/agent-utilities/{skills,prompts,ontologies}/<provider>/...``)
and, whenever a prompt/ontology leg actually changed, drops a small summary
manifest next to it::

    $XDG_DATA_HOME/agent-utilities/install-manifest.json
    {"generated_at": "<iso8601>",
     "prompts": {"<provider>": <n_files>, ...},
     "ontologies": {"<provider>": <n_ttls>, ...}}

The installer's own ``SKILL.md`` ("Ontology / skill / prompt auto-extension
hook for the KG") explicitly documents this as a deliberate half-measure --
the installer makes the artifacts *discoverable* (materialized into the
canonical tree + the manifest as a change signal) but never itself parses the
manifest, mints KG nodes, or calls ``source_sync`` ("that's the graph-os/
epistemic-graph side's job"). **This module is that graph-os-side consumer.**

Design -- reuse, never reimplement:

The manifest is read purely as a **change signal** (its content hash is the
dedup watermark via the existing :class:`~.manifest.DeltaManifest`); the
manifest does NOT enumerate individual file paths, so this module does not
try to diff files itself. Instead it re-drives the three ingestion primitives
AU already has for each leg -- each is independently idempotent (upsert-keyed
by stable id / content hash), so re-running them costs a no-op write for
anything unchanged:

* **prompts**    -> :func:`agent_utilities.agent.registry_builder.ingest_prompts_to_graph`
  (the same base+fleet+overlay prompt-registry reload the CLI/boot path uses).
* **ontologies** -> :func:`agent_utilities.mcp.tools.ontology_tools._sync_package_ontologies`
  (the same federation-runtime reload ``graph_ontology action='sync_packages'``
  and graph-os boot already call).
* **skills**     -> :func:`agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest.ingest_skill_workflows`
  (the corpus-wide workflow-skill leg -- the one existing "*.md skill corpus ->
  KG" ingestion path). The manifest itself does not yet itemize atomic
  (non-workflow) skills per provider -- a documented upstream gap in the
  installer, not something this module papers over with new ingestion logic.

Registered as source ``"package_install"`` in
:data:`agent_utilities.knowledge_graph.core.source_sync._DELTA_HANDLERS`
(CONCEPT:AU-KG.ingest.enterprise-source-extractor) so it rides the ONE
existing ``source_sync`` MCP tool / REST twin (``/source/sync``) -- no new
tool or route. It is also registered as a durable ``:Schedule`` (mirroring
``engine_tasks.py::_register_maintenance_schedules``) so it runs on the ONE
supervised unified-scheduler plane instead of a new daemon.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "install-manifest.json"
_DEDUP_CATEGORY = "package_install_manifest"


def manifest_path() -> Path:
    """The one global ``install-manifest.json`` the universal-installer writes."""
    from agent_utilities.core.paths import data_dir

    return data_dir() / _MANIFEST_NAME


def _read_manifest() -> tuple[dict[str, Any] | None, bytes]:
    path = manifest_path()
    if not path.is_file():
        return None, b""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.warning("package_install: could not read %s: %s", path, exc)
        return None, b""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        logger.warning("package_install: %s is not valid JSON: %s", path, exc)
        return None, raw
    return data if isinstance(data, dict) else None, raw


def _graph_name(engine: Any) -> str:
    gc = getattr(engine, "graph_compute", None)
    return getattr(gc, "graph_name", None) or "__commons__"


def _ingest_prompts_leg() -> dict[str, Any]:
    """Re-drive the existing prompt-registry reload (base + fleet + overlay)."""
    import asyncio

    from agent_utilities.agent.registry_builder import ingest_prompts_to_graph

    try:
        asyncio.run(ingest_prompts_to_graph())
        return {"status": "ok"}
    except RuntimeError as exc:
        # Already inside a running event loop (unexpected for this sync
        # dispatch path, but fail soft rather than crash the whole sync).
        logger.warning("package_install: prompts leg skipped (%s)", exc)
        return {"status": "skipped", "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 - one leg failing must not fail the rest
        logger.warning("package_install: prompts leg failed: %s", exc)
        return {"status": "error", "reason": str(exc)}


def _ingest_ontologies_leg(engine: Any) -> dict[str, Any]:
    """Re-drive the existing ontology-federation reload (`sync_packages`)."""
    try:
        from agent_utilities.knowledge_graph.ontology.lifecycle import (
            OntologyLifecycle,
        )
        from agent_utilities.mcp.tools.ontology_tools import (
            _sync_package_ontologies,
        )

        report = _sync_package_ontologies(OntologyLifecycle(engine=engine))
        report.setdefault("status", "ok")
        return report
    except Exception as exc:  # noqa: BLE001
        logger.warning("package_install: ontologies leg failed: %s", exc)
        return {"status": "error", "reason": str(exc)}


def _ingest_skills_leg(engine: Any) -> dict[str, Any]:
    """Re-drive the existing workflow-skill corpus reload."""
    try:
        from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
            ingest_skill_workflows,
        )

        report = ingest_skill_workflows(engine)
        report.setdefault("status", "ok")
        return report
    except Exception as exc:  # noqa: BLE001
        logger.warning("package_install: skills leg failed: %s", exc)
        return {"status": "error", "reason": str(exc)}


def sync_package_install(
    engine: Any,
    *,
    mode: str = "delta",
    ids: list[str] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Ingest a newly-installed/updated package's skills+prompts+ontology into the KG.

    CONCEPT:AU-KG.ingest.package-install-autoingest. The ``_DELTA_HANDLERS``-shaped
    entrypoint for source ``"package_install"`` -- called by
    :func:`~agent_utilities.knowledge_graph.core.source_sync.sync_source` (and
    therefore reachable via the ``source_sync`` MCP tool / ``/source/sync`` REST
    twin) and by the ``package_install_ingest`` maintenance ``:Schedule``.

    Dedup: the manifest's own content hash is the watermark (via the shared
    :class:`~.manifest.DeltaManifest`, category ``"package_install_manifest"``),
    so an unchanged manifest is a cheap no-op -- this makes the periodic
    ``:Schedule`` tick safe to run on a fixed cadence with no new state.
    ``mode="full"`` or a non-empty ``ids`` (an operator/agent explicitly asking
    to (re)ingest specific provider(s) right now) both bypass the watermark
    check. ``ids`` narrows the *reported* provider set for traceability; the
    reused ingestion primitives themselves always reconcile every registered
    provider (each is independently idempotent, so this is safe, just not
    maximally granular -- narrowing further would mean reimplementing their
    discovery, which this module deliberately does not do).

    Returns a report dict; always ``status="ok"`` unless the manifest itself
    is unreadable -- an individual leg's failure is recorded under
    ``legs.<leg>.status`` rather than failing the whole sync (one bad leg
    must not block the others, matching every other ``_DELTA_HANDLERS`` entry).
    """
    manifest, raw = _read_manifest()
    if manifest is None:
        return {
            "status": "skipped",
            "source": "package_install",
            "reason": (
                f"no {_MANIFEST_NAME} found under data_dir() -- nothing "
                "installed via the universal-installer yet, or the manifest "
                "is unreadable"
            ),
        }

    from .manifest import DeltaManifest

    backend = getattr(engine, "backend", None)
    dm = DeltaManifest(backend=backend)
    graph_name = _graph_name(engine)
    manifest_uri = str(manifest_path())
    content_hash = hashlib.sha256(raw).hexdigest()

    requested_providers = sorted({str(i) for i in (ids or []) if i})
    force = mode == "full" or bool(requested_providers)

    if not force and dm.seen(graph_name, _DEDUP_CATEGORY, manifest_uri, content_hash):
        return {
            "status": "ok",
            "source": "package_install",
            "mode": mode,
            "delta_capable": True,
            "generated_at": manifest.get("generated_at"),
            "skipped_unchanged": True,
            "reason": "install-manifest.json unchanged since last ingest",
        }

    manifest_providers = sorted(
        set(manifest.get("prompts") or {}) | set(manifest.get("ontologies") or {})
    )

    legs = {
        "prompts": _ingest_prompts_leg(),
        "ontologies": _ingest_ontologies_leg(engine),
        "skills": _ingest_skills_leg(engine),
    }

    dm.record(graph_name, _DEDUP_CATEGORY, manifest_uri, content_hash)

    errors = [leg for leg, res in legs.items() if res.get("status") == "error"]
    return {
        "status": "ok",
        "source": "package_install",
        "mode": mode,
        "delta_capable": True,
        "generated_at": manifest.get("generated_at"),
        "manifest_providers": manifest_providers,
        "requested_providers": requested_providers,
        "skipped_unchanged": False,
        "legs": legs,
        "failed_legs": errors,
    }
