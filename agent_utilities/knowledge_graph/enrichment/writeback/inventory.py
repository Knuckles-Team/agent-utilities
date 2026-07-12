"""Cross-source inventory push + multi-SoR asset mirror (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

The capstone of bidirectional enrichment: take the KG's reconciled technology
inventory (infra/topology nodes + LeanIX ITComponents/Applications + TRM products/
assets) and create, in a target CMDB/ERP, the items that don't yet exist there.

Reconciliation is the OWL layer's job — ``ALIGNED_WITH`` identity (the same
mechanism Camunda⇆ARIS⇆Egeria use) collapses an infra server, its LeanIX
ITComponent, and its CMDB CI into one identity. Here we skip any candidate that is
already represented in the target — its ``domain`` is the target, it carries the
target's round-trip ``<target>_ci_id`` stamp (see :meth:`WritebackContext.stamp_external_id`),
or it is ``ALIGNED_WITH`` a node that is — and propose the rest as creations through
the unified, fail-closed :func:`run_writeback`.

**Multi-SoR mirror.** The canonical ``:Asset``/CI model stays in the graph; each
enabled system-of-record (ServiceNow / ERPNext / Egeria / Twenty) is a *projection*
of it. :func:`run_asset_mirror` fans one inventory pass out to every sink named in
``ASSET_MIRROR_TARGETS`` (the per-sink gate, modeled on ``CONTINUOUS_STARDOG_MIRROR``),
each still fail-closed on its own ``<SINK>_ENABLE_WRITE`` and dry-run-first by default.
The ``<target>_ci_id`` stamp makes a re-run a no-op/update rather than a re-create,
per sink, so the fan-out is idempotent across all sinks.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent_utilities.core.config import setting

from .core import run_writeback

logger = logging.getLogger(__name__)

# KG node types that constitute the technology inventory pushed upstream.
#
# MAINTAINED ALLOWLIST — the union of every asset/CI ``type`` string the fleet's
# ingest producers actually emit (grepped from their ``kg_ingest`` modules), plus
# the generic EA/TRM inventory types and the unified-infra ontology classes:
#   * container-manager-mcp: Container, ContainerImage, ContainerNetwork,
#     ContainerVolume, Deployment, Host, K8sService, Namespace, Pod, SwarmNode,
#     SwarmService  (Docker + Podman + Swarm + K8s)
#   * tunnel-manager:        Host, HostGroup   (SshKey is a credential, NOT a CI — excluded)
#   * portainer-agent:       Container, Stack, EndpointGroup, Environment
#   * dockerhub-api:         ContainerImage, Namespace, Repository
#   * unified-infra ontology: Node, Workload, Service, Tunnel, NetworkPath,
#     NetworkInterface, DiskVolume
#   * EA / TRM / generic:    Server, ITComponent, Application, TechnologyProduct,
#     AssetInstance, ConfigurationItem, HardwareNode
# Keep this exhaustive: a type missing here is silently excluded from every CMDB
# push. Add new producer types here when a new fleet ingestor lands.
INVENTORY_TYPES: tuple[str, ...] = (
    # generic EA / TRM / CMDB
    "Server",
    "Service",
    "HardwareNode",
    "ITComponent",
    "Application",
    "TechnologyProduct",
    "AssetInstance",
    "ConfigurationItem",
    # hosts / nodes
    "Host",
    "HostGroup",
    "Node",
    "SwarmNode",
    # container / orchestration workloads
    "Container",
    "ContainerImage",
    "ContainerNetwork",
    "ContainerVolume",
    "Pod",
    "Deployment",
    "Workload",
    "Namespace",
    "K8sService",
    "SwarmService",
    "Stack",
    "EndpointGroup",
    "Environment",
    "Repository",
    # network / storage components
    "NetworkInterface",
    "DiskVolume",
    "Tunnel",
    # NetworkPath is a measurement/Process (not a Continuant CI); included for
    # completeness of the fleet's emitted types — sinks map it to a generic CI.
    "NetworkPath",
)


def collect_inventory_creations(
    backend: Any,
    target: str,
    *,
    node_types: tuple[str, ...] | None = None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Inventory candidates not yet present in ``target`` → ``[{type,name}]``.

    Best-effort over the backend: a node is a candidate when it is named, of an
    inventory type, and its ``domain`` is not already the target. The
    ``ALIGNED_WITH`` cross-source-identity exclusion is applied when the backend
    can serve it (degrades to the domain check otherwise).
    """
    if backend is None:
        return []
    types = node_types or INVENTORY_TYPES
    rows: list[dict] = []
    try:
        rows = (
            backend.execute(
                "MATCH (n) WHERE n.name IS NOT NULL "
                "AND (n.domain IS NULL OR n.domain <> $t) "
                "RETURN n.type AS type, n.name AS name, n.id AS id LIMIT $limit",
                {"t": target, "limit": limit},
            )
            or []
        )
    except Exception:  # noqa: BLE001 - tolerant: no candidates rather than crash
        logger.debug("inventory candidate query failed", exc_info=True)
        return []

    type_set = {t.lower() for t in types}
    aligned = _aligned_to_target_ids(backend, target)
    stamped = _stamped_ids(backend, target)
    creations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        ntype = str(r.get("type") or "")
        name = r.get("name")
        nid = str(r.get("id") or "")
        if ntype.lower() not in type_set or not name:
            continue
        if nid in aligned:  # already represented upstream via ALIGNED_WITH identity
            continue
        if (
            nid in stamped
        ):  # already created in target (round-trip <target>_ci_id stamp)
            continue
        if name in seen:
            continue
        seen.add(name)
        creations.append({"type": ntype, "name": name, "node": nid})
    return creations


# ``<sink>_ci_id`` stamp key must be a safe Cypher identifier (validated before use).
_CI_ID_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def ci_id_key(target: str) -> str:
    """The per-sink round-trip stamp property name (``<target>_ci_id``)."""
    return f"{(target or '').lower().strip()}_ci_id"


def _stamped_ids(backend: Any, target: str) -> set[str]:
    """Node ids already stamped with this target's ``<target>_ci_id`` (best-effort).

    This is what makes a re-run idempotent per sink: once a create round-trips the
    SoR's id back onto the source node (:meth:`WritebackContext.stamp_external_id`),
    the node is excluded from the next collection, so it is skipped (not re-created).
    """
    if backend is None:
        return set()
    key = ci_id_key(target)
    if not _CI_ID_KEY_RE.match(key):  # defensive: never build a query from junk
        return set()
    try:
        rows = backend.execute(
            f"MATCH (n) WHERE n.{key} IS NOT NULL RETURN n.id AS id",
            {},
        )
        return {
            str(r["id"]) for r in (rows or []) if isinstance(r, dict) and r.get("id")
        }
    except Exception:  # noqa: BLE001 - stamp exclusion is optional
        logger.debug("stamped-id query failed for %s", target, exc_info=True)
        return set()


def _aligned_to_target_ids(backend: Any, target: str) -> set[str]:
    """Node ids ALIGNED_WITH a node already in ``target`` (best-effort)."""
    try:
        rows = backend.execute(
            "MATCH (n)-[:ALIGNED_WITH]-(m) WHERE m.domain = $t RETURN n.id AS id",
            {"t": target},
        )
        return {
            str(r["id"]) for r in (rows or []) if isinstance(r, dict) and r.get("id")
        }
    except Exception:  # noqa: BLE001 - alignment exclusion is optional
        return set()


def push_inventory(
    target: str,
    *,
    backend: Any = None,
    engine: Any = None,
    node_types: tuple[str, ...] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Collect the reconciled inventory and create the missing items in ``target``.

    Fail-closed + dry-run-first via :func:`run_writeback` (the target's
    ``*_ENABLE_WRITE`` gate still applies).
    """
    creations = collect_inventory_creations(backend, target, node_types=node_types)
    result = run_writeback(
        target, backend=backend, engine=engine, dry_run=dry_run, creations=creations
    )
    if isinstance(result, dict):
        result["inventory_candidates"] = len(creations)
    return result


# Sinks that project the canonical KG :Asset/CI model into an external SoR CMDB.
# The graph stays the authority; each of these is a mirror/projection, opted in
# per-deployment via ``ASSET_MIRROR_TARGETS``.
ASSET_MIRROR_SINKS: tuple[str, ...] = ("servicenow", "erpnext", "egeria", "twenty")


def enabled_mirror_targets() -> list[str]:
    """The sinks opted into the asset mirror via ``ASSET_MIRROR_TARGETS``.

    Modeled on ``CONTINUOUS_STARDOG_MIRROR`` (the ONE explicit switch for the
    Stardog mirror): here a single ``ASSET_MIRROR_TARGETS`` list names which CMDB
    sinks receive the projection. Empty by default, so every sink — ServiceNow
    included — stays available-but-inert until a deployment opts it in. Unknown
    names are dropped (only the real CMDB sinks can mirror).
    """
    raw = setting("ASSET_MIRROR_TARGETS", None)
    names: list[str]
    if isinstance(raw, list | tuple):
        names = [str(t).strip().lower() for t in raw if str(t).strip()]
    else:
        s = str(raw or "").strip()
        if not s:
            return []
        # tolerant parse: JSON array (config.json env injection) OR comma list.
        parsed: Any = None
        try:
            import json as _json

            parsed = _json.loads(s)
        except Exception:  # noqa: BLE001
            parsed = None
        if isinstance(parsed, list):
            names = [str(t).strip().lower() for t in parsed if str(t).strip()]
        else:
            names = [
                t.strip().strip("[]\"'").lower()
                for t in s.split(",")
                if t.strip().strip("[]\"'")
            ]
    known = set(ASSET_MIRROR_SINKS)
    out: list[str] = []
    for n in names:
        if n in known and n not in out:
            out.append(n)
        elif n not in known:
            logger.warning(
                "asset-mirror: target %r is not a known CMDB sink %s; skipping.",
                n,
                ASSET_MIRROR_SINKS,
            )
    return out


def run_asset_mirror(
    *,
    backend: Any = None,
    engine: Any = None,
    targets: list[str] | None = None,
    node_types: tuple[str, ...] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Fan ONE inventory pass out to every enabled CMDB sink (multi-SoR mirror).

    The canonical CI model lives in the graph; each enabled sink is a projection.
    Gating is layered and fail-closed:

    * ``ASSET_MIRROR_TARGETS`` selects which sinks participate (empty ⇒ no-op).
    * each sink still enforces its own ``<SINK>_ENABLE_WRITE`` for live writes
      (via :func:`run_writeback`), and ``dry_run`` (default True) previews only.

    Returns one manifest keyed by sink plus a rollup, so a single scheduled pass
    (the ``asset-mirror`` CronJob / ``python -m ...writeback.asset_mirror``)
    reports intended-writes for every projection at once.
    """
    selected = [t.lower() for t in targets] if targets else enabled_mirror_targets()
    per_sink: dict[str, Any] = {}
    created = enriched = skipped = errors = candidates = 0
    for target in selected:
        res = push_inventory(
            target,
            backend=backend,
            engine=engine,
            node_types=node_types,
            dry_run=dry_run,
        )
        per_sink[target] = res
        if isinstance(res, dict):
            created += int(res.get("created") or 0)
            enriched += int(res.get("enriched") or 0)
            skipped += int(res.get("skipped") or 0)
            errors += int(res.get("errors") or 0)
            candidates += int(res.get("inventory_candidates") or 0)
    return {
        "status": "completed",
        "dry_run": dry_run,
        "targets": selected,
        "sinks": per_sink,
        "created": created,
        "enriched": enriched,
        "skipped": skipped,
        "errors": errors,
        "inventory_candidates": candidates,
    }
