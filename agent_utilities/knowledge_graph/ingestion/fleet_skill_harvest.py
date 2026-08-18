"""Promote skills harvested from fleet MCP children into runnable resources.

CONCEPT:AU-ECO.mcp.cross-process-skill-harvest — Cross-Process Skill Harvest.

THE GAP THIS CLOSES
-------------------
``core.providers.resolve_skill_provider_dirs()`` discovers skill providers with
an in-process ``importlib.metadata.entry_points()`` walk, so it only ever sees
what is installed in **graph-os's own serving venv**. The ~62 fleet
``agents/*`` packages (``servicenow-api`` among them) are deliberately NOT
co-installed there — that is ``AGENTS.md`` §"Dependency discipline" working as
designed, not a bug. Their skills are therefore *structurally* invisible to
in-process discovery, no matter how correct the downstream ingestion is.

The fix does not install anything. Each fleet child ALREADY publishes its
skills as fastmcp-4 ``skill://{name}/SKILL.md`` resources (via
``server_factory._register_skill_providers``), and the multiplexer ALREADY
holds a live session to it during its probe sweep. So graph-os *pulls* the
bodies over that existing connection
(:meth:`~...mcp.multiplexer.MCPMultiplexer._harvest_skill_bodies`) and this
module promotes them through the EXISTING, already-correct
:func:`~.skill_workflow_ingest.ingest_runnable_skill` primitive. Nothing about
that primitive changes; it simply finally gets called for fleet skills.

FAIL CLOSED, WITH A NAMED PRECONDITION
--------------------------------------
The multiplexer no longer conflates "child process is running" with "its tools
are dispatchable", and a harvested skill must not recreate that lie. A skill is
promoted to a runnable ``CallableResource`` ONLY when every precondition in
:data:`RUNNABLE_PRECONDITIONS` holds. When one does not, the skill is still
recorded — as a ``Skill`` node carrying ``runnable_blocked_by=<precondition>``
— so the execution path fails closed against a *named* reason instead of a
generic "not found or runnable". A skill that is present but not dispatchable
must be loudly un-runnable, never quietly half-runnable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Ordered: the FIRST unmet precondition is the one reported, because a later
# check is meaningless once an earlier one has failed (a child that never
# answered cannot also be said to have served an empty body).
RUNNABLE_PRECONDITIONS: tuple[str, ...] = (
    "server_reachable",
    "skill_body_served",
    "server_exposes_tools",
    "no_local_provider_conflict",
)


def _local_provider_owns(engine: Any, resource_id: str, server_name: str) -> bool:
    """Whether a LOCALLY-installed provider already owns this skill identity.

    A co-installed provider is the stronger authority: it was verified by
    ``resolve_skill_provider_dirs()`` against a real asset manifest, whereas a
    harvested body is whatever a child chose to serve. So a name collision
    resolves in the local provider's favour and the harvest declines rather
    than overwriting it — the fleet child does not get to silently redefine a
    skill graph-os already ships.
    """
    # ``backend.execute`` is the parameterized read path agent resolution uses;
    # ``engine.query_cypher`` does not bind a ``WHERE`` predicate the same way on
    # every backend build, and a silently-empty result here would disable the
    # conflict guard entirely rather than merely mis-order one skill.
    rows = engine.backend.execute(
        "MATCH (r:CallableResource) WHERE r.id = $rid "
        "RETURN r.provider_ref AS provider_ref, r.mcp_server AS mcp_server",
        {"rid": resource_id},
    )
    if not rows:
        return False
    row = rows[0] or {}
    # Already harvested from THIS same child → a refresh, not a conflict.
    return str(row.get("mcp_server") or "") != server_name


def evaluate_skill_dispatchable(
    engine: Any,
    server_name: str,
    server_info: dict[str, Any],
    entry: dict[str, Any],
    *,
    resource_id: str,
) -> str:
    """Return the first unmet precondition, or ``""`` when all of them hold."""
    if server_info.get("error"):
        return "server_reachable"
    if not str(entry.get("instructions") or "").strip():
        return "skill_body_served"
    # A skill's instructions direct an LLM to CALL this child's tools. A child
    # that exposes none cannot execute the skill, so advertising it as runnable
    # would be exactly the mounted-vs-callable lie this gate exists to prevent.
    #
    # This deliberately does NOT call ``MCPMultiplexer.tool_dispatchable()``,
    # even though that is the correct single source of truth for "can this
    # session dispatch this tool". ``tool_dispatchable`` is only meaningful on
    # the SERVING multiplexer instance: on a freshly-constructed one (which is
    # what a harvest / ``source_sync`` builds) ``_exposed`` and
    # ``_session_loaded`` are empty and ``_server_for_prefixed`` resolves
    # nothing, so every name falls through to its final "unknown to our
    # bookkeeping → unconditionally callable" branch. Verified live in-pod: it
    # returned True for real probed tools AND for the invented name
    # ``servicenow_servicenow_account``. Gating on it here would therefore
    # assert callability for tools that do not exist — re-creating the exact lie
    # it was written to remove, one layer up. The probe's own tool list is the
    # honest signal at ingestion time; per-session dispatchability is enforced
    # later, at the tools/call gate, where the serving state actually exists.
    if not (server_info.get("tools") or []):
        return "server_exposes_tools"
    if _local_provider_owns(engine, resource_id, server_name):
        return "no_local_provider_conflict"
    return ""


def promote_harvested_skills(engine: Any, catalog: dict[str, dict]) -> dict[str, Any]:
    """Promote every dispatchable harvested fleet skill to a runnable resource.

    ``catalog`` is the multiplexer's probe result — ``{server: {"tools": [...],
    "skills": [{name, uri, description, instructions|harvest_error}],
    "error": str|None}}``.

    Returns ``{"promoted", "blocked", "blocked_detail", "errors",
    "error_detail"}``. ``blocked_detail`` maps ``"<server>/<skill>"`` to the
    named unmet precondition so the reason a skill is not runnable is queryable
    rather than inferred.
    """
    from .skill_workflow_ingest import skill_reference

    promoted: list[str] = []
    blocked_detail: dict[str, str] = {}
    error_detail: dict[str, str] = {}

    for server_name, info in (catalog or {}).items():
        if not isinstance(info, dict):
            continue
        for entry in info.get("skills") or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            key = f"{server_name}/{name}"
            slug = skill_reference(name).removeprefix("skill://")
            resource_id = f"resource:skill:{slug}"
            try:
                unmet = evaluate_skill_dispatchable(
                    engine, server_name, info, entry, resource_id=resource_id
                )
            except Exception as exc:  # noqa: BLE001 — one skill's precondition
                # check must not abort the sweep, but the cause is kept verbatim
                # (never reduced to a class name) and logged with its traceback.
                error_detail[key] = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Skill harvest precondition check failed for %s (%s)",
                    key,
                    type(exc).__name__,
                    exc_info=True,
                )
                continue
            if unmet:
                # The harvest itself may have recorded a more specific reason
                # (e.g. WHY the body could not be served); keep it alongside the
                # precondition name so the failure is diagnosable, not just
                # categorised.
                harvest_error = str(entry.get("harvest_error") or "").strip()
                blocked_detail[key] = (
                    f"{unmet}: {harvest_error}" if harvest_error else unmet
                )
                logger.warning(
                    "Fleet skill %s is NOT runnable — unmet precondition %s%s",
                    key,
                    unmet,
                    f" ({harvest_error})" if harvest_error else "",
                )
                continue
            try:
                promote_one(
                    engine,
                    server_name=server_name,
                    name=name,
                    description=str(entry.get("description") or ""),
                    instructions=str(entry["instructions"]),
                )
            except Exception as exc:  # noqa: BLE001 — see above: recorded and
                # logged with the chained traceback, never swallowed.
                error_detail[key] = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Failed to promote harvested skill %s (%s)",
                    key,
                    type(exc).__name__,
                    exc_info=True,
                )
                continue
            promoted.append(key)

    logger.info(
        "Cross-process skill harvest: %d promoted, %d blocked, %d errors",
        len(promoted),
        len(blocked_detail),
        len(error_detail),
    )
    return {
        "promoted": len(promoted),
        "promoted_skills": sorted(promoted),
        "blocked": len(blocked_detail),
        "blocked_detail": blocked_detail,
        "errors": len(error_detail),
        "error_detail": error_detail,
    }


def promote_one(
    engine: Any,
    *,
    server_name: str,
    name: str,
    description: str,
    instructions: str,
) -> str:
    """Write ONE harvested skill through the existing runnable-ingest primitive.

    Also links the serving ``MCPServer`` node to the runnable resource so the
    binding between "this skill" and "the child whose tools it calls" is an
    edge in the graph, not an inference from a string property. The server node
    is upserted here rather than assumed: promotion runs BEFORE the atomic
    catalog slice write (and must still stand if that write is rejected), so it
    cannot depend on the catalog having created its link target.
    """
    from ...models.company_brain import DataClassification
    from ...protocols.source_connectors.base import ExternalAccess
    from ..core.session import resolve_session
    from .skill_workflow_ingest import ingest_runnable_skill

    resource_id = ingest_runnable_skill(
        engine,
        name=name,
        description=description,
        instructions=instructions,
        provider=f"mcp:{server_name}",
        mcp_server=server_name,
        # A fleet-harvested skill has no local SKILL.md frontmatter to read
        # ``skill_type`` from — it is, by construction, an atomic runnable
        # skill served over Skills-over-MCP, so it is classified explicitly
        # rather than defaulted (CONCEPT:AU-KG.ingest.fleet-catalog-relational-tables).
        skill_type="mcp_skill",
    )
    session = resolve_session(required_scope="kg:write")
    server_node_id = f"mcp_server_{server_name}"
    engine._upsert_node(
        "MCPServer",
        server_node_id,
        {
            "tenant_id": session.tenant,
            "classification": DataClassification.PUBLIC.value,
            "external_access": ExternalAccess(is_public=True).model_dump(mode="json"),
            "name": server_name,
        },
    )
    engine.link_nodes(server_node_id, resource_id, "SERVES", session=session)
    return resource_id
