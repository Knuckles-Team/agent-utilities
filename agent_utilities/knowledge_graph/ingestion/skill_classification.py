"""Skill classification write-back (CONCEPT:AU-KG.ingest.skill-classification-writeback).

Lets an operator turn an "unclassified" skill (one whose declared/derived
``skill_type`` is outside the known set -- see
:func:`~.skill_workflow_ingest.classify_skill_type`) into a classified
one, and persist that choice so it survives re-ingestion rather than being
silently overwritten by the next ``fleet-tool-schema-sync`` pass.

**Two-tier persistence, both real, neither silent:**

1. **Source-of-truth attempt**: try to write the ``skill_type`` frontmatter
   field directly on the skill's own ``SKILL.md``. This is the "real" fix --
   it is what every ingester reads from on a fresh corpus checkout. It only
   succeeds where the process actually has write access to the
   ``universal-skills`` source tree (true in local/dev checkouts; **false in
   every deployed profile**, where ``/skills`` is an NFS export mounted
   ``readOnly: true`` -- see ``services/graph-os/k8s/graph-os.deployment.yaml``
   -- and empirically confirmed unwritable even for volumes whose mount
   entry does not itself say ``readOnly`` (``/au`` was tested: the mount
   flag is absent, the underlying NFS export is still read-only). This
   module never trusts a mount flag or an ``os.access()`` permission-bit
   check for this decision -- both can lie about an NFS export's real
   read-only state. It ALWAYS attempts the actual write and reports exactly
   what happened.

2. **Durable override** (EH-345: ``epistemic_graph``'s
   ``FleetCatalogClient.set_override`` -- was
   ``fleet_catalog_tables.write_skill_classification_override``): written
   unconditionally, regardless of whether (1) succeeded. This lands in EG's
   own fleet-catalog projection, which this process can always write (it is
   not the NFS-mounted source tree); the projection applies it at READ
   time, so it is consulted on every future read with no separate
   "refresh the row" write needed, and it survives the next re-sync even
   when the source file itself could not be touched.

``reclassify_skill`` returns which of the two actually landed. **Fail
closed**: the result's ``persisted`` field is ``True`` only when at least one
of them provably succeeded; a caller (REST/MCP/UI) MUST treat ``persisted:
False`` as a failure and MUST NOT report success -- there is no silent
"queued, will apply eventually" state here, only "wrote the file", "wrote a
durable override that the catalog will honor from now on", or "failed,
here is why".
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, TypedDict

from .skill_workflow_ingest import (
    classify_skill_type,
    discover_atomic_skill_files,
    discover_workflow_skill_files,
    parse_workflow_skill,
    skill_reference,
)

logger = logging.getLogger(__name__)

# The only skill_type values a human may assign through this capability.
# `mcp_skill` is a system-assigned kind for fleet-harvested skills (there is
# no SKILL.md to attribute it to -- the skill lives on a child MCP server),
# so it is deliberately not offered here.
ALLOWED_SKILL_TYPES = frozenset({"skill", "workflow", "graph"})

_SKILL_TYPE_LINE_RE = re.compile(r"^skill_type\s*:.*$", re.MULTILINE)


class SkillClassificationResult(TypedDict):
    """Typed contract for :func:`reclassify_skill`'s return value.

    A typed shape (not a bare ``dict``) so the producer/consumer key contract
    cannot silently drift between this function, the ``skill_classify`` MCP
    tool that serializes it, and its REST/UI callers.
    """

    persisted: bool
    reason: str | None
    skill_type: str
    classification: str
    persisted_to_source_file: bool
    persisted_as_durable_override: bool
    catalog_refreshed: bool


class SkillClassificationError(ValueError):
    """The requested classification itself is malformed.

    Distinct from a persistence failure (reported via the result dict's
    ``persisted``/``reason`` fields): this is raised for a caller error --
    an unknown ``skill_type`` -- before any write is attempted.
    """


def _fleet_catalog_lookup_row(
    fleet_catalog: Any, component_id: str
) -> dict[str, Any] | None:
    """One skill's fleet-catalog row as a plain dict, or ``None`` on a miss.

    EH-345 replacement for the deleted ``fleet_catalog_tables.get_skill_row``.
    ``fleet_catalog`` may be ``None`` (EG's fleet-catalog surface not yet
    landed on this deployment) or the lookup may legitimately find nothing
    (no ConnectorPack import has covered ``component_id`` yet) -- both are
    reported as a miss (``None``), never raised, matching the deleted
    function's own "unknown id -> None" contract.
    """
    if fleet_catalog is None:
        return None
    try:
        answer = fleet_catalog.lookup([component_id])
    except Exception as exc:  # noqa: BLE001 — lookup is best-effort
        logger.error(
            "fleet catalog lookup failed for %s (%s: %s)",
            component_id,
            type(exc).__name__,
            exc,
        )
        return None
    rows = getattr(answer, "rows", None) or ()
    for entry in rows:
        row = getattr(entry, "row", None)
        component = getattr(row, "component", None)
        if component is None or getattr(component, "id", None) != component_id:
            continue
        return {
            "name": getattr(component, "name", ""),
            "description": getattr(component, "description", ""),
            "uri": getattr(row, "uri", ""),
            "provider": str(getattr(component, "connector", "") or ""),
            "mcp_server": getattr(component, "server_name", ""),
            "enabled": bool(getattr(component, "enabled", True)),
            "tenant_id": getattr(getattr(component, "acl", None), "tenant_id", ""),
            "skill_type": str(getattr(row, "skill_type", "") or ""),
        }
    return None


def _iter_all_skill_files(root: str | None) -> list[Path]:
    """Every discoverable ``SKILL.md`` under ``root`` (default corpus), deduped."""
    seen: set[Path] = set()
    files: list[Path] = []
    for f in (*discover_atomic_skill_files(root), *discover_workflow_skill_files(root)):
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            files.append(f)
    return files


def _find_skill_file(name: str, root: str | None) -> Path | None:
    """Resolve a skill by name to its ``SKILL.md`` path via a fresh corpus scan.

    No stored path exists anywhere for a skill (deliberately --
    ``skill_reference``'s own docstring: filesystem locations are runtime
    discovery details and must never be written to graph nodes, reports,
    logs, or traces). This re-derives the SAME stable reference every
    ingester already computes from a name (:func:`skill_reference`) and
    matches on it, so it finds exactly the file an ingester would have used
    for that name.
    """
    target_ref = skill_reference(name)
    for f in _iter_all_skill_files(root):
        parsed = parse_workflow_skill(f)
        if parsed is not None and parsed.get("source_ref") == target_ref:
            return f
    return None


def _write_frontmatter_skill_type(
    skill_md: Path, value: str
) -> tuple[bool, str | None]:
    """Rewrite ONLY the ``skill_type`` frontmatter line, verified after write.

    Always makes a real write attempt -- this IS the write-access probe
    (mount flags and ``os.access()`` are not trusted; only an actual
    ``OSError`` from a real write proves a mount is read-only). Writes via a
    same-directory temp file + atomic ``os.replace`` so a failure never
    leaves a half-written ``SKILL.md`` behind. Re-reads the file afterward
    and refuses to report success unless the new value is actually present
    on disk.
    """
    try:
        content = skill_md.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"could not read the skill file ({type(exc).__name__})"

    if not content.startswith("---"):
        return False, "the skill file has no YAML frontmatter block to edit"
    parts = content.split("---", 2)
    if len(parts) < 3:
        return False, "the skill file's frontmatter block is malformed"

    fm_text = parts[1]
    new_line = f"skill_type: {value}"
    if _SKILL_TYPE_LINE_RE.search(fm_text):
        new_fm_text = _SKILL_TYPE_LINE_RE.sub(new_line, fm_text, count=1)
    else:
        new_fm_text = f"{fm_text.rstrip(chr(10))}\n{new_line}\n"
    new_content = f"---{new_fm_text}---{parts[2]}"

    tmp_path = skill_md.parent / f".{skill_md.name}.classify.tmp"
    try:
        tmp_path.write_text(new_content, encoding="utf-8")
        os.replace(tmp_path, skill_md)
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:  # noqa: BLE001 -- best-effort cleanup of a
            # leftover temp file after the write itself already failed; the
            # real cause is the write failure below (returned to the
            # caller), this is just tidiness and must not mask it.
            logger.debug(
                "skill classification: temp-file cleanup failed for %s (%s)",
                skill_reference(skill_md.parent.name),
                cleanup_exc,
            )
        return False, (
            "the skills source tree is not writable from this process "
            f"({type(exc).__name__})"
        )

    try:
        verify = skill_md.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"wrote the file but could not verify it ({type(exc).__name__})"
    if new_line not in verify:
        return (
            False,
            "post-write verification failed: the file does not read back the new classification",
        )
    return True, None


def reclassify_skill(
    engine: Any,
    *,
    skill_id: str,
    skill_type: str,
    principal: str,
    root: str | None = None,
) -> SkillClassificationResult:
    """Classify a skill and persist the choice so it survives the next sync.

    Args:
        engine: the live ``IntelligenceGraphEngine`` (or a test double
            exposing an equivalent ``graph_compute``/``client`` surface).
        skill_id: EH-345 (was the AU SQL-tier bound id, e.g.
            ``"skill:my-skill__tenant_local"``): now the EG fleet-catalog
            component id (``mcp:<connector>/skill/<name>``) a ConnectorPack
            import assigns the skill. Until ``eg-pack`` has imported a pack
            covering this skill, no such id resolves and this returns
            ``persisted=False`` honestly (see the lookup miss below) rather
            than guessing one.
        skill_type: one of :data:`ALLOWED_SKILL_TYPES`.
        principal: the authenticated caller. EH-345: kept for the caller
            contract and any future audit-trail projection, though EG's
            ``set_override`` binds the actual writer from the verified
            request context itself rather than this argument (never a
            caller-supplied identity).
        root: optional explicit corpus root override (tests only; production
            always uses the installed ``universal_skills`` package).

    Returns:
        ``{"persisted": bool, "reason": str | None, "skill_type": str,
        "classification": str, "persisted_to_source_file": bool,
        "persisted_as_durable_override": bool, "catalog_refreshed": bool}``.

        ``persisted`` is ``True`` iff at least one of the source file or the
        durable override was provably written. Callers MUST render
        ``persisted: False`` as a failure -- never as success, and never as
        a silent "queued" state.

    Raises:
        SkillClassificationError: ``skill_type`` is not one of
            :data:`ALLOWED_SKILL_TYPES` -- a caller error, checked before any
            write is attempted.
    """
    normalized = str(skill_type or "").strip().lower()
    if normalized not in ALLOWED_SKILL_TYPES:
        raise SkillClassificationError(
            f"skill_type must be one of {sorted(ALLOWED_SKILL_TYPES)}, got {skill_type!r}"
        )
    _, classification = classify_skill_type(normalized)

    fleet_catalog = getattr(
        getattr(getattr(engine, "graph_compute", None), "client", None),
        "fleet_catalog",
        None,
    )
    row = _fleet_catalog_lookup_row(fleet_catalog, skill_id)
    if row is None:
        return {
            "persisted": False,
            "reason": "unknown skill_id -- no catalog row found for this tenant",
            "skill_type": normalized,
            "classification": classification,
            "persisted_to_source_file": False,
            "persisted_as_durable_override": False,
            "catalog_refreshed": False,
        }
    name = str(row.get("name") or "")

    # 1. Best-effort source-of-truth write. Always attempted -- never assumed
    # impossible from a mount flag or permission bit (see module docstring).
    skill_md = _find_skill_file(name, root)
    disk_reason: str | None
    if skill_md is None:
        disk_written = False
        disk_reason = "no on-disk SKILL.md could be located for this skill"
    else:
        disk_written, disk_reason = _write_frontmatter_skill_type(skill_md, normalized)
        if not disk_written:
            logger.info(
                "skill classification: source-file write refused for %s (%s)",
                skill_reference(name),
                disk_reason,
            )

    # 2. Durable override -- always attempted, regardless of (1)'s outcome.
    # This is what makes the classification survive the next re-sync even
    # when the source file could not be touched. EH-345:
    # FleetCatalogClient.set_override; the projection applies it at READ
    # time (never touched by a pack re-import), so there is no separate
    # "refresh the row" step to run afterward -- re-lookup below just
    # confirms what the override write already guarantees.
    override_written = False
    catalog_refreshed = False
    if fleet_catalog is not None:
        try:
            from epistemic_graph.generated.fleet_catalog import (
                FleetOverrideSetRequest,
            )

            receipt = fleet_catalog.set_override(
                FleetOverrideSetRequest(
                    component_id=skill_id,
                    value={"field": "skill_type", "skill_type": normalized},
                )
            )
            override_written = str(getattr(receipt, "disposition", "")) in {
                "written",
                "replayed",
            }
        except Exception as exc:  # noqa: BLE001 — override write is best-effort
            logger.error(
                "skill classification override write failed for %s (%s: %s)",
                skill_id,
                type(exc).__name__,
                exc,
            )

    persisted = disk_written or override_written
    if persisted and override_written:
        refreshed_row = _fleet_catalog_lookup_row(fleet_catalog, skill_id)
        catalog_refreshed = bool(
            refreshed_row is not None and refreshed_row.get("skill_type") == normalized
        )

    reason: str | None
    if persisted:
        reason = None
    elif skill_md is None:
        reason = disk_reason
    else:
        reason = disk_reason or "override write failed"

    return {
        "persisted": persisted,
        "reason": reason,
        "skill_type": normalized,
        "classification": classification,
        "persisted_to_source_file": disk_written,
        "persisted_as_durable_override": override_written,
        "catalog_refreshed": catalog_refreshed,
    }
