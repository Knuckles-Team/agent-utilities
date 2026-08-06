"""Promote prompts harvested from fleet MCP children into the Prompt corpus.

CONCEPT:AU-ECO.mcp.cross-process-prompt-harvest — the ``prompt://`` sibling of
``fleet_skill_harvest``'s CONCEPT:AU-ECO.mcp.cross-process-skill-harvest.

THE GAP THIS CLOSES
-------------------
``core.providers.resolve_prompt_provider_dirs()`` (used by
:func:`agent_utilities.agent.registry_builder.ingest_prompts_to_graph`)
discovers prompt providers with an in-process ``importlib.metadata
.entry_points()`` walk, so it only ever sees what is installed in
**graph-os's own serving venv**. The ~65-68 fleet ``agents/*`` packages that
declare an ``agent_utilities.prompt_providers`` entry point are deliberately
NOT co-installed there (``AGENTS.md`` "Dependency discipline" — the same
precondition ``fleet_skill_harvest`` documents), so their ``prompts/*.json``
files are *structurally* invisible to in-process discovery: live-pod
measurement (2026-08-06) found exactly 1 of 68 declared fleet prompt
providers (``langfuse-agent``) resolvable via ``entry_points()`` inside the
``graph-os`` container, which is why the ``:Prompt`` corpus (96 nodes) is
exactly the packaged base and nothing from the fleet's 143 on-disk
``prompts/*.json`` files.

The fix does not install anything and does not read the workspace filesystem
(a hostPath dependency graph-os only happens to have in THIS homelab
deployment — ``fleet_skill_harvest`` explicitly rejected that path for the
same reason). Instead, every fleet MCP server built via
``mcp.server_factory.build_server`` now ALSO exposes its own
``prompts/*.json`` files as ``prompt://{provider}/{stem}`` MCP resources
(mirroring ``_register_skill_providers``' ``skill://`` resources), and the
multiplexer's existing probe session for that child reads them back
(:meth:`~agent_utilities.mcp.multiplexer.MCPMultiplexer._harvest_prompt_bodies`)
before this module writes them through the SAME ``PromptNode`` shape/id the
boot sweep uses (:func:`agent_utilities.agent.registry_builder
.ingest_prompt_node`) — one writer, two discovery paths.

Unlike a skill, a harvested prompt needs no dispatchability precondition: it
is static text, not a tool invocation. The gate here is only "did the body
actually arrive and parse as the JSON blueprint schema" — anything else is
recorded per-prompt so the reason is queryable, never silently dropped.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def promote_harvested_prompts(engine: Any, catalog: dict[str, dict]) -> dict[str, Any]:
    """Promote every harvested fleet prompt body into the ``:Prompt`` corpus.

    ``catalog`` is the multiplexer's probe result — ``{server: {"tools": [...],
    "skills": [...], "prompts": [{name, provider, uri, body|harvest_error}],
    "error": str|None}}``.

    Returns ``{"promoted", "promoted_prompts", "blocked", "blocked_detail",
    "errors", "error_detail"}``. ``blocked_detail`` maps
    ``"<server>/<provider>/<name>"`` to the reason the body never became a
    node (no body served, or the body did not parse as the prompt blueprint
    schema) so the cause is queryable rather than inferred from an absence.
    """
    from ...agent.registry_builder import ingest_prompt_node

    promoted: list[str] = []
    blocked_detail: dict[str, str] = {}
    error_detail: dict[str, str] = {}

    for server_name, info in (catalog or {}).items():
        if not isinstance(info, dict):
            continue
        for entry in info.get("prompts") or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            provider = str(entry.get("provider") or server_name).strip()
            if not name or not provider:
                continue
            key = f"{server_name}/{provider}/{name}"

            body = entry.get("body")
            if not isinstance(body, str) or not body.strip():
                reason = str(
                    entry.get("harvest_error") or "prompt body not served"
                ).strip()
                blocked_detail[key] = reason
                logger.warning("Fleet prompt %s was NOT promoted — %s", key, reason)
                continue

            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("prompt body is not a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                error_detail[key] = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Fleet prompt %s body did not parse as the prompt "
                    "blueprint schema (%s: %s)",
                    key,
                    type(exc).__name__,
                    exc,
                )
                continue

            data.setdefault("name", name)
            data.setdefault("type", "prompt")
            data.setdefault("source", provider)
            try:
                node_id = ingest_prompt_node(
                    engine, source_label=provider, stem=name, data=data
                )
            except Exception as exc:  # noqa: BLE001 — one prompt's write must
                # not abort the sweep, but the cause is kept verbatim (never
                # reduced to a class name) and logged with its traceback.
                error_detail[key] = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Failed to promote harvested prompt %s (%s)",
                    key,
                    type(exc).__name__,
                    exc_info=True,
                )
                continue
            promoted.append(node_id)

    logger.info(
        "Cross-process prompt harvest: %d promoted, %d blocked, %d errors",
        len(promoted),
        len(blocked_detail),
        len(error_detail),
    )
    return {
        "promoted": len(promoted),
        "promoted_prompts": sorted(promoted),
        "blocked": len(blocked_detail),
        "blocked_detail": blocked_detail,
        "errors": len(error_detail),
        "error_detail": error_detail,
    }
