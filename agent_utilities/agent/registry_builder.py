#!/usr/bin/python
"""Agent Registry Builder Module.

CONCEPT:ORCH-1.2

This module provides logic to ingest specialized prompt metadata into the
Knowledge Graph as ``PromptNode`` entries. Prompts are JSON blueprints that
live under ``agent_utilities/prompts/*.json``; each file is read as a dict
and its top-level keys (``name``, ``description``, ``capabilities``,
``tags``, ``content``) are projected directly onto a ``PromptNode``.

Two JSON schemas are accepted in ``_resolve_fields`` because the repository
currently mixes two valid shapes:

* The modern blueprint schema: ``{name, description, capabilities, content}``.
* The ``StructuredPrompt`` schema still used by ~40 prompts:
  ``{task, goal, tools, input}``.

Markdown fallbacks have been removed. Only ``*.json`` files under
``prompts/`` are ingested.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from agent_utilities.core.workspace import (
    get_agent_workspace,
)
from agent_utilities.models.knowledge_graph import RegistryNodeType

logger = logging.getLogger(__name__)


RESERVED_CORE_NODES: set[str] = {
    "usage_guard",
    "router",
    "dispatcher",
    "synthesizer",
    "onboarding",
    "error_recovery",
    "parallel_batch_processor",
    "expert_executor",
    "research_joiner",
    "execution_joiner",
    "mcp_router",
    "mcp_server_execution",
    "approval_gate",
    "memory_selection",
    "expert_dispatch",
}


def _load_prompt_metadata(pfile: Path) -> dict[str, Any] | None:
    """Load a single JSON prompt file and return a unified metadata dict.

    Only the modern JSON blueprint schema (``*.json`` with top-level
    ``name``/``description``/``capabilities``/``content`` keys, or the
    ``StructuredPrompt`` variant with ``task``/``goal``/``tools``/``input``)
    is supported. Returns ``None`` if the file cannot be parsed.
    """
    if pfile.suffix != ".json":
        logger.debug("Skipping unsupported prompt file: %s", pfile.name)
        return None

    try:
        raw = pfile.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(f"Failed to read prompt file {pfile.name}: {e}")
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse prompt JSON {pfile.name}: {e}")
        return None

    if not isinstance(data, dict):
        logger.warning(f"Prompt {pfile.name} is not a JSON object")
        return None

    data.setdefault("name", pfile.stem)
    data.setdefault("type", "prompt")
    return data


def _resolve_fields(data: dict[str, Any], stem: str) -> tuple[str, str, list[str], str]:
    """Extract ``(name, description, capabilities, system_prompt)`` from data.

    Accepts both the modern blueprint schema
    (``{name, description, capabilities, content}``) and the
    ``StructuredPrompt`` schema (``{task, goal, tools, input}``) because the
    repository currently ships both valid JSON shapes side-by-side.
    """
    name = str(data.get("name") or data.get("task") or stem)
    description = str(data.get("description") or data.get("goal") or "")
    capabilities_raw: Any = (
        data.get("capabilities") or data.get("tools") or data.get("skills") or []
    )
    if isinstance(capabilities_raw, list):
        capabilities = [str(c) for c in capabilities_raw]
    elif isinstance(capabilities_raw, str):
        capabilities = [capabilities_raw]
    else:
        capabilities = []

    # CONCEPT:ORCH-1.80 — resolve the body via the single canonical resolver so
    # decomposed ``instructions.core_directive`` prompts are not stored empty.
    from agent_utilities.prompting.structured import resolve_body

    system_prompt = resolve_body(data)
    return name, description, capabilities, system_prompt


# Source label for the packaged base prompts (``agent_utilities/prompts/``).
_BASE_SOURCE = "agent_utilities"
# Source label for the operator XDG overlay (``prompts_dir()``).
_OVERLAY_SOURCE = "__user__"


def _prompt_id(source_label: str, name: str, data: dict[str, Any]) -> str:
    """Compute the namespaced ``PromptNode`` id for a prompt (CONCEPT:KG-2.141).

    * Packaged base prompts keep the bare ``prompt:<name>`` id (preserving the
      existing ~90 ids and any references to them).
    * Fleet-contributed prompts are namespaced ``prompt:<provider>/<name>`` so
      80 packages never collide.
    * Overlay (user) prompts override the base namespace by default, or a named
      provider namespace when the file declares a ``provider`` field.
    """
    if source_label in (_BASE_SOURCE, _OVERLAY_SOURCE):
        provider = data.get("provider") or data.get("source")
        if isinstance(provider, str) and provider and ":" not in provider:
            return f"prompt:{provider}/{name}"
        return f"prompt:{name}"
    return f"prompt:{source_label}/{name}"


def _iter_prompt_sources() -> list[tuple[str, Path]]:
    """Build the ordered ``(source_label, json_file)`` list for prompt ingestion.

    Precedence is list order, low→high; because ``_upsert_node`` is keyed on the
    namespaced id, a later same-id source naturally overwrites an earlier one:

      1. packaged base   — ``agent_utilities/prompts/*.json``
      2. fleet-contributed — each ``agent_utilities.prompt_providers`` dir
      3. operator overlay  — ``prompts_dir()`` (``~/.config/agent-utilities/prompts``)
    """
    from agent_utilities.core.paths import prompts_dir
    from agent_utilities.core.providers import resolve_prompt_provider_dirs

    sources: list[tuple[str, Path]] = []

    base = Path(__file__).parent.parent / "prompts"
    if base.exists():
        sources += [(_BASE_SOURCE, f) for f in sorted(base.glob("*.json"))]

    # XDG-first (CONCEPT:OS-5.78): the materialized unified tree when populated, else
    # live ``prompt_providers`` entry-point discovery.
    for provider_name, pdir in resolve_prompt_provider_dirs():
        try:
            sources += [(provider_name, f) for f in sorted(pdir.glob("*.json"))]
        except OSError as e:  # pragma: no cover - defensive
            logger.debug("Could not list prompt provider %s: %s", provider_name, e)

    try:
        overlay = prompts_dir()
        if overlay.exists():
            sources += [(_OVERLAY_SOURCE, f) for f in sorted(overlay.glob("*.json"))]
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not list prompt overlay dir: %s", e)

    return sources


# CONCEPT:ORCH-1.100 — Built-in, KG-bound, dispatchable AgentTemplates.
#
# A prompt blueprint is the persona; an ``AgentTemplate`` is what makes that
# persona a *runnable* agent — it binds the system-prompt node, the toolsets,
# and the model preference so ``graph_orchestrate(action='execute_agent')``
# resolves it (``orchestration/agent_runner._resolve_agent_from_kg``). The
# ``agent-utilities-expert`` is the resident ecosystem expert: it is bound to its
# base prompt, the platform's own management toolsets, and the local qwen model.
_BUILTIN_AGENT_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "at:agent-utilities-expert",
        "name": "agent-utilities-expert",
        "role": "ecosystem-expert",
        "description": (
            "Resident expert of the entire agent-utilities ecosystem — understands "
            "the 5 pillars, the one-engine ontology-driven Knowledge Graph, the dev "
            "discipline, SDD, graph-os + the multiplexer, connectors/ingestion, and "
            "the evolution loop; manages and evolves the platform as code."
        ),
        "system_prompt_id": "prompt:agent-utilities-expert",
        "toolset_ids": [
            "graph-os",
            "mcp-multiplexer",
            "repository-manager-mcp",
            "data-science-mcp",
            "scholarx-mcp",
        ],
        # Default to the LOCAL fleet model (the model router gets final say).
        "model_preference": "qwen/qwen3.6-27b",
        "execution_tier": "standard",
    },
]


def seed_builtin_agent_templates(engine: Any) -> int:
    """Upsert the built-in AgentTemplate nodes + their USES_PROMPT edges.

    CONCEPT:ORCH-1.100 — makes the packaged expert personas *dispatchable* agents
    rather than bare prompt blueprints. Each template (1:1 with its base prompt)
    is upserted via the same engine path the prompts use, and wired to its
    ``prompt:<name>`` node with a ``USES_PROMPT`` edge so resolution can recover
    the system prompt. Best-effort and idempotent (keyed on the stable node id);
    returns the number of templates seeded.
    """
    from agent_utilities.models.knowledge_graph import (
        AgentTemplateNode,
        RegistryEdgeType,
    )

    backend = getattr(engine, "backend", None)
    if backend is None:
        return 0

    seeded = 0
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for tmpl in _BUILTIN_AGENT_TEMPLATES:
        try:
            node = AgentTemplateNode(
                id=tmpl["id"],
                name=tmpl["name"],
                description=tmpl.get("description", ""),
                role=tmpl.get("role", ""),
                system_prompt_id=tmpl.get("system_prompt_id", ""),
                toolset_ids=list(tmpl.get("toolset_ids", [])),
                model_preference=tmpl.get("model_preference", ""),
                execution_tier=tmpl.get("execution_tier", "standard"),
                timestamp=ts,
                is_permanent=True,
            )
            data = engine._serialize_node(node, label="AgentTemplate")
            engine._upsert_node("AgentTemplate", node.id, data)
            prompt_id = tmpl.get("system_prompt_id", "")
            if prompt_id:
                engine.add_edge(node.id, prompt_id, RegistryEdgeType.USES_PROMPT.value)
            seeded += 1
            logger.debug("[ORCH-1.100] Seeded AgentTemplate %s", node.id)
        except Exception as e:  # noqa: BLE001 — never break prompt ingestion
            logger.warning("Failed to seed AgentTemplate %s: %s", tmpl.get("id"), e)
    return seeded


async def ingest_prompts_to_graph():
    """Ingest JSON prompt blueprints into the Knowledge Graph prompt library.

    Replaces the legacy NODE_AGENTS.md registry with a dynamic KG-native
    discovery mechanism. Ingests, in precedence order (CONCEPT:KG-2.141):
    the packaged base prompts (``agent_utilities/prompts/``), every
    fleet-contributed ``agent_utilities.prompt_providers`` package, and the
    operator XDG overlay (``prompts_dir()``). Later sources override earlier
    ones of the same namespaced id.
    """

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.models.knowledge_graph import PromptNode

    workspace = get_agent_workspace()

    try:
        engine = IntelligenceGraphEngine.get_active()
        if not engine:
            db_path = str(workspace / "knowledge_graph.db")
            engine = IntelligenceGraphEngine(db_path=db_path)

        if engine.backend:
            engine.backend.create_schema()

        backend = engine.backend
        if backend is None:
            logger.warning("Graph backend not available, skipping prompt ingestion.")
            return None

        sources = _iter_prompt_sources()
        logger.debug("Found %d prompt files across all sources", len(sources))

        for source_label, pfile in sources:
            stem = pfile.stem
            # The reserved-core skip applies only to the packaged base layer;
            # fleet/overlay prompts are namespaced and cannot shadow core nodes.
            if pfile.name.startswith("_"):
                continue
            if source_label == _BASE_SOURCE and stem in RESERVED_CORE_NODES:
                logger.debug(f"Skipping reserved/internal node: {stem}")
                continue

            data = _load_prompt_metadata(pfile)
            if data is None:
                continue
            # Record provenance so the source is queryable on the node.
            data.setdefault(
                "source",
                source_label if source_label not in (_OVERLAY_SOURCE,) else "user",
            )

            try:
                name, description, capabilities, system_prompt = _resolve_fields(
                    data, stem
                )
                node_id = _prompt_id(source_label, name, data)
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                node = PromptNode(
                    id=node_id,
                    name=name,
                    description=description,
                    system_prompt=system_prompt,
                    json_blueprint=data,
                    capabilities=capabilities,
                    type=RegistryNodeType.PROMPT,
                    timestamp=ts,
                    is_permanent=True,
                )
                logger.debug("Ingesting prompt %s via engine.upsert", node_id)
                serialized_data = engine._serialize_node(node, label="Prompt")
                engine._upsert_node("Prompt", node.id, serialized_data)
            except Exception as e:
                logger.warning(f"Failed to ingest prompt {pfile.name}: {e}")

        # CONCEPT:ORCH-1.100 — once the prompt nodes exist, seed the built-in
        # AgentTemplates that bind them into dispatchable agents (USES_PROMPT).
        try:
            seeded = seed_builtin_agent_templates(engine)
            if seeded:
                logger.debug("Seeded %d built-in AgentTemplate(s)", seeded)
        except Exception as e:  # noqa: BLE001 — best-effort; never break ingest
            logger.warning("AgentTemplate seeding failed: %s", e)

        logger.info("Prompt library ingested into the Knowledge Graph.")
    except Exception as e:
        logger.info(f"Failed to ingest prompts: {e}")
    return None


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(ingest_prompts_to_graph())
