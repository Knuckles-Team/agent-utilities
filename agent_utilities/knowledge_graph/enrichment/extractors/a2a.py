"""A2A (Agent-to-Agent) agent-card source extractor (CONCEPT:KG-2.10).

Self-registering extractor that maps externally-defined **A2A AgentCards** into
the uniform ``ExtractionBatch`` shape (typed ``GraphNode`` + ``EnrichmentEdge``)
so the KG can discover and reason over external A2A agents — and their exposed
skills — through the one generic writer, with no edits to any shared hub file.

AgentCards are **injected** via ``config["cards"]`` (a list of standard A2A
agent-card dicts) or referenced as a path / list of paths to JSON card files,
which are read and ``json.load``-ed tolerantly. Standard card keys handled
(all optional): ``name``, ``description``, ``url`` / ``endpoint``, ``version``,
``provider``, ``capabilities`` (dict), ``skills`` (list of
``{id|name, description, tags}``), ``defaultInputModes`` /
``defaultOutputModes``, ``authentication`` / ``securitySchemes``. Field access
is tolerant of missing keys. This module performs **no network calls**.

Mapping per card::

    card  -> GraphNode(type="A2AAgentCard", id="a2a:<slug(name)>",
                       props={name, description, url, version, provider})
    skill -> GraphNode(type="Skill",
                       id="skill:a2a:<slug(card)>:<slug(skill)>",
                       props={description, tags})
             + EnrichmentEdge(card_id -> skill_id, rel_type="EXPOSES_SKILL")
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

try:  # keep self-contained but reuse the canonical helper when available
    from .document import slug as _slug
except Exception:  # pragma: no cover - defensive fallback
    import re

    def _slug(text: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
        return s[:80] or "agent"


CATEGORY = "a2a"


def _get(record: Any, key: str, default: Any = None) -> Any:
    """Tolerant field access for dict records (or attr-style objects)."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _scalar(value: Any) -> str | None:
    """Normalise a possibly-nested value to a scalar string (or None)."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = (
            value.get("name")
            or value.get("organization")
            or value.get("id")
            or value.get("value")
        )
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tags(value: Any) -> list[str]:
    """Coerce a skill's tags field into a clean list of strings."""
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list | tuple | set):
        return []
    out: list[str] = []
    for item in value:
        text = _scalar(item)
        if text:
            out.append(text)
    return out


def _load_json(path: str) -> list[dict]:
    """Read+json.load a card file, returning a list of card dicts (tolerant)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    if isinstance(data, dict):
        # A file may hold a single card, or a wrapper with a "cards" key.
        inner = data.get("cards")
        if isinstance(inner, list):
            return [c for c in inner if isinstance(c, dict)]
        return [data]
    if isinstance(data, list):
        return [c for c in data if isinstance(c, dict)]
    return []


def _collect_cards(config: Any) -> list[dict]:
    """Resolve ``config`` into a flat list of AgentCard dicts.

    Accepts: a dict carrying ``cards`` (list of card dicts and/or path strings),
    a single path string, or a list of paths / card dicts.
    """
    raw: Any
    if isinstance(config, str):
        raw = [config]
    elif isinstance(config, list | tuple):
        raw = list(config)
    else:
        raw = _get(config, "cards", []) or []
        if isinstance(raw, str | dict):
            raw = [raw]

    cards: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            cards.append(item)
        elif isinstance(item, str):
            if os.path.exists(item):
                cards.extend(_load_json(item))
        else:  # attr-style object: best-effort dict view
            data = _get(item, "__dict__")
            if isinstance(data, dict):
                cards.append(data)
    return cards


def extract(config: Any) -> ExtractionBatch:
    """Extract A2A agent cards into a uniform ``ExtractionBatch``.

    ``config`` carries ``cards`` (a list of AgentCard dicts) or is a path / list
    of paths to JSON card files. Each card becomes an ``A2AAgentCard`` node; each
    declared skill becomes a ``Skill`` node linked back via ``EXPOSES_SKILL``.
    """
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []

    for card in _collect_cards(config):
        name = _scalar(_get(card, "name"))
        if not name:
            continue
        card_slug = _slug(name)
        card_id = f"a2a:{card_slug}"
        url = _scalar(_get(card, "url")) or _scalar(_get(card, "endpoint"))
        nodes.append(
            GraphNode(
                id=card_id,
                type="A2AAgentCard",
                props={
                    "name": name,
                    "description": _scalar(_get(card, "description")),
                    "url": url,
                    "version": _scalar(_get(card, "version")),
                    "provider": _scalar(_get(card, "provider")),
                },
            )
        )

        skills = _get(card, "skills") or []
        if isinstance(skills, dict):
            skills = list(skills.values())
        if not isinstance(skills, list | tuple):
            continue
        for skill in skills:
            skill_name = _scalar(_get(skill, "name")) or _scalar(_get(skill, "id"))
            if not skill_name:
                continue
            skill_id = f"skill:a2a:{card_slug}:{_slug(skill_name)}"
            nodes.append(
                GraphNode(
                    id=skill_id,
                    type="Skill",
                    props={
                        "description": _scalar(_get(skill, "description")),
                        "tags": _tags(_get(skill, "tags")),
                    },
                )
            )
            edges.append(
                EnrichmentEdge(
                    source=card_id,
                    target=skill_id,
                    rel_type="EXPOSES_SKILL",
                )
            )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY,
    extract,
    description="A2A agent cards (external agents) → KG",
)
