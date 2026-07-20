"""Deployment model registry extractor (CONCEPT:AU-KG.enrichment.a2a-capability-extraction).

Ingests the agent-utilities ``config.json`` model registry (``chat_models`` +
``embedding_models``) into the KG as ``Model`` nodes with their capabilities, so
the agent-synthesis layer can choose the right model per task (lite for routing,
KG-capable for heavy work) via OWL relationships instead of hard-coding.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent_utilities.core.config import setting

from ..models import ExtractionBatch, GraphNode
from ..registry import register_extractor


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:80] or "model"


def _default_config_path() -> str:
    base = setting("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "agent-utilities", "config.json")


def extract(config: Any = None) -> ExtractionBatch:
    """Read the config.json model registry → Model GraphNodes.

    ``config`` may be: a dict with ``data`` (parsed config) or ``path``; a path
    string; or None (use the XDG default path).
    """
    data: dict | None = None
    path = None
    if isinstance(config, dict):
        data = config.get("data")
        path = config.get("path")
    elif isinstance(config, str):
        path = config
    if data is None:
        path = path or _default_config_path()
        try:
            data = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ExtractionBatch(category="config_models")

    nodes: list[GraphNode] = []
    for m in data.get("chat_models", []) or []:
        mid = m.get("id")
        if not mid:
            continue
        nodes.append(
            GraphNode(
                id=f"model:{_slug(mid)}",
                type="Model",
                props={
                    "name": mid,
                    "kind": "chat",
                    "provider": m.get("provider", ""),
                    "base_url": m.get("base_url", ""),
                    "intelligence_level": m.get("intelligence_level", "normal"),
                    "can_route": bool(m.get("can_route", False)),
                    "can_kg": bool(m.get("can_kg", False)),
                    "tools_enabled": bool(m.get("tools_enabled", False)),
                    "reasoning": bool(m.get("reasoning", False)),
                    "vision": bool(m.get("vision", False)),
                    "context_window": m.get("context_window", 0),
                    "parallel_instances": m.get("parallel_instances", 1),
                    "max_parallel_calls": m.get("max_parallel_calls", 1),
                },
            )
        )
    for m in data.get("embedding_models", []) or []:
        mid = m.get("id")
        if not mid:
            continue
        nodes.append(
            GraphNode(
                id=f"model:{_slug(mid)}",
                type="Model",
                props={
                    "name": mid,
                    "kind": "embedding",
                    "provider": m.get("provider", ""),
                    "base_url": m.get("base_url", ""),
                    "context_window": m.get("context_window", 0),
                    "parallel_instances": m.get("parallel_instances", 1),
                    "max_parallel_calls": m.get("max_parallel_calls", 1),
                },
            )
        )
    return ExtractionBatch(category="config_models", nodes=nodes)


register_extractor(
    "config_models",
    extract,
    description="agent-utilities config.json model registry → Model nodes",
)
