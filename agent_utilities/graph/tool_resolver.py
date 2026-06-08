#!/usr/bin/python
from __future__ import annotations

"""Tool gap-fill resolver for workflow materialization (CONCEPT:ECO-4.0 / ORCH-1.8).

A skill-workflow template declares the tools each step needs (``AgentSpec.tools``).
When `graph_orchestrate` materializes the template into a live agent graph, a needed
tool may be absent (e.g. a workflow wants a *gitlab-pr* tool that isn't bound). This
resolver decides, per requested tool: **available** (bind it), **filled** (substitute
an available tool that provides the same capability, via the capability index), or
**missing** (surface a precise gap).

`resolve_tools` is pure + deterministic (inject ``available`` + ``designate_fn``), so
it is fully testable without a live engine. `build_designate_fn` / `available_tags`
derive those from a live engine best-effort; when availability can't be determined
they return ``None`` and the resolver passes the requested tools through unchanged
(no regression on the execution hot path).

Concept: tool-resolver
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

DesignateFn = Callable[[str], list[str]]


@dataclass
class ToolResolution:
    """Per-agent tool resolution outcome."""

    requested: list[str] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)  # what to actually bind
    filled: dict[str, str] = field(default_factory=dict)  # requested → substitute
    missing: list[str] = field(default_factory=list)  # no available tool/substitute

    @property
    def has_gaps(self) -> bool:
        return bool(self.missing)


def resolve_tools(
    requested: list[str] | None,
    *,
    available: set[str] | list[str] | None = None,
    designate_fn: DesignateFn | None = None,
) -> ToolResolution:
    """Resolve requested tools against availability, gap-filling via capability.

    - ``available is None`` → availability is unknown → pass requested through
      (deduped); nothing is marked missing (safe default for the hot path).
    - otherwise: present → resolved; absent but a capability substitute exists →
      filled; else → missing.
    """
    req = [t for t in dict.fromkeys(requested or []) if t]
    if available is None:
        return ToolResolution(requested=req, resolved=list(req))
    avail = set(available)
    res = ToolResolution(requested=req)
    for t in req:
        if t in avail:
            res.resolved.append(t)
            continue
        sub = None
        if designate_fn is not None:
            try:
                sub = next((c for c in (designate_fn(t) or []) if c in avail), None)
            except Exception:  # pragma: no cover - designate is best-effort
                sub = None
        if sub:
            res.filled[t] = sub
            res.resolved.append(sub)
        else:
            res.missing.append(t)
    res.resolved = list(dict.fromkeys(res.resolved))
    return res


def build_designate_fn(engine: Any) -> DesignateFn | None:
    """Best-effort capability→candidate-tool resolver from the engine's index."""
    idx = getattr(engine, "capability_index", None)
    designate = getattr(idx, "designate", None)
    embed = getattr(engine, "embed", None) or getattr(engine, "embed_text", None)
    if not callable(designate) or not callable(embed):
        return None

    def _fn(capability: str) -> list[str]:
        try:
            vec = embed(capability)
            out = designate(vec, {capability}, top_k=3)
            ids: list[str] = []
            for d in out:
                did = getattr(d, "id", None)
                if did:
                    ids.append(str(did))
            return ids
        except Exception:  # pragma: no cover - best-effort
            return []

    return _fn


def available_tags(engine: Any) -> set[str] | None:
    """Best-effort set of bindable tool tags; ``None`` when undeterminable.

    NOTE (staged): the live tool-tag universe is spread across MCP servers + skills
    with no single registry, so this returns ``None`` today → the resolver passes
    tools through. Full live gap-fill requires reconciling the tool-tag namespace
    with the capability namespace (see WORKFLOW_ABSTRACTION_STRATEGY.md).
    """
    fn = getattr(engine, "available_tool_tags", None)
    if callable(fn):
        try:
            tags = fn()
            return set(tags) if tags else None
        except Exception:  # pragma: no cover
            return None
    return None


def resolve_agent_tools(engine: Any, requested: list[str] | None) -> ToolResolution:
    """Convenience: resolve an agent's tools against the live engine (defensive)."""
    try:
        return resolve_tools(
            requested,
            available=available_tags(engine),
            designate_fn=build_designate_fn(engine),
        )
    except Exception:  # pragma: no cover - never break the caller
        return ToolResolution(
            requested=list(requested or []), resolved=list(requested or [])
        )


__all__ = [
    "ToolResolution",
    "resolve_tools",
    "build_designate_fn",
    "available_tags",
    "resolve_agent_tools",
]
