"""Canonical inventory for agent-utilities' pre-bundled workflow skills."""

from __future__ import annotations

BUNDLED_SKILLS: tuple[str, ...] = (
    "agent-utilities-deployment",
    "agent-utilities-development",
    "agent-utilities-evolution",
    "agent-utilities-self-evolution",
    "agent-utilities-source-integration",
    "autonomous-contribution",
    "graph-engine-and-modalities",
    "graph-ingestion-and-integration",
    "graph-modeling-and-mutation",
    "graph-orchestration-and-automation",
    "graph-query-and-explanation",
    "graph-research-and-analysis",
    "graph-runtime-and-governance",
)

__all__ = ["BUNDLED_SKILLS"]
