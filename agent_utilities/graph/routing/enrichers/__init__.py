"""Routing enrichers — augment routing context with KG-derived signals."""

from .capability_designation import designate_specialists
from .self_model import (
    format_pheromone_affinities,
    format_proficiency_context,
    self_model_context,
)

__all__ = [
    "designate_specialists",
    "format_proficiency_context",
    "format_pheromone_affinities",
    "self_model_context",
]
