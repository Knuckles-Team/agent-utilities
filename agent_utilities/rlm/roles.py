"""CONCEPT:AU-ORCH.routing.conductor-per-step-model (RLM extension) — role-specialized RLM-GEPA model resolution.

Resolves the RLM-GEPA functional roles — ``rlm-executor`` / ``rlm-sublm`` (cheap, run the skill) and
``rlm-proposer`` (strong, reflects on traces and rewrites the skill) — through the ORCH-1.27 model
registry, with a graceful fallback to a model-id string. A skill optimized with a *cheap* executor
still lifts a *strong* executor at eval (the AppWorld RLM-GEPA cost/quality trick), and binding to
roles means the same config works across any provider pool.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# The three RLM-GEPA roles (must exist in models.model_registry._DEFAULT_ROLE_ROUTING).
RLM_ROLES = ("rlm-executor", "rlm-sublm", "rlm-proposer")


def rlm_role_model(role: str, fallback: Any = None) -> Any:
    """Resolve an RLM role to a concrete model via ORCH-1.27, falling back to ``fallback``.

    Returns a pydantic-ai model instance when the registry resolves the role, else ``fallback``
    (typically a model-id string). Never raises — role resolution is best-effort.
    """
    try:
        from agent_utilities.core.model_factory import create_model

        model = create_model(role=role)
        return model if model is not None else fallback
    except Exception as e:  # noqa: BLE001 - resolution is best-effort
        logger.debug("rlm role resolution failed for %r: %s", role, e)
        return fallback
