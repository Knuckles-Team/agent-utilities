"""CONCEPT:AU-ORCH.routing.conductor-per-step-model — role-specialized RLM resolution.

Resolves the depth-specific RLM roles: ``rlm-root`` for the high-capability
reasoning pass and ``rlm-executor`` / ``rlm-sublm`` for economical recursive work.
Binding to roles keeps the same runtime portable across provider pools.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# The three RLM roles (must exist in models.model_registry._DEFAULT_ROLE_ROUTING).
RLM_ROLES = ("rlm-executor", "rlm-sublm", "rlm-root")


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
