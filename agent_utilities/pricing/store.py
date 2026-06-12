"""Pricing refresh orchestration.

CONCEPT:ECO-4.40 — Unified model pricing catalog.

Fetches LiteLLM rates and overlays them onto the process-wide catalog,
optionally persisting the resolved rows into a usage backend's
``model_pricing`` table. Network failure is non-fatal: the offline fallback
remains in effect. Invoked by the consolidated KG daemon (see
``gateway/daemon.py``) on a daily cadence and once at startup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .catalog import PricingCatalog, get_pricing_catalog
from .litellm import LITELLM_URL, fetch_litellm_pricing

if TYPE_CHECKING:  # avoid a hard import cycle with the usage package
    from agent_utilities.usage.backend import UsageBackend

logger = logging.getLogger(__name__)


def refresh_catalog(
    *,
    url: str = LITELLM_URL,
    catalog: PricingCatalog | None = None,
    backend: UsageBackend | None = None,
) -> int:
    """Refresh the catalog from LiteLLM. Returns the number of models merged.

    Best-effort: on any fetch/parse error the offline fallback stays in place
    and 0 is returned. When ``backend`` is supplied the merged rows are also
    persisted so other processes (and SQL aggregation) see current rates.
    """
    catalog = catalog or get_pricing_catalog()
    try:
        entries = fetch_litellm_pricing(url)
    except Exception as exc:  # noqa: BLE001 — refresh must never be fatal
        logger.warning("LiteLLM pricing refresh failed, using fallback: %s", exc)
        return 0
    catalog.merge(entries)
    if backend is not None:
        try:
            backend.upsert_pricing(entries)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Persisting pricing to usage backend failed: %s", exc)
    logger.info("Refreshed pricing catalog: %d models from LiteLLM", len(entries))
    return len(entries)
