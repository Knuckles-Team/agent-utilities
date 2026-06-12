#!/usr/bin/python
from __future__ import annotations

"""Process-wide CompanyBrain runtime + enforcement gate (CONCEPT:KG-2.6).

The :class:`~agent_utilities.knowledge_graph.core.company_brain.CompanyBrain`
infrastructure (trust hierarchy, conflict resolution, provenance, data-level
ACLs, tenancy) is fully implemented but, historically, never instantiated in the
live read/write path. This module is the seam that activates it:

* :func:`get_company_brain` — one lazily-built, process-wide brain configured for
  source-authority conflict resolution and always-on provenance.
* :func:`brain_enforcement_enabled` — reads ``KG_BRAIN_ENFORCE`` (default off) so
  trust/permission enforcement is opt-in and the existing suite stays green.

The default :class:`TrustHierarchy` is seeded so source authority is declarative
("live systems beat stale docs") and can be overridden from XDG ``config.json``.
"""

import logging

from agent_utilities.core.config import setting

from ...models.company_brain import MergeStrategy, TrustHierarchyEntry
from .company_brain import CompanyBrain

logger = logging.getLogger(__name__)

_BRAIN: CompanyBrain | None = None

_TRUTHY = {"1", "true", "yes", "on"}

# Declarative default trust hierarchy: authority_level in [0,1] (higher wins),
# trust_decay_rate per-day (how fast authority ages — live systems barely decay,
# documents decay faster, inspiration decays fastest). Override via config.json
# key ``kg_trust_hierarchy`` (list of these field dicts).
_DEFAULT_TRUST: tuple[dict, ...] = (
    {
        "source_system": "human_review",
        "authority_level": 0.98,
        "trust_decay_rate": 0.0,
        "rationale": "Explicit human judgment is the highest authority.",
    },
    {
        "source_system": "servicenow",
        "authority_level": 0.90,
        "trust_decay_rate": 0.02,
        "rationale": "Live ITSM system of record.",
    },
    {
        "source_system": "erpnext",
        "authority_level": 0.90,
        "trust_decay_rate": 0.02,
        "rationale": "Live ERP system of record.",
    },
    {
        "source_system": "camunda",
        "authority_level": 0.88,
        "trust_decay_rate": 0.02,
        "rationale": "Live process engine.",
    },
    {
        "source_system": "crm",
        "authority_level": 0.88,
        "trust_decay_rate": 0.03,
        "rationale": "Live CRM record.",
    },
    {
        "source_system": "leanix",
        "authority_level": 0.80,
        "trust_decay_rate": 0.03,
        "rationale": "Enterprise-architecture inventory.",
    },
    {
        "source_system": "git",
        "authority_level": 0.85,
        "trust_decay_rate": 0.02,
        "rationale": "Source code is authoritative for code facts.",
    },
    {
        "source_system": "document",
        "authority_level": 0.55,
        "trust_decay_rate": 0.08,
        "rationale": "Docs/SOPs go stale; recency matters.",
    },
    {
        "source_system": "inspiration",
        "authority_level": 0.30,
        "trust_decay_rate": 0.15,
        "rationale": "Reference material, not a source of truth.",
    },
)


def brain_enforcement_enabled() -> bool:
    """Whether trust/permission enforcement is active (``KG_BRAIN_ENFORCE``)."""
    return setting("KG_BRAIN_ENFORCE", False)


def _seed_trust(brain: CompanyBrain) -> None:
    """Load the trust hierarchy from config.json, falling back to defaults."""
    entries: tuple[dict, ...] = _DEFAULT_TRUST
    raw = setting("KG_TRUST_HIERARCHY") or setting("kg_trust_hierarchy")
    if raw:
        try:
            import json

            loaded = json.loads(raw)
            if isinstance(loaded, list) and loaded:
                entries = tuple(loaded)
        except (ValueError, TypeError) as exc:  # pragma: no cover - bad config
            logger.warning("Invalid KG_TRUST_HIERARCHY, using defaults: %s", exc)
    for spec in entries:
        try:
            entry = TrustHierarchyEntry(**spec)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Skipping bad trust entry %s: %s", spec, exc)
            continue
        brain.conflicts.add_trust_entry(entry)
        brain.provenance.add_trust_entry(entry)


def get_company_brain() -> CompanyBrain:
    """Return the lazily-built, process-wide CompanyBrain singleton.

    Configured for ``SOURCE_AUTHORITY_WINS`` conflict resolution (so the trust
    hierarchy actually decides contested writes) with provenance enforced.
    """
    global _BRAIN
    if _BRAIN is None:
        _BRAIN = CompanyBrain(
            default_merge_strategy=MergeStrategy.SOURCE_AUTHORITY_WINS,
            enforce_provenance=True,
        )
        _seed_trust(_BRAIN)
        logger.debug(
            "CompanyBrain runtime initialized (enforcement=%s)",
            brain_enforcement_enabled(),
        )
    return _BRAIN


def reset_company_brain() -> None:
    """Drop the singleton (test helper; not used in production paths)."""
    global _BRAIN
    _BRAIN = None
