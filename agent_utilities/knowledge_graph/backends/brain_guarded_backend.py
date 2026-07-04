#!/usr/bin/python
from __future__ import annotations

"""Trust/provenance-guarded backend wrapper (CONCEPT:AU-KG.research.research-pipeline-runner).

A transparent proxy around any concrete ``GraphBackend`` that activates the
dormant Company Brain on the **write** path without editing the dozens of
writers (``registry.write_batch``, ``pipeline._write_*``, the ingestion engine,
every extractor). It intercepts ``add_node``/``add_edge`` to:

1. **Record provenance** (who/what/when/which-source/confidence) for every write.
2. **Arbitrate source authority** with trust decay (a stale high-authority source
   can lose to a fresher one), replacing effective "last-write-wins" with
   "source-authority-wins".

Two arbitration modes:

* **Field-level survivorship** (the MDM "golden record" model) when the inner
  backend exposes ``get_node_properties`` for a cheap read: each attribute is
  kept from its highest-authority writer, so a low-authority source can still add
  *new* attributes without clobbering a high-authority source's fields. The full
  reconciled record is written back, so it is correct regardless of the backend's
  merge-vs-replace semantics. Per-attribute provenance is tracked.
* **Node-level** arbitration as a transparent fallback when the backend can't
  cheaply read (the whole lower-authority write loses).

Everything else is delegated unchanged. The wrapper is only installed when
``KG_BRAIN_ENFORCE`` is on, so the default path is byte-identical to today.
"""

import json
import logging
import time
from typing import Any

from ...models.company_brain import ActorType, AssertionType, MergeStrategy
from ...security.brain_context import current_actor, current_source
from .base import GraphBackend

logger = logging.getLogger(__name__)

# Map ActorType → the AssertionType that best describes its writes.
_ASSERTION_FOR: dict[ActorType, AssertionType] = {
    ActorType.HUMAN: AssertionType.HUMAN_JUDGMENT,
    ActorType.AI_AGENT: AssertionType.AGENT_INFERENCE,
    ActorType.AUTOMATED_SERVICE: AssertionType.RAW_DATA,
    ActorType.SYSTEM: AssertionType.SYNTHESIZED,
    ActorType.HYBRID_TEAM: AssertionType.AGENT_INFERENCE,
}


# Node-level metadata the guard manages itself — never arbitrated as content.
# ``_field_prov`` is the durable per-attribute provenance map persisted ON the
# node so field-level survivorship survives a process restart.
_PROV_KEYS = {"_source_system", "_actor_id", "_confidence", "_ts", "_field_prov"}


def _stamp_ownership(properties: dict[str, Any], actor: Any) -> None:
    """Stamp private-by-default owner/scope markers (CONCEPT:AU-KG.compute.data-is-private-its).

    Lazy-imported so the backend layer never hard-depends on the sharing module;
    any failure is non-fatal (ownership is additive metadata, not correctness).
    """
    try:
        from ..core.tenant_sharing import stamp_ownership

        stamp_ownership(properties, actor)
    except Exception as exc:  # pragma: no cover - ownership is best-effort
        logger.debug("ownership stamp skipped for write: %s", exc)


class BrainGuardedBackend:
    """Delegating proxy adding provenance + authority arbitration.

    When the inner backend can cheaply read a node's current properties
    (``get_node_properties``), writes are resolved with **field-level
    survivorship** (Option B): each attribute is kept from its highest-authority
    (trust-decayed) writer, so a low-authority source can still contribute *new*
    attributes without clobbering a high-authority source's fields, and the full
    reconciled record is written back (robust to merge-vs-replace semantics).
    When the backend can't read, it falls back to **node-level** arbitration.
    """

    def __init__(self, inner: Any, brain: Any) -> None:
        self._inner = inner
        self._brain = brain
        # node_id -> (source_system, base_authority, monotonic_ts)  [node-level]
        self._seen: dict[str, tuple[str, float, float]] = {}
        # node_id -> field -> (source, wall_clock_epoch)  [field-level, cache of
        # the durable ``_field_prov`` map persisted on each node]
        self._field_owner: dict[str, dict[str, tuple[str, float]]] = {}

    # -- transparent delegation for everything we don't override -----------
    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on the proxy itself.
        return getattr(self._inner, name)

    @property
    def inner(self) -> Any:
        return self._inner

    # -- helpers -----------------------------------------------------------
    def _source(self, actor: Any) -> str:
        return current_source() or actor.actor_id or "unknown"

    @staticmethod
    def _iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def _load_field_prov(existing: dict[str, Any]) -> dict[str, tuple[str, float]]:
        """Parse the durable ``_field_prov`` map persisted on a node.

        Stored as a JSON object ``{field: {"src": source, "ts": epoch}}``. This is
        what lets field-level survivorship survive a process restart: the prior
        owner of each attribute is recovered from the node itself.
        """
        raw = existing.get("_field_prov")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (ValueError, json.JSONDecodeError):
                return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, tuple[str, float]] = {}
        for field, meta in raw.items():
            if isinstance(meta, dict) and meta.get("src"):
                out[field] = (str(meta["src"]), float(meta.get("ts", 0.0)))
        return out

    def _record_node(self, node_id: str, actor: Any, source: str, auth: float) -> None:
        try:
            self._brain.provenance.record_write(
                node_id=node_id,
                actor_id=actor.actor_id,
                actor_type=actor.actor_type,
                source_system=source,
                confidence=auth,
                assertion_type=_ASSERTION_FOR.get(
                    actor.actor_type, AssertionType.AGENT_INFERENCE
                ),
                tenant_id=actor.tenant_id,
            )
        except Exception as exc:  # pragma: no cover - provenance best-effort
            logger.debug("provenance record failed for %s: %s", node_id, exc)

    def _record_field(
        self, node_id: str, field: str, actor: Any, source: str, auth: float
    ) -> None:
        try:
            self._brain.provenance.record_field_write(
                node_id=node_id,
                field=field,
                actor_id=actor.actor_id,
                actor_type=actor.actor_type,
                source_system=source,
                confidence=auth,
                assertion_type=_ASSERTION_FOR.get(
                    actor.actor_type, AssertionType.AGENT_INFERENCE
                ),
                tenant_id=actor.tenant_id,
            )
        except Exception as exc:  # pragma: no cover - provenance best-effort
            logger.debug("field provenance failed for %s.%s: %s", node_id, field, exc)

    def _log_conflict(self, node_id, field, va, vb, actor_a, actor_b, ca, cb) -> None:
        try:
            self._brain.conflicts.detect_conflict(
                node_id=node_id,
                field_name=field,
                value_a=va,
                value_b=vb,
                actor_a=actor_a or "unknown",
                actor_b=actor_b,
                confidence_a=ca,
                confidence_b=cb,
            )
        except Exception as exc:  # pragma: no cover - audit best-effort
            logger.debug("conflict record failed for %s.%s: %s", node_id, field, exc)

    # -- guarded writes ----------------------------------------------------
    def add_node(self, node_id: str, **properties: Any) -> None:
        actor = current_actor()
        source = self._source(actor)
        now_mono = time.monotonic()
        now_wall = time.time()
        base_auth = self._brain.conflicts.effective_authority(source, age_days=0.0)

        reader = getattr(self._inner, "get_node_properties", None)
        if not callable(reader):
            # Backend can't read → whole-node authority arbitration.
            self._add_node_level(
                node_id, properties, actor, source, base_auth, now_mono
            )
            return

        try:
            existing = reader(node_id)
        except Exception:  # pragma: no cover - read best-effort
            existing = None
        if existing is None:  # readable backend, brand-new node
            existing = {}

        # Field-level survivorship (Option B). The in-memory ledger is a cache;
        # the durable owner of each attribute is recovered from the node's
        # persisted ``_field_prov`` map, so survivorship survives a restart.
        merged = dict(existing)
        owners = self._field_owner.setdefault(node_id, {})
        for field, meta in self._load_field_prov(existing).items():
            owners.setdefault(field, meta)

        for field, value in properties.items():
            if field in _PROV_KEYS:
                continue
            prior = owners.get(field)
            if prior is None:
                prior_eff = 0.5 if field in existing else -1.0
                prior_src = (
                    existing.get("_source_system") if field in existing else None
                )
            else:
                p_src, p_ts = prior
                prior_eff = self._brain.conflicts.effective_authority(
                    p_src, age_days=max(0.0, (now_wall - p_ts) / 86400.0)
                )
                prior_src = p_src
            differs = existing.get(field) != value
            if base_auth >= prior_eff:
                merged[field] = value
                owners[field] = (source, now_wall)
                self._record_field(node_id, field, actor, source, base_auth)
                if differs and prior_src not in (None, source):
                    self._log_conflict(
                        node_id,
                        field,
                        existing.get(field),
                        value,
                        prior_src,
                        source,
                        prior_eff,
                        base_auth,
                    )
            elif differs:
                # Prior owner wins: keep existing value, record the contest.
                self._log_conflict(
                    node_id,
                    field,
                    existing.get(field),
                    value,
                    prior_src,
                    source,
                    prior_eff,
                    base_auth,
                )

        # Persist the durable per-attribute provenance map back onto the node.
        merged["_field_prov"] = json.dumps(
            {f: {"src": s, "ts": t} for f, (s, t) in owners.items()}
        )
        # Refresh node-level metadata to the latest touch.
        merged["_source_system"] = source
        merged["_actor_id"] = actor.actor_id
        merged["_confidence"] = round(base_auth, 4)
        merged["_ts"] = self._iso()
        _stamp_ownership(merged, actor)
        self._record_node(node_id, actor, source, base_auth)
        self._seen[node_id] = (source, base_auth, now_mono)
        self._inner.add_node(node_id, **merged)

    def _add_node_level(self, node_id, properties, actor, source, base_auth, now):
        """Whole-node authority arbitration (backends without cheap reads)."""
        prior = self._seen.get(node_id)
        if prior is not None:
            prior_source, _prior_base, prior_ts = prior
            if prior_source != source:
                age_days = max(0.0, (now - prior_ts) / 86400.0)
                prior_eff = self._brain.conflicts.effective_authority(
                    prior_source, age_days=age_days
                )
                strategy = self._brain.conflicts._default_strategy
                if (
                    strategy == MergeStrategy.SOURCE_AUTHORITY_WINS
                    and prior_eff > base_auth
                ):
                    self._log_conflict(
                        node_id,
                        "*",
                        prior_source,
                        source,
                        prior_source,
                        source,
                        prior_eff,
                        base_auth,
                    )
                    logger.info(
                        "Suppressed lower-authority overwrite of %s (%s=%.3f < %s=%.3f)",
                        node_id,
                        source,
                        base_auth,
                        prior_source,
                        prior_eff,
                    )
                    return
        properties.setdefault("_source_system", source)
        properties.setdefault("_actor_id", actor.actor_id)
        properties.setdefault("_confidence", round(base_auth, 4))
        properties.setdefault("_ts", self._iso())
        _stamp_ownership(properties, actor)
        self._record_node(node_id, actor, source, base_auth)
        self._seen[node_id] = (source, base_auth, now)
        self._inner.add_node(node_id, **properties)

    def add_edge(self, source: str, target: str, **properties: Any) -> None:
        actor = current_actor()
        properties.setdefault("_source_system", self._source(actor))
        properties.setdefault("_actor_id", actor.actor_id)
        properties.setdefault("_ts", self._iso())
        self._inner.add_edge(source, target, **properties)


# Make isinstance(guard, GraphBackend) hold for duck-typed consumers.
GraphBackend.register(BrainGuardedBackend)
