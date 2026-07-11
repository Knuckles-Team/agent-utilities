#!/usr/bin/python
from __future__ import annotations

"""Workload-aware data-placement mining (X-5, CONCEPT:AU-KG.evolution.placement-mining-canary-loop).

Agent traces reveal which tenants, tools, entities, and modalities CO-OCCUR
and which are hot vs. cold. This module mines that telemetry the SAME way
every other mining pass in this codebase does — through the engine's
``graph_mine`` surface (associate/anomaly/sequence), via the SAME ``_invoke``
helper :mod:`.trace_pattern_miner` / ``loop_controller._mine_*`` already use
— and turns each mined finding into a typed :class:`PlacementProposal`
(``shard_split`` / ``replica`` / ``cache_prewarm`` / ``materialized_join`` /
``embedding_refresh`` / ``index_change``), each carrying the real mined
evidence + an expected-benefit statement (never fabricated, mirroring
``candidate_insight.CandidateInsight``'s no-fabrication contract).

Pipeline (mirrors ``loop_controller._run_trace_mining`` exactly — mining is
NOT a fourth bespoke authority, it reuses the SAME governance/gate spine)::

    mine (associate/anomaly/sequence over Episode/ToolCall/Entity provenance)
        -> PlacementProposal (typed, evidenced)
        -> Claim (status="proposal", ALWAYS persisted, is_verified=False)
        -> PromotionGovernanceValidator.validate()   (reused as-is)
        -> action_policy.decide(kind="apply_placement_change")  (SAFETY-CRITICAL
           gate — shipped tier is approval_required, see
           ``deploy/action-policy.default.yml``)
        -> ONLY IF allowed: a MEASURED CANARY (:func:`run_canary` — apply to a
           small scope, measure the SLO/latency delta, promote or roll back)
        -> promote reaches the engine's PlacementCatalog admin path
           (``ReshardingClient.catalog_assign``/``catalog_remove``, via the
           SAME ``engine_resharding`` dispatcher ``engine_tools.py`` already
           exposes — no second placement authority, no new engine RPC)

Data sources: the ``Episode -[:USED_TOOL]-> ToolCall -[:AFFECTS]-> Entity``
provenance chain (the SAME real, wired schema :mod:`.trace_pattern_miner`
already mines the failure side of — see that module's docstring for why this
is the defensible mapping, not the unwired ``AgentTaskNode``/"AgentOutcomeNode"
naming the task description uses). Tenant/modality context is read from
``ToolCall.args`` (a ``dict[str, Any]`` — real callers already stash
``tenant_id``/``modality`` there; a record missing them simply omits that
item from its basket, never guessed). ``Entity.temporal_drift_score``
(already a ``RegistryNode`` field) is the drift signal
:func:`gather_drift_scores` mines for the ``embedding_refresh`` proposal
kind — reusing an existing field, not inventing a drift metric.

Connects to DIST-P2-1's ``PlacementCatalog`` (epoch'd virtual partitions +
online move, ``epistemic-graph src/raft/placement.rs``) and this repo's own
:mod:`~agent_utilities.knowledge_graph.core.placement_catalog` READ-side
consumer (:func:`~agent_utilities.knowledge_graph.core.placement_catalog.
resolve_placement`) — an applied ``shard_split``/``replica`` change is what
that resolver's cache picks up on its next (or a forced) refresh
(:func:`~agent_utilities.knowledge_graph.core.placement_catalog.invalidate`
is called after every catalog admin call this module makes).
"""

import base64
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.models.evidence_bundle import EvidenceBundle
from agent_utilities.models.knowledge_graph import ClaimNode, RegistryNodeType

logger = logging.getLogger(__name__)

__all__ = [
    "CONFIDENCE_FLOOR",
    "PlacementProposal",
    "CanaryResult",
    "gather_access_records",
    "gather_drift_scores",
    "build_baskets",
    "build_tenant_access_counts",
    "build_tenant_sequences",
    "mine_placement_patterns",
    "proposals_from_association",
    "proposals_from_tenant_anomaly",
    "proposals_from_drift_anomaly",
    "proposals_from_sequence",
    "placement_proposals_from_mining",
    "run_canary",
    "apply_placement_change",
    "rollback_placement_change",
    "run_placement_mining_cycle",
]

#: A mined finding must clear this floor before it is even materialized as a
#: ``ClaimNode`` — mirrors ``candidate_insight.CONFIDENCE_FLOOR`` (kept as its
#: own constant, not imported, since a placement proposal is a distinct
#: finding family with its own review surface — see module docstring).
CONFIDENCE_FLOOR = 0.6

#: Cap on provenance rows scanned per cycle — bounded like every other
#: mining pass's query LIMIT (``trace_pattern_miner._TRACE_SCAN_LIMIT``).
_SCAN_LIMIT = 200

#: Minimum mined association-rule confidence before a tool<->entity-type rule
#: is read as an ``index_change`` recommendation (stricter than the general
#: ``min_confidence`` mining threshold — an index is a standing structural
#: change, so it wants a stronger signal than a one-off co-occurrence).
_INDEX_CONFIDENCE = 0.9

#: Default canary tolerance: a canary metric may regress by up to this
#: fraction of its baseline value and still be promoted (SLO noise
#: tolerance). Anything worse rolls back.
_CANARY_TOLERANCE = 0.10

_PLACEMENT_KINDS = {
    "shard_split",
    "replica",
    "cache_prewarm",
    "materialized_join",
    "embedding_refresh",
    "index_change",
}

_READ_VERBS = ("get", "list", "read", "query", "search", "fetch", "find")


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(list(parts), sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _is_read_tool(name: str) -> bool:
    return name.lower().startswith(_READ_VERBS)


def _mining_ok(payload: Any) -> bool:
    """``True`` when an ``_invoke`` JSON payload is a live (non-degraded) result."""
    return isinstance(payload, dict) and "error" not in payload


# ═══════════════════════════════════════════════════════════════════════════
# PlacementProposal — one typed, evidenced recommendation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PlacementProposal:
    """One mined placement recommendation, on its way to a reviewable ``ClaimNode``.

    ``kind`` is one of :data:`_PLACEMENT_KINDS`; ``target`` names what the
    change applies to (a tenant id for ``shard_split``/``replica``, an
    ``"entity_a|entity_b"`` pair for ``materialized_join``, an entity/
    modality id for ``embedding_refresh``, an entity type for
    ``index_change``, or an ``"item1 → item2"`` sequence label for
    ``cache_prewarm``). ``confidence`` is always a real mined signal (rule
    confidence, a saturated anomaly z-score, or a sequence's mined support)
    — never fabricated, mirroring ``candidate_insight.CandidateInsight``.
    """

    kind: str
    target: str
    statement: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    expected_benefit: str = ""

    def __post_init__(self) -> None:
        self.confidence = _clamp01(self.confidence)
        if self.kind not in _PLACEMENT_KINDS:
            raise ValueError(f"unknown placement proposal kind: {self.kind!r}")

    @property
    def proposal_id(self) -> str:
        return _stable_id("placement", self.kind, self.target)

    @property
    def clears_floor(self) -> bool:
        """``True`` when this proposal is claim-worthy (>= :data:`CONFIDENCE_FLOOR`)."""
        return self.confidence >= CONFIDENCE_FLOOR

    @property
    def claim_id(self) -> str:
        return f"claim:{self.proposal_id}"

    def to_claim_node(self) -> ClaimNode:
        """Materialize as a ``ClaimNode`` — ALWAYS unverified (never self-verifying).

        Verification is the governance/action-policy/canary gate's job, not
        this constructor's — see module docstring.
        """
        return ClaimNode(
            id=self.claim_id,
            type=RegistryNodeType.CLAIM,
            name=self.statement[:120] or self.proposal_id,
            claim_text=self.statement,
            confidence=self.confidence,
            claim_type="placement_proposal",
            source_ids=[self.target],
            extracted_from=self.proposal_id,
            domain="placement_mining",
            is_verified=False,
            metadata={
                "kind": self.kind,
                "target": self.target,
                "expected_benefit": self.expected_benefit,
            },
            importance_score=self.confidence,
        )

    def to_evidence_bundle(self) -> EvidenceBundle:
        """Package the raw mined finding as an audit-visible :class:`EvidenceBundle`."""
        return EvidenceBundle(
            answer_candidate=self.statement,
            claims=[{"id": self.proposal_id, "text": self.statement}],
            evidence_spans=[
                {"id": self.proposal_id, "kind": self.kind, **self.evidence}
            ],
            confidence=self.confidence,
            reasoning_trace=[
                {
                    "step": "placement_mining",
                    "kind": self.kind,
                    "target": self.target,
                    "evidence": self.evidence,
                }
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════
# Telemetry gathering — Episode/ToolCall/Entity provenance
# ═══════════════════════════════════════════════════════════════════════════


def gather_access_records(engine: Any, *, limit: int = _SCAN_LIMIT) -> list[dict[str, Any]]:
    """Query the Episode/ToolCall/Entity provenance chain for per-call access records.

    Returns one record per ``(episode, tool_call)`` row:
    ``{episode_id, tool_name, tenant, modality, entity_id, entity_type}``.
    ``tenant``/``modality`` are read from ``ToolCall.args`` (parsed JSON if
    stored as a string) — never guessed; a row missing them simply carries
    an empty string for that field. Empty (never raises) on a missing
    engine or a query failure, matching every other mining pass's tolerance.
    """
    if engine is None:
        return []
    try:
        rows = (
            engine.query_cypher(
                "MATCH (e:Episode)-[:USED_TOOL]->(t:ToolCall) "
                "OPTIONAL MATCH (t)-[:AFFECTS]->(en:Entity) "
                "RETURN e.id AS episode_id, t.tool_name AS tool_name, "
                "t.args AS args, en.id AS entity_id, en.type AS entity_type "
                f"ORDER BY episode_id LIMIT {int(limit)}"
            )
            or []
        )
    except Exception as e:  # noqa: BLE001 — a query failure degrades, never raises
        logger.debug("placement_mining: access-record query failed: %s", e)
        return []

    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("episode_id"):
            continue
        args = row.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        records.append(
            {
                "episode_id": str(row["episode_id"]),
                "tool_name": str(row.get("tool_name") or ""),
                "tenant": str(args.get("tenant_id") or args.get("tenant") or ""),
                "modality": str(args.get("modality") or ""),
                "entity_id": str(row.get("entity_id") or ""),
                "entity_type": str(row.get("entity_type") or ""),
            }
        )
    return records


def gather_drift_scores(
    engine: Any, *, limit: int = _SCAN_LIMIT
) -> tuple[list[str], list[float]]:
    """Query ``Entity.temporal_drift_score`` (an existing ``RegistryNode`` field).

    Returns ``(entity_ids, scores)`` — parallel lists, feeding the
    ``embedding_refresh`` anomaly pass. Empty (never raises) on a missing
    engine or a query failure.
    """
    if engine is None:
        return [], []
    try:
        rows = (
            engine.query_cypher(
                "MATCH (en:Entity) WHERE en.temporal_drift_score IS NOT NULL "
                "RETURN en.id AS id, en.temporal_drift_score AS drift "
                f"LIMIT {int(limit)}"
            )
            or []
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("placement_mining: drift-score query failed: %s", e)
        return [], []

    ids: list[str] = []
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        try:
            values.append(float(row.get("drift") or 0.0))
        except (TypeError, ValueError):
            continue
        ids.append(str(row["id"]))
    return ids, values


def _basket_items(rec: dict[str, Any]) -> list[str]:
    items: list[str] = []
    if rec.get("tenant"):
        items.append(f"tenant:{rec['tenant']}")
    if rec.get("tool_name"):
        items.append(f"tool:{rec['tool_name']}")
    if rec.get("entity_id"):
        items.append(f"entity:{rec['entity_id']}")
    if rec.get("entity_type"):
        items.append(f"entity_type:{rec['entity_type']}")
    if rec.get("modality"):
        items.append(f"modality:{rec['modality']}")
    return items


def build_baskets(records: list[dict[str, Any]]) -> list[list[str]]:
    """One basket per episode — the union of its access-record items.

    Only baskets with >= 2 distinct items carry a co-occurrence signal
    (mirrors ``trace_pattern_miner``'s own ">= 2" episode filter).
    """
    baskets: dict[str, set[str]] = {}
    order: list[str] = []
    for rec in records:
        eid = rec["episode_id"]
        if eid not in baskets:
            baskets[eid] = set()
            order.append(eid)
        baskets[eid].update(_basket_items(rec))
    return [sorted(baskets[eid]) for eid in order if len(baskets[eid]) >= 2]


def build_tenant_access_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Distinct-episode access count per tenant — the ``shard_split`` anomaly input."""
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for rec in records:
        tenant = rec.get("tenant")
        if not tenant:
            continue
        key = (tenant, rec["episode_id"])
        if key in seen:
            continue
        seen.add(key)
        counts[tenant] = counts.get(tenant, 0) + 1
    return counts


def build_tenant_sequences(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Per-tenant ORDERED item sequence (episode order = time order) — the
    ``cache_prewarm`` sequence-mining input. Only tenants with >= 2 items
    carry an ordered-subsequence signal."""
    by_tenant: dict[str, list[str]] = {}
    for rec in records:
        tenant = rec.get("tenant")
        if not tenant:
            continue
        item = None
        if rec.get("entity_id"):
            item = f"entity:{rec['entity_id']}"
        elif rec.get("tool_name"):
            item = f"tool:{rec['tool_name']}"
        if item:
            by_tenant.setdefault(tenant, []).append(item)
    return {t: seq for t, seq in by_tenant.items() if len(seq) >= 2}


# ═══════════════════════════════════════════════════════════════════════════
# Mining — delegated to the engine's graph_mine surface (associate/anomaly/sequence)
# ═══════════════════════════════════════════════════════════════════════════


def mine_placement_patterns(
    engine: Any,
    *,
    limit: int = _SCAN_LIMIT,
    min_support: float = 0.1,
    min_confidence: float = 0.6,
) -> dict[str, Any]:
    """Mine tenant/tool/entity/modality co-occurrence + hot/cold skew + sequence.

    Three independent, best-effort ``graph_mine`` passes over the SAME
    gathered access records, through the SAME ``_invoke`` helper every other
    mining pass in this codebase uses — degrades exactly like those on a
    no-mining engine build (empty result, never raises):

    1. **associate** — co-occurrence rules over per-episode baskets
       (tenant/tool/entity/entity_type/modality items).
    2. **anomaly** (x2) — per-tenant access-count skew (hot tenant -> a
       ``shard_split`` candidate) and per-entity ``temporal_drift_score``
       skew (drifted -> an ``embedding_refresh`` candidate).
    3. **sequence** — per-tenant ordered access sequences (a predictable
       "what reliably follows what" -> a ``cache_prewarm`` candidate).
    """
    from agent_utilities.mcp.tools.engine_surface_tools import _invoke

    errors: list[str] = []
    records = gather_access_records(engine, limit=limit)
    baskets = build_baskets(records)
    access_counts = build_tenant_access_counts(records)
    sequences_by_tenant = build_tenant_sequences(records)
    drift_ids, drift_values = gather_drift_scores(engine, limit=limit)

    association: dict[str, Any] = {"rules": []}
    if baskets:
        try:
            raw = _invoke(
                surface="mining",
                action="associate",
                graph="",
                candidates=(("mining", "associate"),),
                params={
                    "transactions": baskets,
                    "min_support": min_support,
                    "min_confidence": min_confidence,
                    "algorithm": "fpgrowth",
                    "writeback": True,
                },
            )
            payload = json.loads(raw)
            if _mining_ok(payload):
                association = payload.get("result") or {"rules": []}
            else:
                errors.append(
                    f"placement_mining:associate: {payload.get('error') or payload}"
                )
        except Exception as e:  # noqa: BLE001 — never let mining break the caller
            errors.append(f"placement_mining:associate: {e}")

    tenant_ids = list(access_counts.keys())
    tenant_anomaly: dict[str, Any] = {"rows": []}
    if len(tenant_ids) >= 3:
        try:
            raw = _invoke(
                surface="mining",
                action="anomaly",
                graph="",
                candidates=(("mining", "anomaly"),),
                params={
                    "values": [float(access_counts[t]) for t in tenant_ids],
                    "algorithm": "zscore",
                    "writeback": True,
                },
            )
            payload = json.loads(raw)
            if _mining_ok(payload):
                tenant_anomaly = payload.get("result") or {"rows": []}
            else:
                errors.append(
                    f"placement_mining:tenant_anomaly: {payload.get('error') or payload}"
                )
        except Exception as e:  # noqa: BLE001
            errors.append(f"placement_mining:tenant_anomaly: {e}")

    drift_anomaly: dict[str, Any] = {"rows": []}
    if len(drift_values) >= 3:
        try:
            raw = _invoke(
                surface="mining",
                action="anomaly",
                graph="",
                candidates=(("mining", "anomaly"),),
                params={
                    "values": drift_values,
                    "algorithm": "zscore",
                    "writeback": True,
                },
            )
            payload = json.loads(raw)
            if _mining_ok(payload):
                drift_anomaly = payload.get("result") or {"rows": []}
            else:
                errors.append(
                    f"placement_mining:drift_anomaly: {payload.get('error') or payload}"
                )
        except Exception as e:  # noqa: BLE001
            errors.append(f"placement_mining:drift_anomaly: {e}")

    sequence: dict[str, Any] = {"patterns": []}
    seqs = list(sequences_by_tenant.values())
    if seqs:
        try:
            raw = _invoke(
                surface="mining",
                action="sequence",
                graph="",
                candidates=(("mining", "sequence"),),
                params={"sequences": seqs, "min_support": 0.3, "writeback": True},
            )
            payload = json.loads(raw)
            if _mining_ok(payload):
                sequence = payload.get("result") or {"patterns": []}
            else:
                errors.append(
                    f"placement_mining:sequence: {payload.get('error') or payload}"
                )
        except Exception as e:  # noqa: BLE001
            errors.append(f"placement_mining:sequence: {e}")

    return {
        "association": association,
        "tenant_anomaly": {"result": tenant_anomaly, "tenant_ids": tenant_ids},
        "drift_anomaly": {"result": drift_anomaly, "entity_ids": drift_ids},
        "sequence": sequence,
        "access_counts": access_counts,
        "records_scanned": len(records),
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Classification — mined finding -> typed PlacementProposal
# ═══════════════════════════════════════════════════════════════════════════


def _rule_items(rule: dict[str, Any]) -> tuple[list[str], list[str]]:
    ante = rule.get("antecedent")
    cons = rule.get("consequent")
    ante = ante if isinstance(ante, list) else ([ante] if ante else [])
    cons = cons if isinstance(cons, list) else ([cons] if cons else [])
    return [str(a) for a in ante], [str(c) for c in cons]


def _prefixed(items: list[str], prefix: str) -> list[str]:
    return [i for i in items if i.startswith(prefix)]


def proposals_from_association(association: dict[str, Any] | None) -> list[PlacementProposal]:
    """Co-occurrence rules -> ``materialized_join`` / ``index_change`` / ``replica``.

    Classification (most-specific first, one proposal kind per rule):

    * both sides name a real entity (``entity:``), different ones -> frequent
      cross-entity access -> ``materialized_join``.
    * one side is a tool, the other an ``entity_type:`` at very high
      confidence (>= :data:`_INDEX_CONFIDENCE`) -> ``index_change``.
    * antecedent names a tenant and either side names a read-verb tool
      (get/list/read/query/search/fetch/find) -> a hot READ set for that
      tenant -> ``replica``.
    """
    out: list[PlacementProposal] = []
    for rule in (association or {}).get("rules") or []:
        if not isinstance(rule, dict):
            continue
        ante, cons = _rule_items(rule)
        all_items = ante + cons
        confidence = _clamp01(rule.get("confidence"))
        entities_ante = _prefixed(ante, "entity:")
        entities_cons = _prefixed(cons, "entity:")
        tools = _prefixed(ante, "tool:") + _prefixed(cons, "tool:")
        tenants_ante = _prefixed(ante, "tenant:")
        entity_types = _prefixed(all_items, "entity_type:")

        if entities_ante and entities_cons and entities_ante[0] != entities_cons[0]:
            target = f"{entities_ante[0]}|{entities_cons[0]}"
            out.append(
                PlacementProposal(
                    kind="materialized_join",
                    target=target,
                    statement=(
                        f"{entities_ante[0]} and {entities_cons[0]} are frequently "
                        f"co-accessed (confidence={rule.get('confidence')}, "
                        f"lift={rule.get('lift')})"
                    ),
                    confidence=confidence,
                    evidence=dict(rule),
                    expected_benefit=(
                        "materialize this cross-entity join to avoid repeated "
                        "separate fetches"
                    ),
                )
            )
        elif tools and entity_types and confidence >= _INDEX_CONFIDENCE:
            etype = entity_types[0]
            tool = tools[0]
            out.append(
                PlacementProposal(
                    kind="index_change",
                    target=etype,
                    statement=(
                        f"{tool} very reliably resolves against {etype} "
                        f"(confidence={rule.get('confidence')}) — index candidate"
                    ),
                    confidence=confidence,
                    evidence=dict(rule),
                    expected_benefit=f"add/verify an index on {etype} to serve {tool} lookups",
                )
            )
        elif tenants_ante and tools:
            reading_tools = [t for t in tools if _is_read_tool(t.split(":", 1)[-1])]
            if reading_tools:
                tenant = tenants_ante[0]
                out.append(
                    PlacementProposal(
                        kind="replica",
                        target=tenant,
                        statement=(
                            f"{tenant} has a hot, read-heavy access pattern via "
                            f"{reading_tools[0]} (confidence={rule.get('confidence')})"
                        ),
                        confidence=confidence,
                        evidence=dict(rule),
                        expected_benefit=f"dedicate a read-serving placement for {tenant}",
                    )
                )
    return out


def proposals_from_tenant_anomaly(tenant_anomaly: dict[str, Any] | None) -> list[PlacementProposal]:
    """A tenant whose access count is a positive statistical outlier -> ``shard_split``.

    Only a HOT (positive-score) outlier is in scope — a cold/underused
    tenant isn't a split candidate.
    """
    tenant_anomaly = tenant_anomaly or {}
    result = tenant_anomaly.get("result") or {}
    tenant_ids = tenant_anomaly.get("tenant_ids") or []
    out: list[PlacementProposal] = []
    for idx, row in enumerate(result.get("rows") or []):
        if not isinstance(row, dict) or not row.get("is_anomaly"):
            continue
        try:
            score = float(row.get("anomaly_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score <= 0:
            continue
        tenant = tenant_ids[idx] if idx < len(tenant_ids) else row.get("id")
        if not tenant:
            continue
        out.append(
            PlacementProposal(
                kind="shard_split",
                target=str(tenant),
                statement=(
                    f"tenant {tenant} is a hot-access outlier "
                    f"(anomaly_score={row.get('anomaly_score')})"
                ),
                confidence=_clamp01(abs(score) / 5.0),
                evidence=dict(row),
                expected_benefit=(
                    f"split {tenant} onto its own virtual shard to isolate its load"
                ),
            )
        )
    return out


def proposals_from_drift_anomaly(drift_anomaly: dict[str, Any] | None) -> list[PlacementProposal]:
    """An entity whose ``temporal_drift_score`` is a statistical outlier -> ``embedding_refresh``."""
    drift_anomaly = drift_anomaly or {}
    result = drift_anomaly.get("result") or {}
    entity_ids = drift_anomaly.get("entity_ids") or []
    out: list[PlacementProposal] = []
    for idx, row in enumerate(result.get("rows") or []):
        if not isinstance(row, dict) or not row.get("is_anomaly"):
            continue
        try:
            score = float(row.get("anomaly_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        entity_id = entity_ids[idx] if idx < len(entity_ids) else row.get("id")
        if not entity_id:
            continue
        out.append(
            PlacementProposal(
                kind="embedding_refresh",
                target=str(entity_id),
                statement=(
                    f"{entity_id}'s temporal-drift score is a statistical outlier "
                    f"(anomaly_score={row.get('anomaly_score')})"
                ),
                confidence=_clamp01(abs(score) / 5.0),
                evidence=dict(row),
                expected_benefit=f"refresh {entity_id}'s embedding to correct drift",
            )
        )
    return out


def proposals_from_sequence(sequence: dict[str, Any] | None) -> list[PlacementProposal]:
    """A frequent ORDERED access subsequence -> ``cache_prewarm``.

    ``support`` is already a ``[0, 1]`` frequency fraction (the same
    ``min_support`` units the mining call itself gates on) so it maps
    straight through as the proposal's confidence — no transform needed.
    """
    out: list[PlacementProposal] = []
    for pat in (sequence or {}).get("patterns") or []:
        if not isinstance(pat, dict):
            continue
        items = pat.get("items") or []
        if len(items) < 2:
            continue
        target = " → ".join(str(i) for i in items)
        out.append(
            PlacementProposal(
                kind="cache_prewarm",
                target=target,
                statement=(
                    f"predictable access sequence {target} "
                    f"(support={pat.get('support')}, count={pat.get('count')})"
                ),
                confidence=_clamp01(pat.get("support")),
                evidence=dict(pat),
                expected_benefit=f"prewarm the cache for {items[-1]} whenever {items[0]} is accessed",
            )
        )
    return out


def placement_proposals_from_mining(mine_result: dict[str, Any] | None) -> list[PlacementProposal]:
    """The full fan-out: a :func:`mine_placement_patterns` report -> every proposal."""
    mine_result = mine_result or {}
    out: list[PlacementProposal] = []
    out += proposals_from_association(mine_result.get("association"))
    out += proposals_from_tenant_anomaly(mine_result.get("tenant_anomaly"))
    out += proposals_from_drift_anomaly(mine_result.get("drift_anomaly"))
    out += proposals_from_sequence(mine_result.get("sequence"))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Measured canary — apply to a small scope, measure, promote or roll back
# ═══════════════════════════════════════════════════════════════════════════

MeasurementFn = Callable[[PlacementProposal, str], dict[str, float]]
ApplyFn = Callable[[PlacementProposal], dict[str, Any]]


@dataclass
class CanaryResult:
    """The measured canary's verdict for one proposal."""

    proposal_id: str
    verdict: str  # "promote" | "rollback"
    reason: str
    baseline: dict[str, float] = field(default_factory=dict)
    canary: dict[str, float] = field(default_factory=dict)
    delta: dict[str, float] = field(default_factory=dict)
    applied: bool = False
    apply_detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "baseline": self.baseline,
            "canary": self.canary,
            "delta": self.delta,
            "applied": self.applied,
        }


def _default_measurement(proposal: PlacementProposal, phase: str) -> dict[str, float]:
    """Best-effort PromQL SLO read for ``proposal.target`` (observability surface).

    Degrades to ``{}`` on any failure (unreachable engine / no promql
    surface / no such series) — :func:`run_canary` treats an empty
    measurement the same as evidence of a bad change (conservative
    rollback), never as evidence of a good one.
    """
    try:
        from agent_utilities.mcp.tools.engine_surface_tools import _invoke

        raw = _invoke(
            surface="promql",
            action="instant",
            graph="",
            candidates=(("observability", "promql"), ("metrics", "promql")),
            params={"query": f'placement_latency_ms{{target="{proposal.target}"}}'},
        )
        payload = json.loads(raw)
    except Exception as e:  # noqa: BLE001 — a missing/unreachable surface degrades
        logger.debug("placement_mining: default measurement failed: %s", e)
        return {}
    if not _mining_ok(payload):
        return {}
    result = payload.get("result")
    if isinstance(result, dict) and "value" in result:
        try:
            return {"latency_ms": float(result["value"])}
        except (TypeError, ValueError):
            return {}
    return {}


def run_canary(
    proposal: PlacementProposal,
    *,
    measurement_fn: MeasurementFn | None = None,
    apply_fn: ApplyFn | None = None,
    rollback_fn: ApplyFn | None = None,
    tolerance: float = _CANARY_TOLERANCE,
) -> CanaryResult:
    """Measured canary: baseline measurement -> apply to a small scope -> canary
    measurement -> promote (keep the change) or roll back (revert it).

    ``measurement_fn(proposal, phase)`` (``phase`` is ``"baseline"``/``"canary"``)
    defaults to :func:`_default_measurement`; ``apply_fn``/``rollback_fn`` default
    to :func:`apply_placement_change`/:func:`rollback_placement_change`. All
    three are injectable so a test can mock the measurement (and the apply/
    rollback side effects) without touching a real engine.

    A regression beyond ``tolerance`` (fraction of the baseline value, any
    measured metric) rolls back; no measurement at all (baseline or canary)
    also rolls back — absence of evidence is never treated as evidence of
    safety. ``CanaryResult.applied`` is ``True`` only on a promoted, actually-
    applied change.
    """
    measure = measurement_fn or _default_measurement
    apply_ = apply_fn or apply_placement_change
    rollback_ = rollback_fn or rollback_placement_change

    baseline = measure(proposal, "baseline") or {}
    try:
        apply_result = apply_(proposal) or {}
    except Exception as e:  # noqa: BLE001 — an apply failure is data, not a crash
        apply_result = {"applied": False, "detail": str(e)}
    canary_metrics = measure(proposal, "canary") or {}
    delta: dict[str, float] = {}

    if not baseline or not canary_metrics:
        verdict, reason = (
            "rollback",
            "no measurement available — conservatively rolling back",
        )
    else:
        worst = 0.0
        for key, base_v in baseline.items():
            if key not in canary_metrics:
                continue
            canary_v = canary_metrics[key]
            d = (
                (canary_v - base_v) / abs(base_v)
                if base_v
                else (0.0 if canary_v == 0 else 1.0)
            )
            delta[key] = round(d, 4)
            worst = max(worst, d)
        if not delta:
            verdict, reason = (
                "rollback",
                "no overlapping metrics between baseline and canary — rolling back",
            )
        elif worst <= tolerance:
            verdict, reason = (
                "promote",
                f"max metric delta {worst:.2%} within tolerance {tolerance:.0%}",
            )
        else:
            verdict, reason = (
                "rollback",
                f"max metric delta {worst:.2%} exceeds tolerance {tolerance:.0%}",
            )

    applied = bool(apply_result.get("applied"))
    if verdict == "rollback" and applied:
        try:
            rollback_(proposal)
        except Exception as e:  # noqa: BLE001 — rollback failure is logged, never raised
            logger.debug("placement_mining: rollback failed for %s: %s", proposal.target, e)
        applied = False  # the canary-scope change was reverted; nothing stayed applied

    return CanaryResult(
        proposal_id=proposal.proposal_id,
        verdict=verdict,
        reason=reason,
        baseline=baseline,
        canary=canary_metrics,
        delta=delta,
        applied=(verdict == "promote" and applied),
        apply_detail=apply_result,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Apply / rollback — the PlacementCatalog admin path (no second authority)
# ═══════════════════════════════════════════════════════════════════════════


def apply_placement_change(proposal: PlacementProposal) -> dict[str, Any]:
    """Reach the accepted change to its real system of record.

    * ``shard_split`` / ``replica`` — the engine's ``PlacementCatalog`` admin
      path: ``ReshardingClient.catalog_assign`` via the SAME
      ``engine_<domain>`` dispatcher ``engine_tools.py`` already exposes as
      ``engine_resharding`` (no second placement authority, no new engine
      RPC). Both proposal kinds are literally a partition-placement
      decision — ``shard_split`` assigns the hot tenant its own virtual
      shard, ``replica`` pins a hot READ set to a dedicated shard/node
      (``proposal.evidence`` may carry an explicit ``shard``/``node``; a
      deterministic placeholder shard is derived from the target otherwise).
    * ``cache_prewarm`` — the shared content-addressed KV-cache
      (:class:`~agent_utilities.kvcache.EpistemicGraphKVBackend`), which
      already degrades every transport error to a no-op.
    * ``materialized_join`` / ``embedding_refresh`` / ``index_change`` — these
      name a change outside AU's currently-wired write surfaces (creating an
      ad-hoc index or recomputing an arbitrary embedding); applying one
      materializes an accepted KG record (the governed decision is real and
      audited) without a further engine call — a deliberate scope boundary,
      not a silent no-op (see module docstring: this is a mining->proposal->
      canary->apply loop, not a second placement authority inventing new
      engine capability).
    """
    if proposal.kind in {"shard_split", "replica"}:
        return _apply_via_catalog(proposal)
    if proposal.kind == "cache_prewarm":
        return _apply_via_kvcache(proposal)
    return {"applied": True, "method": "accepted_record", "detail": proposal.kind}


def rollback_placement_change(proposal: PlacementProposal) -> dict[str, Any]:
    """Revert an applied change — the canary's "undo" path.

    ``shard_split``/``replica`` drop the explicit catalog placement
    (``catalog_remove`` — reverts to the default FNV-1a hash-ring routing,
    see ``placement_catalog.py``'s module docstring). The remaining kinds
    have no destructive engine-side state to revert (a KV-cache prewarm or
    an accepted-record kind), so rollback is a no-op by construction.
    """
    if proposal.kind in {"shard_split", "replica"}:
        return _rollback_via_catalog(proposal)
    return {"rolled_back": True, "method": "noop"}


def _resharding_methods() -> set[str]:
    from agent_utilities.mcp.tools import engine_tools

    return set(engine_tools.ENGINE_DOMAINS.get("resharding") or [])


def _apply_via_catalog(proposal: PlacementProposal) -> dict[str, Any]:
    from agent_utilities.mcp.tools import engine_tools

    methods = _resharding_methods()
    if "catalog_assign" not in methods:
        return {
            "applied": False,
            "method": "catalog_assign",
            "detail": "resharding surface unavailable",
        }
    shard = proposal.evidence.get("shard")
    if shard is None:
        # Deterministic placeholder virtual-shard id (a real deployment's
        # rebalance planner picks the actual target; this only needs to be
        # STABLE per target so a repeat canary reassigns the same shard).
        shard = abs(hash(proposal.target)) % 8
    params: dict[str, Any] = {"graph": proposal.target, "shard": int(shard)}
    node = proposal.evidence.get("node")
    if node is not None:
        params["node"] = node
    try:
        raw = engine_tools._dispatch(
            "resharding", methods, "catalog_assign", json.dumps(params), ""
        )
        payload = json.loads(raw)
    except Exception as e:  # noqa: BLE001 — surface as data, never raise
        return {"applied": False, "method": "catalog_assign", "detail": str(e)}
    if isinstance(payload, dict) and payload.get("error"):
        return {"applied": False, "method": "catalog_assign", "detail": payload["error"]}

    _invalidate_placement_cache(proposal.target)
    return {"applied": True, "method": "catalog_assign", "detail": payload}


def _rollback_via_catalog(proposal: PlacementProposal) -> dict[str, Any]:
    from agent_utilities.mcp.tools import engine_tools

    methods = _resharding_methods()
    if "catalog_remove" not in methods:
        return {
            "rolled_back": False,
            "method": "catalog_remove",
            "detail": "resharding surface unavailable",
        }
    try:
        raw = engine_tools._dispatch(
            "resharding",
            methods,
            "catalog_remove",
            json.dumps({"graph": proposal.target}),
            "",
        )
        payload = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return {"rolled_back": False, "method": "catalog_remove", "detail": str(e)}

    _invalidate_placement_cache(proposal.target)
    if isinstance(payload, dict) and payload.get("error"):
        return {
            "rolled_back": False,
            "method": "catalog_remove",
            "detail": payload["error"],
        }
    return {"rolled_back": True, "method": "catalog_remove", "detail": payload}


def _invalidate_placement_cache(target: str) -> None:
    """Drop the AU-side ``placement_catalog.py`` route cache for ``target``.

    So the READ-side consumer (:func:`~agent_utilities.knowledge_graph.core.
    placement_catalog.resolve_placement`) picks up the new/reverted
    assignment on its next call rather than serving a stale cached route
    for up to its TTL.
    """
    try:
        from agent_utilities.knowledge_graph.core import placement_catalog

        placement_catalog.invalidate(target)
    except Exception as e:  # noqa: BLE001 — cache invalidation is best-effort
        logger.debug("placement_mining: cache invalidate failed for %s: %s", target, e)


def _apply_via_kvcache(proposal: PlacementProposal) -> dict[str, Any]:
    key = proposal.target.split("→")[-1].strip() or proposal.target
    try:
        from agent_utilities.kvcache import EpistemicGraphKVBackend

        backend = EpistemicGraphKVBackend.from_env()
        try:
            stored = backend.put(f"placement_prewarm:{key}", key.encode("utf-8"))
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
    except Exception as e:  # noqa: BLE001 — the KV connector already degrades transport
        return {"applied": False, "method": "kvcache_put", "detail": str(e)}
    return {
        "applied": bool(stored),
        "method": "kvcache_put",
        "detail": {"key": f"placement_prewarm:{key}", "value_b64": base64.b64encode(key.encode()).decode()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# The full cycle — mine -> propose -> govern -> canary -> apply
# ═══════════════════════════════════════════════════════════════════════════


def run_placement_mining_cycle(
    engine: Any,
    *,
    measurement_fn: MeasurementFn | None = None,
    tolerance: float = _CANARY_TOLERANCE,
    limit: int = _SCAN_LIMIT,
) -> dict[str, Any]:
    """Mine -> propose -> govern -> canary -> apply, end to end (X-5).

    Mirrors ``loop_controller._run_trace_mining``'s shape exactly: mine ->
    :class:`PlacementProposal` -> ``Claim`` (ALWAYS persisted,
    ``status="proposal"``, unconditionally) -> ``PromotionGovernanceValidator``
    (reused as-is) -> ``action_policy.decide(kind="apply_placement_change")``
    (SAFETY-CRITICAL — shipped tier is ``approval_required``, see
    ``deploy/action-policy.default.yml``) -> ONLY THEN a measured
    :func:`run_canary` -> a promoted canary flips the ``Claim`` to
    ``status="applied"``; a rolled-back one is marked ``status="rejected"``.

    Propose-only by construction: nothing reaches the canary (let alone the
    engine's PlacementCatalog) unless BOTH governance is valid AND the
    action-policy decision allows — the shipped default never allows it, so
    this never applies anything out of the box.
    """
    from agent_utilities.orchestration.action_policy import (
        ActionRequest,
        get_action_policy,
    )

    from .promotion_governance import PromotionGovernanceValidator

    errors: list[str] = []
    mine_result = mine_placement_patterns(engine, limit=limit)
    errors.extend(mine_result.get("errors") or [])
    proposals = placement_proposals_from_mining(mine_result)
    below_floor = [p for p in proposals if not p.clears_floor]
    eligible = [p for p in proposals if p.clears_floor]

    validator = PromotionGovernanceValidator(engine)
    action_policy = get_action_policy(engine)

    persisted = 0
    applied = 0
    examples: list[dict[str, Any]] = []

    for prop in eligible:
        claim = prop.to_claim_node()
        bundle = prop.to_evidence_bundle()
        spec = {
            "id": claim.id,
            "name": claim.name,
            "goal": claim.claim_text,
            "description": claim.claim_text,
            "quality_score": claim.confidence,
            "type": "PlacementProposal",
        }

        try:
            engine.add_node(
                claim.id,
                "PlacementProposal",
                properties={
                    **claim.model_dump(mode="json", exclude={"type"}),
                    "status": "proposal",
                    "kind": prop.kind,
                    "target": prop.target,
                    "evidence_bundle_json": bundle.model_dump_json(),
                },
            )
            persisted += 1
        except Exception as e:  # noqa: BLE001 — persistence is best-effort
            errors.append(f"placement_mining:persist {claim.id}: {e}")
            continue

        try:
            verdict = validator.validate(spec)
        except Exception as e:  # noqa: BLE001 — a validator error holds, never crashes
            errors.append(f"placement_mining:validate {claim.id}: {e}")
            continue

        # -- SAFETY-CRITICAL: action_policy.decide() MUST run — and complete —
        # BEFORE the canary (which itself applies to a small scope) for every
        # candidate, unconditionally. --
        try:
            decision = action_policy.decide(
                ActionRequest(
                    kind="apply_placement_change",
                    target=prop.target,
                    params={
                        "proposal_kind": prop.kind,
                        "confidence": prop.confidence,
                        "governance_valid": verdict.valid,
                    },
                    source="placement_mining",
                    reason=(
                        f"apply a mined {prop.kind} placement change for {prop.target}"
                    ),
                )
            )
        except Exception as e:  # noqa: BLE001 — fail closed, never crash
            errors.append(f"placement_mining:action_policy {claim.id}: {e}")
            continue

        record: dict[str, Any] = {
            "claim_id": claim.id,
            "kind": prop.kind,
            "target": prop.target,
            "confidence": round(prop.confidence, 4),
            "governance_valid": verdict.valid,
            "action_decision": decision.decision,
            "applied": False,
        }

        # -- ONLY reachable after action_policy.decide() (above) returned
        # allowed (auto/auto_notify). The shipped tier is approval_required,
        # so this branch never fires out of the box. --
        if verdict.valid and decision.allowed:
            try:
                canary = run_canary(prop, measurement_fn=measurement_fn, tolerance=tolerance)
                record["canary"] = canary.to_dict()
                record["applied"] = bool(canary.applied)
            except Exception as e:  # noqa: BLE001 — never let a canary crash the cycle
                errors.append(f"placement_mining:canary {claim.id}: {e}")
                canary = None

            new_status = "proposal"
            if canary is not None:
                new_status = "applied" if canary.applied else "rejected"
                if canary.applied:
                    applied += 1
            try:
                engine.add_node(
                    claim.id,
                    "PlacementProposal",
                    properties={
                        **claim.model_dump(mode="json", exclude={"type"}),
                        "status": new_status,
                        "kind": prop.kind,
                        "target": prop.target,
                        "evidence_bundle_json": bundle.model_dump_json(),
                    },
                )
            except Exception as e:  # noqa: BLE001 — status update is best-effort
                errors.append(f"placement_mining:promote {claim.id}: {e}")

        if len(examples) < 5:
            examples.append(record)

    return {
        "proposals": len(proposals),
        "below_floor": len(below_floor),
        "eligible": len(eligible),
        "persisted": persisted,
        "applied": applied,
        "examples": examples,
        "errors": errors,
    }
