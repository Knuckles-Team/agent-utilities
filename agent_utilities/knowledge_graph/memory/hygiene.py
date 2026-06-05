"""CONCEPT:KG-2.17 — Memory Hygiene (decay scanner + semantic merge).

Assimilated from memory-os (ClaudioDrews/memory-os@a4ca094, scripts/decay_scanner.py &
scripts/semantic_dedup.py). A maintenance pass that bounds memory growth WITHOUT destroying
information:

- **Decay scan**: exponential decay with an importance-tiered half-life; low-decay AI-generated
  memory is *archived* by setting KG-2.11 ``valid_to`` (never hard-deleted, so it stays
  as-of-queryable). High-confidence-but-stale items are *alerted* for review instead of archived;
  human/procedural memory is exempt.
- **Semantic merge**: near-duplicate memories (cosine ≥ 0.92) are merged (tag union, max
  importance) with a cheap length-ratio pre-filter to avoid O(n²) on clearly-different sizes.

The decision functions are pure (LLM/engine-free) and unit-tested; ``MemoryHygiene.run`` applies
the plan to the durable backend. Reuses ``core.engine.cosine_similarity``.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# memory-os archival half-lives (days). Distinct from KG-2.1's recall-decay timescales — these
# govern *archival*, not ranking. Importance ≥ 0.3 decays slowly (90d), else fast (30d).
HALF_LIFE_IMPORTANT_DAYS = 90.0
HALF_LIFE_DEFAULT_DAYS = 30.0
ARCHIVE_DECAY_FLOOR = 0.1  # decay below this → archival candidate
CONFIDENCE_EXEMPT = (
    0.7  # importance/confidence ≥ this → exempt (alert instead of archive)
)
EXEMPT_SOURCE_TYPES = frozenset({"human", "procedural"})
MERGE_COSINE = 0.92  # near-duplicate threshold
LENGTH_RATIO_FLOOR = 0.5  # skip cosine if size ratio < this (cheap pre-filter)


def half_life_days(importance: float) -> float:
    return HALF_LIFE_IMPORTANT_DAYS if importance >= 0.3 else HALF_LIFE_DEFAULT_DAYS


def decay_score(age_days: float, importance: float) -> float:
    """Exponential decay ``exp(-ln2 * age / half_life)`` in [0, 1] (1 = fresh)."""
    hl = half_life_days(importance)
    return math.exp(-math.log(2) * max(0.0, age_days) / hl)


def _age_days(created_at: str | None, now: datetime) -> float:
    if not created_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, (now - dt).total_seconds() / 86400.0)


def classify_node(node: dict[str, Any], now: datetime) -> str:
    """Classify a node: ``exempt`` | ``alert`` | ``archive`` | ``keep`` (CONCEPT:KG-2.17)."""
    source_type = str(node.get("source_type") or node.get("source") or "").lower()
    importance = float(node.get("importance_score", node.get("importance", 0.0)) or 0.0)
    if source_type in EXEMPT_SOURCE_TYPES:
        return "exempt"
    age = _age_days(node.get("created_at") or node.get("storage_time"), now)
    d = decay_score(age, importance)
    if d >= ARCHIVE_DECAY_FLOOR:
        return "keep"
    # Stale: archive unless high-confidence (then alert for human review — never silently drop).
    confidence = float(
        node.get("confidence", node.get("importance_score", importance)) or 0.0
    )
    return "alert" if confidence >= CONFIDENCE_EXEMPT else "archive"


def plan_decay(nodes: list[dict[str, Any]], now: datetime) -> dict[str, list[str]]:
    """Bucket node ids by decay decision (pure)."""
    plan: dict[str, list[str]] = {"archive": [], "alert": [], "keep": [], "exempt": []}
    for n in nodes:
        plan[classify_node(n, now)].append(str(n.get("id", "")))
    return plan


def semantic_merge_groups(
    nodes: list[dict[str, Any]], threshold: float = MERGE_COSINE
) -> list[list[str]]:
    """Group near-duplicate node ids (cosine ≥ threshold) with a length-ratio pre-filter.

    Each group is a list of ids; the first is the survivor. Nodes without embeddings are skipped.
    """
    from ..core.engine import cosine_similarity

    items = [n for n in nodes if n.get("embedding")]
    groups: list[list[str]] = []
    claimed: set[int] = set()
    for i, a in enumerate(items):
        if i in claimed:
            continue
        group: list[str] = [str(a.get("id", ""))]
        a_len = len(str(a.get("content") or a.get("name") or "")) or 1
        for j in range(i + 1, len(items)):
            if j in claimed:
                continue
            b = items[j]
            b_len = len(str(b.get("content") or b.get("name") or "")) or 1
            ratio = min(a_len, b_len) / max(a_len, b_len)
            if ratio < LENGTH_RATIO_FLOOR:
                continue  # clearly different sizes — skip the cosine computation
            if cosine_similarity(a["embedding"], b["embedding"]) >= threshold:
                group.append(str(b.get("id", "")))
                claimed.add(j)
        if len(group) > 1:
            groups.append(group)
            claimed.add(i)
    return groups


def merge_plan(
    nodes: list[dict[str, Any]], groups: list[list[str]]
) -> list[dict[str, Any]]:
    """Compute the merge-write for each near-duplicate group (pure, CONCEPT:KG-2.17).

    For each group the first id is the **survivor**. Returns
    ``{"survivor": id, "tags": [union], "importance": max, "retired": [ids]}`` — the survivor absorbs
    the union of tags and the max importance; the rest are retired (soft, pointed at the survivor).
    """
    by_id = {n.get("id"): n for n in nodes}
    plans: list[dict[str, Any]] = []
    for group in groups:
        if len(group) < 2:
            continue
        survivor, *dups = group
        members = [by_id.get(gid, {}) for gid in group]
        tags: list[str] = []
        for m in members:
            for t in m.get("tags", []) or []:
                if t not in tags:
                    tags.append(t)
        importance = max(
            (
                float(m.get("importance_score", m.get("importance", 0.0)) or 0.0)
                for m in members
            ),
            default=0.0,
        )
        plans.append(
            {
                "survivor": survivor,
                "tags": tags,
                "importance": importance,
                "retired": dups,
            }
        )
    return plans


class MemoryHygiene:
    """Applies the decay + merge plan to the durable backend (CONCEPT:KG-2.17)."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def run(
        self, *, now: datetime | None = None, dry_run: bool = False
    ) -> dict[str, Any]:
        """Scan memory nodes, archive stale AI content (set ``valid_to``), and report.

        Archival is a soft close of the bi-temporal interval (KG-2.11) — the node is never
        deleted, so as-of queries before the archival instant still see it.
        """
        now = now or datetime.now(UTC)
        backend = getattr(self.engine, "backend", None)
        if backend is None:
            return {"error": "no backend", "archived": 0, "alerted": 0, "merged": 0}
        try:
            rows = backend.execute(
                "MATCH (n:Memory) WHERE n.embedding IS NOT NULL OR n.content IS NOT NULL "
                "RETURN n.id as id, n as data LIMIT 5000"
            )
        except Exception as e:  # pragma: no cover - backend variance
            logger.debug("Hygiene scan query failed: %s", e)
            return {"error": str(e), "archived": 0, "alerted": 0, "merged": 0}

        nodes: list[dict[str, Any]] = []
        for row in rows or []:
            data = row.get("data") if isinstance(row, dict) else None
            if isinstance(data, dict):
                d = dict(data)
                d["id"] = row.get("id")
                nodes.append(d)

        plan = plan_decay(nodes, now)
        groups = semantic_merge_groups(nodes)
        plans = merge_plan(nodes, groups)
        stamp = now.isoformat()
        retired = 0

        if not dry_run:
            for nid in plan["archive"]:
                try:
                    backend.execute(
                        "MATCH (n) WHERE n.id = $id SET n.valid_to = $vt, n.status = 'ARCHIVED'",
                        {"id": nid, "vt": stamp},
                    )
                except Exception as e:  # pragma: no cover
                    logger.debug("archive failed for %s: %s", nid, e)
            # CONCEPT:KG-2.17 — APPLY the semantic merge: survivor absorbs union(tags)+max(importance);
            # each duplicate is soft-retired (status=MERGED + valid_to) and linked MERGED_INTO survivor
            # (never hard-deleted, so as-of queries before the merge still resolve the duplicate).
            for p in plans:
                try:
                    backend.execute(
                        "MATCH (s) WHERE s.id = $sid SET s.tags = $tags, s.importance_score = $imp",
                        {
                            "sid": p["survivor"],
                            "tags": p["tags"],
                            "imp": p["importance"],
                        },
                    )
                    for dup in p["retired"]:
                        backend.execute(
                            "MATCH (d) WHERE d.id = $id "
                            "SET d.status = 'MERGED', d.valid_to = $vt, d.merged_into = $sid",
                            {"id": dup, "vt": stamp, "sid": p["survivor"]},
                        )
                        try:
                            self.engine.link_nodes(dup, p["survivor"], "MERGED_INTO")
                        except Exception:  # noqa: BLE001 - edge is best-effort
                            pass
                        retired += 1
                except Exception as e:  # pragma: no cover
                    logger.debug("merge apply failed for %s: %s", p["survivor"], e)
        else:
            retired = sum(len(p["retired"]) for p in plans)

        return {
            "scanned": len(nodes),
            "archived": len(plan["archive"]),
            "alerted": len(plan["alert"]),
            "exempt": len(plan["exempt"]),
            "kept": len(plan["keep"]),
            "merge_groups": len(groups),
            "merged": retired,
            "dry_run": dry_run,
        }


def run_hygiene(engine: Any, *, dry_run: bool = False) -> dict[str, Any]:
    """CLI/daemon entry point for a memory-hygiene pass (CONCEPT:KG-2.17)."""
    return MemoryHygiene(engine).run(dry_run=dry_run)
