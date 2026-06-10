"""Bounded subgraph checkout with dirty-delta write-back (CONCEPT:KG-2.7 P2).

The tiered store loads a *bounded* working subgraph from the durable tier into the
fast L1 compute engine (``engine.checkout_subgraph``). This module wraps that
working copy so that mutations are tracked, and only the **deltas** — not the whole
graph — are written back to the durable tier:

    sub = engine.checkout_subgraph("MATCH (n)-[*1..2]-(t {id:$id}) ... RETURN n, r")
    sub.add_node("x", {...}); sub.add_edge("x", "y", {...})
    sub.flush_deltas_to_durable()          # writes 2 rows, not the full subgraph

This replaces full-graph enumeration on the hot path (``reconcile_to_durable``
stays the periodic safety net), supports a non-blocking background flush, and
stamps a per-node baseline version at checkout for conflict detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _sanitize_label(label: str) -> str:
    s = re.sub(r"\W+", "_", str(label or "Node")).strip("_")
    return s or "Node"


class CheckedOutSubgraph:
    """A write-back-capable working copy of a bounded subgraph.

    Wraps a detached :class:`GraphComputeEngine`. Reads/algorithms delegate to the
    inner engine; mutations are applied to it **and** recorded in a dirty ledger
    so :meth:`flush_deltas_to_durable` writes only what changed.
    """

    def __init__(
        self,
        engine: Any,
        *,
        durable: Any = None,
        baseline: dict[str, str] | None = None,
    ) -> None:
        self._engine = engine
        self._durable = durable
        self._baseline: dict[str, str] = dict(baseline or {})
        # Dirty ledger
        self._dirty_nodes: dict[str, dict[str, Any]] = {}
        self._deleted_nodes: dict[str, str] = {}  # id -> label (captured at delete)
        self._dirty_edges: dict[tuple[str, str], dict[str, Any]] = {}
        self._deleted_edges: set[tuple[str, str]] = set()

    # ── checkout baseline / versioning ────────────────────────────────────
    @staticmethod
    def _version_of(props: dict[str, Any] | None) -> str:
        """Stable short hash of a node's properties (for conflict detection)."""
        blob = json.dumps(props or {}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]  # noqa: S324 - not security

    # ── inner-engine delegation ───────────────────────────────────────────
    @property
    def engine(self) -> Any:
        return self._engine

    def __getattr__(self, name: str) -> Any:
        # Only fires for names not defined on this wrapper → delegate reads,
        # algorithms (pagerank, get_blast_radius, ...) to the inner engine.
        engine = self.__dict__.get("_engine")
        if engine is not None and hasattr(engine, name):
            return getattr(engine, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # ── mutating ops (apply + record delta) ───────────────────────────────
    def add_node(self, node_id: str, properties: Any = None, **kwargs: Any) -> None:
        self._engine.add_node(node_id, properties, **kwargs)
        self._deleted_nodes.pop(node_id, None)
        # Record the post-write props so the flush mirrors exactly what L1 holds.
        try:
            current = dict(self._engine._get_node_properties(node_id) or {})
        except Exception:  # noqa: BLE001
            current = dict(properties or {})
        self._dirty_nodes[node_id] = current

    def add_edge(
        self, source_id: str, target_id: str, properties: Any = None, **kwargs: Any
    ) -> None:
        self._engine.add_edge(source_id, target_id, properties, **kwargs)
        key = (source_id, target_id)
        self._deleted_edges.discard(key)
        self._dirty_edges[key] = dict(properties or {})

    def remove_node(self, node_id: str) -> None:
        # Capture the label before removal so the durable delete can target it.
        label = "Node"
        try:
            props = self._engine._get_node_properties(node_id) or {}
            label = _sanitize_label(props.get("type") or props.get("label") or "Node")
        except Exception:  # noqa: BLE001
            pass
        self._engine.remove_node(node_id)
        self._dirty_nodes.pop(node_id, None)
        self._deleted_nodes[node_id] = label

    def remove_edge(
        self, source_id: str, target_id: str, key: Any = None
    ) -> None:
        self._engine.remove_edge(source_id, target_id, key)
        k = (source_id, target_id)
        self._dirty_edges.pop(k, None)
        self._deleted_edges.add(k)

    # ── dirty introspection ───────────────────────────────────────────────
    def is_dirty(self) -> bool:
        return bool(
            self._dirty_nodes
            or self._deleted_nodes
            or self._dirty_edges
            or self._deleted_edges
        )

    def delta_summary(self) -> dict[str, int]:
        return {
            "nodes_changed": len(self._dirty_nodes),
            "nodes_deleted": len(self._deleted_nodes),
            "edges_changed": len(self._dirty_edges),
            "edges_deleted": len(self._deleted_edges),
        }

    def clear_deltas(self) -> None:
        self._dirty_nodes.clear()
        self._deleted_nodes.clear()
        self._dirty_edges.clear()
        self._deleted_edges.clear()

    # ── conflict detection ────────────────────────────────────────────────
    def _durable_node_version(self, durable: Any, node_id: str) -> str | None:
        try:
            rows = durable.execute("MATCH (n {id: $id}) RETURN n", {"id": node_id})
            if rows and isinstance(rows[0], dict):
                n = rows[0].get("n", rows[0])
                if isinstance(n, dict):
                    return self._version_of(
                        {k: v for k, v in n.items() if k != "id"}
                    )
        except Exception:  # noqa: BLE001
            return None
        return None

    def detect_conflicts(self, durable: Any | None = None) -> list[str]:
        """Node ids whose durable version changed since checkout (best-effort).

        Only nodes that existed at checkout (have a baseline) can conflict; nodes
        created during the session have no baseline and are never flagged.
        """
        durable = durable or self._durable
        if durable is None:
            return []
        conflicts: list[str] = []
        candidates = set(self._dirty_nodes) | set(self._deleted_nodes)
        for nid in candidates:
            base = self._baseline.get(nid)
            if base is None:
                continue
            cur = self._durable_node_version(durable, nid)
            if cur is not None and cur != base:
                conflicts.append(nid)
        return conflicts

    # ── write-back ────────────────────────────────────────────────────────
    def flush_deltas_to_durable(
        self,
        durable: Any | None = None,
        *,
        on_conflict: str = "overwrite",
    ) -> dict[str, Any]:
        """Write only the recorded deltas to the durable tier.

        ``on_conflict`` ∈ {``overwrite`` (last-writer-wins, default; logs),
        ``skip`` (leave conflicted nodes untouched), ``error`` (raise)}. Uses the
        transpiler-recognized upsert forms (``MERGE (n:Label {id}) SET ...`` and
        ``MATCH (a),(b) MERGE (a)-[r:REL]->(b)``). Clears the ledger on completion.
        """
        durable = durable or self._durable
        if durable is None:
            raise RuntimeError("CheckedOutSubgraph: no durable backend to flush to")

        summary: dict[str, Any] = {
            "nodes_written": 0,
            "edges_written": 0,
            "nodes_deleted": 0,
            "edges_deleted": 0,
            "conflicts": 0,
            "skipped_conflicts": 0,
            "errors": 0,
        }

        conflicts: set[str] = set()
        if on_conflict in ("skip", "error"):
            conflicts = set(self.detect_conflicts(durable))
            summary["conflicts"] = len(conflicts)
            if conflicts and on_conflict == "error":
                raise RuntimeError(
                    f"CheckedOutSubgraph: {len(conflicts)} node(s) changed in the "
                    f"durable tier since checkout: {sorted(conflicts)[:10]}"
                )
            if conflicts:
                logger.warning(
                    "CheckedOutSubgraph: skipping %d conflicted node(s) on flush",
                    len(conflicts),
                )

        # Changed nodes → upsert (delta only).
        for nid, props in self._dirty_nodes.items():
            if nid in conflicts and on_conflict == "skip":
                summary["skipped_conflicts"] += 1
                continue
            try:
                label = _sanitize_label(
                    props.get("type") or props.get("label") or "Node"
                )
                scalars = {
                    k: v
                    for k, v in props.items()
                    if k != "id"
                    and isinstance(v, (str, int, float, bool))
                    and v is not None
                }
                scalars.setdefault("name", str(props.get("name") or nid))
                scalars.setdefault("type", label)
                set_clause = ", ".join(f"n.{k} = ${k}" for k in scalars)
                durable.execute(
                    f"MERGE (n:{label} {{id: $id}}) SET {set_clause}",
                    {"id": nid, **scalars},
                )
                summary["nodes_written"] += 1
            except Exception as exc:  # noqa: BLE001 — durability is best-effort
                summary["errors"] += 1
                logger.debug("flush node %s failed: %s", nid, exc)

        # Changed edges → upsert via the transpiler-recognized MATCH+MERGE form.
        for (src, dst), edata in self._dirty_edges.items():
            try:
                rel = _sanitize_label((edata or {}).get("type", "RELATED_TO"))
                durable.execute(
                    f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
                    f"MERGE (a)-[r:{rel}]->(b)",
                    {"source_id": src, "target_id": dst},
                )
                summary["edges_written"] += 1
            except Exception as exc:  # noqa: BLE001
                summary["errors"] += 1
                logger.debug("flush edge %s->%s failed: %s", src, dst, exc)

        # Deleted nodes → DETACH DELETE by label+id.
        for nid, label in self._deleted_nodes.items():
            if nid in conflicts and on_conflict == "skip":
                summary["skipped_conflicts"] += 1
                continue
            try:
                durable.execute(
                    f"MATCH (n:{label} {{id: $id}}) DETACH DELETE n", {"id": nid}
                )
                summary["nodes_deleted"] += 1
            except Exception as exc:  # noqa: BLE001
                summary["errors"] += 1
                logger.debug("flush delete node %s failed: %s", nid, exc)

        # Deleted edges → best-effort.
        for src, dst in self._deleted_edges:
            try:
                durable.execute(
                    "MATCH (a {id: $source_id})-[r]->(b {id: $target_id}) DELETE r",
                    {"source_id": src, "target_id": dst},
                )
                summary["edges_deleted"] += 1
            except Exception as exc:  # noqa: BLE001
                summary["errors"] += 1
                logger.debug("flush delete edge %s->%s failed: %s", src, dst, exc)

        logger.info("CheckedOutSubgraph flush: %s", summary)
        # On completion, the durable tier reflects our deltas — reset the ledger.
        if not (on_conflict == "skip" and conflicts):
            self.clear_deltas()
        return summary

    def flush_deltas_async(
        self, durable: Any | None = None, *, on_conflict: str = "overwrite"
    ) -> threading.Thread:
        """Flush in a background thread so the agent loop never blocks on L3.

        Snapshots the ledger and clears it immediately, so further mutations after
        the call accumulate a fresh delta set (they are NOT lost). Returns the
        worker thread (join it if you need to await durability).
        """
        snapshot = CheckedOutSubgraph(self._engine, durable=durable or self._durable)
        snapshot._dirty_nodes = self._dirty_nodes
        snapshot._deleted_nodes = self._deleted_nodes
        snapshot._dirty_edges = self._dirty_edges
        snapshot._deleted_edges = self._deleted_edges
        snapshot._baseline = self._baseline
        # Reset our ledger; the snapshot owns the in-flight deltas.
        self._dirty_nodes = {}
        self._deleted_nodes = {}
        self._dirty_edges = {}
        self._deleted_edges = set()

        def _worker() -> None:
            try:
                snapshot.flush_deltas_to_durable(on_conflict=on_conflict)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CheckedOutSubgraph async flush failed: %s", exc)

        t = threading.Thread(
            target=_worker, name="subgraph-delta-flush", daemon=True
        )
        t.start()
        return t
