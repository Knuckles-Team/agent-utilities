#!/usr/bin/python
"""Bounded node iteration (CONCEPT:KG-2.261).

A whole-graph ``graph.nodes(data=True)`` on the live multi-tenant engine materializes
EVERY node (166K+ on ``__commons__``, with 1024-dim embeddings) into one MessagePack
frame — a gigabyte-scale payload that overloads and resets the connection. The fix is
to iterate by TYPE through the engine-side **bounded** label fetch (CONCEPT:KG-2.51,
``get_nodes_by_label``), which scopes the wire payload per label instead of dumping the
whole graph.

``iter_nodes_by_types`` is the one helper every type-filtered reader should use. It:

* uses ``get_nodes_by_label`` when the graph exposes it (the live engine facade),
  trying the common label casings (live labels are inconsistently cased, e.g.
  ``article`` vs ``Concept``), and TRUSTS that bounded result — it does NOT fall back
  to a full scan on an empty result, because falling back would re-introduce the very
  166K-node pull we are avoiding for a legitimately-empty type;
* degrades to a plain ``graph.nodes(data=True)`` only when the graph has NO bounded
  fetch — i.e. a small in-memory/test/pipeline graph, where a full pass is cheap and
  correct.

So a reader keyed on ``type == "team"`` becomes O(#teams), not O(graph), with no
behavior change on local graphs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def _type_value(t: Any) -> str:
    """The stored ``type`` string for an enum member or a plain string."""
    return str(getattr(t, "value", t))


def _label_casings(value: str) -> set[str]:
    """Common label casings to probe (live labels are inconsistently cased)."""
    return {value, value.lower(), value.upper(), value.capitalize(), value.title()}


def get_node_data(graph: Any, node_id: str) -> dict[str, Any] | None:
    """Fetch ONE node's data by id (CONCEPT:KG-2.261) — a single engine round-trip via
    the facade's per-id properties fetch, NEVER a whole-graph scan to find one node.
    Returns ``None`` if missing. Degrades to the NX node view on a local/test graph.
    """
    for meth in ("get_node_properties", "_get_node_properties"):
        fn = getattr(graph, meth, None)
        if callable(fn):
            try:
                data = fn(node_id)
            except Exception:  # noqa: BLE001 — try the next access path
                continue
            return data if isinstance(data, dict) and data else None
    # local/test graph: index the node view directly (no full scan)
    try:
        nodes = graph.nodes
        view = nodes() if callable(nodes) else nodes
        data = view[node_id]
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def iter_nodes_by_types(
    graph: Any, *types: Any
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(node_id, data)`` for nodes whose ``type`` is one of ``types``.

    BOUNDED on the engine (per-label fetch); a full pass only on a graph with no
    ``get_nodes_by_label`` (small/local). See module docstring.
    """
    wanted = {_type_value(t).lower() for t in types}
    by_label = getattr(graph, "get_nodes_by_label", None)
    if wanted and callable(by_label):
        out: dict[str, dict[str, Any]] = {}
        for t in types:
            for lbl in _label_casings(_type_value(t)):
                try:
                    rows = by_label(lbl, 0) or []
                except Exception:  # noqa: BLE001 — try the next casing
                    continue
                for row in rows:
                    if isinstance(row, list | tuple) and len(row) >= 2:
                        nid, data = str(row[0]), row[1]
                        if (
                            nid not in out
                            and isinstance(data, dict)
                            and str(data.get("type", "")).lower() in wanted
                        ):
                            out[nid] = data
        # TRUST the bounded result — do NOT full-scan an empty type (that is the
        # 166K-node pull this helper exists to prevent).
        yield from out.items()
        return
    # No bounded fetch (small/local/test graph) → a full pass is cheap + correct.
    try:
        node_iter = graph.nodes(data=True)
    except (TypeError, AttributeError):  # pragma: no cover - non-standard graph
        return
    for nid, data in node_iter:
        if isinstance(data, dict) and (
            not wanted or str(data.get("type", "")).lower() in wanted
        ):
            yield nid, data
