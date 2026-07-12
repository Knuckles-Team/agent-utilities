#!/usr/bin/python
from __future__ import annotations

"""``capability`` context-plane provider — CONCEPT:AU-KG.retrieval.capability-power-descriptor.

Surfaces the generated Capability Power Descriptors (CPDs — see
``capability_power_descriptor.py``) on the SAME universal context plane
(``agent_utilities.knowledge_graph.retrieval.context_plane``) that
``graph_explain``/``graph_context`` already serve — no new MCP tool, no new
REST route: ``graph_explain(action="explain", target="capability:<id>")`` (or
``target="capability:list"``) is both surfaces for free, since
``graph_explain``/``graph_context`` are already mounted on MCP
(``/graph/explain``) AND REST. Reads the checked-in, gate-kept-fresh
``docs/capabilities-power.json`` — never rebuilds the tool registry per
request (that's ``scripts/gen_capability_power.py``'s job, run at commit time).
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_JSON_PATH = Path(__file__).resolve().parents[3] / "docs" / "capabilities-power.json"

__all__ = ["capability_power_context", "load_cpds"]


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, Any]:
    if not _JSON_PATH.exists():
        return {"count": 0, "capabilities": []}
    return json.loads(_JSON_PATH.read_text(encoding="utf-8"))


def load_cpds() -> dict[str, dict[str, Any]]:
    """``{capability_id: cpd_dict}`` from the checked-in generated CPD set."""
    return {c["id"]: c for c in _load_raw().get("capabilities", [])}


def capability_power_context(
    engine: Any,  # noqa: ARG001 — signature parity with every other context_plane provider
    *,
    query: str = "",
    intent: str = "how",  # noqa: ARG001 — one shape regardless of intent; kept for provider parity
    **opts: Any,
) -> dict[str, Any]:
    """Context-plane provider: look up one CPD by id, or list all ids.

    ``query`` (or ``opts["node_id"]``) is the capability id (an MCP tool name,
    e.g. ``"graph_query"``). Empty/``"list"`` returns every capability's id +
    one-liner (the browsable index) instead of one full record.
    """
    cpds = load_cpds()
    cap_id = (query or opts.get("node_id") or "").strip()
    if not cap_id or cap_id == "list":
        return {
            "status": "ok",
            "answer": f"{len(cpds)} graph-os capabilities available.",
            "capabilities": [
                {"id": c["id"], "one_line": c.get("one_line", "")}
                for c in cpds.values()
            ],
            "citations": ["docs/capabilities-power.md", "docs/capabilities-power.json"],
        }
    cpd = cpds.get(cap_id)
    if cpd is None:
        return {
            "status": "error",
            "answer": f"No CPD for capability id {cap_id!r}.",
            "available_ids": sorted(cpds),
        }
    return {
        "status": "ok",
        "answer": cpd.get("one_line", ""),
        "cpd": cpd,
        "citations": [f"docs/capabilities-power.md#{cap_id.replace('_', '')}"],
    }
