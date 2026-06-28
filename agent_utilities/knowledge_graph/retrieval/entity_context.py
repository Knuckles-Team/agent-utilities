#!/usr/bin/python
from __future__ import annotations

"""Generic entity context provider (CONCEPT:KG-2.139).

The ``entity`` domain of the context plane: answers *"what's in the world-model /
how many X / show me recent X"* over **any** node type present in the KG — Code,
Document, Concept, Task today; Ticket, Incident, Deployment, Process as connectors
populate them tomorrow. This is the enterprise-cockpit mechanism: a new domain
(tickets, deploys, finance) is this provider registered under that name with a
label filter — *more providers on the one plane*, not a new subsystem — so the
breadth grows with the ingested data, never speculatively ahead of it.

Pure Cypher reads (best-effort, never raises); degrades to "nothing ingested yet".
"""

from typing import Any

from agent_utilities.knowledge_graph.retrieval.context_plane import read_rows

VALID_INTENTS = ("health", "list", "why")

# Domain alias → candidate node labels (extend as connectors land their types).
DOMAIN_LABELS: dict[str, tuple[str, ...]] = {
    "tickets": ("Ticket", "Issue", "Incident"),
    "deploys": ("Deployment", "Service", "Stack"),
    "process": ("Process", "BpmnProcess", "ProcessInstance"),
    "research": ("Paper", "ResearchPaper", "Document"),
}


def _as_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def entity_context(
    engine: Any,
    *,
    query: str = "",
    intent: str = "health",
    domain: str = "entity",
    top_k: int = 12,
    **_opts: Any,
) -> dict[str, Any]:
    """Synthesize a count/list view over KG node types (see module docstring)."""
    intent = (intent or "health").strip().lower()
    if intent not in VALID_INTENTS:
        intent = "health"
    limit = max(1, min(50, int(top_k)))

    # "why" is a relational, multi-hop question ("why/how is X connected …") — answer
    # it by *actively reconstructing* the query-relevant subgraph from a seed cue
    # rather than a flat census (CONCEPT:KG-2.275, MRAgent). Falls through to the
    # census view when nothing resolves, so it never regresses the other intents.
    if intent == "why":
        reconstructed = _reconstruct_answer(engine, query, domain=domain, top_k=limit)
        if reconstructed is not None:
            return reconstructed

    labels = DOMAIN_LABELS.get(domain, ())
    by_type = [
        {"type": str(r.get("label")), "count": _as_int(r.get("n"))}
        for r in read_rows(
            engine,
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS n "
            "ORDER BY n DESC LIMIT 30",
            {},
        )
        if r.get("label")
    ]
    # Focus: a label named in the query, or the domain's candidate labels.
    focus_labels = [
        t["type"] for t in by_type if str(t["type"]).lower() in (query or "").lower()
    ]
    if not focus_labels and labels:
        focus_labels = [t["type"] for t in by_type if t["type"] in labels]

    recent: list[dict[str, Any]] = []
    if focus_labels:
        recent = [
            {"id": r.get("id"), "name": r.get("name"), "type": focus_labels[0]}
            for r in read_rows(
                engine,
                f"MATCH (n:{focus_labels[0]}) RETURN n.id AS id, n.name AS name "
                f"LIMIT {limit}",
                {},
            )
        ]

    answer = _synthesize(domain, by_type, focus_labels, recent)
    citations = [{"type": "node_type", **t} for t in by_type[:top_k]] + [
        {"type": "node", **r} for r in recent
    ]
    return {
        "status": "ok",
        "domain": domain,
        "intent": intent,
        "query": query,
        "answer": answer,
        "citations": citations,
        "sections": {"by_type": by_type, "recent": recent},
        "capability_id": f"entity:{domain}:{intent}",
        "used_primitives": ["type_census"] + (["recent"] if recent else []),
    }


def _synthesize(domain, by_type, focus_labels, recent) -> str:
    if not by_type:
        return (
            f"The world-model has no '{domain}' entities ingested yet — wire the "
            "connector (source_sync) and they appear here automatically."
        )
    total = sum(t["count"] for t in by_type)
    head = (
        f"World-model: {total} nodes across {len(by_type)} type(s). "
        "Top: " + ", ".join(f"{t['type']}={t['count']}" for t in by_type[:8]) + "."
    )
    if focus_labels:
        focus = focus_labels[0]
        cnt = next((t["count"] for t in by_type if t["type"] == focus), 0)
        tail = f" {focus}: {cnt} node(s)."
        if recent:
            tail += " Recent: " + ", ".join(
                str(r.get("name") or r.get("id")) for r in recent[:6]
            )
        return head + tail
    if domain in DOMAIN_LABELS:
        return (
            head
            + f" No {domain} entities ({'/'.join(DOMAIN_LABELS[domain])}) ingested yet."
        )
    return head


def _reconstruct_answer(
    engine: Any, query: str, *, domain: str, top_k: int
) -> dict[str, Any] | None:
    """Active reconstruction for a "why"/relational question (CONCEPT:KG-2.275).

    Resolve seed cues from the query, walk the Cue-Tag-Content subgraph
    accumulating evidence, and synthesize a cited answer with the reconstruction
    trajectory. Returns ``None`` (so the caller falls back to the census) when no
    seed resolves or the walk surfaces no evidence.
    """
    from agent_utilities.knowledge_graph.retrieval.active_reconstruction import (
        engine_neighbor_fn,
        reconstruct,
        resolve_seeds,
    )

    if not (query or "").strip():
        return None
    seeds = resolve_seeds(engine, query, top_k=3)
    seed_ids = [str(s["id"]) for s in seeds if s.get("id")]
    if not seed_ids:
        return None

    recon = reconstruct(
        query,
        seed_ids,
        neighbor_fn=engine_neighbor_fn(engine),
        content_top_k=top_k,
    )
    if not recon.evidence:
        return None

    seed_name = next((s.get("name") for s in seeds if s.get("name")), seed_ids[0])
    top = recon.top(top_k)
    path = " ".join(f"--[{e.via_tag}]--> {e.text}" for e in top[:3])
    answer = (
        f"Reconstructed {len(recon.evidence)} evidence node(s) from seed "
        f"'{seed_name}' over {len(recon.steps)} hop(s) "
        f"(stopped: {recon.stop_reason or 'complete'}). "
        f"Trajectory: {seed_name} {path}. "
        "Top evidence: " + ", ".join(str(e.text) for e in top[:6]) + "."
    )
    citations = [
        {
            "type": "evidence",
            "id": e.id,
            "text": e.text,
            "via_tag": e.via_tag,
            "hop": e.hop,
            "label": e.label,
            "score": round(e.score, 4),
        }
        for e in top
    ]
    return {
        "status": "ok",
        "domain": domain,
        "intent": "why",
        "query": query,
        "answer": answer,
        "citations": citations,
        "sections": {
            "seeds": seed_ids,
            "trajectory": [
                {
                    "hop": s.hop,
                    "activated_tags": s.activated_tags,
                    "added": s.added_ids,
                    "pruned": s.pruned,
                }
                for s in recon.steps
            ],
            "stop_reason": recon.stop_reason,
        },
        "capability_id": f"entity:{domain}:why",
        "used_primitives": ["active_reconstruction"],
    }
