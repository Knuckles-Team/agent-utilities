"""Connector-grounded harness evolution + ARA-Seal (CONCEPT:KG-2.108).

HarnessX evolves over benchmark verifiers only, and its co-evolution needs one team
owning harness+model ("impractical without cross-team coordination"). Our ONE
ontology-driven KG is that cross-team substrate: a harness variant is grounded in
evidence from the WHOLE connector fleet via the ARA `grounded_in` chain (transitive,
already reasoned by owl_bridge), sealed by held-out certification (L1/L2/L3), and its
behavioral dimensions link to the live `ecosystem_topology` services they touch — so
reasoning chains harness-edit → dimension → service → node, something a siloed
per-agent harness cannot.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.harness.superhuman_gate import CertificationResult


def ground_variant(variant_id: str, evidence_ids: list[str]) -> list[dict[str, Any]]:
    """`grounded_in` edges from a HarnessVariant to its connector-fleet evidence
    (traces / test-results / metric-reports). Because `grounded_in` is transitive
    (KG-2.80), reasoning materialises variant → source from variant → evidence."""
    return [
        {"source": variant_id, "target": e, "type": "grounded_in"}
        for e in evidence_ids
        if e
    ]


def seal_level_for(cert: CertificationResult) -> str:
    """Map a held-out certification to a Seal level: L3 (certified with a clear
    margin), L2 (certified), L1 (uncertified/baseline) (CONCEPT:AHE-3.56)."""
    if not getattr(cert, "certified", False):
        return "L1"
    margin = (getattr(cert, "ci_lower", 0.0) or 0.0) - (
        getattr(cert, "human_baseline", 0.0) or 0.0
    )
    return "L3" if margin >= 0.1 else "L2"


def seal_variant(
    variant_id: str, cert: CertificationResult
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Seal a promoted variant by held-out certification. Returns (nodes, edges,
    level): a seal_certificate node grounded in the variant + the certificationLevel."""
    level = seal_level_for(cert)
    seal_id = f"seal_certificate:{variant_id}:{level}"
    nodes = [
        {
            "id": seal_id,
            "type": "seal_certificate",
            "level": level,
            "mean_reward": getattr(cert, "mean_reward", 0.0),
            "ci_lower": getattr(cert, "ci_lower", 0.0),
        },
        {"id": variant_id, "type": "harness_variant", "certificationLevel": level},
    ]
    edges = [{"source": seal_id, "target": variant_id, "type": "grounded_in"}]
    return nodes, edges, level


def link_dimension_to_service(dimension_id: str, service_id: str) -> dict[str, Any]:
    """Link a HarnessDimension to the deployed ecosystem Service it touches (via the
    transitive `grounded_in` chain), so reasoning chains dimension → service → node."""
    return {"source": dimension_id, "target": service_id, "type": "grounded_in"}
