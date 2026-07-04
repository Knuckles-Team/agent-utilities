#!/usr/bin/python
from __future__ import annotations

"""Finance microstructure ontology objects — CONCEPT:AU-KG.ontology.kyle-insider-stealth-surveillance: Kyle insider/stealth surveillance signal ontology (interfaces + typed links).

Makes the Kyle insider/stealth-trading surveillance work (engine kernel KG-2.20k,
detector EE-042, gate EE-043 — distilling arXiv:2605.27684) *ontologically
driven*: a short-horizon microstructure signal and its surveillance
specialization become first-class ontology **Interfaces**, linked to the research
paper (``Article``) and the ecosystem capability ``Concept`` registry via typed
**Links** the OWL bridge reasons over:

* ``GROUNDED_IN`` — the signal is grounded in the research paper (inverse SUPPORTS),
* ``RELATES_TO`` — the signal relates to an ecosystem concept.

Registered into the import-populated default registries so ``kg.ontology``
discovers them with no configuration (KG-2.38 interfaces, KG-2.26 links). The
emerald-exchange MCP keeps writing raw ``microstructure_signal`` nodes for
robustness; this layer gives them a governed ontology schema (conformance, link
cardinality, OWL/SHACL emission, cross-domain reasoning). DEFENSIVE: informed-flow
surveillance + maker adverse-selection protection, not trade concealment.
(CONCEPT:AU-KG.ontology.kyle-insider-stealth-surveillance)
"""

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from .interfaces import (
    DEFAULT_INTERFACE_REGISTRY,
    Interface,
    InterfaceLinkConstraint,
    InterfaceProperty,
    InterfaceRegistry,
)
from .links import (
    DEFAULT_LINK_REGISTRY,
    LinkCardinality,
    LinkType,
    LinkTypeRegistry,
)

_FINANCE_INTERFACES = "MicrostructureSignal", "SurveillanceSignal"


def register_finance_ontology(
    interfaces: InterfaceRegistry, links: LinkTypeRegistry
) -> None:
    """Register the microstructure-signal interfaces + typed links (CONCEPT:AU-KG.ontology.kyle-insider-stealth-surveillance).

    Idempotent: skips an interface/link already registered (re-import safe).
    """
    if interfaces.get("MicrostructureSignal") is None:
        interfaces.register(
            Interface(
                name="MicrostructureSignal",
                description=(
                    "Abstract shape for a short-horizon microstructure alpha with "
                    "measured statistical priors (directional accuracy, deflated "
                    "Sharpe, PBO, decay regime) the backtester writes back so "
                    "signal-fusion weights self-adjust."
                ),
                properties=[
                    InterfaceProperty(
                        name="prediction_horizon",
                        type_ref="string",
                        description="Forecast horizon, e.g. '30s', '1m', '5m'.",
                    ),
                    InterfaceProperty(
                        name="directional_accuracy",
                        type_ref="double",
                        description="Measured directional accuracy prior in [0, 1].",
                    ),
                    InterfaceProperty(
                        name="standalone_sharpe",
                        type_ref="double",
                        description="Measured deflated Sharpe.",
                    ),
                    InterfaceProperty(
                        name="pbo",
                        type_ref="double",
                        description="Probability of backtest overfit.",
                    ),
                    InterfaceProperty(
                        name="decay_regime",
                        type_ref="string",
                        required=False,
                        description="stationary | decaying | regime_dependent.",
                    ),
                ],
            )
        )
        interfaces.register(
            Interface(
                name="SurveillanceSignal",
                description=(
                    "A MicrostructureSignal specialization carrying Kyle "
                    "insider/stealth-trading surveillance scores (engine kernel "
                    "KG-2.20k, distils arXiv:2605.27684): informed-flow share, "
                    "detection hazard and a legal-risk score driving maker "
                    "adverse-selection protection."
                ),
                extends=["MicrostructureSignal"],
                properties=[
                    InterfaceProperty(
                        name="legal_risk_score",
                        type_ref="double",
                        description="Squashed detection·toxicity hazard in [0, 1].",
                    ),
                    InterfaceProperty(
                        name="informed_share",
                        type_ref="double",
                        description="VPIN-estimated informed-flow share α.",
                    ),
                    InterfaceProperty(
                        name="stealth_ratio",
                        type_ref="double",
                        required=False,
                        description="Noise/informed volume (camouflage effectiveness).",
                    ),
                ],
                link_constraints=[
                    InterfaceLinkConstraint(
                        name="grounded_in",
                        edge_type=RegistryEdgeType.GROUNDED_IN,
                        target_type=RegistryNodeType.ARTICLE,
                        min_count=0,
                        description="Grounded in the research paper it distils.",
                    ),
                ],
            )
        )

    for link in (
        LinkType(
            name="signal_grounded_in_paper",
            source_type=RegistryNodeType.MICROSTRUCTURE_SIGNAL,
            target_type=RegistryNodeType.ARTICLE,
            edge_type=RegistryEdgeType.GROUNDED_IN,
            cardinality=LinkCardinality.MANY_TO_MANY,
            inverse_edge_type=RegistryEdgeType.SUPPORTS,
            description="A microstructure signal is grounded in a research paper.",
        ),
        LinkType(
            name="signal_relates_to_concept",
            source_type=RegistryNodeType.MICROSTRUCTURE_SIGNAL,
            target_type=RegistryNodeType.CONCEPT,
            edge_type=RegistryEdgeType.RELATES_TO,
            cardinality=LinkCardinality.MANY_TO_MANY,
            description="A microstructure signal relates to an ecosystem concept.",
        ),
    ):
        if links.get(link.name) is None:
            links.register(link)


# Import-populated, never an empty shell — so kg.ontology discovers them by default.
register_finance_ontology(DEFAULT_INTERFACE_REGISTRY, DEFAULT_LINK_REGISTRY)

__all__ = ["register_finance_ontology"]
