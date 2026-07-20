#!/usr/bin/python
"""SHACL Validation Engine.

CONCEPT:AU-KG.ontology.enterprise-governance-validation — SHACL Governance Validation

Validates the materialized RDF graph against SHACL shapes for enterprise
governance compliance. Supports layered shapes (global + domain overrides)
using the pyshacl library.

References:
    - W3C SHACL: https://www.w3.org/TR/shacl/
    - pyshacl: https://github.com/RDFLib/pySHACL
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SHACLValidator:
    """Validates RDF graphs against SHACL shape constraints.

    CONCEPT:AU-KG.ontology.enterprise-governance-validation — Enterprise Governance Validation

    Supports:
    - Single or multiple shapes files (layered validation)
    - Conformance reporting with violation details
    - Integration with OWLBridge for automatic KG validation

    Example::

        validator = SHACLValidator()
        report = validator.validate(rdf_graph, "shapes/governance.shapes.ttl")
        if not report["conforms"]:
            for v in report["violations"]:
                print(f"  {v['focus_node']}: {v['message']}")
    """

    def validate(
        self,
        data_graph: Any,
        shapes_path: str | Path | Any,
        ont_graph: Any | None = None,
    ) -> dict[str, Any]:
        """Validate an RDF graph against SHACL shapes.

        Args:
            data_graph: An rdflib.Graph to validate.
            shapes_path: SHACL shapes source — either a path to a ``.ttl`` file,
                or an already-parsed ``rdflib.Graph`` of shapes (in-memory). The
                latter lets generated shapes (e.g. the value-type registry, KG-2.39)
                be validated without round-tripping through disk.
            ont_graph: Optional ontology graph for inference during validation.

        Returns:
            Dict with:
                - conforms: bool — whether the data is fully conformant
                - violations: list of violation dicts
                - results_text: human-readable report
        """
        try:
            import pyshacl
        except ImportError:
            logger.warning("pyshacl not installed. Install with: pip install pyshacl")
            return {
                "conforms": True,
                "violations": [],
                "results_text": "SHACL validation skipped: pyshacl not installed.",
            }

        import rdflib

        # Resolve the shapes source: an in-memory rdflib.Graph is handed straight
        # to pyshacl; a path is existence-checked then passed as a file string.
        if isinstance(shapes_path, rdflib.Graph):
            shacl_graph: Any = shapes_path
        else:
            shapes_file = Path(shapes_path)
            if not shapes_file.exists():
                return {
                    "conforms": True,
                    "violations": [],
                    "results_text": f"Shapes file not found: {shapes_path}",
                }
            shacl_graph = str(shapes_file)

        try:
            if isinstance(data_graph, rdflib.Graph):
                mapped_graph = rdflib.Graph()
                mapped_graph.bind("kg", rdflib.Namespace("http://knuckles.team/kg#"))

                casing_map = {
                    # CONCEPT:AU-KG.ontology.ontology-action-system — Ontology Action System
                    "Ontologyaction": "OntologyAction",
                    "Actioninvocation": "ActionInvocation",
                    "Architecturedecisionrecord": "ArchitectureDecisionRecord",
                    "Optimizationpattern": "OptimizationPattern",
                    "Softwarefeature": "SoftwareFeature",
                    "Testcase": "TestCase",
                    "Triggersonostreamevent": "triggersOnEvent",
                    "Scopespath": "scopesPath",
                    "Motivatedby": "motivatedBy",
                    "Teststatus": "testStatus",
                }

                def map_term(term):
                    if isinstance(term, rdflib.URIRef):
                        val = str(term)
                        if val.startswith("http://agent-utilities.dev/ontology#"):
                            local = val.split("#", 1)[1]
                            mapped_local = casing_map.get(local, local)
                            return rdflib.URIRef(
                                f"http://knuckles.team/kg#{mapped_local}"
                            )
                    return term

                for s, p, o in data_graph:
                    mapped_graph.add((map_term(s), map_term(p), map_term(o)))
                data_graph = mapped_graph
        except Exception as e:
            logger.warning("Failed to align RDF graph namespaces: %s", e)

        try:
            conforms, results_graph, results_text = pyshacl.validate(
                data_graph=data_graph,
                shacl_graph=shacl_graph,
                ont_graph=ont_graph,
                inference="rdfs",
                abort_on_first=False,
                serialize_report_graph="turtle",
            )
        except Exception as e:
            logger.error("SHACL validation failed: %s", e)
            return {
                "conforms": False,
                "violations": [{"error": str(e)}],
                "results_text": f"SHACL validation error: {e}",
            }

        violations = self._parse_violations(results_text)

        return {
            "conforms": conforms,
            "violations": violations,
            "results_text": (
                results_text
                if isinstance(results_text, str)
                else results_text.decode("utf-8", errors="replace")
            ),
        }

    def validate_layered(
        self,
        data_graph: Any,
        shapes_paths: list[str | Path | Any],
    ) -> dict[str, Any]:
        """Validate against multiple layered SHACL shapes.

        Runs validation against each shapes layer in order. Global shapes
        are typically applied first, then domain-specific overrides.

        Args:
            data_graph: An rdflib.Graph to validate.
            shapes_paths: Ordered list of SHACL shapes layers — each a file path
                or an in-memory ``rdflib.Graph`` (see :meth:`validate`).

        Returns:
            Combined validation report.
        """
        all_violations: list[dict[str, Any]] = []
        all_texts: list[str] = []
        overall_conforms = True

        for shapes_path in shapes_paths:
            result = self.validate(data_graph, shapes_path)
            if not result["conforms"]:
                overall_conforms = False
            all_violations.extend(result["violations"])
            all_texts.append(result["results_text"])

        return {
            "conforms": overall_conforms,
            "violations": all_violations,
            "results_text": "\n---\n".join(all_texts),
            "layers_checked": len(shapes_paths),
        }

    def validate_kg(self, owl_bridge: Any) -> dict[str, Any]:
        """Convenience: validate the KG via OWLBridge materialization.

        Builds the RDF graph from the LPG and validates against
        the default governance shapes.

        Args:
            owl_bridge: An OWLBridge instance with _build_rdf_graph().

        Returns:
            Validation report dict.
        """
        rdf_graph = owl_bridge._build_rdf_graph()

        # Find shapes directory relative to this module
        shapes_dir = Path(__file__).parent.parent / "shapes"
        governance_shapes = shapes_dir / "governance.shapes.ttl"

        if not governance_shapes.exists():
            return {
                "conforms": True,
                "violations": [],
                "results_text": "No governance shapes found.",
            }

        # CONCEPT:AU-KG.ontology.value-type-shacl-load — enforce the ontology value-type SHACL shapes
        # (EmailAddress sh:pattern, Percentage sh:min/maxInclusive, …) alongside
        # the hand-authored governance shapes. The shapes are rendered from the
        # in-memory VALUE_TYPES registry and validated as an rdflib.Graph — no
        # round-trip through disk, so this works on read-only/packaged installs.
        # Failure to render never blocks the gate.
        layers: list[str | Path | Any] = [governance_shapes]
        try:
            import rdflib

            from ..ontology.value_types import value_types_shapes_ttl

            vt_graph = rdflib.Graph()
            vt_graph.parse(data=value_types_shapes_ttl(), format="turtle")
            layers.append(vt_graph)
        except Exception as exc:  # pragma: no cover - enhancement only
            logger.debug("value-type shapes unavailable: %s", exc)

        if len(layers) == 1:
            return self.validate(rdf_graph, governance_shapes)
        return self.validate_layered(rdf_graph, layers)

    @staticmethod
    def _parse_violations(results_text: str | bytes) -> list[dict[str, Any]]:
        """Parse SHACL results text into structured violations."""
        if isinstance(results_text, bytes):
            results_text = results_text.decode("utf-8", errors="replace")

        violations: list[dict[str, Any]] = []
        current: dict[str, Any] = {}

        for line in results_text.split("\n"):
            line = line.strip()
            if line.startswith("Constraint Violation"):
                if current:
                    violations.append(current)
                current = {"type": "violation"}
            elif line.startswith("Severity:"):
                current["severity"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source Shape:"):
                current["source_shape"] = line.split(":", 1)[1].strip()
            elif line.startswith("Focus Node:"):
                current["focus_node"] = line.split(":", 1)[1].strip()
            elif line.startswith("Value Node:"):
                current["value_node"] = line.split(":", 1)[1].strip()
            elif line.startswith("Message:"):
                current["message"] = line.split(":", 1)[1].strip()
            elif line.startswith("Result Path:"):
                current["result_path"] = line.split(":", 1)[1].strip()

        if current:
            violations.append(current)

        return violations
