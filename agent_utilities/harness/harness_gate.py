"""Harness-evolution SHACL gate (CONCEPT:AU-AHE.evaluation.parity-surpass-scoreboard).

The formal seesaw HarnessX (arXiv:2606.14249) lacks. The paper's per-edit pass@2
gate cannot see *sub-threshold coupling*: its τ³-Bench Telecom run shipped 5
same-dimension edits (R2–R6) whose accumulated coupling caused a tipping-point
−14% regression undetected. We model the harness-evolution facts as RDF and
validate them against the concentration / no-regression / pathology SHACL shapes
— so the gate **detects and blocks concentration before** the tipping point,
reasoned over the harness ontology rather than read off per-task scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import rdflib

from agent_utilities.knowledge_graph.core.shacl_validator import SHACLValidator

_KG = rdflib.Namespace("http://knuckles.team/kg#")
_SHAPES = (
    Path(__file__).resolve().parents[1]
    / "knowledge_graph"
    / "shapes"
    / "harness.shapes.ttl"
)


@dataclass
class GateVerdict:
    """Outcome of the harness gate: whether the candidate harness state ships."""

    passed: bool
    violations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [str(v.get("message", "")).strip() for v in self.violations]


# The two read-only lifecycle hooks (HarnessX Table 1): an edit may not modify a
# field here. Stamped into the data graph so the hook-contract shape is
# self-contained (CONCEPT:AU-KG.ontology.harness-gate).
_READ_ONLY_HOOKS = {"step_end", "task_end"}


def build_evolution_graph(
    edits: list[dict[str, Any]],
    variants: list[dict[str, Any]] | None = None,
    pathologies: list[dict[str, Any]] | None = None,
    processors: list[dict[str, Any]] | None = None,
) -> rdflib.Graph:
    """Build a harness-evolution RDF graph from plain dicts (CONCEPT:AU-AHE.evaluation.parity-surpass-scoreboard).

    ``edits``: ``{id, dimension, round, status?, regresses?:[task_ids],
        at_hook?, modifies_field?, operation?}``.
    ``variants``: ``{id, status, applies:[edit_ids]}``.
    ``pathologies``: ``{id, kind, exhibited_by?:node_id}``.
    ``processors``: ``{id, hook, singleton_group, status?}`` (CONCEPT:AU-KG.ontology.harness-gate).
    """
    g = rdflib.Graph()
    g.bind("kg", _KG)
    for e in edits:
        eid = _KG[e["id"]]
        dim = _KG[e["dimension"]]
        g.add((eid, rdflib.RDF.type, _KG.HarnessEdit))
        g.add((dim, rdflib.RDF.type, _KG.HarnessDimension))
        g.add((eid, _KG.targetsDimension, dim))
        g.add((eid, _KG.editStatus, rdflib.Literal(e.get("status", "shipped"))))
        g.add((eid, _KG.editRound, rdflib.Literal(int(e.get("round", 0)))))
        for t in e.get("regresses", []) or []:
            g.add((eid, _KG.causesRegression, _KG[t]))
        # Substitution-algebra facts (CONCEPT:AU-KG.ontology.harness-gate).
        if e.get("operation"):
            g.add((eid, _KG.editOperation, rdflib.Literal(e["operation"])))
        if e.get("at_hook"):
            hook = _KG[e["at_hook"]]
            g.add((eid, _KG.atHook, hook))
            g.add((hook, rdflib.RDF.type, _KG.HarnessHook))
            g.add(
                (
                    hook,
                    _KG.hookReadOnly,
                    rdflib.Literal(e["at_hook"] in _READ_ONLY_HOOKS),
                )
            )
        if e.get("modifies_field"):
            g.add((eid, _KG.modifiesField, rdflib.Literal(e["modifies_field"])))
    for proc in processors or []:
        pid = _KG[proc["id"]]
        g.add((pid, rdflib.RDF.type, _KG.Processor))
        g.add((pid, _KG.variantStatus, rdflib.Literal(proc.get("status", "accepted"))))
        if proc.get("hook"):
            g.add((pid, _KG.attachedToHook, _KG[proc["hook"]]))
        if proc.get("singleton_group"):
            g.add((pid, _KG.singletonGroup, rdflib.Literal(proc["singleton_group"])))
    for v in variants or []:
        vid = _KG[v["id"]]
        g.add((vid, rdflib.RDF.type, _KG.HarnessVariant))
        g.add((vid, _KG.variantStatus, rdflib.Literal(v.get("status", "pending"))))
        for eid in v.get("applies", []) or []:
            g.add((vid, _KG.appliesEdit, _KG[eid]))
    for p in pathologies or []:
        pid = _KG[p["id"]]
        g.add((pid, rdflib.RDF.type, _KG.HarnessPathology))
        g.add((pid, _KG.pathologyKind, rdflib.Literal(p["kind"])))
        if p.get("exhibited_by"):
            g.add((_KG[p["exhibited_by"]], _KG.exhibitsPathology, pid))
    return g


class HarnessGate:
    """Validate a harness-evolution graph against the SHACL seesaw + concentration
    + pathology shapes. The deterministic acceptance gate of the AEGIS Critic."""

    def __init__(self, shapes_path: str | Path | None = None) -> None:
        self._validator = SHACLValidator()
        self._shapes = str(shapes_path or _SHAPES)

    def check(self, graph: rdflib.Graph) -> GateVerdict:
        report = self._validator.validate(graph, self._shapes)
        return GateVerdict(
            passed=bool(report.get("conforms", True)),
            violations=list(report.get("violations", []) or []),
        )

    def check_facts(
        self,
        edits: list[dict[str, Any]],
        variants: list[dict[str, Any]] | None = None,
        pathologies: list[dict[str, Any]] | None = None,
        processors: list[dict[str, Any]] | None = None,
    ) -> GateVerdict:
        """Convenience: build the graph from dicts and check it."""
        return self.check(
            build_evolution_graph(edits, variants, pathologies, processors)
        )
