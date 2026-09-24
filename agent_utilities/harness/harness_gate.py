"""Harness-evolution SHACL gate (CONCEPT:AU-AHE.evaluation.parity-surpass-scoreboard).

The formal seesaw HarnessX (arXiv:2606.14249) lacks. The paper's per-edit pass@2
gate cannot see *sub-threshold coupling*: its τ³-Bench Telecom run shipped 5
same-dimension edits (R2–R6) whose accumulated coupling caused a tipping-point
−14% regression undetected. We model the harness-evolution facts as RDF and
validate them in Epistemic Graph against its committed concentration /
no-regression / pathology SHACL shapes — so the gate **detects and blocks
concentration before** the tipping point, reasoned over the harness ontology
rather than read off per-task scores. Agent Utilities owns no shapes and no RDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.core.typed_triples import (
    TypedTriple,
    iri,
    kg,
    literal,
    triple,
    typed,
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
# field here. Stamped into the data so the hook-contract shape is
# self-contained (CONCEPT:AU-KG.ontology.harness-gate).
_READ_ONLY_HOOKS = {"step_end", "task_end"}


def build_evolution_triples(
    edits: list[dict[str, Any]],
    variants: list[dict[str, Any]] | None = None,
    pathologies: list[dict[str, Any]] | None = None,
    processors: list[dict[str, Any]] | None = None,
) -> list[TypedTriple]:
    """Harness-evolution facts as typed triples (CONCEPT:AU-AHE.evaluation.parity-surpass-scoreboard).

    ``edits``: ``{id, dimension, round, status?, regresses?:[task_ids],
        at_hook?, modifies_field?, operation?}``.
    ``variants``: ``{id, status, applies:[edit_ids]}``.
    ``pathologies``: ``{id, kind, exhibited_by?:node_id}``.
    ``processors``: ``{id, hook, singleton_group, status?}`` (CONCEPT:AU-KG.ontology.harness-gate).
    """
    triples: list[TypedTriple] = []
    for edit in edits:
        triples.extend(_edit_triples(edit))
    for proc in processors or []:
        triples.extend(_processor_triples(proc))
    for variant in variants or []:
        triples.extend(_variant_triples(variant))
    for pathology in pathologies or []:
        triples.extend(_pathology_triples(pathology))
    return triples


def _edit_triples(edit: dict[str, Any]) -> list[TypedTriple]:
    eid, dim = kg(edit["id"]), kg(edit["dimension"])
    triples = [
        typed(eid, kg("HarnessEdit")),
        typed(dim, kg("HarnessDimension")),
        triple(eid, kg("targetsDimension"), iri(dim)),
        triple(eid, kg("editStatus"), literal(edit.get("status", "shipped"))),
        triple(eid, kg("editRound"), literal(int(edit.get("round", 0)))),
    ]
    triples.extend(
        triple(eid, kg("causesRegression"), iri(kg(task)))
        for task in edit.get("regresses", []) or []
    )
    # Substitution-algebra facts (CONCEPT:AU-KG.ontology.harness-gate).
    if edit.get("operation"):
        triples.append(triple(eid, kg("editOperation"), literal(edit["operation"])))
    if edit.get("at_hook"):
        triples.extend(_hook_triples(eid, edit["at_hook"]))
    if edit.get("modifies_field"):
        triples.append(
            triple(eid, kg("modifiesField"), literal(edit["modifies_field"]))
        )
    return triples


def _hook_triples(eid: str, hook_name: str) -> list[TypedTriple]:
    hook = kg(hook_name)
    return [
        triple(eid, kg("atHook"), iri(hook)),
        typed(hook, kg("HarnessHook")),
        triple(hook, kg("hookReadOnly"), literal(hook_name in _READ_ONLY_HOOKS)),
    ]


def _processor_triples(proc: dict[str, Any]) -> list[TypedTriple]:
    pid = kg(proc["id"])
    triples = [
        typed(pid, kg("Processor")),
        triple(pid, kg("variantStatus"), literal(proc.get("status", "accepted"))),
    ]
    if proc.get("hook"):
        triples.append(triple(pid, kg("attachedToHook"), iri(kg(proc["hook"]))))
    if proc.get("singleton_group"):
        triples.append(
            triple(pid, kg("singletonGroup"), literal(proc["singleton_group"]))
        )
    return triples


def _variant_triples(variant: dict[str, Any]) -> list[TypedTriple]:
    vid = kg(variant["id"])
    triples = [
        typed(vid, kg("HarnessVariant")),
        triple(vid, kg("variantStatus"), literal(variant.get("status", "pending"))),
    ]
    triples.extend(
        triple(vid, kg("appliesEdit"), iri(kg(eid)))
        for eid in variant.get("applies", []) or []
    )
    return triples


def _pathology_triples(pathology: dict[str, Any]) -> list[TypedTriple]:
    pid = kg(pathology["id"])
    triples = [
        typed(pid, kg("HarnessPathology")),
        triple(pid, kg("pathologyKind"), literal(pathology["kind"])),
    ]
    if pathology.get("exhibited_by"):
        triples.append(
            triple(kg(pathology["exhibited_by"]), kg("exhibitsPathology"), iri(pid))
        )
    return triples


class HarnessGate:
    """Validate harness-evolution facts against EG's committed harness shapes
    (seesaw + concentration + pathology, the EG core ``harness-shapes`` source).
    The deterministic acceptance gate of the AEGIS Critic."""

    def __init__(self, *, engine: Any | None = None) -> None:
        self._engine = engine or GraphComputeEngine.get_or_create()

    def check(self, triples: list[TypedTriple]) -> GateVerdict:
        report = self._engine.shacl_validate_committed(data_triples=triples)
        return GateVerdict(
            passed=bool(report.conforms),
            violations=[result.model_dump(mode="json") for result in report.results],
        )

    def check_facts(
        self,
        edits: list[dict[str, Any]],
        variants: list[dict[str, Any]] | None = None,
        pathologies: list[dict[str, Any]] | None = None,
        processors: list[dict[str, Any]] | None = None,
    ) -> GateVerdict:
        """Convenience: build the triples from dicts and check them."""
        return self.check(
            build_evolution_triples(edits, variants, pathologies, processors)
        )
