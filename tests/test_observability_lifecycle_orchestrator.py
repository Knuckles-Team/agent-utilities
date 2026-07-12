"""Regression tests for the enter-anywhere SDLC lifecycle orchestrator
(``agent_utilities.observability.lifecycle_orchestrator``) — the keystone that
diffs any spine node against the required lifecycle shape and emits REPORT-ONLY
gap-fill ``:LifecycleStep`` proposals (``reports/autonomous-sdlc-loop-design.md``
§3). Plus a pyshacl check that the required-shape TTL flags a missing
``:PipelineRun`` for a merged change (§1.3). Mirrors
``test_observability_incidents.py``'s fake-KG style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agent_utilities.knowledge_graph.memory.native_ingest as native_ingest
import agent_utilities.observability.health_ingest as hi
from agent_utilities.observability import lifecycle_orchestrator as lo


class _Capture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, entities, relationships=None, *, source, domain, **kw):
        self.calls.append(
            {
                "entities": entities,
                "relationships": relationships or [],
                "source": source,
            }
        )
        return {"nodes": len(entities), "edges": len(relationships or [])}


class _FakeEngine:
    """Serves out_edges/in_edges/get_nodes_by_label from a flat edge list
    (``(src, rel_type, tgt)``) + a per-label node table."""

    def __init__(
        self,
        edges: list[tuple[str, str, str]] | None = None,
        by_label: dict[str, list[tuple[str, dict]]] | None = None,
    ) -> None:
        self._edges = edges or []
        self._by_label = by_label or {}

    def out_edges(self, node_id: str, data: bool = False):
        return [(s, t, {"rel_type": r}) for (s, r, t) in self._edges if s == node_id]

    def in_edges(self, node_id: str, data: bool = False):
        return [(s, t, {"rel_type": r}) for (s, r, t) in self._edges if t == node_id]

    def get_nodes_by_label(self, label: str, limit: int = 0):
        return self._by_label.get(label, [])


def _complete_edges() -> list[tuple[str, str, str]]:
    """A fully-linked lifecycle: incident -> ... -> validation -resolves-> incident."""
    return [
        ("inc1", "triggers", "tkt1"),
        ("tkt1", "specifies", "spec1"),
        ("spec1", "implements", "cc1"),
        ("cc1", "proposes", "mr1"),
        ("mr1", "triggersPipeline", "pr1"),
        ("pr1", "builds", "img1"),
        ("dep1", "deploys", "img1"),  # deployment -deploys-> image (in-edge from image)
        ("dep1", "validates", "val1"),
        ("val1", "resolves", "inc1"),
    ]


# --- forward gap-fill ----------------------------------------------------- #
def test_enter_at_incident_with_no_ticket_proposes_the_forward_chain(monkeypatch):
    """The headline case: enter at an :Incident with nothing downstream → the
    orchestrator proposes ticket + spec + ... all the way to resolve."""
    monkeypatch.setattr(hi, "_engine", lambda: _FakeEngine())
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    out = lo.run_lifecycle({"id": "inc1", "type": "Incident"})

    names = [s["transition"] for s in out["steps"]]
    assert out["stage"] == "incident"
    # every forward transition from incident onward is proposed
    assert names == [
        "mint_ticket",
        "generate_spec",
        "develop_spec",
        "open_change_request",
        "await_ci",
        "build_image",
        "deploy",
        "validate_deploy",
        "resolve_incident",
    ]
    # report-only: each step is a proposal bound to a fleet capability, nothing run
    assert all(s["status"] == "proposed" for s in out["steps"])
    assert all(s["kind"] == "forward" for s in out["steps"])
    assert out["steps"][0]["boundCapability"].endswith("route_incident")
    # the proposals were written as :LifecycleStep nodes + hasLifecycleStep edges
    assert len(cap.calls) == 1
    assert {e["type"] for e in cap.calls[0]["entities"]} == {"LifecycleStep"}
    assert {r["type"] for r in cap.calls[0]["relationships"]} == {"hasLifecycleStep"}


def test_enter_at_incident_with_existing_ticket_skips_that_hop(monkeypatch):
    """When the incident already has a ticket, mint_ticket is NOT re-proposed; the
    gap starts at the first genuinely-missing downstream stage."""
    engine = _FakeEngine([("inc1", "triggers", "tkt1")])
    monkeypatch.setattr(hi, "_engine", lambda: engine)
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())

    out = lo.run_lifecycle({"id": "inc1", "type": "Incident"}, engine=engine)
    names = [s["transition"] for s in out["steps"]]
    assert "mint_ticket" not in names
    assert names[0] == "generate_spec"


def test_merged_mr_missing_pipeline_run_is_flagged(monkeypatch):
    """Enter at a :MergeRequest whose upstream is present but which has no
    :PipelineRun → the first forward gap is the CI hop (await_ci / triggersPipeline)
    — the required-shape violation the SHACL MergeRequestMergeableShape encodes."""
    engine = _FakeEngine([("cc1", "proposes", "mr1")])  # upstream present, no CI
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())

    out = lo.run_lifecycle({"id": "mr1", "type": "MergeRequest"}, engine=engine)
    forward = [s for s in out["steps"] if s["kind"] == "forward"]
    assert forward[0]["transition"] == "await_ci"
    assert forward[0]["edge"] == "triggersPipeline"
    assert forward[0]["targetType"] == "PipelineRun"


# --- backward gap-fill ---------------------------------------------------- #
def test_enter_at_mr_with_no_upstream_backfills_spec_and_ticket(monkeypatch):
    """An out-of-band :MergeRequest with no linked upstream → backfill proposals
    walk back through code_change / spec / ticket / incident (design §3.3)."""
    engine = _FakeEngine()  # MR floating with nothing attached
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())

    out = lo.run_lifecycle({"id": "mr1", "type": "MergeRequest"}, engine=engine)
    backfills = [s["toStage"] for s in out["steps"] if s["kind"] == "backfill"]
    # upstream chain of a merge_request: code_change <- spec <- ticket <- incident
    assert "spec" in backfills
    assert "ticket" in backfills
    assert "code_change" in backfills


# --- complete lifecycle emits nothing ------------------------------------- #
def test_complete_lifecycle_emits_no_proposals(monkeypatch):
    engine = _FakeEngine(_complete_edges())
    cap = _Capture()
    monkeypatch.setattr(native_ingest, "ingest_entities", cap)

    out = lo.run_lifecycle({"id": "inc1", "type": "Incident"}, engine=engine)
    assert out["proposed"] == 0
    assert out["steps"] == []
    assert cap.calls == []  # nothing written — and nothing executed


# --- idempotency / dedupe ------------------------------------------------- #
def test_gap_fill_is_idempotent_on_signature(monkeypatch):
    """A second sweep over the same entry with the proposals already present does
    not re-propose them."""
    first = _FakeEngine()
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())
    out1 = lo.run_lifecycle({"id": "inc1", "type": "Incident"}, engine=first)
    assert out1["proposed"] == 9

    # seed the engine's LifecycleStep table with the now-written steps.
    seeded = _FakeEngine(
        by_label={
            "LifecycleStep": [
                (s["id"], {"signature": s["signature"]}) for s in out1["steps"]
            ]
        }
    )
    out2 = lo.run_lifecycle({"id": "inc1", "type": "Incident"}, engine=seeded)
    assert out2["proposed"] == 0
    assert out2["deduped"] == 9


# --- guards --------------------------------------------------------------- #
def test_run_lifecycle_no_engine_is_noop(monkeypatch):
    monkeypatch.setattr(hi, "_engine", lambda: None)
    out = lo.run_lifecycle({"id": "inc1", "type": "Incident"})
    assert out["proposed"] == 0
    assert out["steps"] == []


def test_unknown_entry_type_yields_no_proposals():
    engine = _FakeEngine()
    out = lo.run_lifecycle({"id": "x1", "type": "NotAStage"}, engine=engine)
    assert out["proposed"] == 0


def test_sweep_skips_resolved_nodes(monkeypatch):
    engine = _FakeEngine(
        by_label={
            "Incident": [
                ("inc_open", {"status": "open"}),
                ("inc_done", {"status": "resolved"}),
            ]
        }
    )
    monkeypatch.setattr(native_ingest, "ingest_entities", _Capture())
    out = lo.sweep_open_spine(engine=engine)
    assert out["scanned"] == 1  # only the open incident


# --- the SHACL required-shape itself (design §1.3) ------------------------ #
def test_shacl_required_shape_flags_missing_pipeline_run():
    """The declarative source of truth: pyshacl over
    ``shapes/sdlc_lifecycle.shapes.ttl`` flags a :CodeChangeProposal with no
    :triggersPipeline, and conforms once a :PipelineRun is linked."""
    rdflib = pytest.importorskip("rdflib")
    pyshacl = pytest.importorskip("pyshacl")

    import agent_utilities.knowledge_graph as kg

    shapes_path = Path(kg.__file__).parent / "shapes" / "sdlc_lifecycle.shapes.ttl"
    sg = rdflib.Graph().parse(str(shapes_path), format="turtle")

    missing = """@prefix : <http://knuckles.team/kg#> .
:mr1 a :CodeChangeProposal .
"""
    dg = rdflib.Graph().parse(data=missing, format="turtle")
    conforms, _results, text = pyshacl.validate(dg, shacl_graph=sg, inference="none")
    assert conforms is False
    assert "PipelineRun" in text

    present = """@prefix : <http://knuckles.team/kg#> .
:mr1 a :CodeChangeProposal ; :triggersPipeline :pr1 .
:pr1 a :PipelineRun .
"""
    dg2 = rdflib.Graph().parse(data=present, format="turtle")
    conforms2, _r2, _t2 = pyshacl.validate(dg2, shacl_graph=sg, inference="none")
    # the MergeRequestMergeableShape no longer fires (TicketRoutedShape etc. target
    # other classes and have no instances here).
    assert conforms2 is True
