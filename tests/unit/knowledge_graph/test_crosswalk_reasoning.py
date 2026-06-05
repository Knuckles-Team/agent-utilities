#!/usr/bin/python
"""CONCEPT:KG-2.9 — vendor-neutral ArchiMate crosswalk.

Keystone test for the cross-vendor ontology crosswalk: ServiceNow ``:Incident``
and ERPNext ``:ErpNextIssue`` must both resolve to the canonical
``:ApplicationEvent`` so one query returns events regardless of source vendor,
and Camunda ``:BusinessTask`` must resolve to ``:BusinessProcess``.

These assertions check the *asserted* class axioms loaded from the sibling
``ontology*.ttl`` files (owlready2 ``ancestors()``/``equivalent_to``), which is
the correctness of the crosswalk itself. Full HermiT individual-level inference
is exercised separately by ``test_owl_bridge.py``.
"""

from pathlib import Path

import pytest

pytest.importorskip("owlready2")
pytest.importorskip("rdflib")

from agent_utilities.knowledge_graph.backends.owl.owlready2_backend import (
    Owlready2Backend,
)


@pytest.fixture
def ontology_path() -> str:
    # Loading the base ontology globs in every sibling ontology*.ttl
    # (servicenow / erpnext / quant / leanix / archimate) into one world.
    return str(
        Path(__file__).parent.parent.parent.parent
        / "agent_utilities"
        / "knowledge_graph"
        / "ontology.ttl"
    )


@pytest.fixture
def backend(ontology_path):
    be = Owlready2Backend(ontology_path=ontology_path)
    yield be
    be.close()


def _cls(backend, local_name: str):
    """Resolve an owlready2 class by its IRI local name from the world."""
    import owlready2

    for res in backend._world.search(iri=f"*{local_name}"):
        if isinstance(res, owlready2.ThingClass) and res.name == local_name:
            return res
    return None


def test_archimate_crosswalk_loaded(backend):
    """The canonical anchors and the vendor classes all load into one world."""
    for name in (
        "ApplicationEvent",
        "BusinessProcess",
        "BusinessTask",
        "BusinessActor",
        "Incident",
        "ErpNextIssue",
    ):
        assert _cls(backend, name) is not None, f"missing class :{name}"


def test_itsm_incident_crosswalk_is_vendor_neutral(backend):
    """ServiceNow :Incident and ERPNext :ErpNextIssue both ARE :ApplicationEvent."""
    app_event = _cls(backend, "ApplicationEvent")
    incident = _cls(backend, "Incident")
    erp_issue = _cls(backend, "ErpNextIssue")

    assert app_event in incident.ancestors(), (
        "ServiceNow :Incident must be a subclass of canonical :ApplicationEvent"
    )
    assert app_event in erp_issue.ancestors(), (
        "ERPNext :ErpNextIssue must be a subclass of canonical :ApplicationEvent"
    )


def test_incident_and_issue_are_interchangeable(backend):
    """owl:equivalentClass makes the two vendor classes interchangeable."""
    incident = _cls(backend, "Incident")
    erp_issue = _cls(backend, "ErpNextIssue")
    # owlready2 records equivalentClass symmetrically on at least one side.
    equivalent = set(incident.equivalent_to) | set(erp_issue.equivalent_to)
    assert incident in equivalent or erp_issue in equivalent, (
        "ServiceNow :Incident and ERPNext :ErpNextIssue must be owl:equivalentClass"
    )


def test_business_task_is_a_business_process(backend):
    """Camunda :BusinessTask resolves to canonical :BusinessProcess."""
    process = _cls(backend, "BusinessProcess")
    task = _cls(backend, "BusinessTask")
    assert process in task.ancestors(), (
        ":BusinessTask must be a subclass of :BusinessProcess"
    )


def test_change_is_a_business_process(backend):
    """ServiceNow :Change resolves to canonical :BusinessProcess."""
    process = _cls(backend, "BusinessProcess")
    change = _cls(backend, "Change")
    assert process in change.ancestors()
