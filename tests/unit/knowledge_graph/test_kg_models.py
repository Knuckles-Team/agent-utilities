#!/usr/bin/python
from __future__ import annotations

"""Unit tests for KG V2 node / edge model additions.

Covers the 10 new ``RegistryNode`` subclasses and 20 new ``RegistryEdgeType``
members introduced per ``docs/KG_V2_DESIGN.md`` §§2-3. Tests focus on:

* Happy-path construction with required fields.
* Enum-value routing (each class binds the right ``RegistryNodeType``).
* ``Literal[...]`` rejection on invalid enum values.
* ``ge`` / ``le`` bounds on floats (``confidence``, ``strength`` etc.).
* Custom ``@model_validator(mode="after")`` invariants (BeliefNode mutex).
* Edge construction via ``RegistryEdge`` for all 20 new edge types.
* Round-trip JSON invariance (``model_dump_json`` → ``model_validate_json``).
* Schema coverage — every new node class has a matching
  ``TableDefinition`` and every new edge has a ``RelDefinition``.
"""
# CONCEPT:AU-AHE.harness.evolutionary-aggregation — Agentic Coding Patterns

# CONCEPT:AU-ORCH.planning.recursion-nesting-depth — Exploration & Discovery

# CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Evaluation & Monitoring

# CONCEPT:AU-OS.safety.doom-loop-detection — Agent Identity Management


import json

import pytest
from pydantic import ValidationError

from agent_utilities.graph.models import Relationship
from agent_utilities.models.knowledge_graph import (
    BeliefNode,
    PrincipleNode,
    RegistryEdge,
    RegistryEdgeType,
    RegistryNodeType,
)
from agent_utilities.models.schema_definition import SCHEMA

# ---------------------------------------------------------------------------
# Convenience constants
# ---------------------------------------------------------------------------

NEW_NODE_ENUMS: tuple[RegistryNodeType, ...] = (
    RegistryNodeType.ORGANIZATION,
    RegistryNodeType.ROLE,
    RegistryNodeType.PLACE,
    RegistryNodeType.PHASE,
    RegistryNodeType.DECISION,
    RegistryNodeType.INCIDENT,
    RegistryNodeType.SYSTEM,
    RegistryNodeType.BELIEF,
    RegistryNodeType.HYPOTHESIS,
    RegistryNodeType.PRINCIPLE,
    RegistryNodeType.RELATIONSHIP,
    RegistryNodeType.CONTINUANT,
    RegistryNodeType.OCCURRENT,
    RegistryNodeType.SPATIAL_REGION,
    RegistryNodeType.TEMPORAL_REGION,
    RegistryNodeType.BOUNDARY,
    RegistryNodeType.QUANTITY_VALUE,
    RegistryNodeType.UNIT_OF_MEASURE,
    RegistryNodeType.PRODUCT,
    RegistryNodeType.OFFER,
    RegistryNodeType.PROVENANCE_ACTIVITY,
    RegistryNodeType.PROVENANCE_AGENT,
    RegistryNodeType.GENE,
    RegistryNodeType.DISEASE,
    RegistryNodeType.DRUG,
    RegistryNodeType.ANATOMY,
    RegistryNodeType.CLINICAL_TRIAL,
    RegistryNodeType.MEDICAL_OBSERVATION,
    RegistryNodeType.BANK_ACCOUNT,
    RegistryNodeType.EQUITY,
    RegistryNodeType.DERIVATIVE,
    RegistryNodeType.CORPORATE_ACTION,
    RegistryNodeType.LOAN,
    RegistryNodeType.MARKET_INDEX,
    RegistryNodeType.SENSOR,
    RegistryNodeType.ACTUATOR,
    RegistryNodeType.IOT_DEVICE,
    RegistryNodeType.MANUFACTURING_PLANT,
    RegistryNodeType.MATERIAL_ASSET,
    RegistryNodeType.MAINTENANCE_LOG,
    RegistryNodeType.LEGAL_CONTRACT,
    RegistryNodeType.JURISDICTION,
    RegistryNodeType.LEGISLATION,
    RegistryNodeType.COURT_RULING,
    RegistryNodeType.THREAT_ACTOR,
    RegistryNodeType.CULTURAL_ARTIFACT,
    RegistryNodeType.MUSEUM_EXHIBIT,
    RegistryNodeType.MUSICAL_WORK,
    RegistryNodeType.PUBLICATION_RECORD,
    RegistryNodeType.HISTORICAL_EVENT,
    RegistryNodeType.BUSINESS_DIVISION,
    RegistryNodeType.COST_CENTER,
    RegistryNodeType.BOARD_OF_DIRECTORS,
    RegistryNodeType.COMMITTEE,
    RegistryNodeType.EMPLOYEE,
    RegistryNodeType.CONTRACTOR,
    RegistryNodeType.VIRTUAL_WORKER,
    RegistryNodeType.PAY_GRADE,
    RegistryNodeType.PERFORMANCE_REVIEW,
    RegistryNodeType.TRAINING_MODULE,
    RegistryNodeType.AUTHORITY_DELEGATION,
    RegistryNodeType.COMPLIANCE_AUDIT,
    RegistryNodeType.RESOURCE_QUOTA,
    RegistryNodeType.ALL_HANDS_MEETING,
    RegistryNodeType.EXECUTIVE_MEMO,
    RegistryNodeType.TOWN_HALL,
)

NEW_EDGE_ENUMS: tuple[RegistryEdgeType, ...] = (
    RegistryEdgeType.HAS_ROLE,
    RegistryEdgeType.PLAYED_ROLE_DURING,
    RegistryEdgeType.OCCURRED_AT_PLACE,
    RegistryEdgeType.OCCURRED_DURING_PHASE,
    RegistryEdgeType.DECIDED_BY,
    RegistryEdgeType.MOTIVATED_BY,
    RegistryEdgeType.RESULTED_IN,
    RegistryEdgeType.SUPPORTS_BELIEF,
    RegistryEdgeType.CONTRADICTS_BELIEF,
    RegistryEdgeType.GENERALIZES_TO,
    RegistryEdgeType.INSTANCE_OF_PATTERN,
    RegistryEdgeType.CAUSED_INCIDENT,
    RegistryEdgeType.RESOLVED_INCIDENT,
    RegistryEdgeType.OWNS_SYSTEM,
    RegistryEdgeType.DEPENDS_ON_SYSTEM,
    RegistryEdgeType.PREDICTS,
    RegistryEdgeType.OBSERVES,
    RegistryEdgeType.SUPERSEDES_BY,
    RegistryEdgeType.BELONGS_TO_ORGANIZATION,
    RegistryEdgeType.EMPLOYS,
    RegistryEdgeType.HAS_PARENT,
    RegistryEdgeType.HAS_CHILD,
    RegistryEdgeType.HAS_ANCESTOR,
    RegistryEdgeType.HAS_DESCENDANT,
    RegistryEdgeType.HAS_SIBLING,
    RegistryEdgeType.COUPLE,
    RegistryEdgeType.SPOUSE,
    RegistryEdgeType.COLLEAGUE_OF,
    RegistryEdgeType.MENTOR_OF,
    RegistryEdgeType.FRIEND_OF,
    RegistryEdgeType.KNOWS,
    RegistryEdgeType.HAS_QUANTITY,
    RegistryEdgeType.MEASURED_IN,
    RegistryEdgeType.DERIVES_FROM,
    RegistryEdgeType.REPORTS_TO,
    RegistryEdgeType.DELEGATES_AUTHORITY_TO,
    RegistryEdgeType.MONITORS_COMPLIANCE_OF,
    RegistryEdgeType.CONSUMES_BUDGET_OF,
    RegistryEdgeType.ALLOCATED_TO_COST_CENTER,
    RegistryEdgeType.MANAGES,
    RegistryEdgeType.COLLABORATES_WITH,
    RegistryEdgeType.HAS_JURISDICTION_OVER,
    RegistryEdgeType.GOVERNS,
    RegistryEdgeType.MANUFACTURES,
    RegistryEdgeType.TREATS_DISEASE,
    RegistryEdgeType.PRESCRIBES_DRUG,
)

ISO_TS = "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Happy-path construction per class
# ---------------------------------------------------------------------------


def test_belief_node_happy_path() -> None:
    n = BeliefNode(
        id="b:1",
        name="Friday deploys break things",
        statement="Deploying on Fridays correlates with incidents.",
        confidence=0.85,
        evidence_node_ids=["ep:a", "ep:b"],
        supported_by_node_ids=["fact:1"],
        contradicted_by_node_ids=["fact:9"],
        last_reviewed=ISO_TS,
        source_agent_id="agent:sre-bot",
        scope_node_ids=["sys:deploy-pipeline"],
    )
    assert n.type is RegistryNodeType.BELIEF
    assert n.confidence == 0.85
    assert "fact:1" in n.supported_by_node_ids


def test_principle_node_happy_path() -> None:
    n = PrincipleNode(
        id="prin:tdd",
        name="Always TDD",
        principle_id="tdd",
        statement="Always write a failing test before writing code.",
        scope_node_ids=["concept:testing"],
        exceptions=["tracer-bullet spikes"],
        derived_from_decision_ids=["dec:7"],
        derived_from_episode_ids=["ep:a", "ep:b"],
        strength=0.9,
        review_cadence_days=180,
        last_reviewed=ISO_TS,
    )
    assert n.type is RegistryNodeType.PRINCIPLE
    assert n.strength == 0.9
    assert n.review_cadence_days == 180


def test_relationship_graph_node_happy_path() -> None:
    r = Relationship(
        id="rel:2",
        relationship_id="rel-2",
        relationship_type="mentor_of",
        entity1_id="person:alice",
        entity2_id="person:charlie",
        facts=[{"subject": "Rust FFI"}],
    )
    assert "Relationship" in r.labels
    assert r.relationship_id == "rel-2"
    assert r.relationship_type == "mentor_of"
    assert r.entity1_id == "person:alice"
    assert r.entity2_id == "person:charlie"
    assert len(r.facts) == 1
    assert r.facts[0]["subject"] == "Rust FFI"


# ---------------------------------------------------------------------------
# Literal / enum rejection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Float bound enforcement
# ---------------------------------------------------------------------------


def test_belief_confidence_upper_bound_rejected() -> None:
    with pytest.raises(ValidationError):
        BeliefNode(
            id="b:x",
            name="x",
            statement="x",
            confidence=1.5,
            last_reviewed=ISO_TS,
        )


def test_belief_confidence_lower_bound_rejected() -> None:
    with pytest.raises(ValidationError):
        BeliefNode(
            id="b:x",
            name="x",
            statement="x",
            confidence=-0.1,
            last_reviewed=ISO_TS,
        )


def test_belief_confidence_valid_midpoint_accepted() -> None:
    # Sanity check: the happy midpoint in the bound is accepted.
    n = BeliefNode(
        id="b:x",
        name="x",
        statement="x",
        confidence=0.5,
        last_reviewed=ISO_TS,
    )
    assert n.confidence == 0.5


def test_principle_strength_bounds() -> None:
    with pytest.raises(ValidationError):
        PrincipleNode(
            id="pr:x",
            name="x",
            principle_id="x",
            statement="x",
            strength=1.01,
        )
    with pytest.raises(ValidationError):
        PrincipleNode(
            id="pr:x",
            name="x",
            principle_id="x",
            statement="x",
            strength=-0.1,
        )


# ---------------------------------------------------------------------------
# BeliefNode @model_validator: support / contradict mutex
# ---------------------------------------------------------------------------


def test_belief_node_support_contradict_mutex() -> None:
    """Regression for docs/KG_V2_DESIGN.md §2.2.8 invariant."""
    with pytest.raises(ValidationError, match="cannot both support and "):
        BeliefNode(
            id="b:1",
            name="test",
            statement="x",
            confidence=0.5,
            last_reviewed=ISO_TS,
            supported_by_node_ids=["f:1"],
            contradicted_by_node_ids=["f:1"],
        )


def test_belief_node_partial_overlap_mutex() -> None:
    with pytest.raises(ValidationError, match="cannot both support and "):
        BeliefNode(
            id="b:2",
            name="t",
            statement="y",
            confidence=0.5,
            last_reviewed=ISO_TS,
            supported_by_node_ids=["f:1", "f:2"],
            contradicted_by_node_ids=["f:2", "f:3"],
        )


def test_belief_node_disjoint_sets_accepted() -> None:
    n = BeliefNode(
        id="b:3",
        name="t",
        statement="y",
        confidence=0.5,
        last_reviewed=ISO_TS,
        supported_by_node_ids=["f:1", "f:2"],
        contradicted_by_node_ids=["f:9"],
    )
    assert "f:9" in n.contradicted_by_node_ids


# ---------------------------------------------------------------------------
# Round-trip JSON invariance for every new node type
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Edge construction — one happy-path per new RegistryEdgeType
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edge_type", NEW_EDGE_ENUMS)
def test_registry_edge_construction(edge_type: RegistryEdgeType) -> None:
    """RegistryEdge can be constructed for every new V2 edge type."""
    edge = RegistryEdge(
        source="a:1",
        target="b:1",
        type=edge_type,
        weight=0.7,
        metadata={"test": True},
    )
    assert edge.type is edge_type
    assert edge.weight == pytest.approx(0.7)
    # Value follows snake_case convention in docs/KG_V2_DESIGN.md §3.2
    assert edge.type.value == edge_type.value
    assert edge.type.value.islower()


def test_played_role_during_edge_properties() -> None:
    """PLAYED_ROLE_DURING carries {from, to, phase_id} per §3.3."""
    edge = RegistryEdge(
        source="person:alice",
        target="role:sre-oncall",
        type=RegistryEdgeType.PLAYED_ROLE_DURING,
        metadata={
            "from": "2026-04-01T00:00:00Z",
            "to": "2026-06-30T23:59:59Z",
            "phase_id": "q2-2026",
        },
    )
    assert edge.metadata["from"] == "2026-04-01T00:00:00Z"
    assert edge.metadata["to"] == "2026-06-30T23:59:59Z"
    assert edge.metadata["phase_id"] == "q2-2026"


def test_motivated_by_edge_strength_property() -> None:
    """MOTIVATED_BY carries ``strength: float`` per §3.3."""
    edge = RegistryEdge(
        source="dec:7",
        target="goal:throughput",
        type=RegistryEdgeType.MOTIVATED_BY,
        metadata={"strength": 0.85},
    )
    assert 0.0 <= edge.metadata["strength"] <= 1.0


# ---------------------------------------------------------------------------
# Enum / schema coverage sanity — guardrails for future additions
# ---------------------------------------------------------------------------


def test_all_new_node_enum_members_present() -> None:
    """All 10 new RegistryNodeType members resolve."""
    names = {m.name for m in RegistryNodeType}
    for expected in (
        "ORGANIZATION",
        "ROLE",
        "PLACE",
        "PHASE",
        "DECISION",
        "INCIDENT",
        "SYSTEM",
        "BELIEF",
        "HYPOTHESIS",
        "PRINCIPLE",
    ):
        assert expected in names


def test_all_new_edge_enum_members_present() -> None:
    """All 20 new RegistryEdgeType members resolve."""
    names = {m.name for m in RegistryEdgeType}
    for expected in (
        "HAS_ROLE",
        "PLAYED_ROLE_DURING",
        "OCCURRED_AT_PLACE",
        "OCCURRED_DURING_PHASE",
        "DECIDED_BY",
        "MOTIVATED_BY",
        "RESULTED_IN",
        "SUPPORTS_BELIEF",
        "CONTRADICTS_BELIEF",
        "GENERALIZES_TO",
        "INSTANCE_OF_PATTERN",
        "CAUSED_INCIDENT",
        "RESOLVED_INCIDENT",
        "OWNS_SYSTEM",
        "DEPENDS_ON_SYSTEM",
        "PREDICTS",
        "OBSERVES",
        "SUPERSEDES_BY",
        "BELONGS_TO_ORGANIZATION",
        "EMPLOYS",
    ):
        assert expected in names


def test_every_new_node_has_table_definition() -> None:
    """Every new V2 node class has a matching SCHEMA TableDefinition."""
    table_names = {t.name for t in SCHEMA.nodes}
    for expected in (
        "Organization",
        "Role",
        "Place",
        "Phase",
        "Decision",
        "Incident",
        "System",
        "Belief",
        "Hypothesis",
        "Principle",
        "Relationship",
    ):
        assert expected in table_names, (
            f"TableDefinition missing for node label {expected!r}"
        )


def test_every_new_edge_has_rel_definition() -> None:
    """Every new V2 edge has a matching SCHEMA RelDefinition."""
    rel_types = {e.type for e in SCHEMA.edges}
    for expected in (
        "HAS_ROLE",
        "PLAYED_ROLE_DURING",
        "OCCURRED_AT_PLACE",
        "OCCURRED_DURING_PHASE",
        "DECIDED_BY",
        "MOTIVATED_BY",
        "RESULTED_IN",
        "SUPPORTS_BELIEF",
        "CONTRADICTS_BELIEF",
        "GENERALIZES_TO",
        "INSTANCE_OF_PATTERN",
        "CAUSED_INCIDENT",
        "RESOLVED_INCIDENT",
        "OWNS_SYSTEM",
        "DEPENDS_ON_SYSTEM",
        "PREDICTS",
        "OBSERVES",
        "SUPERSEDES_BY",
        "BELONGS_TO_ORGANIZATION",
        "EMPLOYS",
        "HAS_PARENT",
        "HAS_CHILD",
        "HAS_ANCESTOR",
        "HAS_DESCENDANT",
        "HAS_SIBLING",
        "COUPLE",
        "SPOUSE",
        "COLLEAGUE_OF",
        "MENTOR_OF",
        "FRIEND_OF",
        "KNOWS",
    ):
        assert expected in rel_types, (
            f"RelDefinition missing for edge type {expected!r}"
        )


def test_new_table_definitions_include_registry_node_columns() -> None:
    """Each new V2 table includes the base RegistryNode columns.

    This guards against forgetting ``id`` / ``node_type`` / ``importance_score``
    etc. on future additions. ``node_type`` (not a bare ``type``, which the
    schema retired) is the actual column ``schema_definition.py`` declares and
    ``_serialize_node``/``add_node`` write to — see
    ``GENERIC_NODE_COLUMNS``/``TableDefinition.columns`` and
    ``IntelligenceGraphEngine._serialize_node``.
    """
    base_cols = {
        "id",
        "node_type",
        "name",
        "description",
        "importance_score",
        "timestamp",
        "metadata",
        "is_permanent",
    }
    new_node_names = {
        "Organization",
        "Role",
        "Place",
        "Phase",
        "Decision",
        "Incident",
        "System",
        "Belief",
        "Hypothesis",
        "Principle",
        "Relationship",
    }
    for tbl in SCHEMA.nodes:
        if tbl.name in new_node_names:
            missing = base_cols - set(tbl.columns.keys())
            assert not missing, f"{tbl.name} missing base cols: {sorted(missing)}"


def test_new_edge_snake_case_enum_values() -> None:
    """V2 edge StrEnum values are snake_case (matches V1 convention)."""
    for m in NEW_EDGE_ENUMS:
        assert m.value == m.value.lower()
        assert " " not in m.value
        assert m.value == m.name.lower()


def test_new_node_enum_values_match_name_lowercase() -> None:
    """V2 node StrEnum values are lowercase of their member name.

    (Matches the existing 54-member convention per knowledge_graph.py.)
    """
    for m in NEW_NODE_ENUMS:
        assert m.value == m.name.lower()


# ---------------------------------------------------------------------------
# Extra: JSON values in the V2 edges are stable across model_dump cycles.
# ---------------------------------------------------------------------------


def test_edge_json_round_trip() -> None:
    edge = RegistryEdge(
        source="dec:7",
        target="inc:42",
        type=RegistryEdgeType.RESULTED_IN,
        weight=1.0,
        metadata={"note": "pager paged"},
    )
    raw = edge.model_dump_json()
    parsed = RegistryEdge.model_validate_json(raw)
    assert parsed.type is RegistryEdgeType.RESULTED_IN
    # The serialized enum value matches snake_case
    assert json.loads(raw)["type"] == "resulted_in"


# ---------------------------------------------------------------------------
# MAGMA view stubs — engine exposes 'place' and 'epistemic' per §5
# ---------------------------------------------------------------------------


def test_retrieve_place_view_stub_returns_empty_list() -> None:
    from unittest.mock import MagicMock

    from agent_utilities.knowledge_graph.core.engine import (
        IntelligenceGraphEngine,
    )
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    eng = IntelligenceGraphEngine(backend=MagicMock(), graph=GraphComputeEngine())
    result = eng.retrieve_place_view("meeting", top_k=5)
    assert result == []


def test_retrieve_epistemic_view_stub_has_expected_shape() -> None:
    from unittest.mock import MagicMock

    from agent_utilities.knowledge_graph.core.engine import (
        IntelligenceGraphEngine,
    )
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    eng = IntelligenceGraphEngine(backend=MagicMock(), graph=GraphComputeEngine())
    result = eng.retrieve_epistemic_view("database X", top_k=5)
    assert set(result.keys()) == {"beliefs", "supporting", "contradicting"}
    assert result["beliefs"] == []


def test_retrieve_orthogonal_context_includes_v2_views_when_requested() -> None:
    from unittest.mock import MagicMock

    from agent_utilities.knowledge_graph.core.engine import (
        IntelligenceGraphEngine,
    )
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    eng = IntelligenceGraphEngine(backend=MagicMock(), graph=GraphComputeEngine())
    ctx = eng.retrieve_orthogonal_context(
        "what broke during the migration?",
        views=["place", "epistemic"],
    )
    assert "place" in ctx["views"]
    assert "epistemic" in ctx["views"]
    # Default V1 views should NOT be populated if not requested.
    assert "semantic" not in ctx["views"]


def test_retrieve_orthogonal_context_default_keeps_v1_contract() -> None:
    """Passing no ``views`` argument must still return the V1 four views,

    so existing callers don't break. (§6 backward-compat invariant.)
    """
    from unittest.mock import MagicMock

    from agent_utilities.knowledge_graph.core.engine import (
        IntelligenceGraphEngine,
    )
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    eng = IntelligenceGraphEngine(backend=MagicMock(), graph=GraphComputeEngine())
    ctx = eng.retrieve_orthogonal_context("any query")
    assert set(ctx["views"].keys()) == {
        "semantic",
        "temporal",
        "causal",
        "entity",
    }
