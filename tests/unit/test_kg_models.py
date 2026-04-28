#!/usr/bin/python
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

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agent_utilities.models.knowledge_graph import (
    BeliefNode,
    DecisionNode,
    HypothesisNode,
    IncidentNode,
    OrganizationNode,
    PhaseNode,
    PlaceNode,
    PrincipleNode,
    RegistryEdge,
    RegistryEdgeType,
    RegistryNodeType,
    RoleNode,
    SystemNode,
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
)

ISO_TS = "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Happy-path construction per class
# ---------------------------------------------------------------------------


def test_organization_node_happy_path() -> None:
    n = OrganizationNode(
        id="org:acme",
        name="Acme Corp",
        org_id="acme-corp",
        legal_name="Acme, Inc.",
        domain="acme.example.com",
        org_type="company",
        website="https://acme.example.com",
    )
    assert n.type is RegistryNodeType.ORGANIZATION
    assert n.org_id == "acme-corp"
    assert n.org_type == "company"
    assert n.parent_org_id is None


def test_role_node_happy_path() -> None:
    n = RoleNode(
        id="role:sre-oncall",
        name="SRE On-Call",
        role_id="sre-oncall",
        title="Site Reliability Engineer On-Call",
        responsibilities=["pager", "runbook"],
        organization_id="org:acme",
        seniority="senior",
    )
    assert n.type is RegistryNodeType.ROLE
    assert n.role_id == "sre-oncall"
    assert n.responsibilities == ["pager", "runbook"]
    assert n.seniority == "senior"


def test_place_node_happy_path_physical() -> None:
    n = PlaceNode(
        id="place:paris-hq",
        name="Paris HQ",
        place_id="paris-hq",
        kind="physical",
        address="1 Rue X, 75001 Paris",
        geo_lat=48.8566,
        geo_lon=2.3522,
    )
    assert n.type is RegistryNodeType.PLACE
    assert n.kind == "physical"
    assert n.geo_lat == pytest.approx(48.8566)


def test_place_node_virtual_kind() -> None:
    n = PlaceNode(
        id="place:platform-eng",
        name="#platform-eng",
        place_id="platform-eng",
        kind="virtual",
        address="teams://.../channel/xyz",
    )
    assert n.kind == "virtual"
    assert n.geo_lat is None


def test_place_node_contextual_kind() -> None:
    n = PlaceNode(
        id="place:onboarding",
        name="during onboarding",
        place_id="onboarding",
        kind="contextual",
    )
    assert n.kind == "contextual"
    assert n.address is None


def test_phase_node_happy_path() -> None:
    n = PhaseNode(
        id="phase:q2-2026",
        name="Q2 2026",
        phase_id="q2-2026",
        started_at="2026-04-01T00:00:00Z",
        ended_at="2026-06-30T23:59:59Z",
        phase_kind="calendar",
    )
    assert n.type is RegistryNodeType.PHASE
    assert n.phase_kind == "calendar"


def test_phase_node_ongoing_defaults() -> None:
    n = PhaseNode(
        id="phase:kafka-migration",
        name="Kafka Migration",
        phase_id="kafka-migration",
        started_at="2026-04-01T00:00:00Z",
    )
    assert n.ended_at is None
    assert n.phase_kind == "custom"  # default
    assert n.parent_phase_id is None


def test_decision_node_happy_path() -> None:
    n = DecisionNode(
        id="dec:7",
        name="Adopt Kafka",
        decision_id="dec-7",
        statement="Migrate from Kinesis to Kafka for event bus.",
        motivation=["goal:throughput"],
        alternatives_considered=["Keep Kinesis", "Use RabbitMQ"],
        chosen_alternative="Kafka",
        confidence=0.8,
        decided_by=["person:alice"],
        decided_at=ISO_TS,
        reversible=False,
    )
    assert n.type is RegistryNodeType.DECISION
    assert n.confidence == 0.8
    assert n.reversible is False


def test_decision_node_defaults() -> None:
    n = DecisionNode(
        id="dec:8",
        name="Default test",
        decision_id="dec-8",
        statement="x",
        decided_at=ISO_TS,
    )
    assert n.confidence == 0.5  # default per §2.2.5
    assert n.reversible is True
    assert n.motivation == []
    assert n.alternatives_considered == []


def test_incident_node_happy_path() -> None:
    n = IncidentNode(
        id="inc:42",
        name="Prod DB outage",
        incident_id="inc-42",
        severity="critical",
        detected_at="2026-04-02T10:00:00Z",
        resolved_at="2026-04-02T11:30:00Z",
        status="postmortem",
        postmortem_article_id="art:pm-42",
        affected_system_ids=["sys:auth", "sys:token"],
        root_cause_summary="Expired cert on token-vault",
    )
    assert n.type is RegistryNodeType.INCIDENT
    assert n.severity == "critical"
    assert n.status == "postmortem"


def test_system_node_happy_path() -> None:
    n = SystemNode(
        id="sys:auth",
        name="auth-service",
        system_id="auth-service",
        tech_stack=["python", "fastapi", "postgres"],
        owner_role_ids=["role:platform-team"],
        owner_org_id="org:acme",
        depends_on_system_ids=["sys:token"],
        repo_urls=["https://github.com/acme/auth-service"],
        criticality="tier1",
    )
    assert n.type is RegistryNodeType.SYSTEM
    assert n.criticality == "tier1"
    assert "python" in n.tech_stack


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


def test_hypothesis_node_happy_path() -> None:
    n = HypothesisNode(
        id="hyp:3",
        name="Friday deploy → incidents",
        prediction="If we deploy on Friday, we'll regret it by Monday.",
        preconditions_node_ids=["b:1"],
        observation_outcome_ids=["inc:42"],
        falsifiable=True,
        verdict="open",
        confidence_prior=0.6,
    )
    assert n.type is RegistryNodeType.HYPOTHESIS
    assert n.falsifiable is True
    assert n.verdict == "open"
    assert n.confidence_posterior is None  # open ↔ posterior None


def test_hypothesis_node_resolved_posterior() -> None:
    n = HypothesisNode(
        id="hyp:4",
        name="Friday deploy hyp",
        prediction="x",
        verdict="confirmed",
        confidence_prior=0.5,
        confidence_posterior=0.92,
    )
    assert n.verdict == "confirmed"
    assert n.confidence_posterior == pytest.approx(0.92)


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


# ---------------------------------------------------------------------------
# Literal / enum rejection
# ---------------------------------------------------------------------------


def test_organization_invalid_org_type_rejected() -> None:
    with pytest.raises(ValidationError):
        OrganizationNode(
            id="org:x", name="x", org_id="x", org_type="invalid-value"  # type: ignore[arg-type]
        )


def test_place_invalid_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        PlaceNode(id="p:x", name="x", place_id="x", kind="nope")  # type: ignore[arg-type]


def test_incident_invalid_severity_rejected() -> None:
    with pytest.raises(ValidationError):
        IncidentNode(
            id="i:x",
            name="x",
            incident_id="x",
            severity="catastrophic",  # type: ignore[arg-type]
            detected_at=ISO_TS,
        )


def test_system_invalid_criticality_rejected() -> None:
    with pytest.raises(ValidationError):
        SystemNode(
            id="s:x",
            name="x",
            system_id="x",
            criticality="tier0",  # type: ignore[arg-type]
        )


def test_hypothesis_invalid_verdict_rejected() -> None:
    with pytest.raises(ValidationError):
        HypothesisNode(
            id="h:x",
            name="x",
            prediction="x",
            verdict="maybe",  # type: ignore[arg-type]
        )


def test_phase_invalid_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        PhaseNode(
            id="ph:x",
            name="x",
            phase_id="x",
            started_at=ISO_TS,
            phase_kind="quarter",  # type: ignore[arg-type]
        )


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


def test_decision_confidence_upper_bound_rejected() -> None:
    with pytest.raises(ValidationError):
        DecisionNode(
            id="d:x",
            name="x",
            decision_id="x",
            statement="x",
            confidence=1.1,
            decided_at=ISO_TS,
        )


def test_hypothesis_prior_and_posterior_bounds() -> None:
    with pytest.raises(ValidationError):
        HypothesisNode(
            id="h:x",
            name="x",
            prediction="x",
            confidence_prior=1.5,
        )
    with pytest.raises(ValidationError):
        HypothesisNode(
            id="h:x",
            name="x",
            prediction="x",
            confidence_posterior=-0.01,
        )


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


@pytest.fixture(scope="module")
def sample_nodes() -> dict[type, object]:
    """One minimally-constructed instance of each of the 10 new node types."""
    return {
        OrganizationNode: OrganizationNode(
            id="org:a", name="A", org_id="a"
        ),
        RoleNode: RoleNode(id="role:a", name="A", role_id="a", title="A"),
        PlaceNode: PlaceNode(
            id="place:a", name="A", place_id="a", kind="physical"
        ),
        PhaseNode: PhaseNode(
            id="phase:a", name="A", phase_id="a", started_at=ISO_TS
        ),
        DecisionNode: DecisionNode(
            id="dec:a",
            name="A",
            decision_id="a",
            statement="x",
            decided_at=ISO_TS,
        ),
        IncidentNode: IncidentNode(
            id="inc:a",
            name="A",
            incident_id="a",
            severity="low",
            detected_at=ISO_TS,
        ),
        SystemNode: SystemNode(id="sys:a", name="A", system_id="a"),
        BeliefNode: BeliefNode(
            id="b:a",
            name="A",
            statement="x",
            confidence=0.5,
            last_reviewed=ISO_TS,
        ),
        HypothesisNode: HypothesisNode(
            id="hyp:a", name="A", prediction="x"
        ),
        PrincipleNode: PrincipleNode(
            id="prin:a", name="A", principle_id="a", statement="x"
        ),
    }


@pytest.mark.parametrize(
    "cls",
    [
        OrganizationNode,
        RoleNode,
        PlaceNode,
        PhaseNode,
        DecisionNode,
        IncidentNode,
        SystemNode,
        BeliefNode,
        HypothesisNode,
        PrincipleNode,
    ],
)
def test_node_json_round_trip(
    cls: type, sample_nodes: dict[type, object]
) -> None:
    """model_dump_json → model_validate_json is an identity for V2 nodes."""
    original = sample_nodes[cls]
    raw = original.model_dump_json()  # type: ignore[attr-defined]
    parsed = cls.model_validate_json(raw)  # type: ignore[attr-defined]
    # Type stays correct
    assert parsed.type is original.type  # type: ignore[attr-defined]
    # JSON bytes are stable
    assert parsed.model_dump_json() == raw


@pytest.mark.parametrize(
    "cls",
    [
        OrganizationNode,
        RoleNode,
        PlaceNode,
        PhaseNode,
        DecisionNode,
        IncidentNode,
        SystemNode,
        BeliefNode,
        HypothesisNode,
        PrincipleNode,
    ],
)
def test_node_dict_round_trip(
    cls: type, sample_nodes: dict[type, object]
) -> None:
    """model_dump (python dict) round-trips through model_validate as well."""
    original = sample_nodes[cls]
    dump = original.model_dump()  # type: ignore[attr-defined]
    parsed = cls.model_validate(dump)  # type: ignore[attr-defined]
    assert parsed.model_dump() == dump


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
    ):
        assert expected in rel_types, (
            f"RelDefinition missing for edge type {expected!r}"
        )


def test_new_table_definitions_include_registry_node_columns() -> None:
    """Each new V2 table includes the base RegistryNode columns.

    This guards against forgetting ``id`` / ``type`` / ``importance_score``
    etc. on future additions.
    """
    base_cols = {
        "id",
        "type",
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
    }
    for tbl in SCHEMA.nodes:
        if tbl.name in new_node_names:
            missing = base_cols - set(tbl.columns.keys())
            assert not missing, (
                f"{tbl.name} missing base cols: {sorted(missing)}"
            )


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
    import networkx as nx

    from agent_utilities.knowledge_graph.engine import (
        IntelligenceGraphEngine,
    )

    eng = IntelligenceGraphEngine(nx.MultiDiGraph())
    result = eng.retrieve_place_view("meeting", top_k=5)
    assert result == []


def test_retrieve_epistemic_view_stub_has_expected_shape() -> None:
    import networkx as nx

    from agent_utilities.knowledge_graph.engine import (
        IntelligenceGraphEngine,
    )

    eng = IntelligenceGraphEngine(nx.MultiDiGraph())
    result = eng.retrieve_epistemic_view("database X", top_k=5)
    assert set(result.keys()) == {"beliefs", "supporting", "contradicting"}
    assert result["beliefs"] == []


def test_retrieve_orthogonal_context_includes_v2_views_when_requested() -> (
    None
):
    import networkx as nx

    from agent_utilities.knowledge_graph.engine import (
        IntelligenceGraphEngine,
    )

    eng = IntelligenceGraphEngine(nx.MultiDiGraph())
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
    import networkx as nx

    from agent_utilities.knowledge_graph.engine import (
        IntelligenceGraphEngine,
    )

    eng = IntelligenceGraphEngine(nx.MultiDiGraph())
    ctx = eng.retrieve_orthogonal_context("any query")
    assert set(ctx["views"].keys()) == {
        "semantic",
        "temporal",
        "causal",
        "entity",
    }
