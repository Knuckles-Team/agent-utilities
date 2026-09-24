"""Typed cross-layer clients route every call through a generated EG surface."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from epistemic_graph.generated.decision import (
    AbstainReasonInsufficientConfidence,
    AssemblyResult,
    CandidateSourceRecordAgentLibrary,
    DecisionOutcomeAbstained,
    DecisionQuestion,
    DecisionRecord,
    DerivationClass,
    EvidenceClass,
    ResolutionKind,
)
from epistemic_graph.generated.decision_commit import (
    AgentComponentCommittedResult,
    AgentComponentEntry,
    DecisionCommitResult,
)
from epistemic_graph.generated.decision_commit import (
    AgentComponentKind as CommitAgentComponentKind,
)

from agent_utilities.decide.consumers.assembly import Assembler, abstain_reasons, solved
from agent_utilities.layers import clients
from agent_utilities.layers.clients import LayerClients, LayerUnavailable


class _Session:
    tenant = "tenant:t"
    graph = "tenant-graph"


def _clients(eg: Any = None) -> LayerClients:
    session: Any = _Session()
    return LayerClients.for_session(eg or object(), session)


@pytest.fixture
def fake_generated(monkeypatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def sender(name: str):
        async def send(client, params, graph=None, *, idempotency_key=None):
            calls[name] = {"params": params, "graph": graph, "key": idempotency_key}
            return name

        return send

    storage = types.ModuleType("epistemic_graph.generated.storage")
    for name in (
        "send_agent_assemble",
        "send_decision_commit",
        "send_agent_component_search",
        "send_agent_graph",
    ):
        setattr(storage, name, sender(name))
    query = types.ModuleType("epistemic_graph.generated.query")
    for name, send in {"send_explain_provenance_by_ids": sender("provenance")}.items():
        setattr(query, name, send)
    monkeypatch.setitem(sys.modules, "epistemic_graph.generated.storage", storage)
    monkeypatch.setitem(sys.modules, "epistemic_graph.generated.query", query)
    return calls


class _Request:
    def model_dump(self, *, mode: str, exclude_none: bool) -> dict[str, Any]:
        return {"tenant_id": "tenant:t", "mode": mode}


async def test_l3_assemble_and_commit_use_the_generated_decide_senders(
    fake_generated,
) -> None:
    layer = _clients()
    assert await layer.graphs.assemble(_Request()) == "send_agent_assemble"
    sent = fake_generated["send_agent_assemble"]
    assert sent["params"] == {"request": {"tenant_id": "tenant:t", "mode": "json"}}
    assert sent["graph"] == "tenant-graph"
    await layer.graphs.commit_decision(_Request(), idempotency_key="k1")
    assert fake_generated["send_decision_commit"]["key"] == "k1"


async def test_l0_provenance_is_typed(fake_generated) -> None:
    await _clients().knowledge.explain_provenance(["n1", "n2"])
    assert fake_generated["provenance"]["params"] == {"ids": ["n1", "n2"]}


async def test_missing_generated_surface_fails_closed(monkeypatch) -> None:
    empty = types.ModuleType("epistemic_graph.generated.storage")
    monkeypatch.setitem(sys.modules, "epistemic_graph.generated.storage", empty)
    with pytest.raises(LayerUnavailable, match="send_agent_assemble"):
        await _clients().graphs.assemble(_Request())


async def test_l5_outcome_read_requires_the_served_method() -> None:
    class _WorkItems:
        async def get_outcome(self, *, tenant: str, work_item_id: str):
            return {"tenant": tenant, "id": work_item_id}

    class _Eg:
        work_items = _WorkItems()

    assert await _clients(_Eg()).runs.outcome("wi-1") == {
        "tenant": "tenant:t",
        "id": "wi-1",
    }
    with pytest.raises(LayerUnavailable):
        await _clients(object()).runs.outcome("wi-1")


def test_an_unbound_session_is_refused() -> None:
    session: Any = types.SimpleNamespace(tenant="t", graph="")
    bound = clients.KnowledgeClient(object(), session)
    with pytest.raises(LayerUnavailable):
        _ = bound.graph


async def test_l3_publish_graph_carries_the_decision_as_synthesis_evidence(
    fake_generated,
) -> None:
    evidence = {
        "component_id": "decision:abc",
        "kind": "decision_record",
        "definition_digest": "sha256:rec",
    }
    layer = _clients()
    await layer.graphs.publish_graph(
        {"graph_id": "g"}, {"ctx": 1}, evidence=evidence, idempotency_key="p1"
    )
    sent = fake_generated["send_agent_graph"]
    assert sent["key"] == "p1" and sent["graph"] == "tenant-graph"
    request = sent["params"]["op"]["request"]
    assert sent["params"]["op"]["op"] == "publish"
    assert request["graph"] == {"graph_id": "g", "synthesis_evidence": evidence}
    assert request["context"] == {"ctx": 1}


# EH-377(a) consumer audit: ``send_agent_assemble``/``send_decision_commit`` return
# typed ``AssemblyResult``/``DecisionCommitResult`` pydantic models, not dicts, but
# ``agent_utilities.decide.consumers.assembly`` indexes the result as a Mapping
# (``.get``/``[]``). Fixed by decoding at the ``AgentGraphClient`` boundary. These
# tests drive REAL generated model instances (not the dict fakes in
# ``tests/unit/decide/fakes.py``, which is why the bug was not caught there) through
# the real ``assemble``/``commit_decision`` methods and the real consumer helpers.


def _real_decision_record() -> DecisionRecord:
    return DecisionRecord.model_construct(
        caller_principal="agent:t",
        candidate_source=CandidateSourceRecordAgentLibrary(
            source="agent_library", kinds=[]
        ),
        created_at_ms=0,
        derivation_class=DerivationClass.PROOF,
        derivations=[],
        eliminated=[],
        evidence_class=EvidenceClass.CLAIM,
        inputs_digest="sha256:" + "0" * 64,
        outcome=DecisionOutcomeAbstained(
            outcome="abstained",
            reasons=[
                AbstainReasonInsufficientConfidence(reason="insufficient_confidence")
            ],
        ),
        premises=[],
        question=DecisionQuestion.ASSEMBLE,
        record_digest="sha256:" + "1" * 64,
        record_id="decision:abc123",
        resolution_kind=ResolutionKind.ABSTENTION,
        schema_version=1,
        tenant_id="tenant-t",
        why_not=[],
    )


def _real_assembly_result() -> AssemblyResult:
    return AssemblyResult.model_construct(
        record=_real_decision_record(), schema_version=1
    )


def _real_decision_commit_result() -> DecisionCommitResult:
    component = AgentComponentEntry.model_construct(
        actor_scope="agent:t",
        component_id="decision:abc123",
        content_digest="sha256:" + "2" * 64,
        created_at_ms=0,
        definition_digest="sha256:rec",
        entry_revision=1,
        kind=CommitAgentComponentKind.MODEL_PROFILE,
        policy_digest="sha256:" + "3" * 64,
        purpose_id="p1",
        schema_version=1,
        source_revision="r1",
        source_revision_digest="sha256:" + "4" * 64,
        summary="s",
        tenant_id="tenant-t",
        updated_at_ms=0,
        version="v1",
    )
    committed_component = AgentComponentCommittedResult.model_construct(
        batch_id="b1", committed_version=1, component=component, schema_version=1
    )
    return DecisionCommitResult.model_construct(
        component=committed_component,
        record_id="decision:abc123",
        replayed=False,
        schema_version=1,
    )


@pytest.fixture
def fake_generated_typed_models(monkeypatch) -> None:
    """The real generated senders, answering with REAL typed models (as EG does)."""

    async def send_agent_assemble(client, params, graph=None, *, idempotency_key=None):
        return _real_assembly_result()

    async def send_decision_commit(client, params, graph=None, *, idempotency_key=None):
        return _real_decision_commit_result()

    storage = types.ModuleType("epistemic_graph.generated.storage")
    for name, send in {
        "send_agent_assemble": send_agent_assemble,
        "send_decision_commit": send_decision_commit,
    }.items():
        setattr(storage, name, send)
    monkeypatch.setitem(sys.modules, "epistemic_graph.generated.storage", storage)


async def test_assemble_normalizes_a_real_typed_model_to_json(
    fake_generated_typed_models,
) -> None:
    result = await _clients().graphs.assemble(_Request())
    # Before the fix, ``result`` was the raw ``AssemblyResult`` model and every
    # Mapping-style read below raised ``AttributeError``.
    assert isinstance(result, dict)
    assert result["record"]["outcome"]["outcome"] == "abstained"
    assert solved(result) is False
    assert abstain_reasons(result) == ["insufficient_confidence"]


async def test_commit_decision_normalizes_a_real_typed_model_to_json(
    fake_generated_typed_models,
) -> None:
    committed = await _clients().graphs.commit_decision(_Request())
    assert isinstance(committed, dict)
    assert committed["record_id"] == "decision:abc123"
    assert committed["component"]["component"]["definition_digest"] == "sha256:rec"


async def test_assembler_survives_a_real_assembly_result_end_to_end(
    fake_generated_typed_models,
) -> None:
    """``Assembler.assemble`` over the real ``AgentGraphClient`` (not ``FakeGraphs``)."""
    commit_calls: list[Any] = []

    async def commit_context(record: Any) -> dict[str, Any]:
        commit_calls.append(record)
        return {"record": record}

    assembler = Assembler(_clients().graphs, "tenant-t", commit_context=commit_context)
    answer = await assembler.assemble(
        {"tenant_id": "tenant-t"}, lambda reasons: {"fallback_for": reasons}
    )
    assert answer.reason == "abstained: insufficient_confidence"
    assert answer.fallback == {"fallback_for": ["insufficient_confidence"]}
    assert commit_calls == []  # abstained: never attempts a commit
