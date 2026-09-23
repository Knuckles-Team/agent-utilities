"""Typed cross-layer clients route every call through a generated EG surface."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

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
