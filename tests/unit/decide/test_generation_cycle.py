"""EH-397 through its live caller: the Loop engine's cycle runs the governed
embedding-generation swap (``KG_LOOP_EMBEDDING_GENERATION``) with the
process's real parts -- the published model, the corpus re-embedder over the
engine graph, the background capacity class -- and activates only with EG's
passing receipt; the vector arm then embeds queries in the active space."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from agent_utilities import decide
from agent_utilities.core import embedding_utilities
from agent_utilities.core.resource_priority import PriorityClass, current_priority
from agent_utilities.decide.learning import generation, generation_cycle
from agent_utilities.decide.learning.generation_cycle import (
    generation_embedder,
    shadow_graph_of,
)
from agent_utilities.knowledge_graph.research.loop_controller import (
    LoopController,
    _CycleOptions,
)
from tests.unit.decide.fakes import FakeTransport, runner
from tests.unit.decide.test_retrieval_learning_governed import ENTRY, JUDGED, _key

SHADOW = shadow_graph_of("kg", "tuned-m")


class _Embedder:
    def __init__(self, model_name: str, priorities: list[Any]) -> None:
        self.model_name = model_name
        self._priorities = priorities

    def get_text_embedding(self, text: str) -> list[float]:
        self._priorities.append(current_priority())
        return [1.0, 0.0] if self.model_name == "base-m" else [0.0, 1.0]


class _Store:
    """The engine graph and its named-graph views over one in-memory store."""

    def __init__(self, stores: dict[str, dict[str, Any]], name: str = "kg") -> None:
        self.graph_name = name
        self._stores = stores
        stores.setdefault(name, {})

    def for_graph(self, name: str) -> _Store:
        return _Store(self._stores, name)

    def node_ids(self) -> list[str]:
        return sorted(self._stores[self.graph_name])

    def _get_node_properties_batch(self, ids: list[str]) -> dict[str, Any]:
        nodes = self._stores[self.graph_name]
        return {i: dict(nodes[i]["props"]) for i in ids}

    def add_node(self, node_id: str, properties: Mapping[str, Any]) -> None:
        self._stores[self.graph_name][node_id] = {"props": dict(properties)}

    def add_embedding(self, node_id: str, vector: list[float]) -> None:
        self._stores[self.graph_name][node_id]["vector"] = vector


class _Engine:
    def __init__(self, priorities: list[Any]) -> None:
        self.stores: dict[str, dict[str, Any]] = {
            "kg": {
                "n1": {"props": {"description": "billing runbook", "embedding": [1]}},
                "n2": {"props": {"name": "on-call rota"}},
                "n3": {"props": {"weight": 3}},
            }
        }
        self.graph = _Store(self.stores)
        retriever = type("Retriever", (), {})()
        retriever.embed_model = _Embedder("base-m", priorities)
        retriever.engine = self
        self.hybrid_retriever = retriever


def _transport(pointer: list[str]) -> FakeTransport:
    evaluated = {
        "recorded": "generation",
        "receipt_digest": "sha256:g",
        "receipt": {"passed": True},
    }

    def answer(op: Mapping[str, Any]) -> Any:
        key = _key(op)
        if key == "move_pointer:activate":
            pointer.append(op["write"]["movement"]["target"])
            return {"recorded": "pointer", "key": "k", "active": {"target": SHADOW}}
        return {"get": ENTRY, "evaluate_generation": evaluated}[key]

    def query(text: str) -> Any:
        if "decision_pointers" in text:
            return {"columns": ["target"], "rows": [[p] for p in pointer[-1:]]}
        return JUDGED

    return FakeTransport(log_answer=answer, sql_answer=query)


@pytest.fixture
def cycle_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    priorities: list[Any] = []
    monkeypatch.setenv("KG_LOOP_EMBEDDING_GENERATION", "true")
    monkeypatch.setenv("EMBEDDING_GENERATION_MODEL", "tuned-m")
    monkeypatch.setattr(
        embedding_utilities,
        "create_embedding_model",
        lambda model: _Embedder(model, priorities),
    )
    generation._RESOLVED.clear()
    generation_cycle._EMBEDDERS.clear()
    yield priorities
    decide.install_runner(None)
    generation._RESOLVED.clear()
    generation_cycle._EMBEDDERS.clear()


def _run_stage(engine: _Engine) -> dict[str, Any]:
    report: dict[str, Any] = {}
    opts = _CycleOptions(
        insight_validation=False, trace_mining=False, belief_revision=False
    )
    LoopController(engine)._cycle_insight_stages(report, lambda n, fn: fn(), opts)
    return report


def test_the_loop_swaps_the_generation_with_a_passing_receipt(cycle_env) -> None:
    pointer: list[str] = []
    transport = _transport(pointer)
    decide.install_runner(runner(transport))
    engine = _Engine(cycle_env)

    report = _run_stage(engine)

    assert report["embedding_generation"]["activated"], report
    assert pointer == [SHADOW]
    shadow = engine.stores[SHADOW]
    assert sorted(shadow) == ["n1", "n2"], "units with no text are not embedded"
    assert shadow["n1"]["vector"] == [0.0, 1.0]
    assert "embedding" not in shadow["n1"]["props"], "stored vectors never copied"
    assert PriorityClass.BACKGROUND_INGESTION in cycle_env, "under the lease"
    request = next(
        op["write"]["request"]
        for op in transport.ops
        if _key(op) == "evaluate_generation"
    )
    assert (request["active_graph"], request["shadow_graph"]) == ("kg", SHADOW)
    assert request["items"][0]["shadow_q16"] == [0, 65536]

    tuned = generation_embedder(engine.hybrid_retriever)
    assert tuned.model_name == "tuned-m", "queries are embedded in the active space"

    again = _run_stage(engine)["embedding_generation"]
    assert (again["activated"], again["stage"]) == (False, "active")


def test_the_stage_is_opt_in_and_needs_a_published_model(
    cycle_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _Engine(cycle_env)
    monkeypatch.setenv("KG_LOOP_EMBEDDING_GENERATION", "false")
    assert "embedding_generation" not in _run_stage(engine)
    monkeypatch.setenv("KG_LOOP_EMBEDDING_GENERATION", "true")
    monkeypatch.delenv("EMBEDDING_GENERATION_MODEL")
    assert _run_stage(engine)["embedding_generation"]["stage"] == "train"
    assert generation_embedder(engine.hybrid_retriever).model_name == "base-m"
