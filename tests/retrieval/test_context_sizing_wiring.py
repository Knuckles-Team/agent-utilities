"""EH-399 on the live path: every model invocation's evidence is compiled by
``compile_model_context``, which sizes it for the INVOKED model -- its
registry capacity, its exact tokenizer, EG's certified ``Solve`` over the
installed decision transport."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from agent_utilities import decide
from agent_utilities.core import contextual_model as cm
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import use_session
from agent_utilities.knowledge_graph.ontology.permissioning import (
    clear_markings,
    use_marking_authority,
)
from agent_utilities.knowledge_graph.retrieval import context_knapsack as ck
from agent_utilities.models import model_registry
from agent_utilities.models.model_registry import ModelDefinition, ModelRegistry
from tests.retrieval.test_context_compiler import (
    FakeRetriever,
    _FakeMarkingStore,
    _grant_public,
    _session,
)
from tests.unit.decide.fakes import FakeTransport, runner

MODEL = "gpt-4o-mini"


@pytest.fixture(autouse=True)
def _policy_state() -> Any:
    reset_company_brain()
    with use_marking_authority(_FakeMarkingStore()):
        yield
    reset_company_brain()
    clear_markings()


class _SolvingTransport(FakeTransport):
    """The decision transport, answering EG ``Solve`` with the greedy mask
    certified optimal (EG's solver is exercised in EG's own tests)."""

    def __init__(self) -> None:
        super().__init__()
        self.solved: list[Mapping[str, Any]] = []

    async def solve(self, request: Mapping[str, Any]) -> Any:
        self.solved.append(request)
        model = request["model"]
        return {
            "certificate": {
                "status": "optimal",
                "incumbent": {
                    "selected": [True] + [False] * (len(model["variables"]) - 1)
                },
            },
            "certificate_digest": "sha256:cert",
        }


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> None:
    definition = ModelDefinition(
        id="small",
        name="Small",
        provider="openai",
        model_id=MODEL,
        context_window=400,
        max_output_tokens=100,
    )
    registry = ModelRegistry(models=[definition])
    monkeypatch.setattr(model_registry, "_ACTIVE_REGISTRY", registry)


def _nodes() -> list[dict[str, Any]]:
    nodes = [
        {
            "id": f"n{i}",
            "type": "Doc",
            "name": f"Doc {i}",
            "description": "word " * 200,
            "score": 0.9 - i * 0.01,
        }
        for i in range(4)
    ]
    _grant_public(nodes)
    return nodes


def test_a_model_invocation_is_sized_by_its_model_and_certified_by_eg(
    registered: None,
) -> None:
    sizer = ck.sizer_for_model(MODEL)
    assert sizer is not None and sizer.capacity.tokens() == 300
    assert sizer.counter.identity.startswith("tiktoken:")
    transport = _SolvingTransport()
    decide.install_runner(runner(transport))
    session = _session()
    try:
        with use_session(session):
            bundle = cm.compile_model_context(
                "what do the docs say?",
                session=session,
                engine=FakeRetriever(_nodes()),
                model_version=MODEL,
            )
    finally:
        decide.install_runner(None)
    assert len(transport.solved) == 1, "EG certified the sizing"
    assert len(bundle.items) == 1 and bundle.tokens_used <= 300
    assert ck.current_sizer() is None, "the sizing is scoped to the call"


def test_an_unregistered_model_keeps_the_greedy_fit(registered: None) -> None:
    assert ck.sizer_for_model("some-local-model") is None
    assert ck.sizer_for_model("") is None
