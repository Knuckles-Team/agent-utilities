"""Per-LLM-call GenerationNode capture via the model wrap (CONCEPT:OS-5.68).

create_model wraps the model (when a KG sink is installed) so every LLM request records
a GenerationNode — CI-safe here with pydantic-ai's TestModel (no network).
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from agent_utilities.harness import tracing
from agent_utilities.harness.trace_backend import KGTraceBackend


def test_wrap_is_noop_without_sink():
    tracing.set_kg_trace_sink(None)
    m = TestModel()
    assert tracing.wrap_model_for_tracing(m) is m  # unchanged, zero overhead


def test_wrapped_model_records_generation(monkeypatch):
    prev = tracing.get_kg_trace_sink()
    be = KGTraceBackend()
    tracing.set_kg_trace_sink(be)
    try:
        wrapped = tracing.wrap_model_for_tracing(TestModel())
        assert type(wrapped).__name__ == "_TracingModel"
        Agent(wrapped).run_sync("hello")
        gens = [g for e in be._traces.values() for g in e["generations"]]
        assert gens, "expected a GenerationNode from the wrapped request"
        assert gens[0].type == "generation"
    finally:
        tracing.set_kg_trace_sink(prev)
