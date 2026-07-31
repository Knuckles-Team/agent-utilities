"""Provider prompt-cache usage surfaced on the always-on per-call trace
(CONCEPT:AU-ORCH.optimization.provider-prompt-cache), via the SAME model wrap
``test_model_tracing_wrap.py`` already proves records a ``GenerationNode`` per LLM request.

Proves:

* ``GenerationNode``/``TraceNode`` default to zero cache tokens (no behavior change for a
  provider/response that doesn't report cache usage — e.g. ``TestModel``).
* When the underlying response DOES carry ``usage.cache_read_tokens``/``cache_write_tokens``
  (the pydantic-ai ``RequestUsage`` shape Anthropic/OpenAI populate), ``_TracingModel.request``
  extracts them and both the ``GenerationNode`` AND the owning ``TraceNode`` rollup carry them —
  the measurable "cache-hit tokens in usage/cost telemetry" the constitution asks for.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel

from agent_utilities.harness import tracing
from agent_utilities.harness.trace_backend import KGTraceBackend


@pytest.fixture
def trace_sink():
    prev = tracing.get_kg_trace_sink()
    backend = KGTraceBackend()
    tracing.set_kg_trace_sink(backend)
    try:
        yield backend
    finally:
        tracing.set_kg_trace_sink(prev)


@pytest.mark.asyncio
async def test_no_cache_usage_defaults_to_zero(
    trace_sink: KGTraceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A response whose ``usage`` doesn't carry cache fields (the common case, e.g.
    ``pydantic_ai.usage.RequestUsage`` defaults) must not change existing behavior."""
    fake_response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=5, output_tokens=1)
    )

    async def _fake_request(self, messages, model_settings, mrp):  # noqa: ANN001, ARG001
        return fake_response

    monkeypatch.setattr(WrapperModel, "request", _fake_request)

    wrapped = tracing.wrap_model_for_tracing(TestModel())
    await wrapped.request([], None, None)

    gens = [g for e in trace_sink._traces.values() for g in e["generations"]]
    assert gens, "expected a GenerationNode from the wrapped request"
    assert gens[0].cache_read_tokens == 0
    assert gens[0].cache_write_tokens == 0


@pytest.mark.asyncio
async def test_cache_usage_is_extracted_and_rolled_up_onto_the_trace(
    trace_sink: KGTraceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_usage = SimpleNamespace(
        input_tokens=100, output_tokens=20, cache_read_tokens=64, cache_write_tokens=8
    )
    fake_response = SimpleNamespace(usage=fake_usage)

    async def _fake_request(self, messages, model_settings, mrp):  # noqa: ANN001, ARG001
        return fake_response

    monkeypatch.setattr(WrapperModel, "request", _fake_request)

    wrapped = tracing.wrap_model_for_tracing(TestModel())
    await wrapped.request([], None, None)

    gens = [g for e in trace_sink._traces.values() for g in e["generations"]]
    assert gens, "expected a GenerationNode from the wrapped request"
    assert gens[0].cache_read_tokens == 64
    assert gens[0].cache_write_tokens == 8

    trace = next(iter(trace_sink._traces.values()))["trace"]
    assert trace.cache_read_tokens == 64
    assert trace.cache_write_tokens == 8
