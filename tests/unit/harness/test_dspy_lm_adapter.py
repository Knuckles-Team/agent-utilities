"""Live-path proof: the DSPy optimizer's LM call is endpoint-safe.

CONCEPT:AU-AHE.optimization.endpoint-safe-dspy-optimization. DSPy defaults to its OWN
litellm-backed LM when nothing is configured — which would call the fleet's shared vLLM
endpoint DIRECTLY, bypassing ``model_concurrency`` + the resource-priority edict. This
proves the actual call path — ``ConcurrencyBoundDSPyLM.forward()``, the same method every
DSPy optimizer compile (``BootstrapFewShot``/``MIPROv2``/self-supervised optimizers)
invokes under the hood — really does:

1. acquire the shared per-model priority gate, bounded to
   ``model_concurrency.resolve_capacity`` (the SAME resolver every other LLM fan-out in
   this codebase uses) — i.e. it cannot exceed the model's declared capacity;
2. record usage telemetry tagged ``source=dspy_optimization``.

No network call is made — ``dspy.clients.lm.litellm_completion`` is replaced by a fake so
this is deterministic and offline, while every other layer (the real ``dspy.LM``
machinery, the real ``PriorityModelGate``, the real ``TokenUsageTracker``) runs for real.
"""

from __future__ import annotations

import pytest

dspy = pytest.importorskip("dspy")

from agent_utilities.core import model_concurrency as mc
from agent_utilities.core import resource_priority as rp
from agent_utilities.harness import dspy_lm_adapter as adapter


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FakeChoice:
    finish_reason = "stop"


class _FakeResponse(dict):
    """Minimal stand-in for a litellm ``ModelResponse``: subscriptable (for
    ``dspy.LM._check_truncation``'s ``results["choices"]``) AND carries a real
    ``.usage`` attribute (for our telemetry hook)."""

    def __init__(self, usage: _FakeUsage) -> None:
        super().__init__(choices=[_FakeChoice()])
        self.usage = usage


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    mc.reset_controllers()
    # A deterministic, distinctive capacity so the assertion proves the gate was
    # sized off model_concurrency.resolve_capacity — not some incidental default.
    monkeypatch.setattr(mc, "resolve_capacity", lambda model=None, default=1: 4)
    yield
    mc.reset_controllers()


def test_dspy_lm_forward_acquires_model_concurrency_gate_and_records_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # --- spy on the shared PriorityModelGate (the sync face model_concurrency's
    # priority_slot_sync acquires — bounded to resolve_capacity) -----------------
    calls: list[tuple[str, object]] = []
    real_acquire = rp.PriorityModelGate.acquire_sync
    real_release = rp.PriorityModelGate.release_sync

    def spy_acquire(self: rp.PriorityModelGate, priority: rp.PriorityClass) -> None:
        calls.append(("acquire", priority))
        real_acquire(self, priority)

    def spy_release(self: rp.PriorityModelGate) -> None:
        calls.append(("release", None))
        real_release(self)

    monkeypatch.setattr(rp.PriorityModelGate, "acquire_sync", spy_acquire)
    monkeypatch.setattr(rp.PriorityModelGate, "release_sync", spy_release)

    # --- fake the network edge only: litellm never actually runs ----------------
    fake_response = _FakeResponse(_FakeUsage(prompt_tokens=11, completion_tokens=7))

    def fake_litellm_completion(request, num_retries, cache=None):
        return fake_response

    monkeypatch.setattr(
        dspy.clients.lm, "litellm_completion", fake_litellm_completion
    )

    # --- spy on the real TokenUsageTracker instance the adapter records through --
    tracker = adapter._tracker()
    recorded: list[object] = []
    real_record = tracker.record

    def spy_record(record):
        recorded.append(record)
        return real_record(record)

    monkeypatch.setattr(tracker, "record", spy_record)

    # --- the live path: build the SAME LM the optimization guard installs -------
    lm = adapter.build_dspy_lm()
    assert lm is not None, "DSPy is installed in this test env; adapter must build"

    with rp.priority_scope(rp.PriorityClass.BACKGROUND_INGESTION):
        result = lm.forward(prompt="say hi", cache=False)

    # (1) the real network edge was reached through OUR adapter, not bypassed.
    assert result is fake_response

    # (2) the shared per-model gate — bounded to model_concurrency.resolve_capacity
    # — was acquired then released exactly once around the call.
    assert calls == [
        ("acquire", rp.PriorityClass.BACKGROUND_INGESTION),
        ("release", None),
    ]

    # (3) usage telemetry was recorded, tagged for the shared visibility/cap.
    assert len(recorded) == 1
    rec = recorded[0]
    assert rec.agent_name == "dspy_optimization"
    assert rec.prompt_tokens == 11
    assert rec.response_tokens == 7
    assert rec.metadata.get("source") == "dspy_optimization"


def test_dspy_lm_forward_yields_to_interactive_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An untagged/DSPy call classes itself BACKGROUND_INGESTION by default, so a
    saturating DSPy fan-out yields admission to an interactive/orchestration call
    contending on the SAME model gate (CONCEPT:AU-ORCH.scheduling.resource-priority-edict) — the actual
    endpoint-safety guarantee, not just a capacity count.
    """
    fake_response = _FakeResponse(_FakeUsage(prompt_tokens=1, completion_tokens=1))
    monkeypatch.setattr(
        dspy.clients.lm, "litellm_completion", lambda request, num_retries, cache=None: fake_response
    )

    lm = adapter.build_dspy_lm()
    assert lm is not None
    model_key = lm._au_model_key

    # The autouse fixture resolves capacity=4 for this model ⇒ the SAME gate
    # priority_slot_sync(model_key) resolves internally (auto-reserve carves out 1
    # of the 4 permits — see resource_priority._auto_reserve).
    gate = rp.get_priority_gate(model_key, capacity=4)
    assert gate.reserve == 1

    # Saturate background up to its reserved-minus headroom (capacity - reserve),
    # then confirm: an interactive/orchestration call still lands (the reserved
    # slot), while one more background call must yield.
    for _ in range(gate.capacity - gate.reserve):
        gate.acquire_sync(rp.PriorityClass.BACKGROUND_INGESTION)
    try:
        assert gate._can_admit(is_high=True)  # interactive still gets in
        assert not gate._can_admit(is_high=False)  # background yields — no headroom left
    finally:
        for _ in range(gate.capacity - gate.reserve):
            gate.release_sync()

    # No ambient priority set ⇒ the adapter itself defaults DSPy to BACKGROUND_INGESTION.
    result = lm.forward(prompt="hi", cache=False)
    assert result is fake_response
