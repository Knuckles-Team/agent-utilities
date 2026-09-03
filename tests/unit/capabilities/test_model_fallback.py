from __future__ import annotations

"""Unit tests for the caller-level model/schema fallback chain
(``capabilities/model_fallback.py``).

Covers:

- ``run_fallback_chain``: first attempt succeeds (no fallback needed); first
  attempt exhausts and a LATER attempt succeeds (the actual "model swap"
  proof); every attempt exhausts (``FallbackChainExhausted``, carrying every
  attempt's repair history); a non-``StructuredOutputRepairExhausted``
  exception propagates immediately instead of triggering fallback; an empty
  chain raises ``ValueError``.
"""

import pytest

from agent_utilities.capabilities.model_fallback import (
    FallbackChainExhausted,
    run_fallback_chain,
)
from agent_utilities.capabilities.output_repair import (
    RepairAttempt,
    StructuredOutputRepairExhausted,
)


def _exhausted(model_id: str) -> StructuredOutputRepairExhausted:
    attempts = [
        RepairAttempt(
            classification="schema_invalid",
            attempt=1,
            action="exhausted",
            detail=f"{model_id} could not produce valid output",
        )
    ]
    return StructuredOutputRepairExhausted(
        f"structured output repair exhausted for {model_id}",
        attempts=attempts,
    )


# ─────────────────────────── run_fallback_chain ───────────────────────────


class TestRunFallbackChain:
    async def test_first_attempt_succeeds_no_fallback(self):
        calls: list[str] = []

        async def primary() -> str:
            calls.append("primary")
            return "ok"

        async def never_called() -> str:
            calls.append("never")
            return "should not run"

        result = await run_fallback_chain([primary, never_called])
        assert result == "ok"
        assert calls == ["primary"]

    async def test_falls_back_to_a_later_model_on_exhaustion(self):
        """The concrete 'actual model swap + successful completion' proof
        (D-47's own acceptance bar): the primary attempt exhausts repair, and
        the SECOND attempt (a fresh Agent bound to a different model) actually
        runs and its result is what the caller gets back."""
        calls: list[str] = []

        async def primary_model() -> str:
            calls.append("primary_model")
            raise _exhausted("primary_model")

        async def fallback_model() -> str:
            calls.append("fallback_model")
            return "fallback succeeded"

        result = await run_fallback_chain(
            [primary_model, fallback_model],
            labels=["primary_model", "fallback_model"],
        )
        assert result == "fallback succeeded"
        assert calls == ["primary_model", "fallback_model"]

    async def test_every_attempt_exhausted_raises_chain_exhausted(self):
        async def attempt_a() -> str:
            raise _exhausted("model-a")

        async def attempt_b() -> str:
            raise _exhausted("model-b")

        with pytest.raises(FallbackChainExhausted) as exc_info:
            await run_fallback_chain(
                [attempt_a, attempt_b], labels=["model-a", "model-b"]
            )

        err = exc_info.value
        assert [r.label for r in err.records] == ["model-a", "model-b"]
        # Every attempt's own repair history is preserved, not just the last.
        assert all(r.error.attempts for r in err.records)
        assert "model-a" in str(err)
        assert "model-b" in str(err)
        # __cause__ chains to the LAST attempt's error for a readable traceback.
        assert err.__cause__ is err.records[-1].error

    async def test_non_repair_exception_propagates_without_fallback(self):
        """Fallback is scoped to StructuredOutputRepairExhausted ONLY — any other
        failure (a network error, a bug) must propagate immediately, never be
        silently retried against a different model."""
        calls: list[str] = []

        async def broken() -> str:
            calls.append("broken")
            raise RuntimeError("not a structured-output failure")

        async def never_called() -> str:
            calls.append("never")
            return "unreachable"

        with pytest.raises(RuntimeError, match="not a structured-output failure"):
            await run_fallback_chain([broken, never_called])
        assert calls == ["broken"]

    async def test_empty_chain_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one attempt"):
            await run_fallback_chain([])

    async def test_mismatched_labels_length_raises_value_error(self):
        async def attempt() -> str:
            return "ok"

        with pytest.raises(ValueError, match="same length"):
            await run_fallback_chain([attempt], labels=["a", "b"])
