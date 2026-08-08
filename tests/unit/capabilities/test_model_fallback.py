from __future__ import annotations

"""Unit tests for the D-47 caller-level model/schema fallback chain
(``capabilities/model_fallback.py``).

Covers:

- ``run_fallback_chain``: first attempt succeeds (no fallback needed); first
  attempt exhausts and a LATER attempt succeeds (the actual "model swap"
  proof); every attempt exhausts (``FallbackChainExhausted``, carrying every
  attempt's repair history); a non-``StructuredOutputRepairExhausted``
  exception propagates immediately instead of triggering fallback; an empty
  chain raises ``ValueError``.
- ``model_fallback_chain``: builds its attempt order from ``ModelRegistry``'s
  own tier-fallback ranking (never a hardcoded model list, never able to
  disagree with ``explain_pick_for_task``'s ``chosen_model_id``); respects
  ``max_fallbacks``; ``.run()`` convenience wiring.
"""

import pytest

from agent_utilities.capabilities.model_fallback import (
    FallbackChainExhausted,
    model_fallback_chain,
    run_fallback_chain,
)
from agent_utilities.capabilities.output_repair import (
    RepairAttempt,
    StructuredOutputRepairExhausted,
)
from agent_utilities.models.model_registry import (
    ModelCostRate,
    ModelDefinition,
    ModelRegistry,
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


@pytest.fixture
def sample_registry() -> ModelRegistry:
    return ModelRegistry(
        models=[
            ModelDefinition(
                id="local-fast",
                name="Local LM Studio",
                provider="openai",
                model_id="llama-3.2-3b-instruct",
                base_url="http://localhost:1234/v1",
                tier="light",
                is_default=True,
            ),
            ModelDefinition(
                id="cloud-mini",
                name="GPT-4o Mini",
                provider="openai",
                model_id="gpt-4o-mini",
                api_key_env="OPENAI_API_KEY",
                tier="medium",
                cost=ModelCostRate(input=0.15, output=0.6),
                tags=["code", "tools"],
            ),
            ModelDefinition(
                id="cloud-opus",
                name="Claude 3 Opus",
                provider="anthropic",
                model_id="claude-3-opus-20240229",
                api_key_env="ANTHROPIC_API_KEY",
                tier="heavy",
                cost=ModelCostRate(input=15, output=75),
                tags=["reasoning", "tools"],
            ),
            ModelDefinition(
                id="cloud-reasoning",
                name="o1 Preview",
                provider="openai",
                model_id="o1-preview",
                api_key_env="OPENAI_API_KEY",
                tier="reasoning",
                cost=ModelCostRate(input=15, output=60),
                tags=["reasoning"],
            ),
        ]
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


# ─────────────────────────── model_fallback_chain ───────────────────────────


class TestModelFallbackChain:
    async def test_chain_starts_with_the_registry_chosen_model(
        self, sample_registry: ModelRegistry
    ):
        seen: list[str] = []

        async def build_and_run(model_id: str) -> str:
            seen.append(model_id)
            return f"ran {model_id}"

        chain = model_fallback_chain(
            build_and_run, registry=sample_registry, complexity="medium"
        )
        expected_primary = sample_registry.pick_for_task(complexity="medium").id
        assert chain.model_ids[0] == expected_primary

        result = await chain.run()
        assert result == f"ran {expected_primary}"
        assert seen == [expected_primary]

    async def test_falls_back_across_distinct_registry_models(
        self, sample_registry: ModelRegistry
    ):
        attempted: list[str] = []

        async def build_and_run(model_id: str) -> str:
            attempted.append(model_id)
            if len(attempted) == 1:
                raise _exhausted(model_id)
            return f"succeeded on {model_id}"

        chain = model_fallback_chain(
            build_and_run, registry=sample_registry, complexity="medium"
        )
        assert len(chain.model_ids) >= 2
        assert len(set(chain.model_ids)) == len(chain.model_ids)  # no duplicates

        result = await chain.run()
        assert attempted[0] == chain.model_ids[0]
        assert attempted[1] == chain.model_ids[1]
        assert result == f"succeeded on {chain.model_ids[1]}"

    async def test_max_fallbacks_bounds_the_chain_length(
        self, sample_registry: ModelRegistry
    ):
        async def build_and_run(model_id: str) -> str:
            return model_id

        chain = model_fallback_chain(
            build_and_run,
            registry=sample_registry,
            complexity="medium",
            max_fallbacks=1,
        )
        # primary + 1 fallback = 2, even though the registry has 4 models.
        assert len(chain.model_ids) == 2
        assert len(chain.attempts) == 2

    async def test_all_registry_models_exhausted_raises_chain_exhausted(
        self, sample_registry: ModelRegistry
    ):
        async def build_and_run(model_id: str) -> str:
            raise _exhausted(model_id)

        chain = model_fallback_chain(
            build_and_run,
            registry=sample_registry,
            complexity="medium",
            max_fallbacks=3,
        )
        with pytest.raises(FallbackChainExhausted) as exc_info:
            await chain.run()
        assert [r.label for r in exc_info.value.records] == chain.model_ids

    async def test_empty_registry_raises_value_error(self):
        empty_registry = ModelRegistry(models=[])

        async def build_and_run(model_id: str) -> str:
            return model_id

        with pytest.raises(ValueError):
            model_fallback_chain(build_and_run, registry=empty_registry)
