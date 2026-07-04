"""Pricing catalog + model-name resolution tests (CONCEPT:AU-ECO.toolkit.model-pricing-catalog).

Ports the agentsview ``internal/pricing/normalize_test.go`` and
``fallback_test.go`` cases to guarantee parity of the load-bearing resolver.
"""

from __future__ import annotations

from agent_utilities.pricing import (
    PricingCatalog,
    fallback_pricing,
    get_pricing_catalog,
    normalize_model_name,
    parse_litellm_pricing,
    resolve,
)
from agent_utilities.pricing.catalog import ModelPricing


def test_normalize_model_name():
    cases = {
        "claude-opus-4.7": "claude-opus-4-7",
        "claude-sonnet-4.6": "claude-sonnet-4-6",
        "claude-haiku-4.5": "claude-haiku-4-5",
        "claude-opus-4-8": "claude-opus-4-8",
        "gpt-5.5": "gpt-5-5",
    }
    for inp, want in cases.items():
        assert normalize_model_name(inp) == want, inp


def test_resolve_ordering():
    rates = {
        "claude-opus-4-7": 5,
        "claude-opus-4.6": 99,
        "gemini-3.5": 10,
        "gemini-3.5-flash": 20,
        "openai/gpt-5.5": 30,
        "google/gemini-2.5": 40,
    }

    def r(model):
        return resolve(rates, model)

    assert r("claude-opus-4.7") == (5, True)  # dotted -> normalized
    assert r("claude-opus-4-7") == (5, True)  # exact dashed
    assert r("claude-opus-4.6") == (99, True)  # exact beats normalized
    assert r("CLAUDE-OPUS-4-7") == (5, True)  # case-insensitive
    assert r("Gemini 3.5 Flash (Medium)") == (20, True)  # strip decoration
    assert r("Gemini 3.5 Flash (Low)") == (20, True)
    assert r("Gemini 3.5 Flash") == (20, True)  # longer canonical wins
    assert r("gpt-5.5") == (30, True)  # unqualified -> qualified key
    assert r("google/gemini-2.5") == (40, True)  # same-provider
    assert r("gemini-2.5") == (40, True)  # unqualified -> qualified
    assert r("claude-opus-4.6[1m]") == (99, True)  # bracketed strip
    assert r("claude-opus-4-7-20260101") == (5, True)  # date strip
    assert r("unknown-model") == (None, False)


def test_resolve_provider_prefixes():
    rates = {"openrouter/owl-alpha": 7, "gpt-5.5": 30}
    assert resolve(rates, "other/owl-alpha") == (None, False)
    assert resolve(rates, "owl-alpha") == (7, True)
    assert resolve(rates, "openai/gpt-5.5") == (30, True)


def test_resolve_canonical_determinism():
    rates = {"openai/foo": 1, "other/foo": 2}
    assert resolve(rates, "Foo") == (None, False)  # ambiguous
    assert resolve(rates, "openai/foo[1m]") == (1, True)  # own provider

    with_base = {"openai/bar": 5, "other/bar": 6, "bar": 7}
    assert resolve(with_base, "Bar[1m]") == (7, True)  # unqualified wins

    dupes = {"fo.o": 1, "fo-o": 2}
    assert resolve(dupes, "Foo") == (None, False)  # tied canonical keys


def test_resolve_rejects_arbitrary_substrings():
    rates = {"openai/gpt-5.5": 30, "gemini-3.5-flash": 20}
    assert resolve(rates, "gpt-5.5-codex") == (None, False)
    assert resolve(rates, "wrapped-gemini-3.5-flash-pro") == (None, False)


def test_fallback_is_offline_and_priced():
    catalog = PricingCatalog()
    assert len(catalog) >= len(fallback_pricing())
    # Known models price; unknown stays unpriced.
    cost, priced = catalog.cost_for(
        "claude-opus-4-8", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert priced is True
    assert cost == 5.0 + 25.0  # 1M input @ $5 + 1M output @ $25
    assert catalog.cost_for("totally-made-up-model") == (None, False)


def test_dotted_agent_model_resolves_via_catalog():
    # opencode-style dotted id should price against the dashed fallback key.
    catalog = get_pricing_catalog()
    pricing = catalog.resolve("claude-opus-4.8")
    assert pricing is not None
    assert pricing.input_per_mtok == 5.0


def test_parse_litellm_pricing_converts_per_mtok_and_skips_empty():
    raw = (
        '{"a-model": {"input_cost_per_token": 0.000003,'
        ' "output_cost_per_token": 0.000015,'
        ' "cache_read_input_token_cost": 0.0000003},'
        ' "no-cost-model": {"litellm_provider": "x"}}'
    )
    parsed = parse_litellm_pricing(raw)
    by_pattern = {p.model_pattern: p for p in parsed}
    assert "no-cost-model" not in by_pattern  # missing both costs -> skipped
    m = by_pattern["a-model"]
    assert isinstance(m, ModelPricing)
    assert m.input_per_mtok == 3.0
    assert m.output_per_mtok == 15.0
    assert m.cache_read_per_mtok == 0.3


def test_cost_model_for_model_back_compat():
    from agent_utilities.models.usage import CostModel

    known = CostModel.for_model("claude-opus-4-8")
    assert known.input_token_price == 5.0 / 1_000_000
    # Unknown model falls back to legacy defaults (zero-config safety).
    unknown = CostModel.for_model("nonexistent-xyz")
    assert unknown.input_token_price == 0.00000015
