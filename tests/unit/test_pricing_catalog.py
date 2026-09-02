"""Pricing catalog + model-name resolution tests (CONCEPT:AU-ECO.toolkit.model-pricing-catalog).

Ports the agentsview ``internal/pricing/normalize_test.go`` and
``fallback_test.go`` cases to guarantee parity of the load-bearing resolver.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_utilities.pricing import (
    PricingCatalog,
    get_pricing_catalog,
    normalize_model_name,
    parse_litellm_pricing,
    resolve,
)
from agent_utilities.pricing.catalog import ModelPricing


def test_normalize_model_name():
    cases = {
        "vendor/family.x-7.2": "vendor/family-x-7-2",
        "family-small-4.5": "family-small-4-5",
        "family-large-4-8": "family-large-4-8",
    }
    for inp, want in cases.items():
        assert normalize_model_name(inp) == want, inp


def test_resolve_ordering():
    rates = {
        "family-large-4-7": 5,
        "family-large-4.6": 99,
        "family-rapid-3.5": 10,
        "family-rapid-3.5-plus": 20,
        "vendor-a/family-x-5.5": 30,
        "vendor-b/family-y-2.5": 40,
    }

    def r(model):
        return resolve(rates, model)

    assert r("family-large-4.7") == (5, True)  # dotted -> normalized
    assert r("family-large-4-7") == (5, True)  # exact dashed
    assert r("family-large-4.6") == (99, True)  # exact beats normalized
    assert r("FAMILY-LARGE-4-7") == (5, True)  # case-insensitive
    assert r("Family Rapid 3.5 Plus (Medium)") == (20, True)
    assert r("Family Rapid 3.5 Plus (Low)") == (20, True)
    assert r("Family Rapid 3.5 Plus") == (20, True)
    assert r("family-x-5.5") == (30, True)  # unqualified -> qualified key
    assert r("vendor-b/family-y-2.5") == (40, True)  # same-provider
    assert r("family-y-2.5") == (40, True)  # unqualified -> qualified
    assert r("family-large-4.6[1m]") == (99, True)  # bracketed strip
    assert r("family-large-4-7-20260101") == (5, True)  # date strip
    assert r("unknown-model") == (None, False)


def test_resolve_provider_prefixes():
    rates = {"vendor-a/family-alpha": 7, "family-x-5.5": 30}
    assert resolve(rates, "other/family-alpha") == (None, False)
    assert resolve(rates, "family-alpha") == (7, True)
    assert resolve(rates, "vendor-a/family-x-5.5") == (30, True)


def test_resolve_canonical_determinism():
    rates = {"vendor-a/foo": 1, "vendor-b/foo": 2}
    assert resolve(rates, "Foo") == (None, False)  # ambiguous
    assert resolve(rates, "vendor-a/foo[1m]") == (1, True)  # own provider

    with_base = {"vendor-a/bar": 5, "vendor-b/bar": 6, "bar": 7}
    assert resolve(with_base, "Bar[1m]") == (7, True)  # unqualified wins

    dupes = {"fo.o": 1, "fo-o": 2}
    assert resolve(dupes, "Foo") == (None, False)  # tied canonical keys


def test_resolve_rejects_arbitrary_substrings():
    rates = {"vendor-a/family-x-5.5": 30, "family-rapid-3.5": 20}
    assert resolve(rates, "family-x-5.5-special") == (None, False)
    assert resolve(rates, "wrapped-family-rapid-3.5-pro") == (None, False)


def test_unconfigured_catalog_is_explicitly_unpriced():
    catalog = PricingCatalog()
    assert len(catalog) == 0
    assert catalog.cost_for("operator/not-configured") == (None, False)


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


def test_cost_model_for_model_requires_catalog_pricing(monkeypatch):
    from agent_utilities.models.usage import CostModel
    from agent_utilities.pricing import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "_CATALOG",
        PricingCatalog(
            [ModelPricing(model_pattern="operator/model-v7", input_per_mtok=5)]
        ),
    )
    known = CostModel.for_model("operator/model-v7")
    assert known.input_token_price == 5.0 / 1_000_000
    with pytest.raises(LookupError, match="pricing is not configured"):
        CostModel.for_model("operator/unconfigured")


def test_versioned_operator_catalog_prices_arbitrary_external_id(tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "version": "operator-2026-09-01",
                "models": [
                    {
                        "model_pattern": "provider-z/external-model-v99",
                        "input_per_mtok": 2.0,
                        "output_per_mtok": 6.0,
                    }
                ],
            }
        )
    )

    catalog = PricingCatalog.load_from_file(path)

    assert catalog.version == "operator-2026-09-01"
    assert catalog.cost_for(
        "provider-z/external-model-v99",
        input_tokens=1_000_000,
        output_tokens=500_000,
    ) == (5.0, True)
    assert catalog.cost_for("provider-z/not-configured") == (None, False)


def test_empty_operator_catalog_keeps_unknown_models_unpriced(tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps({"version": "empty-v1", "models": []}))

    catalog = PricingCatalog.load_from_file(path)

    assert len(catalog) == 0
    assert catalog.cost_for("provider-z/unconfigured") == (None, False)


def test_configured_catalog_changes_live_usage_and_trace_consumers(
    tmp_path, monkeypatch
):
    from agent_utilities.core.config import config
    from agent_utilities.harness.trace_backend import KGTraceBackend
    from agent_utilities.pricing import catalog as catalog_module
    from agent_utilities.usage.cost import price_event
    from agent_utilities.usage.models import UsageEvent

    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "version": "operator-v7",
                "models": [
                    {
                        "model_pattern": "operator/external-v7",
                        "input_per_mtok": 2.0,
                        "output_per_mtok": 6.0,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(config, "pricing_catalog_path", str(path))
    monkeypatch.setattr(catalog_module, "_CATALOG", PricingCatalog())
    catalog_module.reset_pricing_catalog()

    catalog = get_pricing_catalog()
    event = price_event(
        UsageEvent(
            session_id="session-1",
            model="operator/external-v7",
            input_tokens=1_000_000,
            output_tokens=500_000,
        )
    )

    assert catalog.version == "operator-v7"
    assert event.cost_usd == 5.0
    assert event.cost_status == "catalog"
    assert KGTraceBackend._cost_usd("operator/external-v7", 1_000_000, 500_000) == 5.0


def test_unknown_pricing_propagates_through_trace_rollups(monkeypatch):
    from agent_utilities.harness.trace_backend import KGTraceBackend
    from agent_utilities.models.knowledge_graph import GenerationNode, TraceNode
    from agent_utilities.pricing import catalog as catalog_module

    monkeypatch.setattr(catalog_module, "_CATALOG", PricingCatalog())
    backend = KGTraceBackend()
    trace = TraceNode(id="trace:unknown", name="run")
    generation = GenerationNode(
        id="generation:unknown",
        name="call",
        trace_id=trace.id,
        model="operator/unconfigured",
        input_tokens=100,
    )

    backend.emit_trace(trace, generations=[generation])

    assert generation.total_cost_usd is None
    assert trace.total_cost_usd is None
    summaries = asyncio.run(backend.get_traces(""))
    assert summaries[0]["total_cost_usd"] is None


def test_fresh_trace_has_unknown_cost_until_every_generation_is_priced():
    from agent_utilities.models.knowledge_graph import TraceNode

    assert TraceNode(id="trace:fresh", name="run").total_cost_usd is None


def test_remote_pricing_refresh_requires_operator_source(monkeypatch):
    from agent_utilities.core.config import config
    from agent_utilities.pricing import store

    monkeypatch.setattr(config, "pricing_litellm_url", "")
    monkeypatch.setattr(
        store,
        "fetch_litellm_pricing",
        lambda _url: pytest.fail("unconfigured refresh attempted network access"),
    )

    assert store.refresh_catalog(catalog=PricingCatalog()) == 0
