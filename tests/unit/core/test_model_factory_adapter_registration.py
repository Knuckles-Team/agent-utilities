"""``_resolve_model_adapter_factory``'s dispatch: registry override, built-in
provider seam, and the fail-closed unknown-provider path (CONCEPT:AU-ORCH.adapter.byok-provider-proxy).

Closes a real coverage gap: nothing in this suite previously exercised
``_resolve_model_adapter_factory`` directly, or ``_BUILTIN_ADAPTER_BUILDERS``
(the explicit provider->builder registration table that replaced a
``globals().get(f"_build_{provider}_model")`` reflection lookup). The
existing ``create_model(provider=...)`` tests exercise the built-in path
*indirectly* by constructing a real model, but never pin the dispatch
contract itself: that a registry-injected factory always wins over a
built-in, that every built-in provider name still resolves to its own
builder after the reflection-to-table refactor, and that an unrecognized
provider fails closed with ``ModelAdapterConfigurationError`` rather than
``AttributeError``/``None`` propagating silently.
"""

from __future__ import annotations

import pytest

from agent_utilities.core import model_factory
from agent_utilities.models import model_registry as registry_module


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Each test gets a fresh process-global registry (no cross-test adapter leakage).

    ``_resolve_model_adapter_factory`` mutates the cached
    ``load_active_registry()`` singleton via ``register_adapter_factory`` when a
    caller injects one — reset before AND after so an injected factory from one
    test can never be observed by a later test in this file or elsewhere.
    """
    registry_module.reset_active_registry()
    yield
    registry_module.reset_active_registry()


@pytest.mark.concept(id="AU-ORCH.adapter.byok-provider-proxy")
@pytest.mark.parametrize(
    ("provider", "builder_name"),
    [
        ("openai", "_build_openai_model"),
        ("ollama", "_build_ollama_model"),
        ("deepseek", "_build_deepseek_model"),
        ("anthropic", "_build_anthropic_model"),
        ("google", "_build_google_model"),
        ("groq", "_build_groq_model"),
        ("mistral", "_build_mistral_model"),
        ("huggingface", "_build_huggingface_model"),
        ("custom", "_build_custom_model"),
    ],
)
def test_every_builtin_provider_resolves_to_its_own_builder(provider, builder_name):
    """Each built-in provider name still dispatches to its dedicated builder.

    Pins the explicit ``_BUILTIN_ADAPTER_BUILDERS`` table against the same
    provider set the retired ``globals().get(f"_build_{provider}_model")``
    reflection lookup resolved, so a future edit that drops or mis-keys an
    entry fails loudly here instead of only at first real use.
    """
    factory = model_factory._resolve_model_adapter_factory(provider)
    assert isinstance(factory, model_factory._BuiltinAdapterFactory)
    assert factory.builder is getattr(model_factory, builder_name)


def test_builtin_adapter_builders_table_has_no_stale_or_missing_entries():
    """The explicit registration table names exactly the module's ``_build_*_model`` set."""
    declared = {
        name[len("_build_") : -len("_model")]
        for name in vars(model_factory)
        if name.startswith("_build_") and name.endswith("_model")
    }
    assert set(model_factory._BUILTIN_ADAPTER_BUILDERS) == declared


def test_unknown_provider_fails_closed_with_configuration_error():
    """No registry entry and no built-in builder -> a typed, fail-closed error."""
    with pytest.raises(model_factory.ModelAdapterConfigurationError):
        model_factory._resolve_model_adapter_factory("not-a-real-provider")


def test_registry_injected_factory_wins_over_the_builtin():
    """An explicitly injected/registered adapter factory takes precedence over the
    built-in table entry for the SAME provider name -- the registry is the
    override seam, the table is only the default."""

    sentinel_calls: list[object] = []

    def _injected(request):
        sentinel_calls.append(request)
        return "sentinel-model"

    factory = model_factory._resolve_model_adapter_factory("openai", injected=_injected)
    assert factory is _injected

    # And the registry now remembers it for a later call with no injection.
    factory_again = model_factory._resolve_model_adapter_factory("openai")
    assert factory_again is _injected


def test_registry_injected_factory_for_a_provider_with_no_builtin_builder():
    """A caller may register a brand-new provider that has no built-in
    ``_build_<provider>_model`` at all -- the registry seam, not the static
    table, is what makes the factory provider-extensible."""

    def _injected(request):
        return "sentinel-model"

    factory = model_factory._resolve_model_adapter_factory(
        "totally-custom-provider", injected=_injected
    )
    assert factory is _injected
