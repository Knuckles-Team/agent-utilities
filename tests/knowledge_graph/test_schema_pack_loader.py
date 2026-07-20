from __future__ import annotations

"""Tests for Schema-Pack active-profile resolution & lifecycle.

CONCEPT:AU-KG.ontology.schema-pack-lifecycle-audit — Schema-Pack Lifecycle, Loader & Audit
"""


from types import SimpleNamespace

import pytest

from agent_utilities.models import schema_pack_loader as loader


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(loader, "config", SimpleNamespace(graph_schema_pack="core"))
    # Reset module state.
    loader._active_pack = None
    loader._listeners.clear()
    yield
    loader._active_pack = None
    loader._listeners.clear()


def test_default_is_core():
    assert loader.resolve_pack_name() == "core"
    assert loader.resolve_active_pack().name == "core"


def test_agent_config_selects_pack():
    loader.config.graph_schema_pack = "research-state"
    assert loader.resolve_pack_name() == "research-state"
    assert loader.resolve_active_pack().name == "research-state"


def test_unknown_configured_pack_fails_closed():
    loader.config.graph_schema_pack = "does-not-exist"
    with pytest.raises(KeyError):
        loader.resolve_active_pack()


def test_set_active_pack_notifies_listeners():
    seen = {}

    def _cb(pack):
        seen["name"] = pack.name

    loader.register_listener(_cb)
    pack = loader.set_active_pack("research-state")
    assert pack.name == "research-state"
    assert seen["name"] == "research-state"
    assert loader.get_active_pack().name == "research-state"


def test_listener_failure_does_not_abort_switch():
    def _bad(_pack):
        raise RuntimeError("boom")

    loader.register_listener(_bad)
    pack = loader.set_active_pack("finance")  # must not raise
    assert pack.name == "finance"
