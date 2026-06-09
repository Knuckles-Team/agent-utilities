from __future__ import annotations

"""Tests for Schema-Pack active-profile resolution & lifecycle.

CONCEPT:KG-2.35 — Schema-Pack Lifecycle, Loader & Audit
"""


import pytest

from agent_utilities.models import schema_pack_loader as loader


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    # Neutralise ambient config so precedence tests are deterministic.
    monkeypatch.delenv("GRAPH_SCHEMA_PACK", raising=False)
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))
    # Reset module state.
    loader._active_pack = None
    loader._listeners.clear()
    yield
    loader._active_pack = None
    loader._listeners.clear()


def test_default_is_core(monkeypatch):
    assert loader.resolve_pack_name() == "core"
    assert loader.resolve_active_pack().name == "core"


def test_env_precedence(monkeypatch):
    monkeypatch.setenv("GRAPH_SCHEMA_PACK", "research-state")
    assert loader.resolve_pack_name() == "research-state"
    assert loader.resolve_active_pack().name == "research-state"


def test_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("GRAPH_SCHEMA_PACK", "research-state")
    assert loader.resolve_pack_name("finance") == "finance"


def test_config_json_precedence(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"graph": {"schema_pack": "biomedical"}}')
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))
    assert loader.resolve_pack_name() == "biomedical"


def test_unknown_pack_falls_back_to_core():
    # Must never raise — a config typo can't take the graph offline.
    assert loader.resolve_active_pack("does-not-exist").name == "core"


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
