"""Tests for the connector-fleet expansion widgets (widget-connector-expansion).

Covers the 20 new ``agent_utilities/gateway/widgets/*.py`` modules added to
close the gap between ``agent-packages/agents/*`` connectors and dashboard
widgets: registry registration, field metadata, the guarded-import
fail-closed contract (a widget whose backing agent-package is not installed
must degrade to ``status="skipped"`` and never raise out of
``fetch_data``/``_safe_fetch``), and the shared ``_optional_client`` helpers.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_utilities.gateway.models import ServiceConfig
from agent_utilities.gateway.registry import Registry
from agent_utilities.gateway.widgets._optional_client import count_items, import_client

NEW_WIDGET_TYPES = [
    "aris",
    "audiobookshelf",
    "camunda",
    "ciso_assistant",
    "clarity",
    "dockerhub",
    "egeria",
    "fan_manager",
    "firefly_iii",
    "freshrss",
    "gramps",
    "hdhomerun",
    "jena",
    "kafka",
    "leanix",
    "okta",
    "onetrust",
    "paperless_ngx",
    "pulselink",
    "rom_manager",
]


def _config(widget_type: str) -> ServiceConfig:
    return ServiceConfig(
        id=widget_type,
        name=widget_type,
        widget_type=widget_type,
        url="https://example.invalid",
        api_key="test-token",  # sanitizer:ignore — synthetic fixture, never a real credential
        username="test-user",
        password="test-pass",  # sanitizer:ignore — synthetic fixture, never a real credential
    )


# --- registry registration --------------------------------------------------


def test_all_new_widgets_are_known_to_the_registry():
    registry = Registry()
    known = registry.list_all_known()
    for widget_type in NEW_WIDGET_TYPES:
        assert widget_type in known


def test_all_new_widgets_load_and_register_metadata():
    """Registration must succeed even though no sibling package is installed:

    the widget module itself must not import the optional agent-package at
    module scope (only lazily, inside fetch_data), or every widget would be
    unregistered in an environment where the fleet isn't installed — exactly
    the failure mode this expansion is meant to avoid.
    """
    registry = Registry()
    available = registry.list_available()
    for widget_type in NEW_WIDGET_TYPES:
        assert widget_type in available, f"{widget_type} failed to register"


@pytest.mark.parametrize("widget_type", NEW_WIDGET_TYPES)
def test_widget_field_metadata_is_well_formed(widget_type):
    registry = Registry()
    widget = registry.get_widget(widget_type)
    assert widget is not None

    fields = widget.get_fields()
    assert len(fields) >= 1
    for field in fields:
        assert field.key
        assert field.label

    assert widget.service_type == widget_type
    assert widget.display_name
    assert widget.env_prefix


# --- fail-closed: guarded import degrades instead of raising ---------------


@pytest.mark.parametrize("widget_type", NEW_WIDGET_TYPES)
def test_fetch_data_skips_cleanly_when_dependency_is_absent(widget_type):
    """The backing agent-package genuinely isn't installed in this venv

    (confirmed separately: none of the 68 agent-packages/agents/* packages
    are declared as au dependencies). Every new widget must therefore report
    status="skipped" here, and _safe_fetch must never surface a raised
    ModuleNotFoundError as an unhandled exception or a false "ok".
    """
    registry = Registry()
    widget = registry.get_widget(widget_type)
    config = _config(widget_type)

    data = widget._safe_fetch(config)

    assert data.status == "skipped"
    assert data.error


@pytest.mark.parametrize("widget_type", NEW_WIDGET_TYPES)
def test_fetch_data_never_raises_out_of_safe_fetch(widget_type):
    """_safe_fetch must catch everything, including on a blank config."""
    registry = Registry()
    widget = registry.get_widget(widget_type)
    blank = ServiceConfig(id=widget_type, name=widget_type, widget_type=widget_type)

    data = widget._safe_fetch(blank)

    assert data.status in {"skipped", "error"}


# --- _optional_client helpers -----------------------------------------------


def test_import_client_returns_package_name_when_missing():
    client_cls, missing = import_client(
        "definitely_not_a_real_package.api_client", "Api"
    )
    assert client_cls is None
    assert missing == "definitely-not-a-real-package"


def test_import_client_returns_class_when_present():
    client_cls, missing = import_client(
        "agent_utilities.gateway.widgets.base", "BaseWidget"
    )
    assert missing is None
    assert client_cls.__name__ == "BaseWidget"


@pytest.mark.parametrize(
    ("value", "key", "expected"),
    [
        ([1, 2, 3], "results", 3),
        ({"results": [1, 2]}, "results", 2),
        ({"data": {"results": [1, 2, 3, 4]}}, "results", 4),
        ({"libraries": [1]}, "libraries", 1),
        ({}, "results", 0),
        (None, "results", 0),
        ("not-a-collection", "results", 0),
    ],
)
def test_count_items_handles_common_envelope_shapes(value, key, expected):
    assert count_items(value, key=key) == expected


def test_count_items_reads_an_attribute_on_a_model_like_object():
    payload = types.SimpleNamespace(results=[1, 2, 3])
    assert count_items(payload) == 3


def test_count_items_is_bounded_and_never_raises_on_cyclic_input():
    cyclic: dict = {"data": {}}
    cyclic["data"]["data"] = cyclic
    assert count_items(cyclic) == 0


# --- representative "ok" paths via injected fake sibling modules -----------


@pytest.fixture
def fake_module(monkeypatch):
    """Register a throwaway module in sys.modules for the duration of a test."""

    created: list[str] = []

    def _make(module_path: str, **attrs):
        module = types.ModuleType(module_path)
        for name, value in attrs.items():
            setattr(module, name, value)
        monkeypatch.setitem(sys.modules, module_path, module)
        created.append(module_path)
        return module

    yield _make


def test_aris_widget_ok_path_counts_models(fake_module):
    calls = {}

    class _FakeArisApi:
        def __init__(self, base_url, token, verify):
            calls["base_url"] = base_url
            calls["token"] = token

        def list_models(self):
            return [{"id": "m1"}, {"id": "m2"}]

    fake_module("aris_mcp.api_client", ArisApi=_FakeArisApi)

    from agent_utilities.gateway.widgets.aris import Widget

    data = Widget().fetch_data(_config("aris"))

    assert data.status == "ok"
    assert data.fields["models"] == 2
    assert calls["base_url"] == "https://example.invalid"
    assert calls["token"] == "test-token"


def test_fan_manager_widget_ok_path_reports_temperature(fake_module):
    class _FakeApi:
        def get_temp(self):
            return {"response": 42.5, "command": "sensors -j", "status": 200}

    fake_module("fan_manager.api_client", Api=_FakeApi)

    from agent_utilities.gateway.widgets.fan_manager import Widget

    data = Widget().fetch_data(_config("fan_manager"))

    assert data.status == "ok"
    assert data.fields["core_temp_c"] == 42.5


def test_fan_manager_widget_reports_error_on_a_failed_sensor_read(fake_module):
    """A degraded-but-non-raising failure from the local command must still

    surface as status="error", not a false "ok" with a zeroed field — the
    fail-closed contract this expansion is required to uphold.
    """

    class _FakeApi:
        def get_temp(self):
            return {
                "response": None,
                "command": "sensors -j",
                "status": 500,
                "error": "x",
            }

    fake_module("fan_manager.api_client", Api=_FakeApi)

    from agent_utilities.gateway.widgets.fan_manager import Widget

    data = Widget()._safe_fetch(_config("fan_manager"))

    assert data.status == "error"


def test_rom_manager_widget_stat_lookup_is_case_insensitive():
    from agent_utilities.gateway.widgets.rom_manager import _stat

    assert _stat({"PLATFORMS": 7, "ROMS": 321}, "PLATFORMS", "platforms") == 7
    assert _stat({"platforms": 7}, "PLATFORMS", "platforms") == 7
    assert _stat({}, "PLATFORMS", "platforms") == 0


def test_camunda_widget_ok_path_uses_v7_client(fake_module):
    class _FakeCamunda7Api:
        def __init__(self, base_url, token, tls_profile):
            pass

        def list_process_definitions(self):
            return [{"id": "d1"}]

        def list_process_instances(self):
            return [{"id": "i1"}, {"id": "i2"}]

    fake_module("camunda_mcp.api.api_client_camunda7", Camunda7Api=_FakeCamunda7Api)
    fake_module("camunda_mcp.api.api_client_camunda8", Camunda8Api=object)

    # camunda_mcp.api_client.Api composes the two lazily; build a minimal
    # stand-in that mirrors that composition instead of importing the real
    # (unavailable) package.
    class _FakeApi:
        def __init__(self, v7_kwargs=None, v8_kwargs=None):
            self._v7_kwargs = v7_kwargs or {}
            self._v7 = None

        @property
        def v7(self):
            if self._v7 is None:
                self._v7 = _FakeCamunda7Api(**self._v7_kwargs)
            return self._v7

    fake_module("camunda_mcp.api_client", Api=_FakeApi)

    from agent_utilities.gateway.widgets.camunda import Widget

    data = Widget().fetch_data(_config("camunda"))

    assert data.status == "ok"
    assert data.fields["process_definitions"] == 1
    assert data.fields["running_instances"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
