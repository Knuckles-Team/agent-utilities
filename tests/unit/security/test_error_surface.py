"""Regression tests for privacy-safe external exception boundaries."""

from __future__ import annotations

import json

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData
from agent_utilities.gateway.widgets.base import BaseWidget
from agent_utilities.security.error_surface import (
    public_error_json,
    public_error_payload,
    public_error_text,
)


class _FailingWidget(BaseWidget):
    service_type = "test"
    display_name = "Test"
    category = ServiceCategory.CUSTOM

    def get_fields(self):
        return []

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        del config
        raise RuntimeError("credential@private.invalid/local/path")


def test_public_error_forms_never_return_or_log_exception_text(caplog) -> None:
    sensitive = "https://identity:credential@private.invalid/local/path"
    exc = RuntimeError(sensitive)

    payload = public_error_payload(exc, code="not-a-public-code")
    encoded = public_error_json(exc)
    rendered = public_error_text(exc)

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "operation_failed"
    assert json.loads(encoded)["schema_version"] == "1"
    assert sensitive not in json.dumps(payload)
    assert sensitive not in encoded
    assert sensitive not in rendered
    assert sensitive not in caplog.text
    assert "RuntimeError" in caplog.text


def test_widget_failure_is_generic_and_correlation_safe(caplog) -> None:
    widget = _FailingWidget()
    config = ServiceConfig(id="test", name="Test", widget_type="test")

    result = widget._safe_fetch(config)

    assert result.status == "error"
    assert result.error == "A required service is unavailable."
    assert result.raw is not None
    assert result.raw["error"]["code"] == "dependency_unavailable"
    assert result.raw["error"]["correlation_id"].startswith("correlation:")
    assert "private.invalid" not in result.model_dump_json()
    assert "private.invalid" not in caplog.text
