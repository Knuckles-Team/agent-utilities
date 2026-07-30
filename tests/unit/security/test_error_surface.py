"""Regression tests for privacy-safe external exception boundaries."""

from __future__ import annotations

import json

import pytest

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData
from agent_utilities.gateway.widgets.base import BaseWidget
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor
from agent_utilities.security.error_surface import (
    public_error_json,
    public_error_payload,
    public_error_text,
    resolve_error_detail,
    resolve_error_detail_for_actor,
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


# ---------------------------------------------------------------------------
# N2 regression — every error envelope must carry enough to diagnose it
# (error_class/failing_layer + a resolvable detail_ref), never the previous
# `{"code":"operation_failed","retryable":false,"correlation_id":...,
# "detail_ref":null}` with zero further signal.
# ---------------------------------------------------------------------------


def test_no_error_envelope_has_a_null_detail_ref_and_no_error_class() -> None:
    """The core N2 invariant: a null `detail_ref` used to be the ONLY shape —
    meaning every failure was equally undiagnosable. Now `error_class` is
    always populated, and `detail_ref` is never null (it is the same,
    genuinely resolvable, correlation identifier)."""

    for code in (
        "operation_failed",
        "invalid_request",
        "dependency_unavailable",
        "permission_denied",
        "unknown-code",
    ):
        payload = public_error_payload(RuntimeError("boom"), code=code)
        error_class = payload.get("error_class")
        detail_ref = payload["error"]["detail_ref"]
        assert not (detail_ref is None and not error_class), (
            f"code={code!r} produced a null detail_ref with no error_class"
        )
        assert error_class == "RuntimeError"
        assert detail_ref is not None


def test_public_error_payload_reports_error_class_and_failing_layer() -> None:
    def _raise_from_this_module() -> None:
        raise ValueError("synthetic failure")

    try:
        _raise_from_this_module()
    except ValueError as exc:
        payload = public_error_payload(exc)

    assert payload["error_class"] == "ValueError"
    # This test module is not under agent_utilities/epistemic_graph, so the
    # inferred layer degrades to "unknown" rather than fabricating one —
    # still non-null and present, which is the actual N2 requirement.
    assert payload["failing_layer"]


def test_detail_ref_equals_correlation_id_and_resolves_a_sanitized_detail() -> None:
    sensitive = "Bearer sk-proj-abcdefghijklmnopqrstuvwxyz012345"

    try:
        raise RuntimeError(sensitive)
    except RuntimeError as exc:
        payload = public_error_payload(exc)

    correlation_id = payload["error"]["correlation_id"]
    detail_ref = payload["error"]["detail_ref"]
    assert detail_ref == correlation_id

    detail = resolve_error_detail(detail_ref)
    assert detail is not None
    assert detail["error_class"] == "RuntimeError"
    assert detail["failing_layer"]
    # The stored detail is sanitized through the SAME persistence privacy
    # gate as any other payload crossing this boundary — no secret material
    # leaks into the resolvable detail store either.
    assert "sk-proj-abcdefghijklmnopqrstuvwxyz012345" not in json.dumps(detail)
    assert "abcdefghijklmnopqrstuvwxyz012345" not in json.dumps(detail)


def test_resolve_error_detail_degrades_cleanly_for_an_unknown_reference() -> None:
    assert resolve_error_detail("correlation:does-not-exist") is None
    assert resolve_error_detail("") is None


def test_public_payload_never_carries_a_message_field_the_detail_store_does() -> None:
    """N2's `error_class`/`failing_layer` diagnostics deliberately do NOT
    include a `message` field on the public payload — the existing exception-
    surface contract (`test_exception_surface_hardening.py` et al.) already
    requires the public envelope to never echo exception text, and free-form
    application text (a query string, a target name, validation detail) is
    not reliably caught by the persistence-privacy pattern list, which targets
    known secret/PII/location shapes rather than arbitrary text. The full
    (sanitized) detail is still available, but only via `resolve_error_detail`
    — never embedded automatically into every response."""

    sensitive = "https://identity:credential@private.invalid/local/path"
    payload = public_error_payload(RuntimeError(sensitive))

    assert "message" not in payload
    assert sensitive not in json.dumps(payload)


# ---------------------------------------------------------------------------
# D-23/D-24 — a caller holding a `detail_ref` can actually resolve it, scoped
# to its own tenant authority (fail closed), through the same tenancy model
# already enforced elsewhere (agent_utilities.knowledge_graph.core.secured_reads).
# ---------------------------------------------------------------------------


def _tenant_actor(tenant_id: str) -> ActorContext:
    return ActorContext(
        actor_id=f"agent:{tenant_id}",
        actor_type=ActorType.AI_AGENT,
        roles=("member",),
        tenant_id=tenant_id,
        authenticated=True,
    )


def test_resolve_error_detail_for_actor_round_trips_within_the_same_tenant() -> None:
    with use_actor(_tenant_actor("acme")):
        try:
            raise RuntimeError("boom in acme")
        except RuntimeError as exc:
            payload = public_error_payload(exc)
        detail_ref = payload["error"]["detail_ref"]
        detail = resolve_error_detail_for_actor(detail_ref)

    assert detail is not None
    assert detail["error_class"] == "RuntimeError"
    assert detail["failing_layer"]


def test_resolve_error_detail_for_actor_refuses_a_different_tenant() -> None:
    with use_actor(_tenant_actor("acme")):
        try:
            raise RuntimeError("boom in acme")
        except RuntimeError as exc:
            payload = public_error_payload(exc)
    detail_ref = payload["error"]["detail_ref"]

    with use_actor(_tenant_actor("globex")):
        # Refused — and deliberately indistinguishable from an unknown ref,
        # so a foreign-tenant caller cannot even confirm the ref exists.
        assert resolve_error_detail_for_actor(detail_ref) is None


def test_resolve_error_detail_for_actor_fails_closed_without_verified_authority() -> (
    None
):
    unauthenticated = ActorContext(
        actor_id="anon", tenant_id="acme", authenticated=False
    )
    with pytest.raises(PermissionError):
        resolve_error_detail_for_actor(
            "correlation:does-not-exist", actor=unauthenticated
        )

    no_tenant = ActorContext(actor_id="agent:x", tenant_id="", authenticated=True)
    with pytest.raises(PermissionError):
        resolve_error_detail_for_actor("correlation:does-not-exist", actor=no_tenant)


def test_resolve_error_detail_for_actor_fails_closed_with_no_bound_actor_at_all(
    monkeypatch,
) -> None:
    """Every served (MCP) call path always has an actor bound by the time a
    tool runs (``ActorContextMiddleware`` / ``verified_tool_session_scope``),
    so this state is unreachable through the served surface. It is still the
    underlying fail-closed contract :func:`current_actor` provides, and worth
    covering directly rather than only through actor-override cases."""
    from agent_utilities.security import brain_context, error_surface

    def _raise_identity_required():
        raise brain_context.IdentityRequiredError("no actor bound")

    monkeypatch.setattr(error_surface, "current_actor", _raise_identity_required)
    with pytest.raises(PermissionError):
        resolve_error_detail_for_actor("correlation:does-not-exist")


def test_resolve_error_detail_for_actor_unknown_ref_returns_none_not_raise() -> None:
    with use_actor(_tenant_actor("acme")):
        assert resolve_error_detail_for_actor("correlation:does-not-exist") is None


def test_resolve_error_detail_for_actor_applies_the_same_sanitization() -> None:
    sensitive = "Bearer sk-proj-abcdefghijklmnopqrstuvwxyz012345"
    with use_actor(_tenant_actor("acme")):
        try:
            raise RuntimeError(sensitive)
        except RuntimeError as exc:
            payload = public_error_payload(exc)
        detail = resolve_error_detail_for_actor(payload["error"]["detail_ref"])

    assert detail is not None
    assert "sk-proj-abcdefghijklmnopqrstuvwxyz012345" not in json.dumps(detail)


def test_detail_persistence_sink_is_called_best_effort_and_never_breaks_recording() -> (
    None
):
    """The D-24 durability seam: an optional sink observes every recorded
    detail, but a broken sink must never turn diagnostics into a second
    failure for the caller whose original exception is being reported."""
    from agent_utilities.security import error_surface

    calls: list[tuple[str, dict]] = []
    error_surface.register_detail_persistence_sink(
        lambda cid, record: calls.append((cid, record))
    )
    try:
        payload = public_error_payload(RuntimeError("sink test"))
    finally:
        error_surface.register_detail_persistence_sink(None)
    assert calls
    assert calls[0][0] == payload["error"]["detail_ref"]
    assert calls[0][1]["error_class"] == "RuntimeError"

    def _broken_sink(_cid: str, _record: dict) -> None:
        raise RuntimeError("sink is down")

    error_surface.register_detail_persistence_sink(_broken_sink)
    try:
        payload2 = public_error_payload(RuntimeError("sink test 2"))
    finally:
        error_surface.register_detail_persistence_sink(None)
    assert payload2["error"]["detail_ref"]
