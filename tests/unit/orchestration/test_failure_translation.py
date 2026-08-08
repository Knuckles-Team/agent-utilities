"""Tests for the error-translation registry (CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency).

Pins :func:`translate_failure`/:func:`build_failure_detail` against the concrete raw error
strings this table was built from — the fleet HTTPS gate (``agent_runner._fleet_server_url``),
the multiplexer's child-tool-error re-raise (``mcp/multiplexer.py``), the retrieval quality
gate (``knowledge_graph/retrieval/retrieval_quality.py``), and a handful of synthesized
messages the router itself composes (reply-budget timeout). Also proves the two hard
requirements: an unmatched error NEVER collapses to a bare "failure", and the raw text is
privacy-sanitized before storage.
"""

from __future__ import annotations

import pytest

from agent_utilities.orchestration.failure_translation import (
    FailureTranslation,
    build_failure_detail,
    translate_failure,
)

# ── Known signatures -> expected category (each entry is a REAL raw error string this
#    registry was built to translate; see failure_translation.py's module docstring for the
#    concrete source of each). ─────────────────────────────────────────────────────────────
_KNOWN_SIGNATURES: tuple[tuple[str, str], ...] = (
    (
        "RuntimeError: fleet MCP endpoint requires HTTPS outside loopback",
        "fleet_https_gate",
    ),
    (
        "ValueError: JWKS transport requires HTTPS outside loopback",
        "fleet_https_gate",
    ),
    (
        "RuntimeError: focused-tools delegation requires FLEET_MCP_URL_TEMPLATE",
        "fleet_config_missing",
    ),
    ("RuntimeError: fleet MCP endpoint is invalid", "fleet_endpoint_invalid"),
    ("ToolError: delegated_child_tool_failed", "fleet_tool_error"),
    (
        "RuntimeError: single-server agent 'github-mcp' exceeded the 300s "
        "wall-clock budget — a bound tool likely blocked",
        "tool_wall_clock_timeout",
    ),
    (
        "RuntimeError: agent 'x' resolved to a single MCP server but has no "
        "bound toolset to invoke",
        "toolset_bind_failed",
    ),
    (
        "Retrieval quality gate FAILED for query 'x' (composite=0.00, "
        "modes=['low_relevance_topk'])",
        "retrieval_quality",
    ),
    (
        "TimeoutError: the run did not finish inside the 45s reply budget",
        "reply_budget_timeout",
    ),
    ("EngineUnreachableError: engine tenants.list() failed", "engine_unreachable"),
    ("ConnectionRefusedError: [Errno 111] Connection refused", "engine_unreachable"),
    ("no KG backend active", "engine_unreachable"),
    (
        "PermissionBootstrapError: permission signing key reference is required "
        "for governed execution",
        "permission_bootstrap",
    ),
    (
        "PermissionBootstrapError: permission context bootstrap failed",
        "permission_bootstrap",
    ),
    (
        "PermissionError: recursive native GraphOS delegation is forbidden",
        "access_denied",
    ),
    ("ToolError: Access denied: component is disabled", "access_denied"),
    ("Security: prompt injection detected (confidence=0.91)", "security_blocked"),
)


@pytest.mark.parametrize("raw,expected_category", _KNOWN_SIGNATURES)
def test_translate_failure_known_signatures(raw: str, expected_category: str) -> None:
    result = translate_failure(raw)
    assert isinstance(result, FailureTranslation)
    assert result.category == expected_category
    # Never a bare/empty translation or hint for a KNOWN signature.
    assert result.translated and len(result.translated) > 10
    assert result.hint and len(result.hint) > 5


def test_translate_failure_is_case_insensitive() -> None:
    lower = translate_failure(
        "runtimeerror: fleet mcp endpoint requires https outside loopback"
    )
    upper = translate_failure(
        "RUNTIMEERROR: FLEET MCP ENDPOINT REQUIRES HTTPS OUTSIDE LOOPBACK"
    )
    assert lower.category == upper.category == "fleet_https_gate"


def test_translate_failure_accepts_a_bare_exception_instance() -> None:
    result = translate_failure(
        RuntimeError("fleet MCP endpoint requires HTTPS outside loopback")
    )
    assert result.category == "fleet_https_gate"


def test_translate_failure_first_match_wins_specific_over_generic() -> None:
    """The wall-clock-timeout signature must win over any generic 'timeout' shadowing,
    since ordering places specific markers before generic ones."""
    result = translate_failure(
        "RuntimeError: single-server agent 'x' exceeded the 300s wall-clock budget"
    )
    assert result.category == "tool_wall_clock_timeout"


# ── The "never a bare failure" contract ──────────────────────────────────────────────────


def test_unmatched_error_falls_back_but_keeps_the_raw_tail() -> None:
    result = translate_failure("Zorblatt exploded on line 42 of the flux capacitor")
    assert result.category == "unknown"
    assert "Zorblatt exploded" in result.translated
    assert result.hint


def test_empty_or_none_error_never_returns_a_bare_failure_string() -> None:
    for raw in (None, "", "   "):
        result = translate_failure(raw)
        assert result.category == "unknown"
        assert result.translated
        assert result.translated.strip().lower() != "failure"
        assert "error detail captured" in result.translated.lower()


def test_translate_failure_never_raises_on_junk_input() -> None:
    for junk in (123, object(), ["a", "list"], {"k": "v"}):
        result = translate_failure(junk)
        assert isinstance(result, FailureTranslation)
        assert result.translated


# ── build_failure_detail: the ready-to-store {raw, translated, category, hint} dict ───────


def test_build_failure_detail_shape() -> None:
    detail = build_failure_detail("delegated_child_tool_failed")
    assert set(detail) == {"raw", "translated", "category", "hint"}
    assert detail["category"] == "fleet_tool_error"
    assert detail["raw"] == "delegated_child_tool_failed"
    assert all(isinstance(v, str) for v in detail.values())


def test_build_failure_detail_sanitizes_endpoints_and_paths_in_raw() -> None:
    """CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — this failure detail can
    ride all the way out to an external chat surface, so the raw text goes through the SAME
    endpoint/path redaction as the internal log-privacy boundary before it is stored."""
    raw = (
        "ConnectionError: could not reach https://internal-secret-host.arpa:8443/v1/x "
        "(config at /home/genius/.config/agent-utilities/secrets.json)"
    )
    detail = build_failure_detail(raw)
    assert "internal-secret-host" not in detail["raw"]
    assert "/home/genius" not in detail["raw"]
    assert "<endpoint>" in detail["raw"] or "<path>" in detail["raw"]


def test_build_failure_detail_caps_raw_length() -> None:
    detail = build_failure_detail("x" * 5000)
    assert len(detail["raw"]) <= 2000


def test_build_failure_detail_never_raises_on_junk() -> None:
    for junk in (None, 123, object()):
        detail = build_failure_detail(junk)
        assert set(detail) == {"raw", "translated", "category", "hint"}
