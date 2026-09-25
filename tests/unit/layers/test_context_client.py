"""The standalone worker accepts only GraphOS's caller-authorized export."""

from __future__ import annotations

import pytest

from agent_utilities.layers.context_client import (
    ContextExportUnavailable,
    GraphOSContextClient,
    parse_context_export,
)


def _export() -> dict:
    url = "https://graphos.example/mcp"
    return {
        "ok": True,
        "meta": {"registry_digest": "a" * 64, "api_version": "v1"},
        "result": {
            "endpoint": {
                "name": "epistemic-graph-context",
                "url": url,
                "transport": "http",
                "bearer_ref": "env://GRAPHOS_BEARER",
            },
            "proof": {
                "endpoint_url": url,
                "registry_digest": "a" * 64,
                "tools": ["ask", "find"],
                "operations": ["context.view", "query.uql"],
            },
        },
    }


def test_context_client_fetches_authorized_export() -> None:
    calls: list[tuple[str, str]] = []

    def post(url: str, token: str) -> dict:
        calls.append((url, token))
        return _export()

    endpoint = GraphOSContextClient(
        "https://graphos.example", lambda: "private-token", post
    ).fetch()
    assert endpoint.url == "https://graphos.example/mcp"
    assert calls == [
        ("https://graphos.example/api/v1/ops/harness.context_endpoint", "private-token")
    ]


@pytest.mark.parametrize("missing", ["endpoint", "proof", "tools", "operations"])
def test_context_client_refuses_incomplete_proof(missing: str) -> None:
    payload = _export()
    if missing in {"endpoint", "proof"}:
        payload["result"].pop(missing)
    else:
        payload["result"]["proof"].pop(missing)
    with pytest.raises(ContextExportUnavailable):
        parse_context_export(payload)


def test_context_client_refuses_untrusted_http_api() -> None:
    with pytest.raises(ContextExportUnavailable, match="HTTPS"):
        GraphOSContextClient("http://graphos.example", lambda: "token").fetch()


def test_context_client_refuses_registry_digest_mismatch() -> None:
    payload = _export()
    payload["meta"]["registry_digest"] = "b" * 64
    with pytest.raises(ContextExportUnavailable, match="incomplete"):
        parse_context_export(payload)
