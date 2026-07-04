"""CONCEPT:AU-ORCH.adapter.byok-provider-proxy — Provider-Normalizing Stream Proxy + SSRF guard + credential resolver.

Pure-unit coverage of the SSRF egress gate (DNS-resolved), three-tier credential resolution, and
canonical stream normalization across providers; plus a live-path test that the FastAPI route rejects
an internal-IP base_url *before* any upstream fetch.
"""

from __future__ import annotations

import pytest

from agent_utilities.core.credentials import CredentialResolver
from agent_utilities.core.execution.adapters.base import ExecEventType
from agent_utilities.core.execution.provider_proxy import (
    check_egress,
    normalize_chunk,
    normalize_stream,
)
from agent_utilities.security.egress import (
    validate_base_url,
    validate_base_url_resolved,
)

pytestmark = pytest.mark.concept(id="AU-ORCH.adapter.byok-provider-proxy")


# ── SSRF egress guard ───────────────────────────────────────────────


def test_ip_literal_private_blocked():
    assert validate_base_url("http://10.0.0.5/v1").allowed is False
    assert validate_base_url("http://169.254.169.254/latest/meta-data").allowed is False


def test_loopback_carve_out():
    assert (
        validate_base_url("http://127.0.0.1:11434/v1", allow_loopback=True).allowed
        is True
    )
    assert (
        validate_base_url("http://127.0.0.1:11434/v1", allow_loopback=False).allowed
        is False
    )


def test_bad_scheme_blocked():
    assert validate_base_url("ftp://example.com").allowed is False
    assert validate_base_url("file:///etc/passwd").allowed is False


def test_public_dns_to_private_ip_blocked():
    # Public hostname that resolves to a private IP — the SSRF vector hostname checks miss.
    def fake_resolver(host, _port=None):
        return [(2, 1, 6, "", ("10.1.2.3", 0))]

    d = validate_base_url_resolved(
        "https://evil.example.com/v1", resolver=fake_resolver
    )
    assert d.allowed is False
    assert "blocked IP" in d.reason


def test_public_dns_to_public_ip_allowed():
    def fake_resolver(host, _port=None):
        return [(2, 1, 6, "", ("93.184.216.34", 0))]

    d = validate_base_url_resolved("https://api.openai.com/v1", resolver=fake_resolver)
    assert d.allowed is True


def test_check_egress_no_custom_url_allows():
    assert check_egress(None).allowed is True


# ── credential resolver (env > file > none) ─────────────────────────


def test_credentials_env_wins(tmp_path):
    cfg = tmp_path / "media-config.json"
    cfg.write_text('{"openai": {"api_key": "from-file"}}')
    r = CredentialResolver(env={"OPENAI_API_KEY": "from-env"}, config_path=cfg)
    res = r.resolve("openai")
    assert res.api_key == "from-env"
    assert res.source == "env"


def test_credentials_file_when_no_env(tmp_path):
    cfg = tmp_path / "media-config.json"
    cfg.write_text('{"openai": {"api_key": "from-file", "base_url": "https://x/v1"}}')
    r = CredentialResolver(env={}, config_path=cfg)
    res = r.resolve("openai")
    assert res.api_key == "from-file"
    assert res.base_url == "https://x/v1"
    assert res.source == "file"


def test_credentials_none(tmp_path):
    r = CredentialResolver(env={}, config_path=tmp_path / "missing.json")
    res = r.resolve("openai")
    assert res.api_key is None and res.source == "none"


# ── stream normalization ────────────────────────────────────────────


def test_normalize_openai_delta():
    evs = normalize_chunk("openai", 'data: {"choices":[{"delta":{"content":"hi"}}]}')
    assert any(e.type is ExecEventType.TEXT_DELTA and e.text == "hi" for e in evs)


def test_normalize_openai_done_sentinel():
    evs = normalize_chunk("openai", "data: [DONE]")
    assert evs and evs[0].type is ExecEventType.END


def test_normalize_anthropic_delta():
    evs = normalize_chunk(
        "anthropic", 'data: {"type":"content_block_delta","delta":{"text":"yo"}}'
    )
    assert any(e.type is ExecEventType.TEXT_DELTA and e.text == "yo" for e in evs)


def test_normalize_google_parts():
    evs = normalize_chunk(
        "google", '{"candidates":[{"content":{"parts":[{"text":"g"}]}}]}'
    )
    assert any(e.type is ExecEventType.TEXT_DELTA and e.text == "g" for e in evs)


def test_normalize_stream_brackets_with_start_end():
    lines = ['data: {"choices":[{"delta":{"content":"a"}}]}', "data: [DONE]"]
    evs = list(normalize_stream("openai", lines))
    assert evs[0].type is ExecEventType.START
    assert evs[-1].type is ExecEventType.END
    assert any(e.type is ExecEventType.TEXT_DELTA for e in evs)


def test_malformed_chunk_skipped_not_raised():
    assert normalize_chunk("openai", "data: not-json{{{") == []
    assert normalize_chunk("openai", ": comment") == []


# ── live-path: route rejects SSRF before fetch ──────────────────────


def test_proxy_route_rejects_internal_ip():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = fastapi.FastAPI()
    from agent_utilities.server.routers import proxy

    app.include_router(proxy.router)
    client = TestClient(app)
    resp = client.post(
        "/api/proxy/openai/stream",
        json={"base_url": "http://10.0.0.9/v1", "model": "x", "messages": []},
    )
    assert resp.status_code == 400
    assert "blocked" in resp.json()["error"]


def test_proxy_route_unsupported_provider():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = fastapi.FastAPI()
    from agent_utilities.server.routers import proxy

    app.include_router(proxy.router)
    client = TestClient(app)
    resp = client.post("/api/proxy/bogus/stream", json={"model": "x", "messages": []})
    assert resp.status_code == 400
