"""Tests for scripts/validate_mcp_config.py (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "validate_mcp_config",
    Path(__file__).resolve().parents[3] / "scripts" / "validate_mcp_config.py",
)
vmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vmc)

CADDY = """
http://github-mcp.example {
    reverse_proxy github-mcp_github-mcp:8000
}

http://container-manager-mcp.example {
    reverse_proxy container-manager-mcp_container-manager-mcp:8000
}

http://lonely-mcp.example {
    reverse_proxy lonely-mcp_lonely-mcp:8000
}
"""


def test_parse_caddy_hosts():
    hosts = vmc.parse_caddy_hosts(CADDY)
    assert hosts["github-mcp.example"] == "github-mcp_github-mcp:8000"
    assert "container-manager-mcp.example" in hosts
    assert "lonely-mcp.example" in hosts


def test_config_url_entries_only_remote():
    config = {
        "mcpServers": {
            "github-mcp": {
                "transport": "streamable-http",
                "url": "http://github-mcp.example/mcp",
            },
            "graph-os": {"command": "/usr/bin/graph-os"},  # stdio — excluded
        }
    }
    entries = vmc.config_url_entries(config)
    assert entries == {"github-mcp": "http://github-mcp.example/mcp"}


def test_validate_all_valid():
    config = {
        "mcpServers": {
            "github-mcp": {"url": "http://github-mcp.example/mcp"},
            "container-manager-mcp": {"url": "http://container-manager-mcp.example/mcp"},
            "graph-os": {"command": "graph-os"},
        }
    }
    # ``managed_suffix`` is an opt-in filter (default "" = coverage-gap checking
    # off, since an unscoped Caddyfile may legitimately route third-party hosts
    # never meant to appear in mcp_config.json); opt in to ".example" to exercise
    # the coverage-gap detection this test is about.
    report = vmc.validate(config, vmc.parse_caddy_hosts(CADDY), managed_suffix=".example")
    assert report["passed"] is True
    assert report["invalid"] == {}
    assert set(report["ok"]) == {"github-mcp", "container-manager-mcp"}
    # lonely-mcp is routed in Caddy but absent from config → coverage gap.
    assert "lonely-mcp.example" in report["missing_from_config"]


def test_validate_flags_invalid_host():
    config = {
        "mcpServers": {
            "github-mcp": {"url": "http://github-mcp.example/mcp"},
            "typo-mcp": {"url": "http://typoo-mcp.example/mcp"},  # not in Caddy
        }
    }
    report = vmc.validate(config, vmc.parse_caddy_hosts(CADDY))
    assert report["passed"] is False
    assert "typo-mcp" in report["invalid"]
    assert "github-mcp" in report["ok"]


def test_validate_live_marks_unreachable(monkeypatch):
    config = {
        "mcpServers": {
            "github-mcp": {"url": "http://github-mcp.example/mcp"},
            "container-manager-mcp": {"url": "http://container-manager-mcp.example/mcp"},
        }
    }

    def fake_probe(url, timeout):
        return (False, "HTTP 502") if "container-manager" in url else (True, "HTTP 200")

    monkeypatch.setattr(vmc, "live_probe", fake_probe)
    report = vmc.validate(config, vmc.parse_caddy_hosts(CADDY), live=True)
    assert report["passed"] is False
    assert "container-manager-mcp" in report["unreachable"]
    assert report["ok"] == ["github-mcp"]


def test_host_of():
    assert vmc._host_of("http://x-mcp.example/mcp") == "x-mcp.example"
    assert vmc._host_of("https://y.example:8443/mcp") == "y.example"
    assert vmc._host_of("z-mcp.example") == "z-mcp.example"
