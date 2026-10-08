"""Ontology providers are served as MCP resources (spec: baseline-ingestion)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastmcp import Client, FastMCP

from agent_utilities.mcp.content_resources import register_ontology_providers

_RESOLVER = "agent_utilities.core.providers.resolve_ontology_provider_dirs"


def _ontology(root: Path, name: str) -> Path:
    directory = root / name / "ontology"
    (directory / "shapes").mkdir(parents=True)
    (directory / f"{name}.ttl").write_text("@prefix ex: <urn:ex#> .\n")
    (directory / "shapes" / "connector.shacl.ttl").write_text(
        "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
    )
    (directory / "certification.json").write_text("{}")
    return directory


async def test_every_provider_serves_ontology_and_shapes(tmp_path: Path) -> None:
    providers = [
        ("demo-mcp", _ontology(tmp_path, "demo")),
        ("other-mcp", _ontology(tmp_path, "other")),
    ]
    mcp: FastMCP[Any] = FastMCP("fleet")
    with patch(_RESOLVER, return_value=providers):
        assert register_ontology_providers(mcp) == 4
    async with Client(mcp) as client:
        uris = {str(resource.uri) for resource in await client.list_resources()}
        body = await client.read_resource("ontology://demo-mcp/demo.ttl")
    assert uris == {
        "ontology://demo-mcp/demo.ttl",
        "shapes://demo-mcp/connector.shacl.ttl",
        "ontology://other-mcp/other.ttl",
        "shapes://other-mcp/connector.shacl.ttl",
    }
    assert "urn:ex#" in body[0].text


def test_one_bad_provider_does_not_sink_the_rest(tmp_path: Path) -> None:
    providers = [
        ("bad-mcp", tmp_path / "bad"),
        ("demo-mcp", _ontology(tmp_path, "demo")),
    ]
    mcp: FastMCP[Any] = FastMCP("fleet")
    calls: list[str] = []

    def _register(server: Any, name: str, directory: Path) -> int:
        calls.append(name)
        if name == "bad-mcp":
            raise OSError("unreadable")
        return 2

    with (
        patch(_RESOLVER, return_value=providers),
        patch(
            "agent_connector_sdk.mcp.content.register_ontology_resources",
            side_effect=_register,
        ),
    ):
        assert register_ontology_providers(mcp) == 2
    assert calls == ["bad-mcp", "demo-mcp"]


def test_resolver_failure_degrades_to_zero() -> None:
    with patch(_RESOLVER, side_effect=RuntimeError("registry down")):
        assert register_ontology_providers(FastMCP("fleet")) == 0
