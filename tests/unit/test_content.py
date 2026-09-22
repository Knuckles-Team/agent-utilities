from __future__ import annotations

import hashlib

import pytest
from agent_connector_sdk.mcp.content import ConnectorContent, register_connector_content
from fastmcp import Client, FastMCP

import agent_utilities
from agent_utilities.content import agent_utilities_content

_RESOURCE_RELATIVE_PATH = "ontology/shapes/governance.shapes.ttl"
_RESOURCE_URI = "shapes://agent-utilities/governance.shapes.ttl"
_RESOURCE_SHA256 = "6c0a88f7d60b0569c0c88d379f0026d434eb7fdadb8146f0ab4a0dbab5fc39e6"


def test_agent_utilities_content_uses_the_canonical_sdk_contract() -> None:
    content = agent_utilities_content()

    assert type(content) is ConnectorContent
    assert content.connector == "agent-utilities"
    assert content.package_version == agent_utilities.__version__
    assert content.manifest_path is None

    resource = content.package_root / _RESOURCE_RELATIVE_PATH
    assert resource.is_file(), _RESOURCE_URI
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == _RESOURCE_SHA256


@pytest.mark.anyio
async def test_agent_utilities_content_registers_the_canonical_shape_resource() -> None:
    mcp = FastMCP("agent-utilities-content-test")
    registration = register_connector_content(mcp, agent_utilities_content())

    assert registration.resources == 1
    async with Client(mcp) as client:
        resource_uris = {
            str(resource.uri) for resource in await client.list_resources()
        }
    assert _RESOURCE_URI in resource_uris
