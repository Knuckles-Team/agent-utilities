"""C8-05 import-direction gates for fleet discovery authority."""

from __future__ import annotations

import ast
from pathlib import Path

from agent_utilities.knowledge_graph.core.discovery_authority import OAuthGrantBinding
from agent_utilities.mcp import remote_oauth_broker


def test_broker_does_not_reexport_the_canonical_lower_binding_type():
    assert "OAuthGrantBinding" not in remote_oauth_broker.__all__


def test_canonical_binding_has_no_grant_digest_compatibility_alias():
    assert "grant_digest" not in OAuthGrantBinding.__dict__


def test_fleet_catalog_has_no_upward_mcp_import():
    root = Path(__file__).parents[4]
    path = root / "agent_utilities/knowledge_graph/core/fleet_catalog_tables.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not any("mcp" in module.split(".") for module in imported_modules)
