"""Focused contract tests for Atlas's governed source catalogue."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core import source_catalog


class _Registry:
    def status(self):
        return {
            "connections": [
                {
                    "name": "pg-main",
                    "backend_type": "postgresql",
                    "connected": True,
                },
                {
                    "name": "puppygraph-prod",
                    "backend_type": "opencypher",
                    "connected": True,
                    "probed": True,
                },
                {
                    "name": "neo4j-prod",
                    "backend_type": "neo4j",
                    "connected": True,
                },
            ]
        }

    def export_specs(self):
        return [
            {
                "name": "pg-main",
                "backend_type": "postgresql",
                "connection_profile_ref": "vault://profiles/postgres",
                "tls_profile_ref": "vault://tls/postgres",
                # These values must never be projected by the catalogue.
                "host": "db.internal.example",
                "password": "do-not-project",
            },
            {
                "name": "puppygraph-prod",
                "backend_type": "opencypher",
                "connection_profile_ref": "secret://profiles/puppygraph",
            },
        ]


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _is_sensitive_key(value: str) -> bool:
    """Match sensitive field names without substring false positives.

    ``supported`` contains the substring ``port`` but is a harmless catalogue
    field.  Normalize the public camelCase projection first, then compare
    complete identifier components so this assertion protects the boundary
    without rejecting ordinary words.
    """

    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", value).casefold()
    parts = set(re.split(r"[_-]+", normalized))
    return bool(
        parts & {"endpoint", "dsn", "uri", "host", "port", "user", "password", "secret"}
    )


def test_catalog_projects_registries_with_truthful_provider_states(monkeypatch):
    monkeypatch.setattr(
        source_catalog,
        "_source_connector_types",
        lambda: ("database", "reader"),
    )
    config = SimpleNamespace(
        external_graph_connectors=[
            {
                "name": "puppygraph-prod",
                "source_alias": "puppygraph-prod",
                "backend": "opencypher",
                "connection_profile_ref": "vault://profiles/puppygraph",
            }
        ],
        provider_configs={
            "postgresql": {
                "enabled": True,
                "endpoint_ref": "env://PG_ENDPOINT",
                "tls_profile_ref": "vault://tls/postgres",
            }
        },
    )

    payload = source_catalog.build_source_catalog(config=config, registry=_Registry())
    providers = {item["provider"]: item for item in payload["sources"]}

    assert providers["postgresql"]["availability"] == "available"
    assert providers["postgresql"]["available"] is True
    assert providers["postgresql"]["connectionProfileRef"] == (
        "vault://profiles/postgres"
    )
    assert providers["puppygraph"]["availability"] == "available"
    assert providers["puppygraph"]["queryMode"] == "cypher"
    assert providers["neo4j"]["availability"] == "available"
    assert providers["neo4j"]["connectionNames"] == ["neo4j-prod"]
    assert providers["spark"]["sync"]["supported"] is False
    assert "compute-only" in providers["spark"]["sync"]["reason"]
    assert providers["teradata"]["availability"] == "unsupported"
    assert providers["teradata"]["available"] is False
    assert payload["querySurfaces"] == [
        {
            "id": "natural_language",
            "available": True,
            "tool": "graph_ask",
            "reason": "Graph-OS natural-language planning is exposed by graph_ask/nl_query",
        },
        {
            "id": "uql",
            "available": True,
            "tool": "engine_query",
            "action": "uql",
            "reason": "the engine_query uql action is the governed cross-modal query seam",
        },
    ]

    assert not any(_is_sensitive_key(key) for key in _walk_keys(payload))
    assert "db.internal.example" not in str(payload)
    assert "do-not-project" not in str(payload)


def test_generic_opencypher_requires_a_probe(monkeypatch):
    monkeypatch.setattr(source_catalog, "_source_connector_types", lambda: ())
    config = SimpleNamespace(
        external_graph_connectors=[
            {
                "name": "graph-read",
                "source_alias": "graph-read",
                "backend": "opencypher",
                "connection_profile_ref": "vault://profiles/graph",
            }
        ],
        provider_configs={},
    )

    payload = source_catalog.build_source_catalog(
        config=config,
        registry=SimpleNamespace(
            status=lambda: {"connections": []}, export_specs=lambda: []
        ),
    )
    opencypher = next(
        item for item in payload["sources"] if item["provider"] == "opencypher"
    )
    assert opencypher["availability"] == "unverified"
    assert opencypher["available"] is False
    assert "probe" in opencypher["reason"]


def test_provider_runtime_profile_projects_refs_without_resolving_values(monkeypatch):
    monkeypatch.setattr(source_catalog, "_source_connector_types", lambda: ())
    config = SimpleNamespace(
        external_graph_connectors=[],
        provider_configs={
            "postgres-primary": {
                "enabled": True,
                "endpoint_ref": "env://PG_ENDPOINT",
                "credential_refs": {"password": "secret://profiles/postgres"},
                "tls_profile_ref": "vault://tls/postgres",
                # A literal endpoint would be invalid AgentConfig, and must
                # never be copied if a permissive fixture supplies one.
                "endpoint": "postgresql://db.internal.example:5432/app",
            }
        },
    )

    payload = source_catalog.build_source_catalog(
        config=config,
        registry=SimpleNamespace(
            status=lambda: {"connections": []}, export_specs=lambda: []
        ),
    )
    postgresql = next(
        item for item in payload["sources"] if item["provider"] == "postgresql"
    )

    assert postgresql["availability"] == "configured"
    assert "connectionProfileRef" not in postgresql
    assert postgresql["authProfileRef"] == "secret://profiles/postgres"
    assert postgresql["tlsProfileRef"] == "vault://tls/postgres"
    assert "db.internal.example" not in str(postgresql)


def test_sync_preview_is_single_source_and_non_executable():
    preview = source_catalog.normalize_source_sync_preview(
        source="database",
        mode="full",
        ids_json='["row-1"]',
        connection="pg-main",
        graph="tenant-main",
    )

    assert preview == {
        "schema_version": "atlas-source-sync-preview.v1",
        "source": "database",
        "mode": "full",
        "ids": ["row-1"],
        "connection": "pg-main",
        "graph": "tenant-main",
        "entrypoint": "source_sync",
        "wouldExecute": False,
        "reason": "preview only; execute the normalized request through source_sync",
    }

    with pytest.raises(ValueError, match="exactly one source"):
        source_catalog.normalize_source_sync_preview(source="all")
    with pytest.raises(ValueError, match="mode"):
        source_catalog.normalize_source_sync_preview(source="database", mode="write")
    with pytest.raises(ValueError, match="neutral aliases"):
        source_catalog.normalize_source_sync_preview(
            source="database", connection="vault://profiles/postgres"
        )
