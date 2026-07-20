"""Contracts for the deterministic native-source capability generator."""

from __future__ import annotations

import base64
import importlib.util
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "generate_native_connector_manifest",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "generate_native_connector_manifest.py",
)
gen = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gen)

_NOW = datetime(2026, 7, 14, tzinfo=UTC)


@pytest.fixture(autouse=True)
def release_signing_key(monkeypatch):
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )


def test_native_manifest_inventory_and_code_pins_are_deterministic():
    first, first_pins = gen.build_manifest(now=_NOW)
    second, second_pins = gen.build_manifest(now=_NOW)

    expected = {
        "ard",
        "database",
        "external_graph",
        "filesystem",
        "graphql_document",
        "reader",
        "rest",
        "rss",
        "web",
    }
    assert set(first_pins) == expected
    assert {sync.tool for sync in first.sync} == expected
    assert all(sync.tool_schema_sha256 == first_pins[sync.tool] for sync in first.sync)
    external_graph = next(sync for sync in first.sync if sync.tool == "external_graph")
    assert external_graph.action == "ingest"
    assert external_graph.raw == {
        "source_type": "external_graph",
        "interface": "external_graph_ingestion_v1",
    }
    assert external_graph.id_field is None
    assert first_pins == second_pins
    assert gen._manifest_yaml(first) == gen._manifest_yaml(second)


def test_native_fingerprint_sidecar_declares_closure_format():
    encoded = json.loads(gen._fingerprints_json({"external_graph": "1" * 64}))

    assert encoded == {
        "format": gen.NATIVE_FINGERPRINT_FORMAT,
        "sources": {"external_graph": "1" * 64},
    }


def test_native_manifest_requires_explicit_release_key(monkeypatch):
    monkeypatch.delenv("ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF", raising=False)

    with pytest.raises(gen.ontology_integrity.ReleaseSigningError):
        gen.build_manifest(now=_NOW)
