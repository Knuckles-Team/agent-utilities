"""Wiring test: a real pack + the real DSL turn a real markdown file's
frontmatter and a real table into graph facts through the LIVE ingest path
(CONCEPT:AU-KG.ingest.domain-pack-framework, CONCEPT:AU-KG.ingest.mapping-dsl).

Nothing in ``domain_packs`` is mocked here: :func:`run_pack` calls the real
``envelope_ingest.ingest_graph_slice`` -> ``ingest_envelope`` ->
``_apply_native_change_envelope``, exercising real identity resolution, real
SHACL/privacy gates, and a real (fake-transport) commit — the same "fake the
wire boundary, run everything above it for real" pattern
``tests/unit/knowledge_graph/ingestion/test_native_envelope_ingest.py`` uses
to test this exact function without a live Rust engine daemon.
"""

from __future__ import annotations

import _fixtures
import pytest

import agent_utilities.knowledge_graph.ingestion.envelope_ingest as envelope_ingest_module
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    reset_session,
    set_session,
)
from agent_utilities.knowledge_graph.domain_packs.envelope_bridge import (
    preview_pack,
    run_pack,
)
from agent_utilities.knowledge_graph.domain_packs.markdown_fragmenter import (
    fragment_markdown_text,
)
from agent_utilities.knowledge_graph.domain_packs.pack_loader import (
    DomainPackError,
    load_pack,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext

RUNBOOK_MD = """---
status: active
owner: alice
---

# Steps

| step | assignee |
| --- | --- |
| Provision VM | bob |

See also [reference](other.md) for background.
"""


# ---------------------------------------------------------------------------
# Minimal native-engine fake — mirrors the pattern in
# tests/unit/knowledge_graph/ingestion/test_native_envelope_ingest.py so this
# test drives the REAL ingest_envelope/ApplyChangeEnvelope code path without a
# live Rust daemon.
# ---------------------------------------------------------------------------


class _Nodes:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, object]] = {}

    def properties(self, node_id: str):
        return self.values.get(node_id)


class _Rdf:
    def __init__(self) -> None:
        self.reports = [{"conforms": True, "results": []}]
        self.validations: list[tuple[str, str]] = []

    def validate_shacl(self, shapes: str, data_graph: str):
        self.validations.append((shapes, data_graph))
        return self.reports[0]


class _Changes:
    def __init__(self, nodes: _Nodes) -> None:
        self.nodes = nodes
        self.records: dict[str, dict[str, object]] = {}
        self.versions: dict[str, dict[str, object]] = {}
        self.cursors: dict[tuple[str, str], dict[str, object]] = {}
        self.applied: list[dict[str, object]] = []

    def get(self, envelope_id: str):
        return self.records.get(envelope_id)

    def content_version(self, object_id: str):
        return self.versions.get(object_id)

    def cursor(self, source: str, partition: str = ""):
        return self.cursors.get((source, partition))

    def apply(self, envelope: dict[str, object]):
        import msgpack

        self.applied.append(envelope)
        mutation = envelope["mutation"]
        for operation in mutation["operations"]:
            method = operation["method"]
            if method["method"] != "AddNode":
                continue
            params = method["params"]
            self.nodes.values[params["node_id"]] = msgpack.unpackb(
                params["properties_msgpack"], raw=False
            )
        version = envelope["content_version"]
        self.versions[str(version["object_id"])] = version
        cursor = envelope.get("cursor")
        if isinstance(cursor, dict):
            self.cursors[(str(cursor["source"]), str(cursor["partition"]))] = cursor
        self.records[str(envelope["envelope_id"])] = envelope
        return {
            "envelope_id": str(envelope["envelope_id"]),
            "batch_id": mutation["batch_id"],
            "replayed": False,
            "projection_pending": False,
            "outbox_count": len(mutation["operations"]) + 2,
        }


class _Client:
    def __init__(self) -> None:
        self.nodes = _Nodes()
        self.changes = _Changes(self.nodes)
        self.rdf = _Rdf()

    def supports(self, operation: str) -> bool:
        return operation in {"ApplyChangeEnvelope", "GetChangeCursor"}


class _Compute:
    def __init__(self, graph: str) -> None:
        self.graph_name = graph
        self.catalog_epoch = 3
        self.placement_group = 8
        self.client = _Client()

    def for_graph(self, graph: str):
        self.graph_name = graph
        return self


class _NeverTouched:
    """Raises on ANY attribute access — proves a dry run never reaches the engine."""

    def __getattr__(self, name):
        raise AssertionError(
            f"dry_run=True must never touch the engine, but {name!r} was accessed"
        )


@pytest.fixture(autouse=True)
def _native_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APP_PROFILE", "dev")
    envelope_ingest_module._NATIVE_GRAPH_VERSIONS.clear()
    envelope_ingest_module._NATIVE_LOCKS.clear()
    actor = ActorContext(
        actor_id="fixture-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="fixture-tenant",
        authenticated=True,
    )
    token = set_session(
        GraphSession(
            actor=actor,
            tenant="fixture-tenant",
            scopes=frozenset({"kg:read", "kg:write"}),
            graph="fixture-graph",
            policy_version="fixture-policy",
            audience="fixture-audience",
        )
    )
    reset_company_brain()
    try:
        yield
    finally:
        reset_session(token)
        reset_company_brain()


def test_dry_run_previews_facts_without_ever_touching_the_engine():
    manifest = _fixtures.build_manifest(evaluation_cases=[])
    artifact, fragments = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")

    preview = run_pack(_NeverTouched(), manifest, artifact, fragments, dry_run=True)

    assert preview == preview_pack(manifest, artifact, fragments)
    entity_types = {e["node_type"] for e in preview.entities}
    assert "Document" in entity_types
    assert "Runbook" in entity_types
    assert "Person" in entity_types
    assert preview.mapping_version == "runbooks@1.0.0"


def test_write_path_commits_real_markdown_frontmatter_and_table_facts(tmp_path):
    manifest = _fixtures.build_manifest(evaluation_cases=[])
    pack_dir = _fixtures.write_pack(tmp_path, manifest)
    loaded = load_pack(
        pack_dir
    )  # proves the pack itself passes every fail-closed check

    artifact, fragments = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")
    compute = _Compute("fixture-graph")

    result = run_pack(compute, loaded.manifest, artifact, fragments, dry_run=False)

    assert result["status"] == "success"

    committed = compute.client.nodes.values
    document_node = committed[artifact.artifact_id]
    assert document_node["node_type"] == "Document"
    assert document_node["status"] == "active"  # from real frontmatter
    assert document_node["classification"] == "internal"  # the pack's default

    row_id = f"{artifact.artifact_id}#row:0"
    assert committed[row_id]["node_type"] == "Runbook"
    assert committed[row_id]["name"] == "Provision VM"  # from a real table cell

    assert committed["person:bob"]["node_type"] == "Person"

    applied_envelope = compute.client.changes.applied[0]
    lineage = applied_envelope["lineage"][0]
    assert lineage["transform_name"] == "domain-pack:runbooks"
    assert lineage["transform_version"] == "runbooks@1.0.0"


def test_pack_declared_classification_overrides_the_default(tmp_path):
    """A pack that declares ``default_classification: confidential`` must
    actually commit its facts as CONFIDENTIAL — not just carry the field
    unused. (Not testing ``public`` here: PUBLIC additionally requires an
    explicit ``source_acl.is_public=True`` proof — CONCEPT:AU-P0-4 — which
    this framework's default quarantined ACL correctly does not supply; a
    pack wanting PUBLIC facts is a follow-up, not a gap in this test.)"""
    manifest = _fixtures.build_manifest(
        evaluation_cases=[], default_classification="confidential"
    )
    pack_dir = _fixtures.write_pack(tmp_path, manifest)
    loaded = load_pack(pack_dir)
    artifact, fragments = fragment_markdown_text(RUNBOOK_MD, source_path="runbook.md")
    compute = _Compute("fixture-graph")

    result = run_pack(compute, loaded.manifest, artifact, fragments, dry_run=False)

    assert result["status"] == "success"
    assert (
        compute.client.nodes.values[artifact.artifact_id]["classification"]
        == "confidential"
    )


def test_invalid_pack_is_refused_before_any_fact_can_reach_the_engine(tmp_path):
    manifest = _fixtures.build_manifest(evaluation_cases=[])
    pack_dir = _fixtures.write_pack(tmp_path, manifest)
    manifest_path = pack_dir / "domain_pack.yml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\ntampered: true\n",
        encoding="utf-8",
    )

    with pytest.raises(DomainPackError):
        load_pack(pack_dir)
    # No engine was ever constructed/called — the refusal happens at load
    # time, strictly before any mapping or ingest step could run.
