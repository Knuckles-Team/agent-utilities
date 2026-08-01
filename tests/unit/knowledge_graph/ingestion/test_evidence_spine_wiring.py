"""The evidence spine is REACHED from the real ingest path — not merely importable.

CONCEPT:AU-KG.ingest.evidence-spine-artifact
CONCEPT:AU-KG.ingest.stable-fragment-address

Wire-First level: these drive ``DocumentProcessor.process`` — the live document
entrypoint the ``document_process`` MCP tool and the ``/document/process`` REST
route both call — with a **real markdown file on disk**, and assert on the rows
that actually reached ``envelope_ingest.ingest_envelope``, inside the *same*
``ChangeEnvelope`` as the document.  The seam is observed, never mocked: the real
``ingest_envelope`` runs and fails downstream at the engine, which is exactly
where a test without an engine should fail.

Asserting on the observed envelope (rather than on the returned
``ProcessedDocument``) is deliberate — it is the payload the transaction would
have committed, so a regression that computed a perfect spine and forgot to
attach it would fail here and pass any in-memory check.

A unit test of ``fragment_markdown`` proves the fragmenter works.  It proves
nothing about whether anything calls it.  That is what these cover.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_utilities.knowledge_graph.ingestion import envelope_ingest
from agent_utilities.knowledge_graph.ingestion.evidence_spine import (
    ARTIFACT_NODE_TYPE,
    FRAGMENT_NODE_TYPE,
    HAS_ARTIFACT_EDGE,
    HAS_FRAGMENT_EDGE,
    NEXT_FRAGMENT_EDGE,
    PARENT_FRAGMENT_EDGE,
    artifact_id_for,
)
from agent_utilities.knowledge_graph.ontology.document_processing import (
    ChunkingConfig,
    DocumentProcessor,
)
from tests.wiring import observe, past_the_seam

MARKDOWN = """# Settlement Runbook

Settlement runs nightly at 02:00 UTC.

## Failure Modes

| Symptom | Action |
|---|---|
| stuck batch | replay the batch |
| duplicate payout | freeze and escalate |

- Check the ledger lag first
- Then check the acquirer feed
"""

CONNECTOR = "git-markdown"
INSTANCE = "runbooks"


@pytest.fixture
def runbook(tmp_path: Path) -> Path:
    path = tmp_path / "settlement-runbook.md"
    path.write_text(MARKDOWN, encoding="utf-8")
    return path


def ingest(path: Path) -> Any:
    """Drive the live entrypoint; return the ChangeEnvelope that reached ingest.

    ``engine=object()`` is a real object, not a mock: the native ingest path
    resolves an authority off it, finds none, and fails closed — downstream of
    the seam under test.  ``embed_fn`` returns empty vectors so the test needs no
    embedding model; the spine does not depend on embeddings either way.
    """
    processor = DocumentProcessor(
        None,
        engine=object(),
        chunking=ChunkingConfig(chunk_size=200, overlap=20),
        embed_fn=lambda texts: [[] for _ in texts],
    )
    with observe(envelope_ingest, "ingest_envelope") as ingested:
        with past_the_seam(must_not_raise="Fragment"):
            processor.process(
                str(path),
                source=str(path),
                connector=CONNECTOR,
                source_instance=INSTANCE,
                persist=True,
            )
    assert ingested.calls, (
        "DocumentProcessor.process must reach envelope_ingest.ingest_envelope — "
        "the evidence spine has no second write path of its own."
    )
    return ingested.calls[0].args[1]


def rows(envelope: Any, node_type: str) -> list[dict[str, Any]]:
    payload = envelope.typed_payload or {}
    return [n for n in payload.get("_nodes", []) if n.get("node_type") == node_type]


def links(envelope: Any, relationship: str) -> list[dict[str, Any]]:
    payload = envelope.typed_payload or {}
    return [e for e in payload.get("_links", []) if e.get("relationship") == relationship]


def test_a_real_markdown_file_reaches_the_ingest_path_as_an_artifact(
    runbook: Path,
) -> None:
    envelope = ingest(runbook)
    artifacts = rows(envelope, ARTIFACT_NODE_TYPE)

    assert len(artifacts) == 1, "exactly one Artifact per retrieved source object"
    artifact = artifacts[0]
    # Keyed to the stable SOURCE OBJECT (the path), never to the Document's
    # content-derived id — otherwise every edit forks a new artifact and orphans
    # every citation that pointed into the old one.
    assert artifact["id"] == artifact_id_for(CONNECTOR, INSTANCE, str(runbook))
    assert artifact["id"] != artifact_id_for(
        CONNECTOR, INSTANCE, envelope.typed_payload["id"]
    )
    assert artifact["content_hash"].startswith("sha256:")
    assert artifact["connector"] == CONNECTOR
    assert artifact["source_instance"] == INSTANCE
    assert artifact["fragment_count"] > 0
    # Governance rode the envelope that gated it, not the payload.
    assert artifact["classification"] == envelope.classification.value
    assert artifact["external_access"] == envelope.source_acl.model_dump()


def test_fragments_ride_the_same_envelope_as_the_document(runbook: Path) -> None:
    """One envelope = one transaction.  A half-landed spine is not a reachable state."""
    envelope = ingest(runbook)
    fragments = rows(envelope, FRAGMENT_NODE_TYPE)
    artifact = rows(envelope, ARTIFACT_NODE_TYPE)[0]

    assert envelope.typed_payload["node_type"] == "Document"
    assert len(fragments) == artifact["fragment_count"] > 0
    assert len(links(envelope, HAS_FRAGMENT_EDGE)) == len(fragments)
    assert len(links(envelope, HAS_ARTIFACT_EDGE)) == 1
    assert links(envelope, HAS_ARTIFACT_EDGE)[0]["source"] == envelope.typed_payload["id"]
    # Nesting and sibling order are explicit edges in the same transaction.
    assert links(envelope, PARENT_FRAGMENT_EDGE)
    assert links(envelope, NEXT_FRAGMENT_EDGE)


def test_the_spine_is_addressable_hashed_and_nested_through_the_live_path(
    runbook: Path,
) -> None:
    envelope = ingest(runbook)
    fragments = rows(envelope, FRAGMENT_NODE_TYPE)
    by_id = {f["id"]: f for f in fragments}

    # Every row satisfies the SHACL FragmentShape's requirements — asserted on
    # what the live path actually produced, not on a hand-built fixture.
    for fragment in fragments:
        assert fragment["address"]
        assert fragment["content_hash"].startswith("sha256:")
        assert fragment["artifact_id"] == rows(envelope, ARTIFACT_NODE_TYPE)[0]["id"]
        assert fragment["fragment_kind"]
        assert isinstance(fragment["sequence"], int)

    table_rows = [f for f in fragments if f["fragment_kind"] == "table_row"]
    assert table_rows, "the runbook has a table"
    table = by_id[table_rows[0]["parent_fragment_id"]]
    section = by_id[table["parent_fragment_id"]]
    assert (table["fragment_kind"], section["fragment_kind"]) == ("table", "heading")
    assert section["label"] == "Failure Modes"
    assert table_rows[0]["depth"] == table["depth"] + 1 == section["depth"] + 2


def test_reingesting_the_unchanged_file_produces_identical_fragment_ids(
    runbook: Path,
) -> None:
    """The load-bearing guarantee: a no-op re-ingest moves no citation."""
    first, second = ingest(runbook), ingest(runbook)

    assert rows(first, ARTIFACT_NODE_TYPE)[0]["id"] == rows(
        second, ARTIFACT_NODE_TYPE
    )[0]["id"]
    assert rows(first, ARTIFACT_NODE_TYPE)[0]["content_hash"] == rows(
        second, ARTIFACT_NODE_TYPE
    )[0]["content_hash"]

    ids = lambda env: [f["id"] for f in rows(env, FRAGMENT_NODE_TYPE)]  # noqa: E731
    hashes = lambda env: [  # noqa: E731
        f["content_hash"] for f in rows(env, FRAGMENT_NODE_TYPE)
    ]
    assert ids(first) == ids(second)
    assert hashes(first) == hashes(second)


def test_editing_one_paragraph_moves_only_that_fragments_hash(runbook: Path) -> None:
    """The other half: a real edit on disk changes exactly what it should."""
    before = ingest(runbook)
    runbook.write_text(
        MARKDOWN.replace("nightly at 02:00 UTC", "nightly at 03:00 UTC"),
        encoding="utf-8",
    )
    after = ingest(runbook)

    # Same source object -> same artifact identity, new revision.
    assert rows(after, ARTIFACT_NODE_TYPE)[0]["id"] == rows(
        before, ARTIFACT_NODE_TYPE
    )[0]["id"]
    assert (
        rows(after, ARTIFACT_NODE_TYPE)[0]["content_hash"]
        != rows(before, ARTIFACT_NODE_TYPE)[0]["content_hash"]
    )

    was = {f["address"]: f for f in rows(before, FRAGMENT_NODE_TYPE)}
    changed = [
        f
        for f in rows(after, FRAGMENT_NODE_TYPE)
        if f["content_hash"] != was[f["address"]]["content_hash"]
    ]
    assert [f["text"] for f in changed] == ["Settlement runs nightly at 03:00 UTC."]
    # …and every fragment id, including the edited one's, survived the edit.
    assert [f["id"] for f in rows(after, FRAGMENT_NODE_TYPE)] == [
        f["id"] for f in rows(before, FRAGMENT_NODE_TYPE)
    ]
