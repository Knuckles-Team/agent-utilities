"""Live-engine proof that an evidence-spine Fragment slice actually COMMITS.

CONCEPT:AU-KG.ingest.evidence-spine-artifact / CONCEPT:AU-KG.ingest.stable-fragment-address

**What this closes.** ``tests/unit/knowledge_graph/ingestion/test_evidence_spine_wiring.py``
proves the WIRING (a real markdown file reaches ``envelope_ingest.ingest_envelope`` as an
``Artifact``/``Fragment`` slice) against ``engine=object()`` — a real object, but one with no
native authority, so the seam is observed and the call never reaches a compiled engine.

That was hiding a real defect: ``Fragment.version_id`` (``f"{fragment_id}@{hash}"``) contains
a literal ``@``, and the compiled engine's ``validate_safe_text``
(``epistemic-graph/crates/eg-types/src/change_envelope.rs:404-415``) blunt-rejects ANY ``@`` in
inline ``ApplyChangeEnvelope`` text as an email/host-leak privacy guard — so EVERY document
with fragments failed its whole commit against a real engine (never caught before because no
test exercised this combination for real; see the deferred ledger's D-GM-4 / D-GS856-6 /
D-MW-1 / D-MW-2 for the full investigation trail).

The fix changes ``Fragment.version_id``'s separator from ``@`` to ``#`` (a content-pin marker
that carries the same meaning without colliding with the privacy scan) rather than loosening
the engine's guard, which is a legitimate PII/leak control this repo does not own the blast
radius to weaken unilaterally. This module is the REAL, ephemeral, redb-backed
(CONCEPT:AU-KG.memory.provides-real-ephemeral-one) proof that the fix actually clears the
compiled engine: a real markdown file, with fragments, committed through the production
``DocumentProcessor.process(..., persist=True)`` entrypoint against a REAL engine, then read
back with real Cypher.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_utilities.knowledge_graph.ingestion.evidence_spine import (
    ARTIFACT_NODE_TYPE,
    FRAGMENT_NODE_TYPE,
)
from agent_utilities.knowledge_graph.ontology.document_processing import (
    ChunkingConfig,
    DocumentProcessor,
)

pytestmark = [pytest.mark.integration, pytest.mark.engine]

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
INSTANCE = "runbooks-live"


def _read(engine: Any, cypher: str, params: dict | None = None) -> list[dict]:
    return engine.backend.execute_read(cypher, {"_clearance_level": 999, **(params or {})})


@pytest.fixture()
def runbook(tmp_path: Path) -> Path:
    path = tmp_path / "settlement-runbook.md"
    path.write_text(MARKDOWN, encoding="utf-8")
    return path


def test_fragment_slice_commits_to_a_real_engine_and_is_queryable(
    engine_graph: Any, runbook: Path
) -> None:
    """The regression proof for D-GM-4 / D-GS856-6 / D-MW-1 / D-MW-2: a document
    WITH fragments (whose ``version_id`` property used to contain an '@') commits
    cleanly through the real native ``ApplyChangeEnvelope`` path -- no rejection,
    no silent swallow -- and is readable back via Cypher on the SAME engine."""
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    real_engine = IntelligenceGraphEngine(
        backend=EpistemicGraphBackend(graph_name=engine_graph.graph_name),
        defer_background_start=True,
    )
    processor = DocumentProcessor(
        None,
        engine=real_engine,
        chunking=ChunkingConfig(chunk_size=200, overlap=20),
        embed_fn=lambda texts: [[] for _ in texts],
    )

    # This used to raise RuntimeError("native document ChangeEnvelope failed: ...
    # persistence privacy policy rejected inline text") the instant the payload
    # carried a Fragment.version_id containing '@'. It must not raise now.
    # ``source`` is deliberately a benign relative label, not the raw absolute
    # tmp-dir path -- an absolute host filesystem path is its OWN, unrelated,
    # correctly-rejected privacy concern (a real machine-path leak) and would
    # confound this test's proof, which is specifically about the Fragment
    # version_id separator.
    processed = processor.process(
        str(runbook),
        source=runbook.name,
        connector=CONNECTOR,
        source_instance=INSTANCE,
        persist=True,
    )
    assert processed.persisted, "document + fragment slice must have actually committed"

    artifact_rows = _read(
        real_engine,
        f"MATCH (a:{ARTIFACT_NODE_TYPE}) WHERE a.connector = $connector "
        "RETURN a.id AS id, a.fragment_count AS fragment_count",
        {"connector": CONNECTOR},
    )
    assert len(artifact_rows) == 1
    assert artifact_rows[0]["fragment_count"] > 0

    fragment_rows = _read(
        real_engine,
        f"MATCH (f:{FRAGMENT_NODE_TYPE}) WHERE f.artifact_id = $artifact_id "
        "RETURN f.id AS id, f.version_id AS version_id",
        {"artifact_id": artifact_rows[0]["id"]},
    )
    assert fragment_rows, "committed artifact must have committed fragments"
    for row in fragment_rows:
        # The whole point: a real, persisted version_id, with the new '#'
        # separator and NOT the '@' that the engine's privacy guard rejects.
        assert row["version_id"].startswith(row["id"] + "#")
        assert "@" not in row["version_id"]
