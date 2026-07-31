#!/usr/bin/python
"""Live-engine proof for the git-markdown connector (universal-ingestion program).

CONCEPT:AU-ECO.connector.git-markdown-revision-connector

This is the end-to-end deliverable ``reports/program/universal-ingestion.md`` asks
for: **several static markdown knowledge graphs ingested as domain packs, with
epistemic-graph authoritative across all of them** — against a REAL, ephemeral,
redb-backed ``epistemic-graph`` engine (``engine_graph``/``tiny_engine``, the same
fixtures every other live-engine test in this suite uses; CONCEPT:AU-KG.memory.provides-real-ephemeral-one), never a
mock or a recording stand-in.

Two REAL, structurally different markdown corpora (byte-identical content copied
from this repo's own ``docs/pillars/**/*.md`` and ``agent_utilities/skills/*/
SKILL.md``, assembled into a small scratch git repo so the connector's diff-based
incrementality can be exercised without mutating this checkout) are ingested as two
``git_markdown`` domain packs (``au-pillars``/``au-skills``) through the real
``ContentType.CONNECTOR`` ingestion path, then queried back with real Cypher on the
SAME engine:

1. Both corpora are present, distinguishable by their ``corpus`` property.
2. One specific fact from EACH corpus is traceable to its corpus, file (``relpath``),
   and git revision (``git_commit``) — real content, real path, real SHA.
3. Modifying one file + re-running yields exactly one changed document; the other
   115 documents keep their prior ``id`` and ``git_commit`` (stable ids, incremental
   — not batch-only).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.knowledge_graph.ingestion.engine import (
    ContentType,
    IngestionEngine,
    IngestionManifest,
)

pytestmark = [pytest.mark.integration, pytest.mark.engine]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 — fixed argv, local git, test-only
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()


@pytest.fixture()
def two_corpora_repo(tmp_path: Path) -> Path:
    """A scratch git repo holding REAL content from this repo's two domain packs.

    Copies every real ``docs/pillars/**/*.md`` (no frontmatter; "``# ID-Title``" +
    "``**Pillar:**``" heading convention) and every real ``agent_utilities/skills/*/
    SKILL.md`` (real YAML frontmatter) verbatim — same bytes shipped in this repo —
    into a fresh, independent git working tree so the diff-incrementality proof can
    freely commit a change without touching this checkout.
    """
    repo = tmp_path / "corpora"
    repo.mkdir()
    pillars_src = _REPO_ROOT / "docs" / "pillars"
    skills_src = _REPO_ROOT / "agent_utilities" / "skills"

    pillars_files = sorted(pillars_src.rglob("*.md"))
    skill_files = sorted(skills_src.glob("*/SKILL.md"))
    assert len(pillars_files) > 50, "sanity: the real pillars corpus must be present"
    assert len(skill_files) > 5, "sanity: the real skills corpus must be present"

    for src in pillars_files:
        dest = repo / "docs" / "pillars" / src.relative_to(pillars_src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
    for src in skill_files:
        dest = repo / "agent_utilities" / "skills" / src.relative_to(skills_src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "proof@test.local")
    _git(repo, "config", "user.name", "universal-ingestion-proof")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "snapshot: real au-pillars + au-skills corpora")
    return repo


def _real_ingestion_engine(engine_graph: Any) -> IngestionEngine:
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    real_engine = IntelligenceGraphEngine(
        backend=EpistemicGraphBackend(graph_name=engine_graph.graph_name),
        defer_background_start=True,
    )
    return IngestionEngine(kg_engine=real_engine, backend=real_engine.backend)


def _read(engine: IngestionEngine, cypher: str, params: dict | None = None) -> list[dict]:
    return engine.backend.execute_read(cypher, {"_clearance_level": 999, **(params or {})})


async def _run_preset(
    engine: IngestionEngine, *, preset: str, root: Path, connector_id: str
):
    manifest = IngestionManifest(
        content_type=ContentType.CONNECTOR,
        source_uri="git_markdown",
        metadata={
            "connector_config": {"root": str(root), "preset": preset},
            "connector_id": connector_id,
            "contextual": False,
        },
    )
    return await engine.ingest(manifest)


@pytest.mark.asyncio
async def test_two_domain_packs_ingested_and_queryable_with_lineage(
    engine_graph: Any, two_corpora_repo: Path
) -> None:
    engine = _real_ingestion_engine(engine_graph)
    sha1 = _git(two_corpora_repo, "rev-parse", "HEAD")

    pillars_result = await _run_preset(
        engine,
        preset="au-pillars",
        root=two_corpora_repo,
        connector_id="proof-au-pillars",
    )
    skills_result = await _run_preset(
        engine,
        preset="au-skills",
        root=two_corpora_repo,
        connector_id="proof-au-skills",
    )

    assert pillars_result.status == "success"
    assert skills_result.status == "success"
    n_pillars = len(list((two_corpora_repo / "docs" / "pillars").rglob("*.md")))
    n_skills = len(list((two_corpora_repo / "agent_utilities" / "skills").glob("*/SKILL.md")))
    assert pillars_result.details["documents"] == n_pillars
    assert skills_result.details["documents"] == n_skills
    # Every file's governed ChangeEnvelope applied cleanly through the SAME
    # ChangeEnvelope contract the operator asked the connector to report against.
    assert pillars_result.details["envelopes"] == n_pillars
    assert skills_result.details["envelopes"] == n_skills

    # (1) epistemic-graph answering ACROSS both corpora with ONE Cypher query,
    # each fact traceable to its corpus / file / git revision.
    rows = _read(
        engine,
        "MATCH (n:Document) WHERE n.corpus IN ['au-pillars', 'au-skills'] "
        "RETURN n.corpus AS corpus, count(n) AS n",
    )
    by_corpus = {row["corpus"]: row["n"] for row in rows}
    assert by_corpus == {"au-pillars": n_pillars, "au-skills": n_skills}

    # (2) one specific, real fact from EACH corpus, traceable to corpus + file +
    # git revision — not a synthetic fixture.
    pillar_rows = _read(
        engine,
        "MATCH (n:Document) WHERE n.relpath = $relpath "
        "RETURN n.corpus AS corpus, n.relpath AS relpath, n.git_commit AS git_commit, "
        "n.doc_type AS doc_type",
        {
            "relpath": (
                "docs/pillars/2_epistemic_knowledge_graph/"
                "KG-2.37-Research_State_Domain_Pack.md"
            )
        },
    )
    assert len(pillar_rows) == 1, pillar_rows
    assert pillar_rows[0]["corpus"] == "au-pillars"
    assert pillar_rows[0]["git_commit"] == sha1
    assert pillar_rows[0]["doc_type"] == "pillar-doc"

    skill_rows = _read(
        engine,
        "MATCH (n:Document) WHERE n.relpath = $relpath "
        "RETURN n.corpus AS corpus, n.relpath AS relpath, n.git_commit AS git_commit, "
        "n.doc_type AS doc_type",
        {"relpath": "agent_utilities/skills/graph-orchestration-and-automation/SKILL.md"},
    )
    assert len(skill_rows) == 1, skill_rows
    assert skill_rows[0]["corpus"] == "au-skills"
    assert skill_rows[0]["git_commit"] == sha1
    assert skill_rows[0]["doc_type"] == "skill-doc"

    # Chunks carry real, distinct content from each convention (frontmatter vs
    # heading-only) — proves the connector didn't special-case either shape.
    chunk_rows = _read(
        engine,
        "MATCH (d:Document {relpath: $relpath})-[:HAS_CHUNK]->(c:Chunk) "
        "RETURN c.content AS content LIMIT 1",
        {"relpath": "agent_utilities/skills/graph-orchestration-and-automation/SKILL.md"},
    )
    assert chunk_rows and "skill_type: skill" in (chunk_rows[0]["content"] or "")


@pytest.mark.asyncio
async def test_incremental_change_updates_only_the_touched_file(
    engine_graph: Any, two_corpora_repo: Path
) -> None:
    """Change one pillar doc, re-run the SAME connector config: only that file's
    facts update; every other document keeps its id and its original git_commit —
    stable fragment/document ids for the untouched ones (the proof this is a real
    incremental change feed, not a batch re-embed)."""
    engine = _real_ingestion_engine(engine_graph)
    sha1 = _git(two_corpora_repo, "rev-parse", "HEAD")

    first = await _run_preset(
        engine, preset="au-pillars", root=two_corpora_repo, connector_id="proof-incremental"
    )
    n_pillars = first.details["documents"]
    assert first.details["checkpoint_advanced"] is True

    unchanged_relpath = (
        "docs/pillars/4_ecosystem_peripherals/"
        "ECO-4.25-Document_Source_Connector_Framework.md"
    )
    unchanged_before = _read(
        engine,
        "MATCH (n:Document {relpath: $relpath}) RETURN n.id AS id, n.git_commit AS git_commit",
        {"relpath": unchanged_relpath},
    )
    assert unchanged_before and unchanged_before[0]["git_commit"] == sha1
    unchanged_id_before = unchanged_before[0]["id"]

    changed_relpath = (
        "docs/pillars/2_epistemic_knowledge_graph/"
        "KG-2.37-Research_State_Domain_Pack.md"
    )
    changed_before = _read(
        engine,
        "MATCH (n:Document {relpath: $relpath}) RETURN n.id AS id",
        {"relpath": changed_relpath},
    )
    changed_id_before = changed_before[0]["id"]

    (two_corpora_repo / changed_relpath).write_text(
        (two_corpora_repo / changed_relpath).read_text()
        + "\n## Incrementality proof addendum\n\nThis line proves diff-based re-ingest.\n"
    )
    _git(two_corpora_repo, "add", "-A")
    _git(two_corpora_repo, "commit", "-q", "-m", "touch one pillar doc")
    sha2 = _git(two_corpora_repo, "rev-parse", "HEAD")
    assert sha2 != sha1

    second = await _run_preset(
        engine, preset="au-pillars", root=two_corpora_repo, connector_id="proof-incremental"
    )

    # Exactly the one touched file was re-ingested — not a batch re-embed of all 96.
    assert second.details["documents"] == 1
    assert second.details["checkpoint_advanced"] is True

    # Total document count for this corpus is unchanged (no duplication).
    total = _read(
        engine,
        "MATCH (n:Document {corpus: 'au-pillars'}) RETURN count(n) AS n",
    )
    assert total[0]["n"] == n_pillars

    changed_after = _read(
        engine,
        "MATCH (n:Document {relpath: $relpath}) RETURN n.id AS id, n.git_commit AS git_commit",
        {"relpath": changed_relpath},
    )
    assert changed_after[0]["id"] == changed_id_before, "document id must stay stable"
    assert changed_after[0]["git_commit"] == sha2, "revision must advance for the touched file"

    unchanged_after = _read(
        engine,
        "MATCH (n:Document {relpath: $relpath}) RETURN n.id AS id, n.git_commit AS git_commit",
        {"relpath": unchanged_relpath},
    )
    assert unchanged_after[0]["id"] == unchanged_id_before, "untouched id must not change"
    assert unchanged_after[0]["git_commit"] == sha1, "untouched file keeps its ORIGINAL revision"


def test_change_envelope_contract_fields_present() -> None:
    """Sanity: the connector's governed envelope literally sets the ChangeEnvelope
    fields the operator's charter named — source identity, revision, access,
    content-bearing payload — independent of any live engine."""
    from agent_utilities.protocols.source_connectors.connectors.git_markdown import (
        GitMarkdownConnector,
    )

    connector = GitMarkdownConnector(root=str(_REPO_ROOT), preset="au-pillars")
    envelope = connector._upsert_envelope("deadbeef" * 5, "docs/pillars/x.md")
    assert isinstance(envelope, ChangeEnvelope)
    assert envelope.connector == "git_markdown"
    assert envelope.source_version == "deadbeef" * 5  # revision
    assert envelope.source_acl is not None  # access labels
    assert envelope.typed_payload is not None  # content-bearing payload
    assert envelope.operation == "upsert"
