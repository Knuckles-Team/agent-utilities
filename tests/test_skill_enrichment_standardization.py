#!/usr/bin/python
from __future__ import annotations

"""Skills, skill-graphs and skill-workflows share one enrichment level (CONCEPT:KG-2.7).

An atomic skill must get the SAME enrichment as a document/book — its instruction body
chunked + embedded (semantic search) and concepts/facts extracted — plus its
skill-type-unique attributes. These run offline (no daemon): DocumentProcessor
materializes chunk structure without a backend and the central seam is exercised via
the returned ``enrichable`` payload.
"""

import asyncio
from pathlib import Path

from agent_utilities.knowledge_graph.ingestion.engine import (
    ContentType,
    IngestionEngine,
    IngestionManifest,
)


def _write_skill(root: Path, name: str, body: str, *, scripts: bool = True) -> Path:
    d = root / name
    if scripts:
        (d / "scripts").mkdir(parents=True)
    else:
        d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A test skill.\ntags: [t1, t2]\n---\n"
        f"# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return d


def test_strip_frontmatter():
    ie = IngestionEngine()
    assert ie._strip_frontmatter("---\nname: x\n---\n# Body\n").strip() == "# Body"
    assert ie._strip_frontmatter("# No frontmatter\n").strip() == "# No frontmatter"


def test_ingest_skill_gets_document_grade_enrichment(tmp_path):
    sd = _write_skill(tmp_path, "my-skill", "Use this skill to do the thing. " * 80)
    ie = IngestionEngine()
    res = asyncio.run(
        ie._ingest_skill(
            IngestionManifest(content_type=ContentType.SKILL, source_uri=str(sd))
        )
    )
    assert res.status == "success"
    # skill-type-unique attributes
    assert res.details["skill_type"] == "atomic"
    # same enrichment level as documents: the body is chunked (+ embedded) ...
    assert res.details["chunks"] >= 1
    assert res.nodes_created == 1 + res.details["chunks"]
    # ... and concepts/facts run via the central seam over the body text.
    assert res.enrichable and res.enrichable[0]["source_type"] == "skill"
    assert "Use this skill" in res.enrichable[0]["text"]  # body, not raw frontmatter


def test_materialize_body_chunks_is_safe_on_empty():
    ie = IngestionEngine()
    assert ie._materialize_body_chunks("x", "", title="x", doc_type="skill") == (0, 0)
