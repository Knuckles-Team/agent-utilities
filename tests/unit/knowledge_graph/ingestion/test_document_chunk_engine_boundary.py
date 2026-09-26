"""The AU document adaptor supplies text chunks to EG's pure projection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent_utilities.knowledge_graph.ingestion.engine import IngestionEngine


def test_verbatim_chunk_slice_consumes_engine_projection() -> None:
    with (
        patch(
            "agent_utilities.knowledge_graph.distillation.distillation_engine.chunk_text",
            return_value=["one", "two"],
        ) as splitter,
        patch(
            "agent_utilities.knowledge_graph.ingestion.engine.verbatim_chunk_slice",
            return_value=([{"id": "eg-node"}], [{"relationship": "PART_OF"}]),
        ) as projector,
    ):
        nodes, edges = IngestionEngine._verbatim_chunk_slice(
            "doc:1", SimpleNamespace(title="Guide"), "raw source", "web-document"
        )

    splitter.assert_called_once_with("raw source")
    projector.assert_called_once_with("doc:1", "Guide", ["one", "two"], "web-document")
    assert nodes == [{"id": "eg-node"}]
    assert edges == [{"relationship": "PART_OF"}]
