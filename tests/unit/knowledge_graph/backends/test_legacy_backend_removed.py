"""The legacy SPARQL backend and database-setup path were retired outright —
``knowledge_graph/backends/sparql/**``,
``knowledge_graph/setup/**``, ``integrations/stardog_sync.py``, and
``integrations/sparql_ingestor.py`` are gone. A caller that still reaches a
path that used to be served by one of these (an explicit ``create_backend``
request, the ETL SPARQL sink, or the ingestion engine's SPARQL adaptor) must
get a typed, reachable "unavailable" answer rather than an import error or a
silently vanished capability.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends import (
    LegacyGraphBackendRemovedError,
    create_backend,
)


@pytest.mark.parametrize("backend_type", ["stardog", "jena_fuseki"])
def test_create_backend_raises_typed_error_for_retired_backend(
    backend_type: str,
) -> None:
    with pytest.raises(LegacyGraphBackendRemovedError, match="retired"):
        create_backend(backend_type=backend_type)


def test_create_backend_still_distinguishes_a_genuinely_unknown_type() -> None:
    """A real typo/unknown type is NOT the retired-backend path: it keeps the
    existing ``None``-returning contract, so the two failure shapes stay
    distinguishable."""
    assert create_backend(backend_type="not-a-real-backend") is None


def test_etl_pipeline_sparql_sink_answers_typed_unavailable() -> None:
    from agent_utilities.knowledge_graph.etl.pipeline import _run_outbound

    class _SparqlBE:
        supports_sparql = True

    result = _run_outbound(
        engine=object(),
        sink="stardog",
        sink_backend=_SparqlBE(),
        sources=None,
        dry_run=True,
        ops={},
    )

    assert result["status"] == "error"
    assert result["error_type"] == "LegacyGraphBackendRemovedError"


@pytest.mark.asyncio
async def test_ingestion_engine_sparql_adaptor_answers_typed_unavailable() -> None:
    from agent_utilities.knowledge_graph.ingestion.engine import (
        ContentType,
        IngestionEngine,
        IngestionManifest,
    )

    engine = IngestionEngine(kg_engine=object())
    manifest = IngestionManifest(
        content_type=ContentType.SPARQL,
        source_uri="https://sparql.invalid/query",
    )

    result = await engine._ingest_sparql(manifest)

    assert result.status == "failed"
    assert result.details["error_type"] == "LegacyGraphBackendRemovedError"
