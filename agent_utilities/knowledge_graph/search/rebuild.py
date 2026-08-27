"""Full-rebuild path: drop and reindex a tenant from a replay of ``eg.cdc.<graph>``.

(CONCEPT:AU-KG.retrieval.opensearch-cdc-indexer, CA-24, DEC-CA-09's P3 acceptance
test: "replay from offset 0 rebuilds an identical index")

Authority: `DEC-CA-01` — eg redb is the sole ACID store; the OpenSearch index
is a derived, rebuildable projection. A full rebuild from offset 0 is always
a valid recovery path, not an exceptional one — this module (both the
:func:`rebuild_index` function and the CLI) is that recovery path, and the
same mechanism CA-43's future ``opensearch-mcp`` tool surface wraps
(the lane doc: "a CLI + MCP-tool-ready function").

**Determinism.** :func:`~.indexer.apply_envelope` is a pure function of
(current index state, next envelope): given the identical ordered record
sequence replayed into a freshly-dropped index, the resulting document set
is byte-identical every time — no batching/parallelism reorders writes
within one graph's single-partition stream.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from typing import Any, TypedDict

from . import doc_shape
from .client import OpenSearchClient, OpenSearchClientConfig
from .indexer import apply_envelope

logger = logging.getLogger(__name__)

__all__ = ["RebuildResult", "rebuild_index", "mcp_reindex", "main"]

_EMPTY_COUNTS = {"applied": 0, "rejected_stale": 0, "quarantined": 0, "failed": 0}


class RebuildResult(TypedDict, total=False):
    """Return shape of :func:`rebuild_index`/:func:`mcp_reindex`. ``indices``/
    ``index_counts`` are exact for the ``records=`` path; the ``consumer=``
    path instead reports ``tenant_document_count`` (see :func:`rebuild_index`'s
    docstring: ``drain_once``'s result carries only aggregate counts, not
    per-record index names)."""

    status: str
    counts: dict[str, int]
    indices: list[str]
    index_counts: dict[str, int | None]
    tenant_document_count: int | None
    details: dict[str, Any]


def rebuild_index(
    tenant: str,
    object_type: str | None = None,
    *,
    from_seq: int = 0,
    graph: str | None = None,
    opensearch: OpenSearchClient | None = None,
    records: list[dict[str, Any]] | None = None,
    consumer: Any = None,
    drop_existing: bool = True,
) -> RebuildResult:
    """Drop and reindex ``tenant`` (optionally scoped to one ``object_type``)
    by replaying ``eg.cdc.<graph>`` from offset 0 (or ``from_seq``).

    Exactly one record source:

    * ``records`` — an explicit, already-decoded, ordered list of envelope
      dicts (the deterministic-replay/test path — no Kafka round trip; also
      the shape a caller re-driving a captured topic dump would use).
    * ``consumer`` — an **unconnected**
      :class:`~.indexer.EgCdcKafkaConsumer` (or a test double exposing the
      same ``connect``/``drain_once``/``disconnect`` async surface). This
      function owns the ENTIRE consumer lifecycle (connect, drain to
      exhaustion, disconnect) inside one bridged coroutine (see
      "async-safety" below) — the live/operator path, matching P3's
      "replay from offset 0" mechanism against the real broker.

      **Topic scoping (measured live, 2026-08-26):** the consumer's topic
      pattern is overridden here to the EXACT ``eg.cdc.<graph>`` topic for
      this rebuild's target graph, never the shared
      ``EG_CDC_TOPIC_PATTERN`` wildcard the always-on production indexer
      uses. A live proof against the real, multi-tenant CA broker showed
      the wildcard pattern picks up every OTHER graph's topic too (230
      quarantined/unrelated records from other lanes' test data on one
      poll) — harmless for the always-on indexer (each record is scoped by
      its own ``graph`` field), but wrong for a scoped "rebuild tenant X"
      call, which must touch only that graph's own topic.

    ``graph`` scopes replay to one eg graph (defaults to ``tenant`` — this
    package treats a graph id as the tenant-scoping unit, see
    ``indexer``'s module doc).

    **Async-safety (a real bug found and fixed via this lane's own live
    proof, 2026-08-26).** An earlier version of this function called
    ``asyncio.run()`` directly for the consumer path. That crashes with
    "asyncio.run() cannot be called from a running event loop" the moment a
    caller (e.g. :func:`mcp_reindex`, or any other async MCP-tool call site)
    invokes this function from inside its own already-running loop. The
    consumer branch now runs through
    ``agent_utilities.protocols.source_connectors.connectors.mcp_package
    ._run_async`` — the SAME "safe whether or not a loop is running" bridge
    CA-21's ``run_cdc_catchup`` already uses for the identical problem — and,
    critically, performs ``connect``/``drain``/``disconnect`` all inside
    ONE coroutine that bridge runs, so the consumer's internal aiokafka
    tasks are never split across two different event loops (which breaks
    aiokafka with a bare ``CancelledError`) the way connecting on a caller's
    loop and draining on a second, separately-bridged loop would.
    """
    if (records is None) == (consumer is None):
        raise ValueError("rebuild_index requires exactly one of records= or consumer=")

    client = opensearch or OpenSearchClient(OpenSearchClientConfig.from_env())
    target_graph = graph or tenant

    if drop_existing:
        pattern = (
            doc_shape.index_name(tenant, object_type)
            if object_type
            else doc_shape.tenant_wildcard(tenant)
        )
        try:
            client.delete_index(pattern)
        except Exception:  # noqa: BLE001 - best-effort drop; ensure_index recreates on first write
            logger.warning(
                "rebuild_index: nothing to drop at %s (or drop failed)", pattern
            )

    counts = dict(_EMPTY_COUNTS)
    touched_indices: set[str] = set()
    last_failure: dict[str, Any] | None = None

    def _handle(envelope: dict[str, Any]) -> bool:
        if envelope.get("graph") != target_graph:
            return True
        if envelope.get("seq") is not None:
            try:
                if int(envelope["seq"]) < from_seq:
                    return True
            except (TypeError, ValueError):  # noqa: BLE001 — non-numeric seq falls through to apply_envelope's own malformed-envelope handling (quarantined, not silently skipped) below; this guard's only job is the from_seq floor
                pass
        result = apply_envelope(client, envelope)
        status = result.get("status", "failed")
        counts[status] = counts.get(status, 0) + 1
        idx = result.get("index")
        if idx:
            touched_indices.add(idx)
        for extra_idx in result.get("indices", []) or []:
            if extra_idx:
                touched_indices.add(extra_idx)
        return status != "failed"

    if records is not None:
        for envelope in records:
            if not _handle(envelope):
                break
    else:
        from ...protocols.source_connectors.connectors.mcp_package import _run_async

        async def _consumer_lifecycle() -> None:
            nonlocal last_failure
            consumer._topic_pattern = re.compile(  # noqa: SLF001 - see docstring: scope to exactly this graph's topic
                re.escape(f"eg.cdc.{target_graph}")
            )
            await consumer.connect()
            try:
                while True:
                    batch = await consumer.drain_once(client)
                    batch_counts = batch.get("counts", {})
                    for key, value in batch_counts.items():
                        counts[key] = counts.get(key, 0) + value
                    if batch.get("status") == "failed":
                        last_failure = batch.get("details", {}).get("failure")
                        break
                    if sum(batch_counts.values()) == 0:
                        break
            finally:
                await consumer.disconnect()

        _run_async(_consumer_lifecycle())

    final_counts: dict[str, Any] = {}
    for idx in sorted(touched_indices):
        try:
            final_counts[idx] = client.count(idx)
        except Exception:  # noqa: BLE001
            final_counts[idx] = None

    result: RebuildResult = {
        "status": "ok" if counts.get("failed", 0) == 0 else "failed",
        "counts": counts,
        "indices": sorted(touched_indices),
        "index_counts": final_counts,
    }
    if consumer is not None:
        # The consumer path's drain_once() result carries only aggregate
        # counts, not per-record index names, so `indices`/`index_counts`
        # (exact for the records= path) stay empty here; this tenant-wide
        # total is the best available diagnostic substitute.
        wildcard = doc_shape.tenant_wildcard(tenant)
        try:
            result["tenant_document_count"] = client.count(wildcard)
        except Exception:  # noqa: BLE001
            result["tenant_document_count"] = None
        if counts.get("failed", 0) and last_failure:
            result["details"] = {"failure": last_failure}
    return result


def mcp_reindex(
    tenant: str,
    object_type: str | None = None,
    *,
    from_seq: int = 0,
    graph: str | None = None,
    drop_existing: bool = True,
) -> RebuildResult:
    """Sync entry point for the MCP/REST surfaces (``graph_ingest`` action
    ``opensearch_reindex`` in ``mcp/tools/write_ingest_tools.py``, reachable
    identically from ``POST /api/graph/ingest`` — both dispatch through
    ``REGISTERED_TOOLS["graph_ingest"]``, the same core, per the lane's
    "two surfaces" requirement).

    A live :class:`~.client.OpenSearchClient` plus an **unconnected**
    :class:`~.indexer.EgCdcKafkaConsumer` are handed to :func:`rebuild_index`,
    which owns the entire consumer connect/drain/disconnect lifecycle itself
    (see that function's async-safety note) — this function does NOT wrap
    :func:`rebuild_index` in a second bridge layer, which is exactly the
    double-wrapping that produced the nested-``asyncio.run`` crash this
    lane's own live proof caught and fixed.
    """
    from .indexer import EgCdcKafkaConsumer

    client = OpenSearchClient(OpenSearchClientConfig.from_env())
    client.connect()
    live_consumer = EgCdcKafkaConsumer(config=None)

    return rebuild_index(
        tenant,
        object_type,
        from_seq=from_seq,
        graph=graph or tenant,
        opensearch=client,
        consumer=live_consumer,
        drop_existing=drop_existing,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild an OpenSearch tenant/object-type index from a live "
            "eg.cdc.<graph> replay (CA-24, DEC-CA-09 P3)."
        )
    )
    parser.add_argument("tenant", help="eg graph id / tenant to rebuild")
    parser.add_argument(
        "--object-type", default=None, help="Scope to one object type only"
    )
    parser.add_argument(
        "--graph", default=None, help="eg graph id (defaults to tenant)"
    )
    parser.add_argument("--from-seq", type=int, default=0)
    parser.add_argument(
        "--no-drop", action="store_true", help="Do not drop the existing index first"
    )
    args = parser.parse_args(argv)

    result = mcp_reindex(
        args.tenant,
        args.object_type,
        from_seq=args.from_seq,
        graph=args.graph or args.tenant,
        drop_existing=not args.no_drop,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
