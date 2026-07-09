#!/usr/bin/python
from __future__ import annotations

"""Typed, validated ETL step output (CONCEPT:AU-KG.etl.result-contract).

Assimilates koheesio's ``Step.Output`` idea (``reports/koheesio-etl-analysis.md``
§3.1) WITHOUT its ``StepMetaClass``/``partialmethod`` machinery: a plain pydantic
model that every ETL entrypoint (:func:`~.pipeline.run_etl`,
:func:`~..core.source_sync.sync_source`,
:func:`~..core.table_ingest.ingest_connector_to_table`) threads its result
through, replacing the duck-typed ad hoc dicts each caller used to guess the
shape of — ``{"nodes": …}`` vs ``{"nodes_hydrated": …}`` vs ``{"created": …}`` vs
``{"rows_written": …}`` — exactly the ambiguity the old module-level
``etl.pipeline._count()`` helper existed to paper over (now :meth:`EtlResult.count_of`).

Backward-compatible by design: every entrypoint still returns a plain ``dict``
(``EtlResult.model_dump()``), so existing callers that index ``out["status"]`` /
``out["inbound"]`` keep working — the model is *internal, validating* plumbing,
not a new consumer-facing type. ``extra="allow"`` so the long tail of
handler-specific fields (``instances``, ``accounts``, ``documents``, ``rows_seen``,
…) that the ~20 ``_sync_*`` handlers in ``core.source_sync`` already emit keep
flowing through unedited — see the koheesio analysis §4: migrate those handlers to
constructing :class:`EtlResult` directly only as touched, never in one mass pass.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# The legacy duck-typed keys the old ``_count()`` helper guessed across (kept as
# one documented list instead of re-deriving it ad hoc at each call site).
_COUNT_KEYS: tuple[str, ...] = ("nodes", "nodes_hydrated", "created", "rows_written")
_EDGE_KEYS: tuple[str, ...] = ("edges", "relations_hydrated")

__all__ = ["EtlResult"]


class EtlResult(BaseModel):
    """One ETL step's validated, typed output.

    Canonical fields are typed and validated; anything else a handler already
    returns (``instances``, ``accounts``, ``rows_seen``, …) survives untouched as
    an extra field because ``extra="allow"`` — this is standardization of the
    *contract*, not a schema straitjacket that would force a mass rewrite of every
    ``_sync_*`` handler in one pass.
    """

    model_config = ConfigDict(extra="allow")

    #: Open vocabulary on purpose — ``ok``/``partial``/``error``/``skipped`` are the
    #: common cases, but delegated sub-pipelines (materialize/hydration/chunked
    #: drain) legitimately emit others (``enqueued``, ``materialized``, ``draining``…).
    status: str = "ok"
    source: str | None = None
    sink: str | None = None
    mode: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    watermark: str | None = None
    error: str | None = None
    lineage: dict[str, Any] | None = None
    inbound: dict[str, Any] | None = None
    outbound: dict[str, Any] | None = None

    @staticmethod
    def count_of(
        result: dict[str, Any] | None, keys: tuple[str, ...] = _COUNT_KEYS
    ) -> int:
        """Best-effort record count across the heterogeneous handler result shapes.

        The direct replacement for the old module-level ``etl.pipeline._count()``
        — same behavior (first matching int key wins), now a documented, reusable
        static method instead of a private free function duplicated by callers.
        """
        if not isinstance(result, dict):
            return 0
        for key in keys:
            val = result.get(key)
            if isinstance(val, int):
                return val
        return 0

    @classmethod
    def coerce(cls, data: Any, **defaults: Any) -> EtlResult:
        """Build an :class:`EtlResult` from a raw handler/pipeline dict.

        ``defaults`` are applied via ``setdefault`` — a handler's own values
        always win over the caller-supplied fallback. When ``counts`` is absent
        it is derived from the legacy duck-typed node/edge keys so callers get a
        real, typed ``counts`` dict without every one of the ``_sync_*`` handlers
        needing to be rewritten first (CONCEPT:AU-KG.etl.result-contract).
        """
        if isinstance(data, EtlResult):
            payload: dict[str, Any] = data.model_dump()
        elif isinstance(data, dict):
            payload = dict(data)
        else:
            payload = {"status": "ok"}

        for key, value in defaults.items():
            if value is not None:
                payload.setdefault(key, value)

        if "counts" not in payload:
            counts: dict[str, int] = {}
            nodes = cls.count_of(payload)
            if nodes:
                counts["nodes"] = nodes
            edges = cls.count_of(payload, _EDGE_KEYS)
            if edges:
                counts["edges"] = edges
            if counts:
                payload["counts"] = counts

        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to the plain ``dict`` every existing surface expects."""
        return self.model_dump()
