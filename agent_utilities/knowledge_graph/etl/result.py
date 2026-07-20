#!/usr/bin/python
from __future__ import annotations

"""Typed, validated ETL step output (CONCEPT:AU-KG.etl.result-contract).

Assimilates koheesio's ``Step.Output`` idea (``reports/koheesio-etl-analysis.md``
§3.1) without its ``StepMetaClass``/``partialmethod`` machinery: a Pydantic model
that every ETL entrypoint (:func:`~.pipeline.run_etl`,
:func:`~..core.source_sync.sync_source`,
:func:`~..core.table_ingest.ingest_connector_to_table`) threads its result
through. The public wire object contains only these fields. Connector-specific
telemetry is namespaced under ``details`` and is never interpreted as counts.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["EtlResult"]


class EtlResult(BaseModel):
    """One ETL step's validated, typed output.

    Unknown top-level fields are rejected. A connector may retain domain-specific
    diagnostics only inside ``details``.
    """

    model_config = ConfigDict(extra="forbid")

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
    reason: str | None = None
    lineage: dict[str, Any] | None = None
    inbound: EtlResult | None = None
    outbound: EtlResult | None = None
    details: dict[str, Any] = Field(default_factory=dict)
