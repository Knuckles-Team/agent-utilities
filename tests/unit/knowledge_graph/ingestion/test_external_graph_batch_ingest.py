"""BUG-CX-010: ``_ingest_envelopes`` (external_graph.py) called
``ingest_envelope`` once per entity in a plain Python ``for`` loop -- N
round-trips over the engine transport, exactly the "N elements in a loop =
N round-trips = catastrophic" defect class the repo's own
``check_no_per_element_ingest_loop.py`` gate exists to catch (per both
``agent-utilities``' and ``epistemic-graph``'s ``AGENTS.md``). The gate
never saw this site: it only scans ``agent_utilities/mcp/`` and a fixed
method-name set that does not include ``ingest_envelope`` -- so this loop
was invisible debt, not a tracked/ratcheted one.

``envelope_ingest.py`` already ships a proper batch primitive,
``ingest_envelopes`` (plural): one native ``ApplyChangeEnvelopes`` round
trip for the whole page, same per-envelope result shape/status vocabulary
as the single-record path, with a built-in graceful fallback to per-record
processing when the native batch method is unavailable. ``_ingest_envelopes``
should route through it instead of hand-rolling the per-element loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.ingestion import external_graph


def test_ingest_envelopes_uses_the_batch_primitive_not_a_per_element_loop(
    monkeypatch,
):
    envelopes = [MagicMock(name=f"envelope-{i}") for i in range(3)]

    per_element_calls: list[object] = []

    def _fake_single(_engine, envelope):
        per_element_calls.append(envelope)
        return {"status": "success"}

    batch_calls: list[list[object]] = []

    def _fake_batch(_engine, envs):
        batch_calls.append(list(envs))
        return [{"status": "success"} for _ in envs]

    monkeypatch.setattr(external_graph, "ingest_envelope", _fake_single)
    monkeypatch.setattr(external_graph, "ingest_envelopes", _fake_batch, raising=False)

    engine = MagicMock(name="authority-engine")
    statuses = external_graph._ingest_envelopes(engine, envelopes)

    assert batch_calls == [envelopes], (
        "_ingest_envelopes must route through the batch primitive "
        "(ingest_envelopes) in a single call, not loop ingest_envelope "
        "per element"
    )
    assert per_element_calls == [], (
        "_ingest_envelopes called the per-element ingest_envelope directly "
        "instead of the batch primitive -- N round trips over the engine "
        "transport for N envelopes"
    )
    assert statuses["success"] == 3
