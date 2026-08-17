"""``EpistemicSyncWorkflow.run_sync_cycle`` must report an honest outcome.

CONCEPT:AU-AHE.evaluation.return-none-on-failure

``run_sync_cycle`` used to swallow every exception from the ingest+flush and
always return ``None`` regardless of whether the cycle actually did anything.
The one live caller, ``graph_analyze(action="epistemic_sync")``
(``agent_utilities/mcp/tools/analysis_tools.py``), never inspected a return
value either — it unconditionally reported
``{"status": "sync_cycle_completed", "message": "... executed successfully."}``
to its caller no matter what ``run_sync_cycle`` actually did. This proves the
now-fixed contract at the module this lane owns: a genuine failure must be
observable as ``status="failed"``, distinct from a genuine success.
"""

from __future__ import annotations

from agent_utilities.workflows.epistemic_sync import EpistemicSyncWorkflow


def _bare_workflow() -> EpistemicSyncWorkflow:
    """Construct without running __init__ (which stands up real KG/backend
    dependencies) — set only the attributes run_sync_cycle actually reads."""
    return object.__new__(EpistemicSyncWorkflow)


class _FakeConfig:
    sparql_endpoints = ["https://example.invalid/sparql"]


async def test_run_sync_cycle_reports_completed_on_success() -> None:
    wf = _bare_workflow()
    wf.config = _FakeConfig()
    wf.ingestor = type(
        "I", (), {"ingest_entities": lambda self, limit=100: 7}
    )()
    wf.engine = type(
        "E", (), {"flush_ledger_to_backend": lambda self, backend: 3}
    )()
    wf.backend = object()

    outcome = await wf.run_sync_cycle()

    assert outcome == {
        "status": "completed",
        "ingested_count": 7,
        "flushed_count": 3,
    }


async def test_run_sync_cycle_reports_failed_not_none_on_ingest_error() -> None:
    wf = _bare_workflow()
    wf.config = _FakeConfig()

    def _raise(self, limit=100):
        raise RuntimeError("simulated SPARQL endpoint failure")

    wf.ingestor = type("I", (), {"ingest_entities": _raise})()
    wf.engine = type(
        "E", (), {"flush_ledger_to_backend": lambda self, backend: 0}
    )()
    wf.backend = object()

    outcome = await wf.run_sync_cycle()

    assert outcome is not None
    assert outcome["status"] == "failed"
    assert outcome["error"] == "RuntimeError"


async def test_run_sync_cycle_reports_failed_on_flush_error() -> None:
    """The flush step failing (ingest succeeded) must also be distinguishable
    from a clean success — not just the ingest step."""
    wf = _bare_workflow()
    wf.config = _FakeConfig()
    wf.ingestor = type(
        "I", (), {"ingest_entities": lambda self, limit=100: 5}
    )()

    def _raise(self, backend):
        raise RuntimeError("simulated LadybugDB flush failure")

    wf.engine = type("E", (), {"flush_ledger_to_backend": _raise})()
    wf.backend = object()

    outcome = await wf.run_sync_cycle()

    assert outcome["status"] == "failed"
    assert outcome["error"] == "RuntimeError"
