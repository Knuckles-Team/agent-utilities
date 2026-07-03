"""Workflow → BusinessProcess lineage close-out (CONCEPT:ORCH-1.43).

When an executed workflow REALIZES a harvested BusinessProcess (ORCH-1.41),
completion writes a (:RunTrace)-[:EXECUTED_PROCESS]->(:BusinessProcess)
provenance edge and feeds the optional injected lineage_sink a normalized
record (the egeria-mcp assert_lineage seam).

@pytest.mark.concept("ORCH-1.43")
"""

from __future__ import annotations

import pytest

from agent_utilities.workflows.runner import StepResult, WorkflowResult, WorkflowRunner

pytestmark = pytest.mark.concept("ORCH-1.43")


class FakeGraph:
    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._edges: list[tuple[str, str, dict]] = []

    def add_node(self, node_id, props):
        self._nodes[node_id] = dict(props)

    def add_edge(self, src, tgt, **props):
        self._edges.append((src, tgt, props))

    @property
    def nodes(self):
        outer = self

        class _View(dict):
            def __call__(self, data=False):
                if data:
                    return list(outer._nodes.items())
                return list(outer._nodes)

        return _View(outer._nodes)

    def out_edges(self, node_id, data=False):
        rows = [(s, t, p) for s, t, p in self._edges if s == node_id]
        return rows if data else [(s, t) for s, t, _ in rows]


class FakeEngine:
    def __init__(self):
        self.graph = FakeGraph()
        self.backend = None

    def add_node(self, node_id, node_type, properties=None, **props):
        self.graph.add_node(node_id, {"type": node_type, **(properties or props or {})})

    def link_nodes(self, source, target, rel_type, properties=None):
        self.graph.add_edge(source, target, type=rel_type, **(properties or {}))


PROC = "bpmn_process:invoice:1:abc"
WID = "workflow:invoice_flow:abc123"


def _engine_with_realizes():
    engine = FakeEngine()
    engine.add_node(
        PROC,
        "BusinessProcess",
        {"name": "Invoice Receipt", "externalToolId": "guid-123"},
    )
    engine.add_node(
        WID, "WorkflowDefinition", {"name": "invoice_flow", "step_count": 2}
    )
    engine.link_nodes(WID, PROC, "REALIZES")
    return engine


def _result(status="completed", session_id="run-42"):
    return WorkflowResult(
        workflow_name="invoice_flow",
        session_id=session_id,
        step_results=[
            StepResult(
                step_index=0,
                node_id="review",
                task="t",
                output="ok",
                status="completed",
                duration_ms=10.0,
            )
        ],
        total_duration_ms=12.5,
        status=status,
    )


class TestCloseOut:
    def test_executed_process_edge_written(self):
        engine = _engine_with_realizes()
        runner = WorkflowRunner()
        runner._close_out_process_lineage(engine, "invoice_flow", _result())

        trace = engine.graph._nodes.get("trace:run-42")
        assert trace is not None
        assert trace["type"] == "RunTrace"
        assert trace["status"] == "completed"
        assert trace["workflow_id"] == WID
        executed = [
            (s, t, p)
            for s, t, p in engine.graph._edges
            if p.get("type") == "EXECUTED_PROCESS"
        ]
        assert len(executed) == 1
        src, tgt, props = executed[0]
        assert (src, tgt) == ("trace:run-42", PROC)
        assert props["status"] == "completed"

    def test_sink_called_with_normalized_record(self):
        engine = _engine_with_realizes()
        records = []
        runner = WorkflowRunner(lineage_sink=records.append)
        runner._close_out_process_lineage(engine, "invoice_flow", _result())

        (rec,) = records
        assert rec["process_id"] == PROC
        assert rec["process_external_id"] == "guid-123"
        assert rec["workflow_id"] == WID
        assert rec["workflow_name"] == "invoice_flow"
        assert rec["run_id"] == "run-42"
        assert rec["status"] == "completed"
        assert rec["completed_steps"] == 1
        assert rec["failed_steps"] == 0
        assert rec["duration_ms"] == 12.5
        assert rec["timestamp"].endswith("Z")

    def test_workflow_without_realizes_is_untouched(self):
        engine = FakeEngine()
        engine.add_node(
            WID, "WorkflowDefinition", {"name": "invoice_flow", "step_count": 2}
        )
        records = []
        runner = WorkflowRunner(lineage_sink=records.append)
        runner._close_out_process_lineage(engine, "invoice_flow", _result())
        assert records == []
        assert "trace:run-42" not in engine.graph._nodes

    def test_sink_failure_never_raises(self):
        engine = _engine_with_realizes()

        def _boom(record):
            raise RuntimeError("lineage SoR down")

        runner = WorkflowRunner(lineage_sink=_boom)
        runner._close_out_process_lineage(engine, "invoice_flow", _result())
        # The KG edge still landed despite the sink failure.
        assert any(
            p.get("type") == "EXECUTED_PROCESS" for _s, _t, p in engine.graph._edges
        )

    def test_failed_run_status_recorded(self):
        engine = _engine_with_realizes()
        records = []
        runner = WorkflowRunner(lineage_sink=records.append)
        runner._close_out_process_lineage(
            engine, "invoice_flow", _result(status="failed", session_id="run-43")
        )
        assert records[0]["status"] == "failed"
        assert engine.graph._nodes["trace:run-43"]["status"] == "failed"

    def test_no_engine_is_noop(self):
        runner = WorkflowRunner(lineage_sink=lambda rec: pytest.fail("called"))
        runner._close_out_process_lineage(None, "invoice_flow", _result())


class TestExecuteWiring:
    async def test_execute_runs_close_out_after_completion(self, monkeypatch):
        """WorkflowRunner.execute() invokes the close-out with the result."""
        from agent_utilities.models.graph import ExecutionStep, GraphPlan

        engine = _engine_with_realizes()
        runner = WorkflowRunner()

        class _WaveResult:
            def __init__(self):
                self.results = []

        class _ExecResult:
            execution_id = "run-99"
            wave_results: list = []
            success = True
            total_duration_ms = 5.0
            mermaid = ""

        async def _fake_parallel(plan, engine, workflow_name="", query=""):
            return _ExecResult()

        monkeypatch.setattr(runner, "execute_via_parallel_engine", _fake_parallel)
        seen = {}

        def _spy(eng, name, result):
            seen["name"] = name
            seen["run_id"] = result.session_id

        monkeypatch.setattr(runner, "_close_out_process_lineage", _spy)
        plan = GraphPlan(steps=[ExecutionStep(id="review")])
        await runner.execute(plan, engine, workflow_name="invoice_flow")
        assert seen == {"name": "invoice_flow", "run_id": "run-99"}
