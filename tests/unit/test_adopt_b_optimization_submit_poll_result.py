"""NE-039 acceptance (AU-ADOPT-B): one valid synthetic optimization run must
reach submit -> poll -> result through the REAL native engine method, not a
stub of it.

``tests/unit/test_optimization_backend.py`` already proves the zero-native-
calls half of ``00ac8be1`` (U-103/U-135) tightly -- ``_MustNotCallEngine``
fails the test outright the instant ``optimize_program`` is invoked for a
no_data/invalid/unavailable disposition, which is a stronger assertion than
counting calls. What it does NOT prove is the "completed" happy path through
the real engine boundary: every "completed"-disposition fixture in that file
(``_NativeEngine``) implements ``optimize_program`` as a single canned
return, bypassing ``GraphComputeEngine.optimize_program``'s own
submit_program_optimization -> (bounded poll loop over) program_optimization_
status -> program_optimization_result pipeline entirely.

This file exercises that real pipeline: :class:`GraphComputeEngine` is
constructed via ``__new__`` (bypassing the transport ``__init__``, exactly the
established convention this repo already uses for a fake ``_client`` --
see e.g. ``tests/unit/knowledge_graph/test_multi_graph_batch_facade.py``,
``tests/unit/knowledge_graph/core/test_graph_compute_provenance.py``) with a
fake ``_client.jobs`` that requires the poll loop to run at least twice
(Submitted -> Running -> Succeeded) before returning a real, fully-validated
``program_candidate`` row shaped exactly like
:data:`agent_utilities.knowledge_graph.core.graph_compute._PROGRAM_RESULT_SCHEMA`
demands. ``try_native_optimization`` is then invoked against that engine with
a valid synthetic request and must reach ``disposition == "completed"``.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.harness.optimization_backend import (
    OptimizationRequest,
    try_native_optimization,
)
from agent_utilities.knowledge_graph.core.graph_compute import (
    _PROGRAM_RESULT_SCHEMA,
    GraphComputeEngine,
)


def _request() -> OptimizationRequest:
    """A valid synthetic request -- same shape as
    ``test_optimization_backend.py``'s own ``_request()`` helper."""
    return OptimizationRequest(
        target="skill",
        objective="skill invocation reliability",
        data={"examples": [{"task": "synthetic-task", "response": "route-a"}]},
    )


class _RecordingJobs:
    """Fakes exactly the ``_client.jobs`` surface
    :meth:`GraphComputeEngine.submit_program_optimization`/
    ``program_optimization_status`` call -- nothing more. The poll loop lives
    in the REAL ``GraphComputeEngine.optimize_program``, not here: this class
    only supplies the durable job-state transitions a real engine would.
    """

    def __init__(self) -> None:
        self.submit_calls: list[tuple[str, bytes, dict[str, Any]]] = []
        self.status_calls: list[str] = []

    def submit_program_optimization(
        self, graph_name: str, payload: bytes, *, purpose: str, max_attempts: int
    ) -> dict[str, Any]:
        self.submit_calls.append(
            (graph_name, payload, {"purpose": purpose, "max_attempts": max_attempts})
        )
        return {"job_id": "job-adopt-b-1", "state": "Submitted"}

    def status(self, job_id: str) -> dict[str, Any]:
        self.status_calls.append(job_id)
        # Force the REAL poll loop in GraphComputeEngine.optimize_program to
        # run at least twice: the first status probe still reports Running,
        # only the second reports Succeeded. A single-call "submit then
        # immediately succeeded" fake would not prove polling actually
        # happens.
        if len(self.status_calls) < 2:
            return {"job_id": job_id, "state": {"Running": {}}}
        row = {
            "id": "eg:cand:0123456789abcdef",
            "kind": "program_candidate",
            "confidence": 0.9,
            "evidence_refs": ["eg:ev:0123456789abcdef"],
            "source_refs": ["eg:src:0123456789abcdef"],
            "proof_ids": [],
            "contradiction_ids": [],
            "program_ref": "eg:program:0123456789abcdef",
            "optimizer": "bootstrap_few_shot",
            "execution": "native_kernel",
            "candidate_role": "proposal",
            "demonstration_refs": ["eg:demo:0123456789abcdef"],
            "artifact_refs": [],
            "composition_refs": [],
            "instruction_ref": None,
            "tool_policy_ref": None,
            "model_profile_ref": None,
            "modalities": ["text"],
            "plan_ref": None,
            "plan_step_kinds": [],
            "plan_executors": [],
            "plan_input_refs": [],
            "plan_output_refs": [],
            "plan_depends_on": [],
            "max_operations": None,
            "selected": True,
        }
        schema = [
            {"name": name, "logical_type": logical_type, "nullable": nullable}
            for name, logical_type, nullable in _PROGRAM_RESULT_SCHEMA
        ]
        return {
            "job_id": job_id,
            "state": {
                "Succeeded": {"result_ref": "eg:program-result:0123456789abcdef"}
            },
            "output": {"schema": schema, "rows": [row]},
        }


class _RecordingClient:
    def __init__(self, jobs: _RecordingJobs) -> None:
        self.jobs = jobs


def test_valid_synthetic_run_reaches_submit_poll_result_through_the_real_engine() -> (
    None
):
    """The positive half NE-039 requires: with valid synthetic data, the
    request travels submit -> poll (>=2 status probes) -> result through the
    REAL ``GraphComputeEngine.optimize_program`` (not a stub of it), and
    ``try_native_optimization`` observes ``disposition == "completed"``."""
    jobs = _RecordingJobs()
    engine = GraphComputeEngine.__new__(GraphComputeEngine)
    engine._client = _RecordingClient(jobs)  # bypass transport __init__
    engine.graph_name = "adopt-b-test-graph"

    result = try_native_optimization(engine, _request())

    assert result.disposition == "completed"
    assert result.error_code == ""
    assert result.payload["status"] == "proposed"
    assert result.payload["result"]["job_id"] == "job-adopt-b-1"
    assert (
        result.payload["result"]["result_ref"] == "eg:program-result:0123456789abcdef"
    )
    assert result.payload["result"]["rows"][0]["id"] == "eg:cand:0123456789abcdef"

    # The point of the fixture: the real submit -> poll(>=2) -> result
    # pipeline actually ran, not a single-shot stub.
    assert len(jobs.submit_calls) == 1
    assert jobs.submit_calls[0][0] == "adopt-b-test-graph"
    assert len(jobs.status_calls) >= 2
    assert jobs.status_calls[0] == "job-adopt-b-1"


def test_valid_synthetic_run_submits_the_exact_program_payload() -> None:
    """The bytes handed to ``jobs.submit_program_optimization`` must be the
    real governed, msgpack-encoded request payload -- proving the request
    that reaches ``submit`` is the one ``OptimizationRequest.to_payload()``
    built, not a placeholder."""
    import msgpack

    jobs = _RecordingJobs()
    engine = GraphComputeEngine.__new__(GraphComputeEngine)
    engine._client = _RecordingClient(jobs)
    engine.graph_name = "adopt-b-test-graph"
    request = _request()
    expected_payload = request.to_payload()

    result = try_native_optimization(engine, request)

    assert result.disposition == "completed"
    submitted_bytes = jobs.submit_calls[0][1]
    decoded = msgpack.unpackb(submitted_bytes, raw=False)
    assert decoded == expected_payload
    assert jobs.submit_calls[0][2] == {
        "purpose": "program-optimization",
        "max_attempts": 1,
    }
