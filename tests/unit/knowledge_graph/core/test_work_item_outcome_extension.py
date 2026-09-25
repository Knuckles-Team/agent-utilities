"""The optional terminal extension crosses AU adapters unchanged."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

from agent_utilities.knowledge_graph.core.engine_tasks import (
    _ControlPlaneWorkItemEngine,
)
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def test_graph_compute_omits_absent_extension_and_forwards_present_one() -> None:
    calls: list[dict[str, Any]] = []
    engine = object.__new__(GraphComputeEngine)
    engine._client = SimpleNamespace(  # type: ignore[attr-defined]
        work_items=SimpleNamespace(
            commit_result=lambda **kwargs: calls.append(kwargs) or {"status": "committed"}
        )
    )
    request = {
        "tenant": "tenant-a",
        "work_item_id": "wi:1",
        "worker_ref": "worker",
        "expected_epoch": 1,
        "fencing_token": 1,
        "idempotency_key": "commit:1",
        "outcome": "succeeded",
        "now_unix": 10.0,
    }
    engine.commit_work_item_result(request)
    assert "outcome_extension" not in calls[-1]

    extension = {"schema_version": "1", "terminal_outcome": {"run_id": "run:1"}}
    engine.commit_work_item_result({**request, "outcome_extension": extension})
    assert calls[-1]["outcome_extension"] is extension


def test_control_plane_work_item_adapter_forwards_complete_commit_request() -> None:
    calls: list[dict[str, Any]] = []
    adapter = object.__new__(_ControlPlaneWorkItemEngine)
    adapter._control_session_scope = lambda: nullcontext()  # type: ignore[method-assign]
    adapter._native_work_item_method = lambda name: (  # type: ignore[method-assign]
        lambda request: calls.append(request) or {"status": "committed"}
    )
    extension = {"schema_version": "1"}
    request = {"work_item_id": "wi:1", "outcome_extension": extension}
    assert adapter.commit_work_item_result(request) == {"status": "committed"}
    assert calls == [request]
