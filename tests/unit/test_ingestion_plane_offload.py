"""Ingestion/serving plane separation (CONCEPT:KG-2.130).

In a SERVING role (not the daemon host), heavy KG writes are enqueued to the durable task
queue instead of embedding+writing inline — so ingestion can never contend with reads/replies
on the shared connection pool (the systemic cause of "received but no reply"). The host /
ingest-worker process performs the actual write.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.core import engine_memory, host_lock


class _StubEngine:
    # Bind the real method under test onto a minimal fake — the serving-role branch only
    # needs ``submit_task`` (it returns before touching the retriever/backend).
    store_memory = engine_memory.MemoryMixin.store_memory

    def __init__(self) -> None:
        self.tasks: list[dict[str, Any]] = []

    def submit_task(self, **kwargs: Any) -> str:
        self.tasks.append(kwargs)
        return "job-test"


def test_serving_role_enqueues_instead_of_writing(monkeypatch) -> None:
    monkeypatch.setattr(host_lock, "effective_daemon_role", lambda: "client")
    eng = _StubEngine()
    mid = eng.store_memory("hello world", memory_type="episodic", agent_id="tester")

    assert mid.startswith("mem:")
    assert len(eng.tasks) == 1
    task = eng.tasks[0]
    assert task["task_type"] == "kg_memory"
    payload = task["extra_meta"]["payload"]
    assert payload["content"] == "hello world"
    assert payload["memory_id"] == mid  # the enqueued id is the returned id


def test_local_flag_never_enqueues(monkeypatch) -> None:
    # The task handler calls store_memory(_local=True) so it must NOT re-enqueue, even in a
    # non-host role. It would proceed to the inline write (which needs more deps), so we only
    # assert the offload branch is skipped.
    monkeypatch.setattr(host_lock, "effective_daemon_role", lambda: "client")
    eng = _StubEngine()
    try:
        eng.store_memory("x", _local=True)
    except Exception:  # noqa: BLE001 — inline path needs a real retriever/backend; fine
        pass
    assert eng.tasks == []  # never enqueued
