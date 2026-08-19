"""AU acceptance A: the public WorkItem facade reaches native durable verbs.

This is intentionally a real-engine test.  The unit doubles in
``tests/unit/orchestration/test_work_item.py`` prove the Python routing and
fencing branches, while this harness proves that the same one-tenant flow is
actually accepted by the installed epistemic-graph server and survives its
native claim/lease authority.  A missing native metadata CAS verb is a hard
failure when an engine is present: silently falling back to a generic graph
CAS would make the acceptance result unsafe.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from _test_engine import TEST_TENANT

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.orchestration import work_item as wi

pytestmark = [pytest.mark.integration, pytest.mark.engine, pytest.mark.timeout(180)]


@pytest.fixture()
def native_work_item_engine(engine_graph: Any):
    """Bind AU's high-level engine to the real engine fixture's graph.

    ``work_item._authority`` resolves this engine's one control-plane view,
    which is the production path used by task/organization callers.  It keeps
    the test on the native WorkItem authority instead of bypassing it with the
    content graph's generic CRUD helpers.
    """

    # The focused command runs this module in isolation.  Refuse to overwrite
    # another process-owned AU engine if a caller embeds this test in a larger
    # process, because doing so would make the result non-deterministic.
    if IntelligenceGraphEngine.get_active() is not None:
        pytest.fail("a process-owned IntelligenceGraphEngine is already active")

    # The generic ``engine_graph`` fixture provisions only its content graph.
    # Production Graph-OS startup materializes ``__control__`` before queue
    # traffic; do the same explicit admin operation here rather than allowing
    # the WorkItem adapter to create a second or implicit authority.
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
    from agent_utilities.knowledge_graph.core.session import current_session

    session = current_session()
    assert session is not None
    GraphComputeEngine._ensure_local_session_graph(  # type: ignore[attr-defined]
        engine_graph._client,
        "__control__",
        session,
    )

    engine = IntelligenceGraphEngine(
        backend=EpistemicGraphBackend(graph_name=engine_graph.graph_name),
        defer_background_start=True,
    )
    try:
        yield engine
    finally:
        IntelligenceGraphEngine.set_active(None)


def _claim(
    engine: IntelligenceGraphEngine,
    item_id: str,
    *,
    worker: str,
    now: float,
    ttl: float,
) -> dict[str, Any]:
    claim = wi.claim_and_start(
        engine,
        item_id,
        token=worker,
        now=now,
        lease_ttl_s=ttl,
    )
    assert claim is not None
    assert claim["_native"] is True
    return claim


def test_native_work_item_submit_checkpoint_input_priority_conflict_and_readback(
    native_work_item_engine: IntelligenceGraphEngine,
) -> None:
    """Drive the complete AU WorkItem contract through the real native client.

    This item exercises priority, checkpoint, input request/response and an
    explicit native CAS conflict before a terminal commit/readback.
    """

    engine = native_work_item_engine
    cas_authority = engine._work_item_engine

    suffix = uuid4().hex
    item_id = wi.submit_work_item(
        engine,
        kind="acceptance_native_cas",
        queue="acceptance-native",
        payload_ref="payload:acceptance-native",
        tenant=TEST_TENANT,
        priority=2,
        work_item_id=f"workitem:acceptance:native-cas:{suffix}",
        idempotency_key=f"acceptance:native-cas:{suffix}",
        now=1000.0,
    )
    assert item_id.endswith(suffix)
    submitted = wi.get_work_item(engine, item_id)
    assert submitted is not None
    assert submitted["tenant"] == TEST_TENANT
    assert submitted["status"] == "ready"

    # Priority is a native metadata CAS while the item is still ready.
    assert wi.set_work_item_priority(engine, item_id, 3, now=1000.1)
    assert wi.get_work_item(engine, item_id)["prio_bucket"] == 3

    claim = _claim(
        engine, item_id, worker="worker:acceptance-a", now=1001.0, ttl=20.0
    )
    assert wi.checkpoint_work_item(
        engine,
        item_id,
        claim,
        "checkpoint:acceptance:1",
        now=1002.0,
        lease_ttl_s=20.0,
    )
    assert wi.get_work_item(engine, item_id)["checkpoint_id"] == (
        "checkpoint:acceptance:1"
    )

    # Bypass only the AU wrapper's bool collapse to observe the native
    # three-way CAS outcome.  The expected checkpoint is intentionally stale;
    # a generic overwrite would make this acceptance unsafe.
    stale_cas = cas_authority.cas_work_item_metadata(
        {
            "tenant": TEST_TENANT,
            "work_item_id": item_id,
            "expected_status": ["leased", "running"],
            "now_ms": 1002500,
            "expected_lease": {
                "worker_ref": claim["lease_owner"],
                "lease_epoch": claim["lease_epoch"],
                "fencing_token": claim["fencing_token"],
            },
            "expected_checkpoint_id": None,
            "set_checkpoint_id": "checkpoint:acceptance:stale",
            "expected_metadata": None,
            "set_metadata": None,
            "expected_prio_bucket": None,
            "set_prio_bucket": None,
        }
    )
    assert stale_cas["outcome"] == "conflict"
    assert wi.get_work_item(engine, item_id)["checkpoint_id"] == (
        "checkpoint:acceptance:1"
    )

    assert wi.request_work_item_input(
        engine,
        item_id,
        claim,
        request={"prompt_ref": "prompt:acceptance"},
        now=1003.0,
        lease_ttl_s=20.0,
    )
    assert wi.submit_work_item_input(
        engine,
        item_id,
        tenant=TEST_TENANT,
        response={"answer_ref": "answer:acceptance"},
        now=1004.0,
    )
    after_input = wi.get_work_item(engine, item_id)
    assert after_input is not None
    assert after_input["metadata"] == {
        "pending_input_response": {"answer_ref": "answer:acceptance"}
    }
    assert (
        wi.commit_result(
            engine,
            item_id,
            claim,
            outcome="succeeded",
            result_ref="result:acceptance-native",
            retryable=False,
            now=1005.0,
        )
        == "committed"
    )
    completed = wi.get_work_item(engine, item_id)
    assert completed is not None
    assert completed["status"] == "succeeded"
    assert completed["result_ref"] == "result:acceptance-native"


def test_native_work_item_expiry_reclaim_old_fence_and_commit_readback(
    native_work_item_engine: IntelligenceGraphEngine,
) -> None:
    """Prove native expiry recovery and fencing with no local state mirror."""

    engine = native_work_item_engine
    suffix = uuid4().hex
    expiring_id = wi.submit_work_item(
        engine,
        kind="acceptance_native_reclaim",
        queue="acceptance-native",
        payload_ref="payload:acceptance-reclaim",
        tenant=TEST_TENANT,
        work_item_id=f"workitem:acceptance:native-reclaim:{suffix}",
        idempotency_key=f"acceptance:native-reclaim:{suffix}",
        now=2000.0,
    )
    old_claim = _claim(
        engine,
        expiring_id,
        worker="worker:acceptance-old",
        now=2000.0,
        ttl=5.0,
    )
    new_claim = _claim(
        engine,
        expiring_id,
        worker="worker:acceptance-new",
        now=2006.0,
        ttl=20.0,
    )
    assert new_claim["lease_epoch"] > old_claim["lease_epoch"]
    assert new_claim["fencing_token"] > old_claim["fencing_token"]

    assert not wi.checkpoint_work_item(
        engine,
        expiring_id,
        old_claim,
        "checkpoint:acceptance:old-fence",
        now=2007.0,
        lease_ttl_s=20.0,
    )
    assert (
        wi.commit_result(
            engine,
            expiring_id,
            old_claim,
            outcome="succeeded",
            result_ref="result:acceptance-stale",
            retryable=False,
            now=2007.0,
        )
        == "fenced"
    )
    assert (
        wi.commit_result(
            engine,
            expiring_id,
            new_claim,
            outcome="succeeded",
            result_ref="result:acceptance-reclaimed",
            retryable=False,
            now=2008.0,
        )
        == "committed"
    )
    reclaimed = wi.get_work_item(engine, expiring_id)
    assert reclaimed is not None
    assert reclaimed["status"] == "succeeded"
    assert reclaimed["result_ref"] == "result:acceptance-reclaimed"
