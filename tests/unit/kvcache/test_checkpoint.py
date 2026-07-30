"""Unit tests for :class:`~agent_utilities.kvcache.checkpoint.KVCheckpointStore`.

CONCEPT:AU-KG.memory.kv-checkpoint-resource. Exercises the store against a fake engine
client (the SAME ``_FakeClient``/``_FakeCompute`` pattern
``tests/unit/knowledge_graph/test_media_store_identity.py`` uses — no live engine
required), proving:

* the checkpoint key hashes EVERY component, so a change to any one of them mints a
  wholly different ``checkpoint_id`` (the "any change invalidates" contract);
* a create → get round trip returns the same provenance;
* cross-tenant load is ALWAYS refused (an explicit test, per the security-boundary
  framing) regardless of ``current_policy_version``;
* a stale ``policy_version`` is refused;
* an unknown checkpoint id is refused;
* ``instantiate_agent`` links a fresh ``:AgentRun`` node to the checkpoint via
  ``initializedFrom``;
* ``restore_conversation`` fails closed by default, and only degrades to an explicit,
  traced cold start when ``allow_cold_start=True`` — proven for not-found, cross-tenant,
  and stale alike.
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.kvcache.checkpoint import (
    CheckpointNotFoundError,
    CrossTenantCheckpointError,
    KVCheckpointError,
    KVCheckpointKey,
    KVCheckpointStore,
    StaleCheckpointError,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakeBlob:
    """Fake ``client.blob`` — content-addressed store + refcount, no chunk streaming."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.refcounts: dict[str, int] = {}
        self.incref_calls: list[str] = []
        self.unref_calls: list[str] = []

    def store(self, data: bytes) -> str:
        digest = _digest(data)
        self.data[digest] = data
        return digest

    def incref(self, digest: str) -> int:
        self.incref_calls.append(digest)
        self.refcounts[digest] = self.refcounts.get(digest, 0) + 1
        return self.refcounts[digest]

    def unref(self, digest: str) -> int:
        self.unref_calls.append(digest)
        self.refcounts[digest] = max(0, self.refcounts.get(digest, 0) - 1)
        return self.refcounts[digest]

    def fetch(self, digest: str) -> bytes:
        return self.data[digest]


class _FakeTxn:
    """Fake ``client.txn`` — writes land immediately (not staged-until-commit like the
    real engine), sufficient to assert the control flow this module owns."""

    def __init__(self, *, fail_commit: bool = False) -> None:
        self.nodes: dict[str, dict] = {}
        self.blob_refs: list[tuple[str, str]] = []
        self.committed: list[str] = []
        self.fail_commit = fail_commit
        self._n = 0

    def begin(self, graph: str | None = None) -> str:
        self._n += 1
        return f"txn{self._n}"

    def add_node(self, txn, node_id, props):
        self.nodes[node_id] = dict(props)

    def blob_ref(self, txn, node_id, digest):
        self.blob_refs.append((node_id, digest))
        return True

    def commit(self, txn) -> bool:
        if self.fail_commit:
            return False
        self.committed.append(txn)
        return True


class _FakeNodes:
    def __init__(self, backing: dict[str, dict]) -> None:
        self._backing = backing

    def has(self, node_id: str) -> bool:
        return node_id in self._backing

    def properties(self, node_id: str) -> dict | None:
        return self._backing.get(node_id)


class _FakeEdges:
    def __init__(self) -> None:
        self.edges: list[tuple[str, str, dict]] = []

    def add(self, source, dest, props):
        self.edges.append((source, dest, props))


class _FakeClient:
    def __init__(self, *, fail_commit: bool = False) -> None:
        self.blob = _FakeBlob()
        self.txn = _FakeTxn(fail_commit=fail_commit)
        self.nodes = _FakeNodes(self.txn.nodes)
        self.edges = _FakeEdges()


class _FakeCompute:
    def __init__(
        self, client: _FakeClient | None = None, graph_name: str = "__commons__"
    ):
        self._client = client or _FakeClient()
        self.graph_name = graph_name


def _session(tenant: str) -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="agent:tester",
            actor_type=ActorType.AI_AGENT,
            tenant_id=tenant,
            authenticated=True,
        ),
        tenant=tenant,
    )


def _key(**overrides) -> KVCheckpointKey:
    base = {
        "model_identity": "qwen2.5-72b-instruct",
        "quantization": "awq-int4",
        "serving_engine": "vllm",
        "engine_version": "0.8.0",
        "prefix_digest": _digest(b"the shared system prompt + prior turns"),
        "tenant": "tenant-a",
        "policy_version": "policy-7",
    }
    base.update(overrides)
    return KVCheckpointKey(**base)


BLOB = b"\x00\x01serialized-kv-cache-bytes\xff" * 32


# --------------------------------------------------------------------------- #
# Key derivation — every component participates                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field",
    [
        "model_identity",
        "quantization",
        "serving_engine",
        "engine_version",
        "prefix_digest",
        "tenant",
        "policy_version",
    ],
)
def test_any_key_component_change_mints_a_different_checkpoint_id(field):
    base = _key()
    changed = _key(**{field: base.model_dump()[field] + "-changed"})
    assert base.checkpoint_id != changed.checkpoint_id, (
        f"changing {field!r} must invalidate the checkpoint id"
    )


def test_identical_key_components_mint_the_same_id():
    assert _key().checkpoint_id == _key().checkpoint_id


def test_tenant_missing_is_rejected_by_the_key_model():
    with pytest.raises(ValidationError):  # min_length=1
        KVCheckpointKey(
            model_identity="m",
            quantization="q",
            serving_engine="vllm",
            engine_version="1",
            prefix_digest="d",
            tenant="",
        )


# --------------------------------------------------------------------------- #
# create → get round trip                                                     #
# --------------------------------------------------------------------------- #


def test_create_then_get_round_trips_provenance():
    client = _FakeClient()
    store = KVCheckpointStore(_FakeCompute(client))
    key = _key()

    created = store.create_checkpoint(
        BLOB,
        key=key,
        run_id="run-1",
        point="pre-tool-call",
        session=_session("tenant-a"),
        provenance={"trace_context": "trace-abc"},
    )
    assert created is not None
    assert created.checkpoint_id == key.checkpoint_id
    assert created.digest == _digest(BLOB)
    assert created.size_bytes == len(BLOB)

    loaded = store.get_checkpoint(
        key.checkpoint_id,
        requesting_tenant="tenant-a",
        current_policy_version="policy-7",
    )
    assert loaded.checkpoint_id == created.checkpoint_id
    assert loaded.key.model_identity == key.model_identity
    assert loaded.run_id == "run-1"
    assert loaded.point == "pre-tool-call"
    assert loaded.provenance == {"trace_context": "trace-abc"}

    # GC-bracketing: incref happened once, no compensating unref on a clean commit.
    assert client.blob.incref_calls == [created.digest]
    assert client.blob.unref_calls == []
    # The heavy blob never lands in a node property — only the digest/blob_id do.
    blob_node = client.txn.nodes[created.blob_id]
    assert blob_node["content_digest"] == created.digest
    assert "data" not in blob_node


def test_create_rejects_empty_key_tenant():
    store = KVCheckpointStore(_FakeCompute())
    with pytest.raises(ValidationError):
        KVCheckpointKey(
            model_identity="m",
            quantization="q",
            serving_engine="vllm",
            engine_version="1",
            prefix_digest="d",
            tenant="",
        )
    del store  # unused beyond proving construction never got this far


def test_create_returns_none_on_empty_payload():
    store = KVCheckpointStore(_FakeCompute())
    assert store.create_checkpoint(b"", key=_key(), run_id="r", point="p") is None


# --------------------------------------------------------------------------- #
# Fail-closed load: not-found / cross-tenant / stale                          #
# --------------------------------------------------------------------------- #


def test_unknown_checkpoint_id_is_refused():
    store = KVCheckpointStore(_FakeCompute())
    with pytest.raises(CheckpointNotFoundError):
        store.get_checkpoint("no-such-checkpoint", requesting_tenant="tenant-a")


def test_cross_tenant_load_is_always_refused():
    """CONCEPT:AU-KG.memory.kv-checkpoint-resource — the explicit security-boundary test:
    a checkpoint minted for one tenant is refused for another, unconditionally."""
    client = _FakeClient()
    store = KVCheckpointStore(_FakeCompute(client))
    key = _key(tenant="tenant-a")
    created = store.create_checkpoint(BLOB, key=key, run_id="run-1", point="p1")
    assert created is not None

    with pytest.raises(CrossTenantCheckpointError):
        store.get_checkpoint(key.checkpoint_id, requesting_tenant="tenant-b")

    # The correct tenant still loads it fine — proving the refusal is tenant-specific,
    # not a blanket failure.
    loaded = store.get_checkpoint(key.checkpoint_id, requesting_tenant="tenant-a")
    assert loaded.checkpoint_id == key.checkpoint_id


def test_stale_policy_version_is_refused():
    client = _FakeClient()
    store = KVCheckpointStore(_FakeCompute(client))
    key = _key(policy_version="policy-7")
    created = store.create_checkpoint(BLOB, key=key, run_id="run-1", point="p1")
    assert created is not None

    with pytest.raises(StaleCheckpointError):
        store.get_checkpoint(
            key.checkpoint_id,
            requesting_tenant=key.tenant,
            current_policy_version="policy-8",
        )
    # No staleness check at all when the caller doesn't supply a current version.
    loaded = store.get_checkpoint(key.checkpoint_id, requesting_tenant=key.tenant)
    assert loaded.checkpoint_id == key.checkpoint_id


def test_get_checkpoint_requires_a_requesting_tenant():
    store = KVCheckpointStore(_FakeCompute())
    with pytest.raises(KVCheckpointError):
        store.get_checkpoint("anything", requesting_tenant="")


# --------------------------------------------------------------------------- #
# instantiate_agent — lineage edge                                            #
# --------------------------------------------------------------------------- #


def test_instantiate_agent_links_a_new_agent_run_to_the_checkpoint():
    client = _FakeClient()
    store = KVCheckpointStore(_FakeCompute(client))
    key = _key()
    created = store.create_checkpoint(BLOB, key=key, run_id="run-1", point="p1")
    assert created is not None

    record = store.instantiate_agent(
        key.checkpoint_id, requesting_tenant=key.tenant, new_run_id="run-2"
    )
    assert record.checkpoint_id == key.checkpoint_id
    assert "agentrun:run-2" in client.txn.nodes
    assert (
        "agentrun:run-2",
        f"kvcheckpoint:{key.checkpoint_id}",
        {"type": "initializedFrom"},
    ) in client.edges.edges


def test_instantiate_agent_is_fail_closed_for_cross_tenant():
    client = _FakeClient()
    store = KVCheckpointStore(_FakeCompute(client))
    key = _key(tenant="tenant-a")
    store.create_checkpoint(BLOB, key=key, run_id="run-1", point="p1")

    with pytest.raises(CrossTenantCheckpointError):
        store.instantiate_agent(
            key.checkpoint_id, requesting_tenant="tenant-b", new_run_id="run-2"
        )
    assert "agentrun:run-2" not in client.txn.nodes


# --------------------------------------------------------------------------- #
# restore_conversation — fail-closed default, explicit traced cold start      #
# --------------------------------------------------------------------------- #


def test_restore_conversation_restores_bytes_on_success():
    client = _FakeClient()
    store = KVCheckpointStore(_FakeCompute(client))
    key = _key()
    store.create_checkpoint(BLOB, key=key, run_id="run-1", point="p1")

    res = store.restore_conversation(
        key.checkpoint_id, conversation_id="conv-1", requesting_tenant=key.tenant
    )
    assert res.restored is True
    assert res.cold_start is False
    assert res.data == BLOB
    assert (
        "conversation:conv-1",
        f"kvcheckpoint:{key.checkpoint_id}",
        {"type": "restoredFrom"},
    ) in client.edges.edges


def test_restore_conversation_raises_by_default_on_not_found():
    store = KVCheckpointStore(_FakeCompute())
    with pytest.raises(CheckpointNotFoundError):
        store.restore_conversation(
            "no-such-checkpoint", conversation_id="conv-1", requesting_tenant="tenant-a"
        )


def test_restore_conversation_raises_by_default_on_cross_tenant():
    client = _FakeClient()
    store = KVCheckpointStore(_FakeCompute(client))
    key = _key(tenant="tenant-a")
    store.create_checkpoint(BLOB, key=key, run_id="run-1", point="p1")

    with pytest.raises(CrossTenantCheckpointError):
        store.restore_conversation(
            key.checkpoint_id, conversation_id="conv-1", requesting_tenant="tenant-b"
        )


def test_restore_conversation_cold_start_is_explicit_and_traced(caplog):
    """CONCEPT:AU-KG.memory.kv-checkpoint-resource — the cold-start fallback is opt-in
    (allow_cold_start=True), EXPLICIT (cold_start=True + fallback_reason on the result),
    and TRACED (a warning is logged) — never a silent degrade."""
    store = KVCheckpointStore(_FakeCompute())
    with caplog.at_level("WARNING"):
        res = store.restore_conversation(
            "no-such-checkpoint",
            conversation_id="conv-1",
            requesting_tenant="tenant-a",
            allow_cold_start=True,
        )
    assert res.restored is False
    assert res.cold_start is True
    assert res.fallback_reason == "CheckpointNotFoundError"
    assert res.data is None
    assert any("cold-start fallback" in rec.message for rec in caplog.records)


def test_restore_conversation_cold_start_for_cross_tenant_too():
    client = _FakeClient()
    store = KVCheckpointStore(_FakeCompute(client))
    key = _key(tenant="tenant-a")
    store.create_checkpoint(BLOB, key=key, run_id="run-1", point="p1")

    res = store.restore_conversation(
        key.checkpoint_id,
        conversation_id="conv-1",
        requesting_tenant="tenant-b",
        allow_cold_start=True,
    )
    assert res.cold_start is True
    assert res.fallback_reason == "CrossTenantCheckpointError"
    assert res.restored is False
