#!/usr/bin/python
"""TMS live-wiring — Seam 3 completion (W3.2, CONCEPT:EG-KG.epistemic.truth-maintenance).

The surpass-6mo audit (``reports/surpass-6mo/01-eg-epistemic-core.md`` item 2)
found the engine side fully shipped (a durable, always-on reasoning-projection
worker auto-registers a materialization the moment a derived node/edge carries
recognized provenance — ``invalidation_deps`` or a ``:DerivedFrom``/
``:GeneratedBy`` edge — and serves ``MaterializationStatus``/
``StaleMaterializations`` reads) but the AU side never actually wired into it:

* ``GraphComputeEngine.register_materialization`` called a
  ``Method::RegisterMaterialization`` wire method THAT DOES NOT EXIST on the
  real engine protocol or client (confirmed absent from both
  ``epistemic_graph/client.py`` and ``crates/eg-types/src/protocol.rs``) — every
  real call raised ``AttributeError``, silently swallowed by every caller's
  best-effort ``except Exception``. Every unit test that appeared to cover
  this used its OWN standalone stub engine that never touched the real
  ``GraphComputeEngine`` class, so the break went completely undetected.
* ``candidate_insight.register_claim_materialization`` and
  ``capability_designation._register_capability_reward_materialization`` both
  called ``engine.add_edge(a, b, relationship_type="DERIVED_FROM")`` — a
  keyword name that does not exist on the real
  ``KnowledgeGraphEngine.add_edge``/``GraphComputeEngine.add_edge`` (both
  require ``rel_type`` positionally and explicitly reject ``relationship_type``
  as a retired alias) — so the SAME provenance edges that were supposed to
  trigger the engine's automatic registration were never actually written
  either. Existing tests used loosely-typed stubs that happened to accept the
  wrong keyword, masking this too.
* Nothing ever consumed ``StaleMaterializations`` — a materialization could go
  ``Stale`` and sit there forever.

This file proves the fix: the two broken calling conventions now match the
real engine contract, and the new ``tms_revalidation`` maintenance task
actually consumes staleness and routes it to the right owner action.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.graph.routing.enrichers.capability_designation import (
    _register_capability_reward_materialization,
)
from agent_utilities.knowledge_graph.adaptation.tms_revalidation import (
    revalidate_stale_materializations,
)
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.research.candidate_insight import (
    register_claim_materialization,
)
from agent_utilities.models.knowledge_graph import ClaimNode, RegistryNodeType

pytestmark = pytest.mark.concept("EG-KG.epistemic.truth-maintenance")


# ---------------------------------------------------------------------------
# 1. The AU-side registration primitive: GraphComputeEngine.register_
#    materialization / materialization_status / stale_materializations.
# ---------------------------------------------------------------------------


class _FakeQueryNamespace:
    """Stands in for ``SyncEpistemicGraphClient.query`` — the real object
    ``self._client.query.<method>`` resolves against."""

    def __init__(self) -> None:
        self.materialization_status_calls: list[str] = []
        self.stale_materializations_calls = 0
        self._status: dict[str, str] = {}
        self._stale_ids: list[str] = []

    def materialization_status(self, id: str) -> dict[str, Any]:  # noqa: A002
        self.materialization_status_calls.append(id)
        status = self._status.get(id)
        return {"status": status, "source_graph_version": 1}

    def stale_materializations(self) -> dict[str, Any]:
        self.stale_materializations_calls += 1
        return {"ids": list(self._stale_ids), "source_graph_version": 1}

    # Confirm the phantom RPC genuinely does not exist on this fake either —
    # a regression back to calling it would raise AttributeError here too,
    # exactly like the real client.
    def __getattr__(self, name: str) -> Any:
        if name == "register_materialization":
            raise AttributeError(name)
        raise AttributeError(name)


class _FakeClient:
    def __init__(self, query: _FakeQueryNamespace) -> None:
        self.query = query


def _bare_engine(query: _FakeQueryNamespace) -> GraphComputeEngine:
    eng = GraphComputeEngine.__new__(GraphComputeEngine)
    eng._client = _FakeClient(query)  # type: ignore[assignment]
    return eng


def test_register_materialization_no_longer_calls_the_phantom_rpc():
    """The historical bug: this must NEVER call ``self._client.query.
    register_materialization`` (it doesn't exist) — it must succeed via the
    real ``materialization_status`` read instead."""
    query = _FakeQueryNamespace()
    query._status["claim:x"] = "Fresh"
    eng = _bare_engine(query)

    result = eng.register_materialization("claim:x")

    assert result == {"id": "claim:x", "status": "Fresh"}
    assert query.materialization_status_calls == ["claim:x"]


def test_register_materialization_reports_none_when_not_yet_registered():
    """Calling this immediately after a write may race the async projection —
    ``None`` (not yet visible) is a legitimate answer, not an error."""
    query = _FakeQueryNamespace()
    eng = _bare_engine(query)

    assert eng.register_materialization("claim:new") == {
        "id": "claim:new",
        "status": None,
    }


def test_materialization_status_reads_through_to_the_real_wire_method():
    query = _FakeQueryNamespace()
    query._status["claim:x"] = "Stale"
    eng = _bare_engine(query)

    assert eng.materialization_status("claim:x") == "Stale"


def test_stale_materializations_unwraps_the_real_ids_field():
    query = _FakeQueryNamespace()
    query._stale_ids = ["eg:reasoning:aaa", "eg:reasoning:bbb"]
    eng = _bare_engine(query)

    assert eng.stale_materializations() == ["eg:reasoning:aaa", "eg:reasoning:bbb"]
    assert query.stale_materializations_calls == 1


def test_stale_materializations_empty_when_nothing_stale():
    eng = _bare_engine(_FakeQueryNamespace())
    assert eng.stale_materializations() == []


# ---------------------------------------------------------------------------
# 2. Producer (b) — candidate_insight.register_claim_materialization.
#    A STRICT double mirroring the REAL KnowledgeGraphEngine.add_edge/
#    add_node signature contract (rel_type positional, retired-alias
#    rejection) — a regression back to `relationship_type=` would raise here
#    exactly like the real engine, unlike the loosely-typed stubs the
#    pre-existing test suite used.
# ---------------------------------------------------------------------------


class _StrictEngine:
    """Mirrors ``KnowledgeGraphEngine.add_node``/``add_edge``'s real contract."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        **_kw: Any,
    ) -> None:
        props = dict(properties or {})
        if "type" in props:
            raise ValueError(
                "node property 'type' is retired; use the node_type argument"
            )
        self.nodes[node_id] = {"node_type": node_type, **props}

    def add_edge(
        self,
        source: str,
        target: str,
        rel_type: str = "",
        ephemeral: bool = False,
        *,
        session: Any = None,
        **properties: Any,
    ) -> None:
        aliases = {"type", "rel_type", "relationship_type", "relation"}.intersection(
            properties
        )
        if aliases:
            raise ValueError(
                f"edge relationship aliases are retired ({', '.join(sorted(aliases))}); "
                "use the rel_type argument"
            )
        if not str(rel_type).strip():
            raise ValueError("rel_type is required")
        self.edges.append((source, target, rel_type))

    def register_materialization(self, derived_id: str) -> dict[str, Any]:
        return {"id": derived_id, "status": None}


def _claim(source_ids: list[str], claim_id: str = "claim:test:1") -> ClaimNode:
    return ClaimNode(
        id=claim_id,
        type=RegistryNodeType.CLAIM,
        name="test claim",
        claim_text="a claim citing real evidence",
        confidence=0.9,
        claim_type="finding",
        source_ids=source_ids,
        is_verified=False,
    )


def test_claim_materialization_writes_derived_from_edges_with_the_real_calling_convention():
    engine = _StrictEngine()
    claim = _claim(["base:fact1", "base:fact2"])
    errors: list[str] = []

    register_claim_materialization(engine, claim, errors, context="test_ctx")

    assert errors == []
    assert set(engine.edges) == {
        ("claim:test:1", "base:fact1", "DERIVED_FROM"),
        ("claim:test:1", "base:fact2", "DERIVED_FROM"),
    }


# ---------------------------------------------------------------------------
# 3. Producer (a) — capability_designation._register_capability_reward_
#    materialization. Same strict-signature proof.
# ---------------------------------------------------------------------------


def test_capability_reward_materialization_writes_derived_from_edge_with_the_real_calling_convention():
    engine = _StrictEngine()

    _register_capability_reward_materialization(
        engine, "tool:search", ["obs:outcome:1"]
    )

    assert engine.edges == [("tool:search", "obs:outcome:1", "DERIVED_FROM")]


# ---------------------------------------------------------------------------
# 4. tms_revalidation — a FAITHFUL in-process TruthMaintenance double: tracks
#    per-id versions and a dependency snapshot at registration time, so a
#    later "committed change" to a base fact is genuinely detectable as
#    staleness — the same fidelity ``test_insight_validation.py``'s
#    ``_TmsAwareInsightStubEngine`` already established for one producer,
#    generalized here across all three owner kinds plus the query/probe
#    surface `tms_revalidation` itself needs.
# ---------------------------------------------------------------------------


class FaithfulTmsEngine:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.deleted_nodes: list[str] = []
        self._versions: dict[str, int] = {}
        self._deps: dict[str, set[str]] = {}
        self._snapshot: dict[str, dict[str, int]] = {}
        self._retracted: set[str] = set()
        self._capability_index_watcher: Any = None

    # -- write surface -----------------------------------------------------
    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        **_kw: Any,
    ) -> None:
        props = dict(properties or {})
        self.nodes[node_id] = {"node_type": node_type, **props}
        self._versions[node_id] = self._versions.get(node_id, 0) + 1
        self._retracted.discard(node_id)
        deps = set(props.get("invalidation_deps") or ())
        if deps:
            self._deps[node_id] = deps
            self._resnapshot(node_id)

    def add_edge(
        self, source: str, target: str, rel_type: str = "", **_kw: Any
    ) -> None:
        self.edges.append((source, target, rel_type))
        if rel_type in ("DERIVED_FROM", "GENERATED_BY"):
            self._deps.setdefault(source, set()).add(target)
            self._retracted.discard(source)
            self._resnapshot(source)

    def delete_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self._deps.pop(node_id, None)
        self._snapshot.pop(node_id, None)
        self._retracted.add(node_id)
        self.deleted_nodes.append(node_id)

    def _resnapshot(self, materialization_id: str) -> None:
        self._snapshot[materialization_id] = {
            dep: self._versions.get(dep, 0) for dep in self._deps[materialization_id]
        }

    # -- test helper ---------------------------------------------------
    def bump(self, node_id: str) -> None:
        """Simulate a committed change to a base fact through the normal
        write path (e.g. a CompareAndSetNodeFields)."""
        self._versions[node_id] = self._versions.get(node_id, 0) + 1

    # -- TMS read surface ----------------------------------------------
    def register_materialization(self, derived_id: str) -> dict[str, Any]:
        return {"id": derived_id, "status": self.materialization_status(derived_id)}

    def materialization_status(self, node_id: str) -> str | None:
        if node_id in self._retracted:
            return "Retracted"
        if node_id not in self._deps:
            return None
        snap = self._snapshot.get(node_id, {})
        if any(self._versions.get(dep, 0) != version for dep, version in snap.items()):
            return "Stale"
        return "Fresh"

    def stale_materializations(self) -> list[str]:
        return [
            f"eg:reasoning:{i}"
            for i, mid in enumerate(self._deps)
            if self.materialization_status(mid) == "Stale"
        ]

    # -- candidate surface ------------------------------------------------
    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        limit = int((params or {}).get("limit", 1_000_000))
        if "Claim" in query:
            rows = [
                {"id": nid}
                for nid, props in self.nodes.items()
                if props.get("node_type") == "Claim"
            ]
        elif "capability_reward" in query:
            rows = [
                {"id": nid}
                for nid, props in self.nodes.items()
                if props.get("capability_reward") is not None
            ]
        elif "ContextBundleMaterialization" in query:
            rows = [
                {"id": nid, "cache_key": props.get("cache_key")}
                for nid, props in self.nodes.items()
                if props.get("node_type") == "ContextBundleMaterialization"
            ]
        else:
            rows = []
        return sorted(rows, key=lambda r: r["id"])[:limit]


class _FakeCapabilityIndex:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def remove(self, entity_id: str) -> bool:
        self.removed.append(entity_id)
        return True


class _FakeWatcher:
    def __init__(self, index: _FakeCapabilityIndex) -> None:
        self.index = index


def test_degrades_soft_when_engine_has_no_stale_materializations():
    class _NoTms:
        pass

    report = revalidate_stale_materializations(_NoTms())
    assert report == {
        "scanned": 0,
        "stale": 0,
        "revalidated": {},
        "errors": ["engine has no stale_materializations"],
    }


def test_degrades_soft_when_stale_materializations_raises():
    class _DeniedTms:
        def stale_materializations(self) -> list[str]:
            raise PermissionError("denied")

    report = revalidate_stale_materializations(_DeniedTms())
    assert report["scanned"] == 0
    assert report["stale"] == 0
    assert any("denied" in e for e in report["errors"])


def test_cheap_gate_skips_every_probe_when_nothing_is_stale():
    """Proves the design's cheap gate: an empty ``stale_materializations()``
    must short-circuit the WHOLE tick — no ``query_cypher`` candidate scan,
    no ``materialization_status`` probe."""

    class _NothingStaleEngine(FaithfulTmsEngine):
        def query_cypher(self, query, params=None):  # noqa: ANN001
            raise AssertionError("must not scan for candidates when nothing is stale")

        def materialization_status(self, node_id):  # noqa: ANN001
            raise AssertionError("must not probe status when nothing is stale")

    eng = _NothingStaleEngine()
    report = revalidate_stale_materializations(eng)
    assert report == {"scanned": 0, "stale": 0, "revalidated": {}, "errors": []}


def test_end_to_end_claim_register_invalidate_tick_revalidates():
    """Register -> invalidate a base fact -> tick -> owner re-validation
    invoked. Propose-only: the Claim node itself is NEVER mutated."""
    eng = FaithfulTmsEngine()
    eng.add_node("base:fact1", "Fact", properties={})
    claim = _claim(["base:fact1"])
    errors: list[str] = []
    eng.add_node(
        claim.id,
        "Claim",
        properties={
            **claim.model_dump(mode="json", exclude={"type"}),
            "status": "proposal",
        },
    )
    register_claim_materialization(eng, claim, errors, context="test")
    assert errors == []
    assert eng.materialization_status(claim.id) == "Fresh"

    # Persistence proof: the durable state is entirely engine-side (redb-
    # backed reasoning projection in production; this fake's own dict here) —
    # nothing AU-side remembers anything between calls, so a FRESH call to
    # `revalidate_stale_materializations` (simulating a process restart, since
    # it is stateless and re-reads from scratch every time) still works.
    eng.bump("base:fact1")
    assert eng.materialization_status(claim.id) == "Stale"

    claim_version_before = dict(eng.nodes[claim.id])
    report = revalidate_stale_materializations(eng)

    assert report["stale"] == 1
    assert report["revalidated"]["claim"] == 1
    proposals = {
        nid: props
        for nid, props in eng.nodes.items()
        if props.get("node_type") == "BeliefRevisionProposal"
    }
    assert len(proposals) == 1
    (proposal,) = proposals.values()
    assert proposal["status"] == "proposal"
    assert proposal["belief_id"] == claim.id

    # Propose-only invariant: the Claim node's own stored properties are
    # byte-for-byte unchanged — no compare_and_set, no re-``add_node`` on the
    # same id, no deletion.
    assert eng.nodes[claim.id] == claim_version_before


def test_end_to_end_capability_index_register_invalidate_tick_evicts_cache():
    eng = FaithfulTmsEngine()
    watcher = _FakeWatcher(_FakeCapabilityIndex())
    eng._capability_index_watcher = watcher

    eng.add_node("base:signal", "Fact", properties={})
    eng.add_node("tool:search", "Tool", properties={"capability_reward": 0.8})
    _register_capability_reward_materialization(eng, "tool:search", ["base:signal"])
    assert eng.materialization_status("tool:search") == "Fresh"

    eng.bump("base:signal")
    assert eng.materialization_status("tool:search") == "Stale"

    report = revalidate_stale_materializations(eng)

    assert report["revalidated"]["capability_index"] == 1
    assert watcher.index.removed == ["tool:search"]
    # A pure cache invalidation: the durable capability_reward property is
    # untouched (never mutated by revalidation).
    assert eng.nodes["tool:search"]["capability_reward"] == 0.8


def test_end_to_end_context_bundle_register_invalidate_tick_drops_from_cache(
    monkeypatch,
):
    eng = FaithfulTmsEngine()
    eng.add_node("claim:cited", "Claim", properties={})
    eng.add_node(
        "context_bundle:ck1",
        "ContextBundleMaterialization",
        properties={"invalidation_deps": ["claim:cited"], "cache_key": "ck1"},
    )
    assert eng.materialization_status("context_bundle:ck1") == "Fresh"

    class _FakeKv:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def delete(self, key: str) -> None:
            self.deleted.append(key)

    kv = _FakeKv()
    monkeypatch.setattr(
        "agent_utilities.core.contextual_model.get_context_compiler_cache",
        lambda: kv,
    )

    eng.bump("claim:cited")
    assert eng.materialization_status("context_bundle:ck1") == "Stale"

    report = revalidate_stale_materializations(eng)

    assert report["revalidated"]["context_bundle"] == 1
    assert kv.deleted == ["ck1"]
    assert eng.deleted_nodes == ["context_bundle:ck1"]
    # Retired, terminal state — never re-surfaces on a later tick.
    assert eng.materialization_status("context_bundle:ck1") == "Retracted"
    report_again = revalidate_stale_materializations(eng)
    assert report_again["revalidated"].get("context_bundle", 0) == 0


def test_context_bundle_revalidation_degrades_soft_without_a_configured_kv_backend(
    monkeypatch,
):
    """No process-wide KV cache configured (e.g. a deployment without
    Seam 6) — dropping from cache is skipped, but the marker node is still
    retired so it never re-surfaces."""
    eng = FaithfulTmsEngine()
    eng.add_node("claim:cited", "Claim", properties={})
    eng.add_node(
        "context_bundle:ck2",
        "ContextBundleMaterialization",
        properties={"invalidation_deps": ["claim:cited"], "cache_key": "ck2"},
    )
    monkeypatch.setattr(
        "agent_utilities.core.contextual_model.get_context_compiler_cache",
        lambda: None,
    )
    eng.bump("claim:cited")

    report = revalidate_stale_materializations(eng)

    assert report["revalidated"]["context_bundle"] == 1
    assert eng.deleted_nodes == ["context_bundle:ck2"]


def test_bounded_per_tick_respects_the_candidate_limit():
    eng = FaithfulTmsEngine()
    eng.add_node("base", "Fact", properties={})
    for i in range(5):
        claim = _claim(["base"], claim_id=f"claim:{i}")
        eng.add_node(
            claim.id,
            "Claim",
            properties={
                **claim.model_dump(mode="json", exclude={"type"}),
                "status": "p",
            },
        )
        register_claim_materialization(eng, claim, [], context="test")
    eng.bump("base")

    report = revalidate_stale_materializations(eng, limit=2)

    assert report["scanned"] == 2  # bounded, not all 5
    assert report["revalidated"]["claim"] == 2


def test_a_single_stale_candidate_probe_failure_does_not_stop_the_rest():
    eng = FaithfulTmsEngine()
    eng.add_node("base", "Fact", properties={})
    claim_ok = _claim(["base"], claim_id="claim:ok")
    eng.add_node(
        claim_ok.id,
        "Claim",
        properties={
            **claim_ok.model_dump(mode="json", exclude={"type"}),
            "status": "p",
        },
    )
    register_claim_materialization(eng, claim_ok, [], context="test")
    eng.add_node("claim:broken", "Claim", properties={"invalidation_deps": ["base"]})
    eng.bump("base")

    # Force the cheap gate to report staleness WITHOUT exercising the
    # per-candidate probe below (that method is about to be made flaky, and
    # this stub's own `stale_materializations` would otherwise call it too).
    eng.stale_materializations = lambda: ["eg:reasoning:placeholder"]  # type: ignore[method-assign]

    real_status = eng.materialization_status

    def _flaky_status(node_id: str) -> str | None:
        if node_id == "claim:broken":
            raise RuntimeError("engine unreachable")
        return real_status(node_id)

    eng.materialization_status = _flaky_status  # type: ignore[method-assign]

    report = revalidate_stale_materializations(eng)

    assert report["revalidated"]["claim"] == 1
    assert any("status probe failed" in e for e in report["errors"])
