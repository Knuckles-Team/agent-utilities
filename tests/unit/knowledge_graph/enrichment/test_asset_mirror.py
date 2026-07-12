"""Multi-SoR asset-mirror tests (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Covers Phase F: the widened INVENTORY_TYPES filter now includes the fleet's real
emitted asset types; the sys_id round-trip stamp makes a second push a no-op/update
(not a re-create); the mirror fans one pass out to every enabled sink; dry-run is the
default (no live calls); and a guarded/no-engine pass is a clean no-op.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.writeback import core, run_writeback
from agent_utilities.knowledge_graph.enrichment.writeback.core import (
    WritebackContext,
    WritebackResult,
    register_sink,
)
from agent_utilities.knowledge_graph.enrichment.writeback import inventory as inv
from agent_utilities.knowledge_graph.enrichment.writeback.inventory import (
    INVENTORY_TYPES,
    ci_id_key,
    collect_inventory_creations,
    enabled_mirror_targets,
    push_inventory,
    run_asset_mirror,
)


# ── fakes ────────────────────────────────────────────────────────────────────
class FakeBackend:
    """Serves inventory candidates + a per-target ``<target>_ci_id`` stamp store."""

    def __init__(self, candidates):
        self._candidates = candidates
        self.stamped: dict[str, set[str]] = {}  # ci_key -> {node ids}

    def execute(self, query, params=None):
        if "ALIGNED_WITH" in query:
            return []
        if "_ci_id IS NOT NULL" in query:
            for key, ids in self.stamped.items():
                if f"n.{key} IS NOT NULL" in query:
                    return [{"id": i} for i in ids]
            return []
        if "n.type AS type" in query:
            return [dict(c) for c in self._candidates]
        return []


class FakeEngine:
    """Records add_node merge-upserts and reflects the ci_id stamp into the backend."""

    def __init__(self, backend):
        self.backend = backend
        self.calls: list[tuple] = []

    def add_node(self, node_id, node_type, properties=None):
        props = dict(properties or {})
        self.calls.append((node_id, node_type, props))
        for k, _v in props.items():
            if k.endswith("_ci_id"):
                self.backend.stamped.setdefault(k, set()).add(node_id)


class FakeSnowClient:
    def __init__(self):
        self.created: list[tuple] = []

    def create_cmdb_instance(self, className, attributes, source):
        self.created.append((className, attributes.get("name")))
        return {"result": {"sys_id": f"SYS-{attributes.get('name')}"}}


class RecordingSink:
    """A minimal fake sink: records its run() and proposes one create per creation."""

    def __init__(self, name):
        self.domain = name
        self.enable_flag = f"{name.upper()}_ENABLE_WRITE"
        self.runs: list[tuple] = []

    def run(self, ctx, ops, *, dry_run):
        creations = ops.get("creations") or []
        self.runs.append((list(creations), dry_run))
        r = WritebackResult(target=self.domain)
        for c in creations:
            r.proposals.append({"op": "create", "name": c.get("name")})
        return r


# ── 1. widened type filter ────────────────────────────────────────────────────
def test_inventory_types_include_real_emitted_types():
    for t in (
        "Host",
        "Pod",
        "Deployment",
        "ContainerImage",
        "SwarmService",
        "SwarmNode",
        "K8sService",
        "Node",
        "Workload",
        "Tunnel",
        "Repository",
        "Stack",
        "NetworkInterface",
        "DiskVolume",
    ):
        assert t in INVENTORY_TYPES


def test_collect_includes_widened_types():
    backend = FakeBackend(
        [
            {"type": "Host", "name": "r510", "id": "host:r510"},
            {"type": "Pod", "name": "graph-os-0", "id": "pod:graph-os-0"},
            {"type": "ContainerImage", "name": "kg:latest", "id": "img:kg"},
            {"type": "Person", "name": "alice", "id": "person:alice"},  # excluded
        ]
    )
    names = {c["name"] for c in collect_inventory_creations(backend, "servicenow")}
    assert names == {"r510", "graph-os-0", "kg:latest"}


# ── 2. sys_id round-trip stamp → idempotent ───────────────────────────────────
def test_creation_stamps_sys_id_back(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    backend = FakeBackend([{"type": "Host", "name": "r510", "id": "host:r510"}])
    engine = FakeEngine(backend)
    client = FakeSnowClient()
    out = run_writeback(
        "servicenow",
        backend=backend,
        engine=engine,
        client=client,
        dry_run=False,
        creations=[{"type": "Host", "name": "r510", "node": "host:r510"}],
    )
    assert out["created"] == 1
    assert client.created == [("cmdb_ci_server", "r510")]
    # the returned sys_id was stamped back onto the source node
    assert engine.calls
    node_id, _label, props = engine.calls[0]
    assert node_id == "host:r510"
    assert props["servicenow_ci_id"] == "SYS-r510"
    assert props["externalToolId"] == "SYS-r510"
    assert props["domain"] == "servicenow"
    # the stamp is now visible to the dedupe query
    assert "host:r510" in backend.stamped[ci_id_key("servicenow")]


def test_second_push_is_noop_after_stamp():
    backend = FakeBackend([{"type": "Host", "name": "r510", "id": "host:r510"}])
    # first pass would create it; simulate that it already round-tripped its sys_id:
    backend.stamped[ci_id_key("servicenow")] = {"host:r510"}
    creations = collect_inventory_creations(backend, "servicenow")
    assert creations == []  # already represented → skipped, never re-created
    out = push_inventory("servicenow", backend=backend, dry_run=True)
    assert out["inventory_candidates"] == 0
    assert out["proposals"] == []


# ── 3. multi-sink fan-out ─────────────────────────────────────────────────────
def test_mirror_fans_out_to_every_enabled_sink():
    a, b = RecordingSink("fake_a"), RecordingSink("fake_b")
    register_sink(a)
    register_sink(b)
    backend = FakeBackend([{"type": "Host", "name": "r510", "id": "host:r510"}])
    out = run_asset_mirror(backend=backend, targets=["fake_a", "fake_b"], dry_run=True)
    assert out["status"] == "completed"
    assert out["targets"] == ["fake_a", "fake_b"]
    assert set(out["sinks"]) == {"fake_a", "fake_b"}
    # every enabled sink was invoked with the collected creation
    assert a.runs and b.runs
    assert a.runs[0][0][0]["name"] == "r510"
    assert b.runs[0][0][0]["name"] == "r510"


def test_mirror_selects_targets_from_env(monkeypatch):
    monkeypatch.setattr(inv, "setting", lambda k, d=None: "servicenow, egeria, bogus")
    # bogus (not a real CMDB sink) is dropped; order preserved
    assert enabled_mirror_targets() == ["servicenow", "egeria"]
    monkeypatch.setattr(inv, "setting", lambda k, d=None: '["twenty","erpnext"]')
    assert enabled_mirror_targets() == ["twenty", "erpnext"]
    monkeypatch.setattr(inv, "setting", lambda k, d=None: None)
    assert enabled_mirror_targets() == []


def test_mirror_empty_by_default(monkeypatch):
    monkeypatch.setattr(inv, "setting", lambda k, d=None: None)
    out = run_asset_mirror(backend=FakeBackend([]), dry_run=True)
    assert out["targets"] == []
    assert out["sinks"] == {}


# ── 4. dry-run default → no live calls ────────────────────────────────────────
def test_dry_run_default_emits_proposals_without_live_calls():
    sink = RecordingSink("fake_dry")
    register_sink(sink)
    backend = FakeBackend([{"type": "Host", "name": "r510", "id": "host:r510"}])
    out = run_asset_mirror(backend=backend, targets=["fake_dry"])  # dry_run defaults True
    assert out["dry_run"] is True
    assert sink.runs[0][1] is True  # ran in dry-run mode
    assert out["sinks"]["fake_dry"]["proposals"]  # intended-writes surfaced
    assert out["created"] == 0  # nothing actually created


# ── 5. guarded / no-engine no-op ──────────────────────────────────────────────
def test_stamp_no_engine_is_noop():
    ctx = WritebackContext(backend=None, engine=None)
    assert ctx.stamp_external_id("host:r510", "servicenow", "SYS-1") is False
    # missing id / external id also no-op, never raising
    assert ctx.stamp_external_id(None, "servicenow", "SYS-1") is False
    assert ctx.stamp_external_id("host:r510", "servicenow", None) is False


def test_mirror_no_backend_no_engine_clean_noop():
    sink = RecordingSink("fake_guard")
    register_sink(sink)
    out = run_asset_mirror(backend=None, engine=None, targets=["fake_guard"], dry_run=True)
    assert out["status"] == "completed"
    assert out["inventory_candidates"] == 0
