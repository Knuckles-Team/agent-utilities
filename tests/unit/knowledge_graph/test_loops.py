"""Unit tests for the Loop model — the long-running-objective unit (CONCEPT:AU-KG.research.these-properties-carry)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.loops import (
    TERMINAL_STATUS,
    active_loops,
    submit_loop,
)


class _Engine:
    def __init__(self, concept_rows=None, addressed=None):
        self.nodes: dict[str, dict] = {}
        self._concept_rows = concept_rows or []
        self._addressed = addressed or []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"type": node_type, **(properties or {})}

    def query_cypher(self, q: str, params: dict | None = None):
        if "ADDRESSED_BY" in q:
            return [{"id": i} for i in self._addressed]
        return self._concept_rows


def test_submit_loop_creates_kinded_node():
    eng = _Engine()
    loop = submit_loop(
        eng, "Make the build pass", kind="develop", validation_cmd="pytest -q"
    )
    assert loop["kind"] == "develop"
    nid = loop["id"]
    node = eng.nodes[nid]
    assert node["loop_kind"] == "develop"
    assert node["validation_cmd"] == "pytest -q"
    assert node["objective"] == "Make the build pass"
    assert node["status"] == "pending"


def test_submit_research_loop_default_kind():
    eng = _Engine()
    loop = submit_loop(eng, "self-improving agent harnesses")
    assert loop["kind"] == "research"
    assert eng.nodes[loop["id"]]["loop_kind"] == "research"


def test_active_loops_dispatch_by_kind_and_status():
    rows = [
        {
            "id": "loop:research:a",
            "name": "A",
            "loop_kind": "research",
            "status": "pending",
        },
        {
            "id": "loop:research:done",
            "name": "D",
            "loop_kind": "research",
            "status": "pending",
        },
        {
            "id": "loop:develop:b",
            "name": "B",
            "loop_kind": "develop",
            "status": "pending",
            "validation_cmd": "pytest",
        },
        {
            "id": "loop:develop:fin",
            "name": "C",
            "loop_kind": "develop",
            "status": "completed",
        },
        {
            "id": "loop:skill:s",
            "name": "S",
            "loop_kind": "skill",
            "status": "pending",
            "skill_ref": "deploy-stack",
        },
        {
            "id": "concept:legacy",
            "name": "legacy topic",
            "loop_kind": None,
            "status": None,
        },
    ]
    eng = _Engine(concept_rows=rows, addressed=["loop:research:done"])
    out = {lp["id"]: lp for lp in active_loops(eng, limit=10)}
    assert "loop:research:a" in out  # research, unaddressed
    assert "loop:research:done" not in out  # research, ADDRESSED_BY → resolved
    assert "loop:develop:b" in out  # develop, non-terminal
    assert "loop:develop:fin" not in out  # develop, completed → terminal
    assert "loop:skill:s" in out and out["loop:skill:s"]["kind"] == "skill"
    assert "concept:legacy" in out  # bare Concept → treated as research loop
    assert out["loop:develop:b"]["validation_cmd"] == "pytest"


def test_terminal_status_set():
    assert "completed" in TERMINAL_STATUS and "running" not in TERMINAL_STATUS


def test_submit_loop_records_priority_bucket():
    from agent_utilities.knowledge_graph.research.loops import prioritize_loop

    eng = _Engine()
    loop = submit_loop(eng, "hot objective", prio_bucket=0)
    assert loop["prio_bucket"] == 0
    assert eng.nodes[loop["id"]]["prio_bucket"] == 0
    # prioritize_loop upserts a new bucket onto the node.
    assert prioritize_loop(eng, loop["id"], 3) is True
    assert eng.nodes[loop["id"]]["prio_bucket"] == 3


def test_submit_loop_coerces_legacy_string_priority():
    """A loop bucket goes through the ONE normalizer (CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task)."""
    eng = _Engine()
    # Legacy string priority on a loop is coerced to the shared 0..3 bucket.
    loop = submit_loop(eng, "hot", prio_bucket="critical")  # type: ignore[arg-type]
    assert loop["prio_bucket"] == 0
    assert eng.nodes[loop["id"]]["prio_bucket"] == 0
    loop2 = submit_loop(eng, "bg", loop_id="loop:x", prio_bucket="low")  # type: ignore[arg-type]
    assert loop2["prio_bucket"] == 3


# ── claim_loop: engine compare-and-set arbitration (CONCEPT:AU-KG.compute.user-override-prompt-library) ───────


class _CASRecorder:
    """A backend whose compare_and_set_node_fields is a controllable mock."""

    def __init__(self, *, win_on: set[str] | None = None, always: bool | None = None):
        self.calls: list[tuple[str, dict, dict]] = []
        self._win_on = win_on
        self._always = always

    def compare_and_set_node_fields(self, node_id, conditions, updates):
        self.calls.append((node_id, dict(conditions), dict(updates)))
        if self._always is not None:
            return self._always
        return conditions.get("status") in (self._win_on or set())


class _CASEngine(_Engine):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend


def test_claim_loop_issues_cas_pending_to_running_and_wins():
    backend = _CASRecorder(win_on={"pending"})
    eng = _CASEngine(backend)
    from agent_utilities.knowledge_graph.research.loops import claim_loop

    assert claim_loop(eng, "loop:develop:b", current_status="pending") is True
    # First CAS is the observed status (pending) → running.
    node_id, conds, updates = backend.calls[0]
    assert node_id == "loop:develop:b"
    assert conds == {"status": "pending"}
    assert updates["status"] == "running"
    # Won on the first try — no further CAS attempts.
    assert len(backend.calls) == 1


def test_claim_loop_lost_race_returns_false():
    # CAS never wins (a peer already flipped the loop) → claim_loop returns False
    # after sweeping every claimable status, and never blind-writes.
    backend = _CASRecorder(always=False)
    eng = _CASEngine(backend)
    from agent_utilities.knowledge_graph.research.loops import (
        CLAIMABLE_STATUS,
        claim_loop,
    )

    assert claim_loop(eng, "loop:develop:b", current_status="pending") is False
    # One CAS per claimable status, all guarded on status=running update.
    tried = {c[1]["status"] for c in backend.calls}
    assert tried == set(CLAIMABLE_STATUS)
    for _nid, _conds, updates in backend.calls:
        assert updates["status"] == "running"


def test_claim_loop_tries_observed_status_first():
    # An orphaned loop (crashed driver) is still claimable; observed status is
    # tried first so the common path is a single CAS.
    backend = _CASRecorder(win_on={"orphaned"})
    eng = _CASEngine(backend)
    from agent_utilities.knowledge_graph.research.loops import claim_loop

    assert claim_loop(eng, "loop:skill:s", current_status="orphaned") is True
    assert backend.calls[0][1] == {"status": "orphaned"}


def test_claim_loop_falls_back_to_blind_flip_without_cas():
    # Older engine with no CAS primitive: single-host blind flip, no regression.
    eng = _Engine()  # _Engine has no .backend
    from agent_utilities.knowledge_graph.research.loops import claim_loop

    assert claim_loop(eng, "loop:develop:b", current_status="pending") is True
    assert eng.nodes["loop:develop:b"]["status"] == "running"


def test_active_loops_emitted_in_priority_order():
    rows = [
        {
            "id": "loop:research:bg",
            "loop_kind": "research",
            "status": "pending",
            "prio_bucket": 3,
        },
        {
            "id": "loop:research:crit",
            "loop_kind": "research",
            "status": "pending",
            "prio_bucket": 0,
        },
        {
            "id": "loop:research:norm",
            "loop_kind": "research",
            "status": "pending",
            "prio_bucket": 2,
        },
    ]
    eng = _Engine(concept_rows=rows)
    order = [lp["id"] for lp in active_loops(eng, limit=10)]
    assert order == ["loop:research:crit", "loop:research:norm", "loop:research:bg"]
