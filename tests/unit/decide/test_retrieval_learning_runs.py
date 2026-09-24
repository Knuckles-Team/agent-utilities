"""EH-394 / EH-395: the live retrieval path attests outcomes and consumes paths.

A retrieval-plan choice EG executed and logged becomes a pending run; what the
run returned is noted; the answer's citations attest it to EG joined to the
committed record. Proven paths of the retriever's task class join the plan
choice; a chosen path runs as a unified plan before any template.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_utilities.decide.learning import runs
from agent_utilities.decide.learning.ops import outcome_op, space_identity
from tests.unit.decide.fakes import FakeTransport, acted, record

PATH = {
    "template_digest": "sha256:p1",
    "successes": 7,
    "failures": 1,
    "template": {
        "task_class": "urn:task:triage",
        "composed_digest": "sha256:schema",
        "policy_version": "1",
        "anchor_class": "Incident",
        "edges": [{"relationship": "AFFECTS", "min_hops": 1, "max_hops": 2}],
        "rank": "fuse_rrf",
        "slots": [],
        "skill_ref": None,
    },
}


class _Embedder:
    model_name = "embed-m"

    def get_text_embedding(self, text: str) -> list[float]:
        return [0.5, -0.25]


class _Graph:
    graph_name = "kg"

    def __init__(self) -> None:
        self.plans: list[Any] = []

    def query_unified(self, plan: Any) -> list[dict[str, Any]]:
        self.plans.append(plan)
        return [{"id": "n1", "score": 0.9}]


class _Retriever:
    def __init__(self) -> None:
        self.embed_model = _Embedder()
        self.engine = type("Engine", (), {"graph": _Graph()})()
        self.path_scope = runs.PathScope("urn:task:triage", "sha256:schema")

    def _batch_node_properties(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        return {
            i: {"description": "a long enough prose description of " + i} for i in ids
        }


def _logged(option: str) -> dict[str, Any]:
    batch = acted(option)
    batch["records"][0]["record_digest"] = "sha256:" + "0" * 64
    return batch


def _answers(paths: list[Mapping[str, Any]]) -> Any:
    def answer(op: Mapping[str, Any]) -> Any:
        action = (op.get("retrieval") or {}).get("action")
        if action == "paths":
            return {"result": "paths", "schema_version": 1, "rows": paths}
        return {"record_id": "logged"}

    return answer


def test_a_proven_path_joins_the_plan_choice_and_runs_first(eg: FakeTransport) -> None:
    eg.log_answer = _answers([PATH])
    eg.answer = _logged("path:sha256:p1")
    retriever = _Retriever()
    mode = runs.plan_retrieval(retriever, "why is billing down?", "hyde")
    assert mode == "standard", "a chosen path skips the HyDE planner"
    options = [o["option_id"] for o in eg.requests[0]["candidates"]["options"]]
    assert "path:sha256:p1" in options
    lists = runs.first_pass(
        retriever, "why is billing down?", 5, lambda: [[{"id": "t"}]]
    )
    assert [n["id"] for n in lists[0]] == ["n1"]
    plan = retriever.engine.graph.plans[0]
    assert plan[0] == {"Scan": {"label": "Incident"}}
    assert plan[1] == {"Traverse": {"rel": "AFFECTS", "min": 1, "max": 2}}
    assert "FuseRrf" in plan[2] and plan[-1] == {"Limit": {"k": 5}}


def test_the_answer_attests_what_the_run_returned_and_cited(eg: FakeTransport) -> None:
    eg.log_answer = _answers([])
    eg.answer = _logged("deep")
    retriever = _Retriever()
    query = "who owns billing?"
    assert runs.plan_retrieval(retriever, query, "hyde") == "deep"
    runs.note_returned(
        retriever,
        query,
        [{"id": "a", "description": "prose about the billing owner team"}, {"id": "b"}],
    )
    assert runs.attest_citations(retriever, query, ["b", "not-returned"])
    sent = eg.ops[-1]["retrieval"]["outcome"]
    assert sent["record_id"] == record({})["record_id"]
    assert [r["evidence_id"] for r in sent["returned"]] == ["a", "b"]
    assert sent["returned"][0]["content_class"] == "prose"
    assert sent["cited"] == ["b"], "only returned units can be cited"
    assert sent["query"]["space_digest"] == space_identity("embed-m", 2)
    assert sent["query"]["q16"] == [32768, -16384]
    assert not runs.attest_citations(retriever, query, ["b"]), "attested once"


def test_an_abstention_leaves_nothing_to_attest(eg: FakeTransport) -> None:
    eg.log_answer = _answers([])
    eg.answer = {"records": [record({"outcome": "abstained", "reasons": []})]}
    retriever = _Retriever()
    assert runs.plan_retrieval(retriever, "q", "hyde") == "hyde"
    runs.note_returned(retriever, "q", [{"id": "a"}])
    assert not runs.attest_citations(retriever, "q", ["a"])


def test_without_a_runner_nothing_is_asked_or_attested() -> None:
    retriever = _Retriever()
    assert runs.proven_paths(retriever.path_scope) == []
    assert runs.plan_retrieval(retriever, "q", "deep") == "deep"
    assert not runs.attest_citations(retriever, "q", [])


def test_the_pending_runs_are_bounded() -> None:
    ledger = runs.RunLedger(limit=2)
    for i in range(3):
        ledger.put(f"q{i}", runs.PendingRun(f"r{i}"))
    assert ledger.get("q0") is None and ledger.get("q2") is not None


def test_an_outcome_keeps_citations_inside_the_returned_set() -> None:
    op = outcome_op("t", "r", [("a", "prose"), ("b", None)], ["b", "b", "c"])
    assert op["retrieval"]["outcome"]["cited"] == ["b"]
    assert op["retrieval"]["action"] == "record_outcome"
