#!/usr/bin/python
from __future__ import annotations

"""Offline tests for the MemoryData graph-os adapter (CONCEPT:AU-AHE.harness.hardening-transparency-surface..3.74).

All tests run against the deterministic ``MockBackendClient`` (``transport="mock"``), so the
suite passes with the live engine down and no extra dependencies. Runnable either under
pytest or as a plain script (``python3 test_adapter_offline.py``) via the ``__main__`` block.
"""

from agent_utilities.harness.memorydata import (
    RETRIEVAL_CONFIGS,
    BakeoffResult,
    GraphOSMemoryMethod,
    GraphOSRouterMethod,
    render_scoreboard,
    run_bakeoff,
)

_STANDARD_KEYS = {
    "output",
    "input_len",
    "output_len",
    "memory_construction_time",
    "query_time_len",
}


def _agent(retrieval: str = "graphos_semantic_hnsw") -> GraphOSMemoryMethod:
    return GraphOSMemoryMethod(
        agent_config={
            "agent_name": f"graphos_{retrieval}",
            "retrieval": retrieval,
            "transport": "mock",
        },
        dataset_config={"sub_dataset": "unit", "dataset": "memorydata"},
    )


def test_memorize_then_recall_returns_standard_shape() -> None:
    agent = _agent()
    mem = agent.send_message(
        "The capital of France is Paris.", memorizing=True, context_id=0
    )
    assert set(mem.keys()) == _STANDARD_KEYS
    assert mem["output"] == ""
    assert mem["memory_construction_time"] >= 0.0

    agent.send_message(
        "Bananas are yellow and grow in bunches.", memorizing=True, context_id=1
    )

    resp = agent.send_message(
        "What is the capital of France?", memorizing=False, query_id="q0"
    )
    assert set(resp.keys()) == _STANDARD_KEYS
    assert isinstance(resp["output"], str) and resp["output"]
    assert "paris" in resp["output"].lower()
    assert isinstance(resp["input_len"], int)
    assert isinstance(resp["output_len"], int)
    assert resp["query_time_len"] >= 0.0


def test_all_six_configs_instantiate_and_answer() -> None:
    assert set(RETRIEVAL_CONFIGS) == {
        "graphos_semantic_hnsw",
        "graphos_bitemporal_asof",
        "graphos_context_plane",
        "graphos_latent",
        "graphos_rlm_facts",
        "graphos_graph_rerank",
    }
    for name in RETRIEVAL_CONFIGS:
        agent = _agent(name)
        agent.send_message(
            "Mercury is the closest planet to the Sun.", memorizing=True, context_id=0
        )
        resp = agent.send_message(
            "Which planet is closest to the Sun?", memorizing=False
        )
        assert set(resp.keys()) == _STANDARD_KEYS
        assert isinstance(resp["output"], str)


def test_unknown_config_rejected() -> None:
    raised = False
    try:
        GraphOSMemoryMethod(
            agent_config={
                "agent_name": "graphos_x",
                "retrieval": "nope",
                "transport": "mock",
            }
        )
    except ValueError:
        raised = True
    assert raised


def _synthetic_family() -> dict:
    return {
        "tag": "membench-update",
        "name": "MemBench-update",
        "context_chunks": [
            "Alice moved to Berlin in 2021.",
            "Bob works as a marine biologist studying coral reefs.",
        ],
        "tasks": [
            {
                "task": "recall",
                "queries": [
                    {"question": "Where did Alice move?", "answer": "Berlin"},
                    {"question": "What is Bob's job?", "answer": "marine biologist"},
                ],
            }
        ],
    }


def test_run_bakeoff_returns_scored_results() -> None:
    results = run_bakeoff(
        configs=["graphos_semantic_hnsw", "graphos_graph_rerank"],
        families=[_synthetic_family()],
        client_transport="mock",
    )
    assert results
    assert all(isinstance(r, BakeoffResult) for r in results)
    for r in results:
        assert 0.0 <= r.exact_match <= 1.0
        assert 0.0 <= r.rouge_l <= 1.0
        assert r.n == 2
        assert r.mean_query_s >= 0.0
    # The scoreboard renders without error and includes the family.
    md = render_scoreboard(results)
    assert "membench-update" in md
    assert "Measured results" in md


def test_router_selects_expected_configs_per_family() -> None:
    cases = {
        "membench-update": "graphos_bitemporal_asof",
        "locomo-singlehop": "graphos_context_plane",
        "longbench-v2": "graphos_latent",
        "conflict-resolution": "graphos_bitemporal_asof",
        "memoryagentbench": "graphos_bitemporal_asof",
        "membench-recall": "graphos_graph_rerank",
        "something-unmapped": "graphos_semantic_hnsw",
    }
    for tag, expected in cases.items():
        router = GraphOSRouterMethod(
            agent_config={"agent_name": "graphos_router", "transport": "mock"},
            dataset_config={"sub_dataset": tag},
            family_tag=tag,
        )
        selected = router._select_config({"family_tag": tag})
        assert selected == expected, f"{tag} → {selected} (expected {expected})"


def test_router_answers_and_records_outcome() -> None:
    router = GraphOSRouterMethod(
        agent_config={"agent_name": "graphos_router", "transport": "mock"},
        dataset_config={"sub_dataset": "locomo-test"},
        family_tag="locomo-test",
    )
    router.send_message(
        "Alice likes hiking on weekends.", memorizing=True, context_id=0
    )
    resp = router.send_message("What does Alice like?", memorizing=False)
    assert set(resp.keys()) == _STANDARD_KEYS
    assert router.retrieval == "graphos_context_plane"
    before = router.weights["graphos_context_plane"]
    after = router.record_outcome("graphos_context_plane", reward=2.0)
    assert after != before


def test_bakeoff_drives_router_config() -> None:
    """run_bakeoff recognizes the meta-config ``graphos_router`` and flags it ``is_router``."""
    results = run_bakeoff(
        configs=["graphos_semantic_hnsw", "graphos_router"],
        families=[_synthetic_family()],
        client_transport="mock",
    )
    by_config = {r.config: r for r in results}
    assert "graphos_router" in by_config
    assert by_config["graphos_router"].is_router is True
    assert by_config["graphos_semantic_hnsw"].is_router is False
    # The scoreboard now emits the router-vs-best-single section.
    md = render_scoreboard(results)
    assert "Router vs best single config" in md


def test_build_client_supports_engine_transport() -> None:
    """The factory exposes the live ``engine`` transport without importing the engine."""
    from agent_utilities.harness.memorydata.client import (
        EngineBackendClient,
        build_client,
    )

    client = build_client({"transport": "engine", "namespace": "unit"})
    assert isinstance(client, EngineBackendClient)
    assert client.namespace == "unit"

    raised = False
    try:
        build_client({"transport": "carrier-pigeon"})
    except ValueError:
        raised = True
    assert raised


def _run_all() -> None:
    fns = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} offline tests passed.")


if __name__ == "__main__":
    _run_all()
