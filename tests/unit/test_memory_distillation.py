"""Unit tests for episodic→procedural memory distillation (CONCEPT:AU-KG.memory.episodic-procedural-memory-distillation).

Drives the AU-side distiller with a MOCK engine (records ``store_memory`` writes),
a MOCK LLM (returns a fixed procedural JSON artifact), and a MOCK flywheel (records
reward-EMA seeds). No live engine or model is required.
"""

from __future__ import annotations

import json
from typing import Any

from agent_utilities.knowledge_graph.memory.distillation import (
    MemoryDistiller,
    MemoryDistillerConfig,
    ProceduralArtifact,
    run_memory_distillation,
)


# ── Mock engine + backend + flywheel ──────────────────────────────────────────
class _MockBackend:
    """Minimal backend exposing the Cypher-subset ``execute`` the reader uses."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self._nodes = nodes

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return [{"id": n["id"], "data": dict(n)} for n in self._nodes]


class _MockEngine:
    """Mock engine exposing the backend reader + the ``store_memory`` write path."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self.backend = _MockBackend(nodes)
        self.store_calls: list[dict[str, Any]] = []
        self._counter = 0

    def store_memory(self, **kwargs: Any) -> str:
        self.store_calls.append(kwargs)
        self._counter += 1
        return f"mem:proc-{self._counter}"


class _MockFlywheel:
    """Mock capability index recording reward-EMA seeds (CONCEPT:AU-KG.memory.episodic-procedural-memory-distillation)."""

    def __init__(self) -> None:
        self.rewards: dict[str, float] = {}

    def record_outcome(
        self, id: str, success: bool | None = None, reward: float | None = None
    ) -> float:
        r = 0.5 if reward is None else float(reward)
        self.rewards[id] = r
        return r


_ARTIFACT_JSON = json.dumps(
    {
        "name": "Rotate deploy credentials",
        "intent": "Safely rotate a service credential across the fleet.",
        "preconditions": ["operator has vault access", "service is healthy"],
        "steps": [
            "Fetch current secret from the vault",
            "Generate a new credential",
            "Push it to all consumers",
            "Verify each consumer reconnects",
        ],
        "tags": ["credentials", "deploy", "rotation"],
    }
)


def _mock_llm(system_prompt: str, user_content: str) -> str:
    return _ARTIFACT_JSON


def _recurring_and_noise() -> list[dict[str, Any]]:
    """A recurring cluster (4 episodes on 'rotate-creds') + noise the distiller skips.

    Noise: a too-rare cluster ('one-off'), a working-memory node (wrong tier), and a
    retired episode — none should form a distillation candidate.
    """
    recurring = [
        {
            "id": f"ep-rot-{i}",
            "memory_type": "episodic",
            "status": "ACTIVE",
            "target_entity": "rotate-creds",
            "content": f"rotated the api token for service {i}",
        }
        for i in range(4)
    ]
    rare = [
        {
            "id": "ep-oneoff-0",
            "memory_type": "episodic",
            "status": "ACTIVE",
            "target_entity": "one-off",
            "content": "did a unique thing once",
        }
    ]
    wrong_tier = [
        {
            "id": "work-0",
            "memory_type": "working",
            "status": "ACTIVE",
            "target_entity": "rotate-creds",
            "content": "scratch",
        }
    ]
    retired = [
        {
            "id": f"ep-old-{i}",
            "memory_type": "episodic",
            "status": "RETIRED",
            "target_entity": "rotate-creds",
            "content": "stale rotation",
        }
        for i in range(4)
    ]
    return recurring + rare + wrong_tier + retired


def _distiller(
    engine: _MockEngine, flywheel: Any = None, **cfg: Any
) -> MemoryDistiller:
    config = MemoryDistillerConfig(enabled=True, min_recurrence=3, **cfg)
    return MemoryDistiller(engine, config=config, llm=_mock_llm, flywheel=flywheel)


# ── select_recurring_clusters ─────────────────────────────────────────────────
def test_kg_2_309_selection_picks_the_recurring_cluster() -> None:
    nodes = _recurring_and_noise()
    engine = _MockEngine(nodes)
    dist = _distiller(engine)

    clusters = dist.select_recurring_clusters(nodes)

    assert len(clusters) == 1
    ids = {n["id"] for n in clusters[0]}
    assert ids == {"ep-rot-0", "ep-rot-1", "ep-rot-2", "ep-rot-3"}
    # Must exclude the rare cluster, the wrong-tier node, and retired episodes.
    assert "ep-oneoff-0" not in ids
    assert "work-0" not in ids
    assert "ep-old-0" not in ids


def test_kg_2_309_selection_empty_when_nothing_recurs() -> None:
    nodes = [
        {
            "id": f"ep-{i}",
            "memory_type": "episodic",
            "status": "ACTIVE",
            "target_entity": f"unique-{i}",
            "content": f"e{i}",
        }
        for i in range(3)
    ]
    dist = _distiller(_MockEngine(nodes))
    assert dist.select_recurring_clusters(nodes) == []


# ── distill: recurring cluster → named procedural artifact ─────────────────────
def test_kg_2_309_distill_records_a_named_procedural_artifact() -> None:
    nodes = _recurring_and_noise()
    engine = _MockEngine(nodes)
    flywheel = _MockFlywheel()
    dist = _distiller(engine, flywheel=flywheel)
    cluster = dist.select_recurring_clusters(nodes)[0]

    res = dist.distill(cluster)

    assert res["status"] == "ok"
    assert res["artifact_name"] == "Rotate deploy credentials"
    assert res["step_count"] == 4
    assert res["capability_id"] == "procedural:rotate-deploy-credentials"

    # Recorded as PROCEDURAL memory via the existing store_memory write path.
    assert len(engine.store_calls) == 1
    call = engine.store_calls[0]
    assert call["memory_type"] == "procedural"
    assert call["name"] == "Rotate deploy credentials"
    assert "STEPS:" in call["content"]
    assert call["extra_props"]["concept"] == "AU-KG.memory.episodic-procedural-memory-distillation"
    assert call["extra_props"]["step_count"] == 4
    # Provenance: the source episode ids are stamped on the artifact.
    assert "ep-rot-0" in call["extra_props"]["distilled_from"]

    # Registered with the reward-EMA flywheel (reward-weighted reinforcement).
    assert flywheel.rewards["procedural:rotate-deploy-credentials"] == 0.5
    assert res["flywheel"]["status"] == "ok"


# ── no-LLM clean fallback ──────────────────────────────────────────────────────
def test_kg_2_309_distill_no_llm_is_a_clean_skip() -> None:
    nodes = _recurring_and_noise()
    engine = _MockEngine(nodes)
    dist = MemoryDistiller(
        engine,
        config=MemoryDistillerConfig(enabled=True, min_recurrence=3),
        llm=lambda s, u: None,  # no model available → clean fallback, no artifact
    )
    cluster = dist.select_recurring_clusters(nodes)[0]

    res = dist.distill(cluster)

    assert res["status"] == "skipped"
    assert res["reason"] == "no_llm"
    assert engine.store_calls == []


def test_kg_2_309_distill_prose_fallback_when_llm_returns_non_json() -> None:
    """A non-JSON LLM response still yields a usable procedure (prose fallback)."""
    nodes = _recurring_and_noise()
    engine = _MockEngine(nodes)
    dist = MemoryDistiller(
        engine,
        config=MemoryDistillerConfig(enabled=True, min_recurrence=3),
        llm=lambda s, u: "Step one: fetch\nStep two: rotate\nStep three: verify",
        flywheel=_MockFlywheel(),
    )
    cluster = dist.select_recurring_clusters(nodes)[0]

    res = dist.distill(cluster)

    assert res["status"] == "ok"
    assert res["step_count"] == 3
    assert engine.store_calls[0]["memory_type"] == "procedural"


# ── run_distillation cycle: wiring + idempotency + disabled + safety ──────────
def test_kg_2_309_run_distillation_cycle_is_idempotent() -> None:
    nodes = _recurring_and_noise()
    engine = _MockEngine(nodes)
    flywheel = _MockFlywheel()
    dist = _distiller(engine, flywheel=flywheel)

    first = dist.run_distillation()
    assert first["status"] == "ok"
    assert first["distilled"] == 1
    assert len(first["artifacts"]) == 1

    # Second cycle over the SAME working set must NOT re-distill the cluster.
    second = dist.run_distillation()
    assert second["status"] == "ok"
    assert second["distilled"] == 0

    assert len(engine.store_calls) == 1  # stored exactly once across both cycles


def test_kg_2_309_run_distillation_disabled_by_default_is_a_noop() -> None:
    engine = _MockEngine(_recurring_and_noise())
    # Default config: enabled=False.
    dist = MemoryDistiller(engine, config=MemoryDistillerConfig(), llm=_mock_llm)

    res = dist.run_distillation()

    assert res == {"status": "disabled"}
    assert engine.store_calls == []


def test_kg_2_309_run_distillation_safe_when_store_unavailable() -> None:
    """A build whose engine lacks a memory write path degrades, never raises."""

    class _BareEngine:
        def __init__(self, ns: list[dict[str, Any]]) -> None:
            self.backend = _MockBackend(ns)

    dist = MemoryDistiller(
        _BareEngine(_recurring_and_noise()),
        config=MemoryDistillerConfig(enabled=True, min_recurrence=3),
        llm=_mock_llm,
    )
    res = dist.run_distillation()

    assert res["status"] == "ok"
    assert res["distilled"] == 0  # nothing recorded, but the cycle is safe


def test_kg_2_309_run_memory_distillation_entrypoint_respects_disabled_default() -> (
    None
):
    engine = _MockEngine(_recurring_and_noise())
    # No config → from_env(), disabled unless the env var is set.
    assert run_memory_distillation(engine) == {"status": "disabled"}


# ── ProceduralArtifact rendering ───────────────────────────────────────────────
def test_kg_2_309_procedural_artifact_renders_slug_and_document() -> None:
    art = ProceduralArtifact(
        name="Rotate Deploy Credentials!",
        intent="rotate safely",
        preconditions=["vault access"],
        steps=["fetch", "rotate", "verify"],
    )
    assert art.slug == "rotate-deploy-credentials"
    doc = art.render()
    assert doc.startswith("PROCEDURE: Rotate Deploy Credentials!")
    assert "PRECONDITIONS:" in doc
    assert "1. fetch" in doc
