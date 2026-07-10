"""Unit + live-path tests for memory→weights distillation (CONCEPT:AU-KG.memory.memory-weights-distillation-export).

Drives the AU-side EXPORT path with a MOCK engine whose backend returns synthetic
consolidated/procedural memory. Asserts:
  * the exported corpus SHAPE (deterministic JSONL of ``{prompt, completion}`` SFT
    examples, or ``{prompt, chosen, rejected}`` preference triples),
  * the typed :class:`DistillationTargetSpec` (base model / adapter rank / scopes),
  * the data-science-mcp hand-off :class:`DistillationJob` contract,
  * the live MCP surface (``graph_analyze action=distill_memory``) — the exact
    coroutine the ``graph-os`` MCP tool + its ``POST /graph/analyze`` REST twin
    dispatch into.

No live engine, model, or torch is required.
"""

from __future__ import annotations

import json
from typing import Any

from agent_utilities.knowledge_graph.memory.weights_distillation import (
    DATA_SCIENCE_MCP_CONTRACT,
    DistillationCorpus,
    DistillationJob,
    DistillationTargetSpec,
    MemoryWeightsDistiller,
    distill_memory_to_weights,
)


# ── Mock engine + backend ──────────────────────────────────────────────────────
class _MockBackend:
    """Minimal backend exposing the Cypher-subset ``execute`` the reader uses."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self._nodes = nodes

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return [{"id": n["id"], "data": dict(n)} for n in self._nodes]


class _MockEngine:
    """Mock engine exposing the backend reader + an ``add_node`` job surface."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self.backend = _MockBackend(nodes)
        self.nodes: dict[str, dict[str, Any]] = {}

    def add_node(self, node_id: str, label: str, properties: dict[str, Any]) -> None:
        self.nodes[node_id] = {"label": label, **properties}

    def get_node(self, node_id: str) -> dict[str, Any]:
        return self.nodes.get(node_id, {})


class _BareEngine:
    """An engine with only a reader (no job surface) — export must still work."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self.backend = _MockBackend(nodes)


def _consolidated_and_noise() -> list[dict[str, Any]]:
    """A procedural artifact + a semantic summary (in scope) + noise to skip."""
    procedural = {
        "id": "proc-rotate-creds",
        "memory_type": "procedural",
        "status": "ACTIVE",
        "target_entity": "rotate-creds",
        "name": "Rotate deploy credentials",
        "content": "PROCEDURE: Rotate deploy credentials\nSTEPS:\n1. fetch\n2. rotate",
        "trust_score": 0.9,
    }
    semantic = {
        "id": "sem-deploy-summary",
        "memory_type": "semantic",
        "status": "ACTIVE",
        "target_entity": "deploy",
        "content": "Deploys succeed when the vault is reachable and services healthy.",
        "trust_score": 0.8,
    }
    # Noise: wrong tier (episodic default-out-of-scope), retired, empty content.
    episodic = {
        "id": "ep-0",
        "memory_type": "episodic",
        "status": "ACTIVE",
        "target_entity": "deploy",
        "content": "one deploy happened",
    }
    retired = {
        "id": "sem-old",
        "memory_type": "semantic",
        "status": "RETIRED",
        "content": "stale summary",
    }
    empty = {"id": "sem-empty", "memory_type": "semantic", "status": "ACTIVE"}
    return [procedural, semantic, episodic, retired, empty]


# ── spec ────────────────────────────────────────────────────────────────────────
def test_kg_2_316_spec_from_params_normalizes() -> None:
    spec = DistillationTargetSpec.from_params(
        {
            "base_model": "qwen/qwen3.6-35b-a3b",
            "adapter_rank": 32,
            "scopes": "procedural, semantic",
            "method": "SFT",
            "time_window_days": 30,
        }
    )
    assert spec.base_model == "qwen/qwen3.6-35b-a3b"
    assert spec.adapter_rank == 32
    assert spec.scopes == ["procedural", "semantic"]
    assert spec.method == "sft"
    assert spec.is_preference is False
    assert spec.time_window_days == 30
    # JSON-safe round trip carries the LoRA target shape.
    d = spec.to_dict()
    assert d["adapter_type"] == "lora" and d["adapter_alpha"] == 32


# ── export: consolidated/procedural memory → SFT corpus ────────────────────────
def test_kg_2_316_export_builds_deterministic_sft_corpus() -> None:
    engine = _MockEngine(_consolidated_and_noise())
    spec = DistillationTargetSpec(
        base_model="Qwen2.5-1.5B", scopes=["procedural", "semantic"]
    )
    dist = MemoryWeightsDistiller(engine, spec=spec)

    corpus = dist.export()

    assert isinstance(corpus, DistillationCorpus)
    # Only the ACTIVE procedural + semantic nodes with content are exported.
    assert corpus.source_ids == ["proc-rotate-creds", "sem-deploy-summary"]
    assert len(corpus.examples) == 2
    # SFT shape: every example is a {prompt, completion} instruction/response pair.
    for ex in corpus.examples:
        assert set(ex.keys()) == {"prompt", "completion", "source_id"}
        assert ex["prompt"] and ex["completion"]
    # The procedural artifact's rendered steps become the completion.
    proc_ex = next(e for e in corpus.examples if e["source_id"] == "proc-rotate-creds")
    assert "STEPS:" in proc_ex["completion"]
    assert "rotate-creds" in proc_ex["prompt"]
    # Deterministic: JSONL + checksum are stable across re-export.
    assert corpus.to_jsonl() == dist.export().to_jsonl()
    assert corpus.checksum == dist.export().checksum
    # JSONL is one valid JSON object per line.
    lines = corpus.to_jsonl().splitlines()
    assert len(lines) == 2
    assert all("prompt" in json.loads(ln) for ln in lines)


def test_kg_2_316_scope_and_entity_filters() -> None:
    engine = _MockEngine(_consolidated_and_noise())
    # Restrict to procedural tier AND the rotate-creds entity only.
    spec = DistillationTargetSpec(
        base_model="m", scopes=["procedural"], target_entities=["rotate-creds"]
    )
    corpus = MemoryWeightsDistiller(engine, spec=spec).export()
    assert corpus.source_ids == ["proc-rotate-creds"]
    assert corpus.stats["by_scope"] == {"procedural": 1}


def test_kg_2_316_preference_corpus_when_nodes_carry_chosen_rejected() -> None:
    nodes = [
        {
            "id": "sem-pref",
            "memory_type": "semantic",
            "status": "ACTIVE",
            "target_entity": "deploy",
            "content": "context",
            "prompt": "How should I deploy?",
            "chosen": "Deploy after the vault check passes.",
            "rejected": "Deploy immediately without checks.",
        }
    ]
    spec = DistillationTargetSpec(base_model="m", method="dpo", scopes=["semantic"])
    corpus = MemoryWeightsDistiller(_MockEngine(nodes), spec=spec).export()
    assert len(corpus.examples) == 1
    ex = corpus.examples[0]
    assert set(ex.keys()) == {"prompt", "chosen", "rejected", "source_id"}
    assert ex["chosen"].startswith("Deploy after")


# ── submit: data-science-mcp hand-off contract ─────────────────────────────────
def test_kg_2_316_default_submit_enqueues_training_job(tmp_path, monkeypatch) -> None:
    # Redirect the memory dir so the corpus/manifest land in a temp dir.
    import agent_utilities.knowledge_graph.memory.memory_engine as me

    monkeypatch.setattr(me, "memory_dir", lambda: tmp_path)

    engine = _MockEngine(_consolidated_and_noise())
    spec = DistillationTargetSpec(base_model="Qwen2.5-1.5B", adapter_rank=8)
    dist = MemoryWeightsDistiller(engine, spec=spec)
    corpus = dist.export()

    job = dist.submit(corpus)

    assert isinstance(job, DistillationJob)
    assert job.status == "enqueued"  # engine has add_node → durable job node
    assert job.example_count == 2
    assert job.checksum == corpus.checksum
    # The JSONL corpus was materialized to disk.
    assert (tmp_path / "distillation" / f"{job.job_id}.jsonl").exists()
    # A durable TrainingJob node was registered for the fleet to pick up.
    assert engine.nodes[job.job_id]["label"] == "TrainingJob"
    assert engine.nodes[job.job_id]["server"] == "data-science-mcp"
    # The hand-off carries the concrete train_model workflow + spec.
    hoff = job.handoff
    assert hoff["contract"] == "AU-KG.memory.memory-weights-distillation-export"
    assert hoff["workflow"]["name"] == "train_model"
    assert hoff["workflow"]["task"]["spec"]["adapter_rank"] == 8
    assert hoff["tools"][1]["tool"] == "train_sft"
    # Poll reports the durable enqueued state.
    assert dist.poll(job.job_id)["status"] == "enqueued"


def test_kg_2_316_default_submit_degrades_without_job_surface(
    tmp_path, monkeypatch
) -> None:
    import agent_utilities.knowledge_graph.memory.memory_engine as me

    monkeypatch.setattr(me, "memory_dir", lambda: tmp_path)
    dist = MemoryWeightsDistiller(
        _BareEngine(_consolidated_and_noise()),
        spec=DistillationTargetSpec(base_model="m"),
    )
    job = dist.submit(dist.export())
    assert job.status == "exported"  # materialized only, nothing enqueued
    assert "no job surface" in job.detail


def test_kg_2_316_injected_submitter_seam() -> None:
    seen: dict[str, Any] = {}

    def _stub(
        corpus: DistillationCorpus, spec: DistillationTargetSpec
    ) -> DistillationJob:
        seen["examples"] = len(corpus.examples)
        seen["base_model"] = spec.base_model
        return DistillationJob(
            job_id="stub-1",
            status="submitted",
            spec=spec.to_dict(),
            corpus_ref="stub",
            checksum=corpus.checksum,
            example_count=len(corpus.examples),
            handoff={"contract": "AU-KG.memory.memory-weights-distillation-export"},
        )

    dist = MemoryWeightsDistiller(
        _MockEngine(_consolidated_and_noise()),
        spec=DistillationTargetSpec(base_model="stubbed"),
        submitter=_stub,
    )
    job = dist.submit(dist.export())
    assert job.job_id == "stub-1" and job.status == "submitted"
    assert seen == {"examples": 2, "base_model": "stubbed"}


def test_kg_2_316_contract_shape() -> None:
    assert DATA_SCIENCE_MCP_CONTRACT["server"] == "data-science-mcp"
    assert DATA_SCIENCE_MCP_CONTRACT["corpus_format"]["sft"] == ["prompt", "completion"]
    assert DATA_SCIENCE_MCP_CONTRACT["mcp_tools"]["train"]["dpo"] == "train_dpo"


# ── action-core entry ───────────────────────────────────────────────────────────
def test_kg_2_316_action_core_export_only() -> None:
    engine = _MockEngine(_consolidated_and_noise())
    res = distill_memory_to_weights(
        engine, params={"base_model": "m", "scopes": ["procedural", "semantic"]}
    )
    assert res["status"] == "ok"
    assert res["concept"] == "AU-KG.memory.memory-weights-distillation-export"
    assert res["corpus"]["example_count"] == 2
    assert res["corpus"]["format"] == "sft"
    # Export-only: a hand-off preview is offered but no job is submitted.
    assert "job" not in res
    assert res["handoff"]["workflow"]["name"] == "train_model"


# ── LIVE PATH: the graph_analyze MCP tool (+ REST twin) dispatch ───────────────
class _FakeMCP:
    """Captures the tool coroutines ``register_analysis_tools`` registers."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *, name: str, description: str = "", tags: Any = None):
        def _decorator(fn):
            self.tools[name] = fn
            return fn

        return _decorator


def test_kg_2_316_live_graph_analyze_action(tmp_path, monkeypatch) -> None:
    """Invoke the REAL graph_analyze tool coroutine with action=distill_memory.

    This is the exact function the graph-os MCP surface and the POST /graph/analyze
    REST twin dispatch into (both funnel through kg_server._execute_tool), so it
    proves the action is wired live on BOTH surfaces from one action-core method.
    """
    import asyncio

    import agent_utilities.knowledge_graph.memory.memory_engine as me
    from agent_utilities.mcp import kg_server
    from agent_utilities.mcp.tools import analysis_tools

    monkeypatch.setattr(me, "memory_dir", lambda: tmp_path)
    engine = _MockEngine(_consolidated_and_noise())
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    fake = _FakeMCP()
    analysis_tools.register_analysis_tools(fake)
    assert (
        kg_server.REGISTERED_TOOLS.get("graph_analyze") is fake.tools["graph_analyze"]
    )

    # Dispatch through the SHARED core (_execute_tool) that BOTH the MCP tool
    # surface and the POST /graph/analyze REST twin funnel through — it resolves
    # the Field() defaults exactly as the live surfaces do.
    params = json.dumps(
        {
            "base_model": "Qwen2.5-1.5B",
            "scopes": ["procedural", "semantic"],
            "adapter_rank": 8,
            "submit": True,
        }
    )
    out = asyncio.run(
        kg_server._execute_tool("graph_analyze", action="distill_memory", query=params)
    )
    payload = json.loads(out)

    assert payload["status"] == "ok"
    assert payload["concept"] == "AU-KG.memory.memory-weights-distillation-export"
    assert payload["corpus"]["example_count"] == 2
    assert payload["corpus"]["spec"]["adapter_rank"] == 8
    # Submitted live: a job with the data-science-mcp hand-off came back.
    assert payload["job"]["status"] == "enqueued"
    assert payload["job"]["handoff"]["workflow"]["name"] == "train_model"
    assert engine.nodes[payload["job"]["job_id"]]["label"] == "TrainingJob"


# ── KG-2.318: LIVE data-science-mcp dispatch + status poll ─────────────────────
def test_kg_2_318_submit_invokes_live_dispatch_and_marks_running(
    tmp_path, monkeypatch
) -> None:
    """The default submit dispatches the train LIVE through the dispatcher seam.

    Injects a MOCK data-science-mcp client (the dispatcher) — standing in for the
    ``train_model`` workflow over ``graph_orchestrate`` — and asserts submit ACTUALLY
    invokes it, marks the job ``running`` with the remote run handle, and registers a
    ``TrainingJob`` node the fleet + poll can read back.
    """
    import agent_utilities.knowledge_graph.memory.memory_engine as me

    monkeypatch.setattr(me, "memory_dir", lambda: tmp_path)

    seen: dict[str, Any] = {}

    def _mock_mcp_dispatch(
        handoff: dict[str, Any],
        corpus: DistillationCorpus,
        spec: DistillationTargetSpec,
    ) -> dict[str, Any]:
        # Assert the concrete data-science-mcp hand-off reaches the client.
        seen["workflow"] = handoff["workflow"]["name"]
        seen["base_model"] = spec.base_model
        seen["examples"] = len(corpus.examples)
        return {
            "dispatched": True,
            "status": "running",
            "run_id": "dsmcp-run-42",
            "via": "execute_workflow",
        }

    engine = _MockEngine(_consolidated_and_noise())
    dist = MemoryWeightsDistiller(
        engine,
        spec=DistillationTargetSpec(base_model="Qwen2.5-1.5B", adapter_rank=8),
        dispatcher=_mock_mcp_dispatch,
    )

    job = dist.submit(dist.export())

    # The mock MCP client was actually invoked with the train_model hand-off.
    assert seen == {
        "workflow": "train_model",
        "base_model": "Qwen2.5-1.5B",
        "examples": 2,
    }
    # Live-dispatched ⇒ the job is running and carries the remote run handle.
    assert job.status == "running"
    assert job.handoff["dispatch"]["dispatched"] is True
    assert job.handoff["dispatch"]["run_id"] == "dsmcp-run-42"
    # The durable TrainingJob node records the dispatch for the fleet + poll.
    node = engine.nodes[job.job_id]
    assert node["label"] == "TrainingJob"
    assert node["status"] == "running"
    assert node["dispatched"] is True
    assert node["run_id"] == "dsmcp-run-42"

    # Poll reads the live state back — and once data-science-mcp finishes the train
    # (updating the node + linking a register_checkpoint node), poll surfaces it.
    assert dist.poll(job.job_id)["status"] == "running"
    engine.nodes[job.job_id]["status"] = "succeeded"
    engine.nodes[job.job_id]["checkpoint_ref"] = "ckpt-1"
    engine.add_node(
        "ckpt-1",
        "ModelCheckpoint",
        {"adapter_path": "/models/lora/ckpt-1", "base_model": "Qwen2.5-1.5B"},
    )
    polled = dist.poll(job.job_id)
    assert polled["status"] == "succeeded"
    assert polled["run_id"] == "dsmcp-run-42"
    assert polled["checkpoint"]["adapter_path"] == "/models/lora/ckpt-1"


def test_kg_2_318_dispatch_degrades_to_enqueued_when_unreachable(
    tmp_path, monkeypatch
) -> None:
    """An unreachable data-science-mcp degrades to a durable ``enqueued`` job.

    A dispatcher that raises (data-science-mcp down) must NOT abort the submit; the
    corpus is still materialized and a ``TrainingJob`` node enqueued for later pickup.
    """
    import agent_utilities.knowledge_graph.memory.memory_engine as me

    monkeypatch.setattr(me, "memory_dir", lambda: tmp_path)

    def _unreachable(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise ConnectionError("data-science-mcp unreachable")

    engine = _MockEngine(_consolidated_and_noise())
    dist = MemoryWeightsDistiller(
        engine,
        spec=DistillationTargetSpec(base_model="m"),
        dispatcher=_unreachable,
    )
    job = dist.submit(dist.export())

    assert job.status == "enqueued"  # durable — fleet can still pick it up
    assert job.handoff["dispatch"]["dispatched"] is False
    assert engine.nodes[job.job_id]["status"] == "enqueued"
    assert engine.nodes[job.job_id]["dispatched"] is False
    # The corpus was still materialized despite the failed dispatch.
    assert (tmp_path / "distillation" / f"{job.job_id}.jsonl").exists()


def test_kg_2_318_default_dispatch_skips_non_orchestration_engine(
    tmp_path, monkeypatch
) -> None:
    """The DEFAULT dispatcher (no injection) is bounded: a mock engine that is not an
    orchestration ``IntelligenceGraphEngine`` degrades straight to ``enqueued`` without
    attempting (or hanging on) a live workflow run."""
    import agent_utilities.knowledge_graph.memory.memory_engine as me

    monkeypatch.setattr(me, "memory_dir", lambda: tmp_path)
    engine = _MockEngine(_consolidated_and_noise())
    dist = MemoryWeightsDistiller(engine, spec=DistillationTargetSpec(base_model="m"))

    job = dist.submit(dist.export())

    assert job.status == "enqueued"
    assert job.handoff["dispatch"]["dispatched"] is False
    assert "no orchestration engine" in job.handoff["dispatch"]["detail"]


# ── GRPO: EG-099 Trajectory/Step reward sequences → GRPO groups ────────────────
def _synthetic_trajectory_steps() -> list[dict[str, Any]]:
    """Two rollout groups (distinct ``state_ref``) of trajectory ``:Step`` nodes.

    Mirrors EG-099 ``:Trajectory``/``:Step{state_ref, action, reward,
    next_state_ref}`` — several sampled actions rolled out from the SAME state,
    each carrying its own reward (the GRPO source shape).
    """
    return [
        {
            "id": "step-a1",
            "memory_type": "trajectory",
            "status": "ACTIVE",
            "trajectory_id": "traj-1",
            "state_ref": "state:deploy-decision",
            "action": "roll forward",
            "reward": 1.0,
            "next_state_ref": "state:post-deploy-ok",
        },
        {
            "id": "step-a2",
            "memory_type": "trajectory",
            "status": "ACTIVE",
            "trajectory_id": "traj-2",
            "state_ref": "state:deploy-decision",
            "action": "roll back",
            "reward": -1.0,
            "next_state_ref": "state:post-deploy-rollback",
        },
        {
            "id": "step-a3",
            "memory_type": "trajectory",
            "status": "ACTIVE",
            "trajectory_id": "traj-3",
            "state_ref": "state:deploy-decision",
            "action": "wait and retry",
            "reward": 0.0,
            "next_state_ref": "state:post-deploy-retry",
        },
        {
            "id": "step-b1",
            "memory_type": "trajectory",
            "status": "ACTIVE",
            "trajectory_id": "traj-4",
            "state_ref": "state:scale-decision",
            "action": "scale up",
            "reward": 2.0,
            "next_state_ref": "state:post-scale-ok",
        },
        {
            "id": "step-b2",
            "memory_type": "trajectory",
            "status": "ACTIVE",
            "trajectory_id": "traj-5",
            "state_ref": "state:scale-decision",
            "action": "no-op",
            "reward": 0.0,
            "next_state_ref": "state:post-scale-noop",
        },
        # Noise: retired step + a step missing an action (unusable).
        {
            "id": "step-old",
            "memory_type": "trajectory",
            "status": "RETIRED",
            "state_ref": "state:deploy-decision",
            "action": "stale",
            "reward": 0.5,
        },
        {
            "id": "step-incomplete",
            "memory_type": "trajectory",
            "status": "ACTIVE",
            "state_ref": "",
            "action": None,
            "reward": 0.1,
        },
    ]


def test_grpo_spec_is_grpo_and_format_name() -> None:
    spec = DistillationTargetSpec(base_model="m", method="GRPO", scopes=["trajectory"])
    assert spec.method == "grpo"
    assert spec.is_grpo is True
    assert spec.is_preference is False
    assert spec.format_name == "grpo"


def test_grpo_renderer_groups_steps_by_state_ref_with_advantage() -> None:
    from agent_utilities.graph.training_signals import batch_normalized_advantage

    spec = DistillationTargetSpec(base_model="m", method="grpo", scopes=["trajectory"])
    dist = MemoryWeightsDistiller(_MockEngine(_synthetic_trajectory_steps()), spec=spec)

    # ``to_grpo_groups`` renders already-SELECTED nodes (status/scope filtering
    # is ``select()``'s job — mirrors the SFT/DPO ``to_example`` contract).
    selected = dist.select()
    records, source_ids = dist.to_grpo_groups(selected)

    assert len(records) == 2
    deploy = next(r for r in records if r["prompt"] == "state:deploy-decision")
    scale = next(r for r in records if r["prompt"] == "state:scale-decision")

    # Valid GRPO shape: {prompt, samples:[{completion, reward, advantage}]}.
    for record in (deploy, scale):
        assert set(record.keys()) == {"prompt", "samples"}
        for sample in record["samples"]:
            assert set(sample.keys()) == {"completion", "reward", "advantage"}

    assert {s["completion"] for s in deploy["samples"]} == {
        "roll forward",
        "roll back",
        "wait and retry",
    }
    # The advantage matches the shared reward-spine primitive exactly.
    rewards = [1.0, -1.0, 0.0]
    expected_adv = batch_normalized_advantage(rewards)
    got_adv = [s["advantage"] for s in deploy["samples"]]
    assert got_adv == expected_adv

    assert len(scale["samples"]) == 2
    # Only steps that landed in an emitted record count as sources (RETIRED /
    # incomplete steps are excluded upstream by ``select()``; this call renders
    # the already-selected nodes directly, so nothing here is dropped).
    assert set(source_ids) == {"step-a1", "step-a2", "step-a3", "step-b1", "step-b2"}


def test_grpo_export_selects_trajectory_scope_and_builds_groups() -> None:
    spec = DistillationTargetSpec(base_model="m", method="grpo", scopes=["trajectory"])
    dist = MemoryWeightsDistiller(_MockEngine(_synthetic_trajectory_steps()), spec=spec)

    corpus = dist.export()

    assert isinstance(corpus, DistillationCorpus)
    assert corpus.stats["format"] == "grpo"
    # RETIRED + incomplete steps are excluded by select(); only the 5 ACTIVE,
    # well-formed steps feed the two GRPO groups.
    assert corpus.stats["selected"] == 5
    assert len(corpus.examples) == 2
    summary = corpus.summary()
    assert summary["format"] == "grpo"
    prompts = {ex["prompt"] for ex in corpus.examples}
    assert prompts == {"state:deploy-decision", "state:scale-decision"}


def test_grpo_export_feeds_build_training_dataset_kind_grpo() -> None:
    """The renderer's output is exactly what ``build_training_dataset kind=grpo``
    (data-science-mcp ``training_data.build_grpo_groups``) accepts as GROUPED
    INPUT, and its shape survives that pass through unchanged (this repo owns
    the same ``batch_normalized_advantage`` primitive data-science-mcp calls, so
    the two are byte-shape-identical without a cross-repo import)."""
    from agent_utilities.graph.training_signals import batch_normalized_advantage

    spec = DistillationTargetSpec(base_model="m", method="grpo", scopes=["trajectory"])
    dist = MemoryWeightsDistiller(_MockEngine(_synthetic_trajectory_steps()), spec=spec)
    corpus = dist.export()

    # Re-derive the equivalent of data-science-mcp's build_grpo_groups() input
    # shape ({prompt, completions, rewards}) from our already-grouped records,
    # and confirm re-normalizing produces the SAME advantages already attached
    # — i.e. our records are a valid, already-normalized `kind=grpo` output.
    for record in corpus.examples:
        rewards = [s["reward"] for s in record["samples"]]
        recomputed = batch_normalized_advantage(rewards)
        assert [s["advantage"] for s in record["samples"]] == recomputed


def test_grpo_handoff_routes_to_train_grpo_tool() -> None:
    spec = DistillationTargetSpec(
        base_model="m", method="grpo", adapter_rank=8, scopes=["trajectory"]
    )
    dist = MemoryWeightsDistiller(_MockEngine([]), spec=spec)
    handoff = dist._build_handoff("inline:test")
    assert handoff["workflow"]["task"]["corpus_format"] == ["prompt", "samples"]
    assert handoff["tools"][1]["tool"] == "train_grpo"


def test_kg_2_318_action_core_poll_reads_status_back(tmp_path, monkeypatch) -> None:
    """The ``poll_job_id`` param on the action-core reads a job's live state back —
    the status-poll surface both the MCP tool and its REST twin dispatch into."""
    import agent_utilities.knowledge_graph.memory.memory_engine as me

    monkeypatch.setattr(me, "memory_dir", lambda: tmp_path)
    engine = _MockEngine(_consolidated_and_noise())

    submitted = distill_memory_to_weights(
        engine,
        params={"base_model": "m", "scopes": ["procedural", "semantic"]},
        submit=True,
    )
    job_id = submitted["job"]["job_id"]

    # data-science-mcp advances the train.
    engine.nodes[job_id]["status"] = "running"
    polled = distill_memory_to_weights(engine, params={"poll_job_id": job_id})
    assert polled["concept"] == "AU-KG.memory.live-data-science-mcp"
    assert polled["poll"]["status"] == "running"
    assert polled["poll"]["job_id"] == job_id
