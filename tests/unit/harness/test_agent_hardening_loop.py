"""Live-path test for the agent-hardening loop (CONCEPT:AHE-3.71/3.72/3.73).

The closed cycle, end-to-end for ONE agent and offline-deterministic (no LM):

    synthetic action_outcomes → per-agent trainset → optimize → candidate prompt
    → evaluate vs baseline on the agent's eval slice → gated apply → audit trail.

Asserts a better prompt is produced and **written under the gate**, held in shadow when
the gate is off, and rejected when it does not beat baseline — leaving a queryable
ProposedPromptChange audit record in every case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_utilities.harness.eval_corpus import EvalCorpus
from agent_utilities.harness.evolve_agent import EvolveAgent
from agent_utilities.knowledge_graph.adaptation.feedback import FeedbackService

pytestmark = pytest.mark.asyncio

# Real deployment outcomes the agent reached — they double as the trainset (demos) and
# the eval slice the candidate is scored against.
_OUTCOMES = [
    ("how do I deploy the api service?", "run kubectl apply -f api.yaml then verify rollout status"),
    ("restart the worker", "scale the worker deployment to zero then back to three replicas"),
    ("check the database health", "query pg_stat_activity and confirm replication lag is under one second"),
]


def _seed_agent_outcomes(fb: FeedbackService, agent_id: str) -> None:
    for query, expected in _OUTCOMES:
        fb.record_action_outcome(
            f"deploy:{agent_id}",
            success=True,
            expected=expected,
            observed=expected,
            query=query,
            agent_id=agent_id,
        )


def _write_baseline_prompt(path: Path, *, core_directive: str) -> None:
    blueprint = {
        "task": "deploy_agent",
        "type": "prompt",
        "prompt_version": "0.1.0",
        "instructions": {"core_directive": core_directive},
    }
    path.write_text(json.dumps(blueprint, indent=2), encoding="utf-8")


def _make_evolver(tmp_path: Path) -> tuple[EvolveAgent, FeedbackService]:
    import subprocess

    # A real git workspace so the apply path's commit succeeds (true live path).
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=False)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=False)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=False)
    fb = FeedbackService(backend=None, eval_corpus=EvalCorpus(backend=None))
    evolver = EvolveAgent(workspace_path=str(tmp_path), feedback_service=fb)
    return evolver, fb


# ── G7/G3: per-agent attribution pools real outcomes into a trainset ─────────


async def test_per_agent_trainset_pools_only_that_agents_outcomes(tmp_path):
    _, fb = _make_evolver(tmp_path)
    _seed_agent_outcomes(fb, "deploy-agent")
    fb.record_action_outcome(
        "other", success=True, expected="irrelevant", query="q", agent_id="other-agent"
    )
    train = fb.build_agent_trainset("deploy-agent")
    assert len(train) == len(_OUTCOMES)  # scoped to THIS agent, not the other
    assert fb.build_agent_trainset("nobody") == []


# ── G1/G8/G2: optimize → better prompt → WRITTEN under the auto-apply gate ───


async def test_cycle_applies_better_prompt_when_gate_on(tmp_path):
    evolver, fb = _make_evolver(tmp_path)
    _seed_agent_outcomes(fb, "deploy-agent")
    prompt_path = tmp_path / "deploy_agent.json"
    _write_baseline_prompt(prompt_path, core_directive="You are a deploy assistant.")

    outcome = await evolver.harden_agent_prompt(
        "deploy-agent", "deploy_agent.json", auto_apply=True
    )

    # a strictly better candidate was produced and promoted
    assert outcome.candidate_score > outcome.baseline_score
    assert outcome.promote is True
    assert outcome.status == "applied" and outcome.applied is True
    assert outcome.trainset_size == len(_OUTCOMES)

    # the optimized body was actually written to source via StructuredPrompt.save()
    written = json.loads(prompt_path.read_text(encoding="utf-8"))
    body = written["instructions"]["core_directive"]
    assert "LEARNED EXEMPLARS" in body
    assert "kubectl apply" in body  # a real outcome folded in
    assert written["prompt_version"] == "0.1.1"  # version bumped

    # audit trail: a ProposedPromptChange record exists with before/after + decision
    proposals = list((tmp_path / ".specify" / "proposals").glob("*.json"))
    assert len(proposals) == 1
    rec = json.loads(proposals[0].read_text(encoding="utf-8"))
    assert rec["status"] == "applied" and rec["applied"] is True
    assert rec["delta"] > 0 and rec["agent_id"] == "deploy-agent"


# ── transparency: gate OFF ⇒ propose-only / shadow, live prompt untouched ────


async def test_cycle_is_shadow_when_gate_off(tmp_path):
    evolver, fb = _make_evolver(tmp_path)
    _seed_agent_outcomes(fb, "deploy-agent")
    prompt_path = tmp_path / "deploy_agent.json"
    _write_baseline_prompt(prompt_path, core_directive="You are a deploy assistant.")
    before = prompt_path.read_text(encoding="utf-8")

    outcome = await evolver.harden_agent_prompt(
        "deploy-agent", "deploy_agent.json", auto_apply=False
    )

    assert outcome.promote is True  # it WOULD beat baseline
    assert outcome.status == "proposed" and outcome.applied is False
    # the live prompt is NOT modified — a rewrite is never silent
    assert prompt_path.read_text(encoding="utf-8") == before

    # but the candidate is queryable + approvable in the audit record
    proposals = list((tmp_path / ".specify" / "proposals").glob("*.json"))
    assert len(proposals) == 1
    rec = json.loads(proposals[0].read_text(encoding="utf-8"))
    assert rec["status"] == "proposed"
    assert "LEARNED EXEMPLARS" in rec["candidate_blueprint"]["instructions"][
        "core_directive"
    ]

    # ...and approval applies it on demand (steerable, not auto)
    approved = evolver.approve_proposed_change(rec["id"])
    assert approved["approved"] is True
    written = json.loads(prompt_path.read_text(encoding="utf-8"))
    assert "LEARNED EXEMPLARS" in written["instructions"]["core_directive"]


# ── promotion gate: a candidate that doesn't beat baseline is rejected ───────


async def test_cycle_rejects_when_no_improvement(tmp_path):
    evolver, fb = _make_evolver(tmp_path)
    _seed_agent_outcomes(fb, "deploy-agent")
    prompt_path = tmp_path / "deploy_agent.json"
    # baseline ALREADY contains the outcomes, so folding them adds no coverage
    baseline_body = "You are a deploy assistant. " + " ".join(e for _, e in _OUTCOMES)
    _write_baseline_prompt(prompt_path, core_directive=baseline_body)
    before = prompt_path.read_text(encoding="utf-8")

    outcome = await evolver.harden_agent_prompt(
        "deploy-agent", "deploy_agent.json", auto_apply=True, min_delta=0.05
    )

    assert outcome.promote is False
    assert outcome.status == "rejected" and outcome.applied is False
    assert prompt_path.read_text(encoding="utf-8") == before  # untouched
    rec = json.loads(
        next((tmp_path / ".specify" / "proposals").glob("*.json")).read_text("utf-8")
    )
    assert rec["status"] == "rejected"


# ── no per-agent corpus ⇒ clean no_data, never raises ────────────────────────


async def test_cycle_no_data_without_outcomes(tmp_path):
    evolver, _ = _make_evolver(tmp_path)
    prompt_path = tmp_path / "deploy_agent.json"
    _write_baseline_prompt(prompt_path, core_directive="You are a deploy assistant.")
    outcome = await evolver.harden_agent_prompt("ghost-agent", "deploy_agent.json")
    assert outcome.status == "no_data" and outcome.applied is False
