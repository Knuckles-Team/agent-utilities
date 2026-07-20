"""Cross-harness GRPO co-evolution (CONCEPT:AU-AHE.harness.co-evolution-loop, CONCEPT:AU-AHE.harness.kg-held-out-certification).

HarnessX co-evolves harness + model over one shared replay buffer via cross-harness
GRPO — grouping trajectories by *task identity across harness versions* so the model
internalises strategies that succeeded under successive scaffolds. We realise it by
reusing our existing primitives: `PrioritizedReplayBuffer`, the GRPO
`batch_normalized_advantage(group_ids=…)` (the cross-harness grouping criterion is
exactly `group_ids = task`), `SubstrateTrainer` (the GRPO corpus → deferred GPU job
— "replay reuse at no added rollout cost"), and `SuperhumanCertifier` (held-out
bootstrap-CI promotion — the held-out evaluation HarnessX lacks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_utilities.graph.training_signals import batch_normalized_advantage
from agent_utilities.harness.replay_buffer import PrioritizedReplayBuffer
from agent_utilities.harness.substrate_trainer import GrpoSample
from agent_utilities.harness.superhuman_gate import (
    CertificationResult,
    SuperhumanCertifier,
)


@dataclass
class Trajectory:
    """One scored rollout, tagged with the (model, harness) that produced it so the
    shared buffer mixes scaffold versions (CONCEPT:AU-AHE.harness.co-evolution-loop)."""

    task: str
    harness_version: str
    model_ckpt: str
    reward: float
    prompt: str = ""


@dataclass
class CoEvolutionState:
    trajectories: list[Trajectory] = field(default_factory=list)


class CrossHarnessCoEvolution:
    """Accumulate trajectories across harness versions in one buffer; expose the
    cross-harness GRPO corpus and held-out certification gate."""

    def __init__(
        self, *, capacity: int = 4096, certifier: SuperhumanCertifier | None = None
    ) -> None:
        self._buffer = PrioritizedReplayBuffer(capacity=capacity)
        self._certifier = certifier or SuperhumanCertifier()
        self.trajectories: list[Trajectory] = []

    def observe(self, traj: Trajectory) -> None:
        """Append a scored trace; the buffer keys by task so successive rounds
        accumulate rather than overwrite."""
        self.trajectories.append(traj)
        self._buffer.add(traj, key=traj.task)

    def cross_harness_advantages(self) -> list[tuple[Trajectory, float]]:
        """Group-relative advantage with the cross-harness grouping criterion:
        `group_ids = task` (across harness versions), so within-group variation is
        the strategy contrast between scaffolds, not sampling noise (Eq. 2/3)."""
        rewards = [t.reward for t in self.trajectories]
        groups: list[Any] = [t.task for t in self.trajectories]
        adv = batch_normalized_advantage(rewards, group_ids=groups)
        return list(zip(self.trajectories, adv, strict=True))

    def grpo_corpus(self) -> list[GrpoSample]:
        """The GRPO training corpus (deferred GPU step via the substrate trainer —
        replay reuse at no added rollout cost)."""
        return [
            GrpoSample(task_key=t.task, prompt=t.prompt, reward=t.reward, advantage=a)
            for t, a in self.cross_harness_advantages()
        ]

    def certify_promotion(
        self, held_out_rewards: list[float], human_baseline: float | None
    ) -> CertificationResult:
        """Gate variant promotion on a HELD-OUT split (the evaluation HarnessX
        lacks): certified only if the bootstrap CI lower bound clears the baseline."""
        return self._certifier.certify(held_out_rewards, human_baseline)
