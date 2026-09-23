"""The external training path for open-weight policy evolution (EH-347).

AU emits a digest-bound job, an EXTERNAL trainer differentiates, and EG records
the outcome. Nothing here runs gradient descent, loads a checkpoint or moves a
serving pointer:

1. the EG-attested capability must have its ``train`` control on (a typed
   ``POLICY_TRAIN_DISABLED`` refusal otherwise, before any side effect);
2. :meth:`SubstrateTrainer.policy_job` emits the job spec naming immutable EG
   capture ids and digests (KLPO is one method beside GRPO/DPO/SFT; LoRA only);
3. the injected :class:`ExternalPolicyTrainer` runs it and reports an outcome;
4. the ``TrainingRun`` receipt is committed to EG whatever the outcome;
5. only a succeeded run's new adapter is registered as a ``ModelPolicyVersion``.
   A failed or cancelled run has no output and can never become servable.

Promotion (release-pointer compare-and-swap after a held-out
``PolicyEvaluation``) belongs to graph-os and stays off here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from epistemic_graph.generated.policy_evolution import (
    ModelPolicyVersion,
    PolicyRecordReceipt,
    TrainingRun,
)

from ..substrate_trainer import PolicyJobInputs, SubstrateTrainer, TrainingJobSpec
from .capture import PolicyRecords, attested_capability

__all__ = [
    "ExternalPolicyTrainer",
    "PolicyTrainingPath",
    "PolicyTrainingRecords",
    "PolicyTrainingRequest",
    "PolicyTrainingResult",
    "TrainerOutcome",
]


class PolicyTrainingRecords(PolicyRecords, Protocol):
    """The generated ``PolicyEvolutionClient`` writes the training path needs."""

    async def commit_training_run(self, run: TrainingRun) -> PolicyRecordReceipt: ...

    async def register_model_policy_version(
        self, version: ModelPolicyVersion
    ) -> PolicyRecordReceipt: ...


@dataclass(frozen=True)
class TrainerOutcome:
    """What the external trainer reports for one job.

    ``status`` is ``succeeded``, ``failed`` or ``cancelled``. Only a succeeded
    outcome carries the new adapter's digest and artifact reference.
    """

    status: str
    trainer_image_digest: str
    resources: dict[str, int] = field(default_factory=dict)
    artifact_digest: str | None = None
    artifact_ref: str | None = None


class ExternalPolicyTrainer(Protocol):
    """The gradient substrate: runs one digest-bound job outside AU and EG."""

    async def run(self, spec: TrainingJobSpec) -> TrainerOutcome: ...


@dataclass(frozen=True)
class PolicyTrainingRequest:
    """One requested adapter-training attempt, owned by a native WorkItem."""

    capability_id: str
    base_version_id: str
    capture_ids: tuple[str, ...]
    work_item_id: str
    method: dict[str, Any]
    adapter: dict[str, Any]
    hyperparameters: dict[str, Any]
    tokenizer_digest: str
    checkpoint_digest: str


@dataclass(frozen=True)
class PolicyTrainingResult:
    """The job that ran, its EG run receipt and, on success, the new version."""

    job: TrainingJobSpec
    run: PolicyRecordReceipt
    version: PolicyRecordReceipt | None


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class PolicyTrainingPath:
    """Gate, emit, run externally, and record one policy training attempt."""

    def __init__(
        self,
        records: PolicyTrainingRecords,
        trainer: ExternalPolicyTrainer,
        substrate: SubstrateTrainer | None = None,
    ) -> None:
        self._records = records
        self._trainer = trainer
        self._substrate = substrate or SubstrateTrainer()

    async def run(self, request: PolicyTrainingRequest) -> PolicyTrainingResult:
        capability = await attested_capability(
            self._records, request.capability_id, "train"
        )
        job = self._substrate.policy_job(
            PolicyJobInputs(
                capability_id=request.capability_id,
                base_version_id=request.base_version_id,
                capture_ids=request.capture_ids,
                method=request.method,
                adapter=request.adapter,
                hyperparameters_digest=_digest(request.hyperparameters),
                output_destination_ref=str(capability.artifact_destination_ref),
            )
        )
        outcome = await self._trainer.run(job)
        run_receipt = await self._records.commit_training_run(
            self._run_record(request, outcome)
        )
        version = None
        if outcome.status == "succeeded":
            version = await self._records.register_model_policy_version(
                self._version_record(request, run_receipt, outcome)
            )
        return PolicyTrainingResult(job=job, run=run_receipt, version=version)

    @staticmethod
    def _run_record(
        request: PolicyTrainingRequest, outcome: TrainerOutcome
    ) -> TrainingRun:
        status: dict[str, Any] = {"status": outcome.status}
        if outcome.status == "succeeded":
            status["output"] = {
                "artifact_digest": outcome.artifact_digest,
                "artifact_ref": outcome.artifact_ref,
            }
        return TrainingRun.model_validate(
            {
                "capability_id": request.capability_id,
                "work_item_id": request.work_item_id,
                "base_version_id": request.base_version_id,
                "input_capture_ids": list(request.capture_ids),
                "method": request.method,
                "adapter": request.adapter,
                "trainer_image_digest": outcome.trainer_image_digest,
                "hyperparameters_digest": _digest(request.hyperparameters),
                "resources": outcome.resources or None,
                "status": status,
            }
        )

    @staticmethod
    def _version_record(
        request: PolicyTrainingRequest,
        run_receipt: PolicyRecordReceipt,
        outcome: TrainerOutcome,
    ) -> ModelPolicyVersion:
        return ModelPolicyVersion.model_validate(
            {
                "checkpoint_digest": request.checkpoint_digest,
                "adapter_digest": outcome.artifact_digest,
                "tokenizer_digest": request.tokenizer_digest,
                "artifact_ref": outcome.artifact_ref,
                "parent_version_id": request.base_version_id,
                "origin": {
                    "origin": "trained",
                    "training_run_id": run_receipt.record_id,
                },
            }
        )
