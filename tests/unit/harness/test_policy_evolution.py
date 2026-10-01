"""Policy evolution, AU side (AU-HARNESS-R001/AU-HARNESS-R002): capture and the external training path.

No GPU, no vLLM, no engine: the sampler transport, EG's ``PolicyEvolutionClient``
and Blob CAS are in-memory fakes that speak the GENERATED EG wire types.
"""

from __future__ import annotations

import asyncio
import hashlib
import struct
from typing import Any

import pytest
from epistemic_graph.generated.policy_evolution import (
    ModelPolicyVersion,
    OpenWeightPolicyCapability,
    PolicyCapture,
    PolicyRecordReceipt,
    PolicyRecordView,
    TrainingRun,
    VersionOriginTrained,
)
from epistemic_graph.policy_evolution import PolicyEvolutionRefused

from agent_utilities.harness.policy_evolution import (
    CaptureSpec,
    EpisodeTokens,
    PolicyCaptureRecorder,
    PolicyTrainingPath,
    PolicyTrainingRequest,
    TrainerOutcome,
    VllmLogprobSampler,
)
from agent_utilities.harness.substrate_trainer import SubstrateTrainer, TrainingJobSpec

_KINDS: dict[type[Any], tuple[str, str]] = {
    OpenWeightPolicyCapability: ("capability", "polcap"),
    PolicyCapture: ("capture", "polcapture"),
    ModelPolicyVersion: ("model_policy_version", "polver"),
    TrainingRun: ("training_run", "poltrain"),
}


def _hex(byte: int) -> str:
    return f"{byte:02x}" * 32


class FakePolicyEvolution:
    """An in-memory stand-in for the generated ``PolicyEvolutionClient``."""

    def __init__(self) -> None:
        self.records: dict[str, Any] = {}

    def _put(self, record: Any) -> PolicyRecordReceipt:
        kind, prefix = _KINDS[type(record)]
        body = record.model_dump_json()
        record_id = f"{prefix}:{hashlib.sha256(body.encode()).hexdigest()}"
        disposition = "replayed" if record_id in self.records else "written"
        self.records[record_id] = record
        return PolicyRecordReceipt.model_validate(
            {
                "record_id": record_id,
                "kind": kind,
                "disposition": disposition,
                "observed_at_ms": 1,
            }
        )

    async def get(self, record_id: str) -> PolicyRecordView | None:
        record = self.records.get(record_id)
        if record is None:
            return None
        kind, _ = _KINDS[type(record)]
        return PolicyRecordView.model_validate(
            {
                "record_id": record_id,
                "recorded_by": "principal:sha256:tests",
                "recorded_at_ms": 1,
                "record": {"kind": kind, "record": record.model_dump(mode="json")},
            }
        )

    async def commit_capture(self, capture: PolicyCapture) -> PolicyRecordReceipt:
        return self._put(capture)

    async def commit_training_run(self, run: TrainingRun) -> PolicyRecordReceipt:
        return self._put(run)

    async def register_model_policy_version(
        self, version: ModelPolicyVersion
    ) -> PolicyRecordReceipt:
        return self._put(version)

    def put_capability(self, *, capture: bool, train: bool) -> str:
        def control(enabled: bool) -> dict[str, Any]:
            return {"enabled": enabled, "scope": "policy-ops" if enabled else None}

        capability = OpenWeightPolicyCapability.model_validate(
            {
                "provider": "vllm",
                "endpoint_ref": "endpoint:gb10",
                "base_checkpoint_digest": _hex(1),
                "tokenizer_digest": _hex(2),
                "decode_params_digest": _hex(3),
                "artifact_destination_ref": "artifacts:policy-new",
                "logprobs": {"chosen_token": True},
                "controls": {"capture": control(capture), "train": control(train)},
                "probe_digest": _hex(4),
                "probed_at_ms": 1,
            }
        )
        return self._put(capability).record_id


class FakeBlobs:
    """Blob CAS: content-addressed bytes."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    async def store(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        self.blobs[digest] = data
        return digest


def _vllm_response(logprobs: bool = True) -> dict[str, Any]:
    content = [
        {"token": "token_id:7", "logprob": -0.25, "top_logprobs": []},
        {"token": "token_id:9", "logprob": -1.5, "top_logprobs": []},
    ]
    choice: dict[str, Any] = {"token_ids": [7, 9], "message": {"content": "ok"}}
    if logprobs:
        choice["logprobs"] = {"content": content}
    return {"prompt_token_ids": [1, 2, 3], "choices": [choice]}


def _sampler(response: dict[str, Any]) -> tuple[VllmLogprobSampler, list[dict]]:
    sent: list[dict[str, Any]] = []

    async def transport(body: dict[str, Any]) -> dict[str, Any]:
        sent.append(body)
        return response

    return VllmLogprobSampler(transport, model="policy-base"), sent


def _spec(capability_id: str, **overrides: Any) -> CaptureSpec:
    fields: dict[str, Any] = {
        "capability_id": capability_id,
        "sampler_version_id": f"polver:{_hex(6)}",
        "trajectory_id": "trajectory:0001",
        "trajectory_steps": 1,
        "completion": "terminal",
        "purpose": "training",
        "captured_at_ms": 5,
        "reward": {
            "value_micros": 1_000_000,
            "verifier_id": "unit-tests",
            "verifier_digest": _hex(9),
            "evidence": "independent_verifier",
        },
    }
    fields.update(overrides)
    return CaptureSpec(**fields)


async def _episode() -> tuple[EpisodeTokens, list[dict[str, Any]]]:
    sampler, sent = _sampler(_vllm_response())
    episode = EpisodeTokens()
    episode.add_turn(
        await sampler.sample([{"role": "user", "content": "hi"}], max_tokens=2)
    )
    episode.add_context([40, 41])  # tool output: masked from the policy loss
    return episode, sent


def test_capture_round_trips_through_blob_cas_and_the_generated_record() -> None:
    engine, blobs = FakePolicyEvolution(), FakeBlobs()
    capability_id = engine.put_capability(capture=True, train=False)
    episode, sent = asyncio.run(_episode())
    assert sent[0]["logprobs"] is True and sent[0]["return_token_ids"] is True
    assert "top_logprobs" not in sent[0]

    receipt = asyncio.run(
        PolicyCaptureRecorder(engine, blobs).commit(_spec(capability_id), episode)
    )
    capture = engine.records[receipt.record_id]
    assert isinstance(capture, PolicyCapture)
    assert (capture.token_count, capture.policy_token_count) == (7, 2)
    ids = blobs.blobs[str(capture.token_ids.digest)]
    assert struct.unpack("<7I", ids) == (1, 2, 3, 7, 9, 40, 41)
    assert blobs.blobs[str(capture.action_mask.digest)] == bytes([0, 0, 0, 1, 1, 0, 0])
    log_q = struct.unpack("<2f", blobs.blobs[str(capture.log_q.digest)])
    assert log_q == pytest.approx((-0.25, -1.5))
    # The committed record survives the wire unchanged.
    wire = capture.model_dump(mode="json")
    assert PolicyCapture.model_validate(wire) == capture


def test_capture_under_a_refused_capability_is_typed_and_side_effect_free() -> None:
    engine, blobs = FakePolicyEvolution(), FakeBlobs()
    episode, _ = asyncio.run(_episode())
    disabled = engine.put_capability(capture=False, train=True)
    before = set(engine.records)
    with pytest.raises(PolicyEvolutionRefused) as refused:
        asyncio.run(
            PolicyCaptureRecorder(engine, blobs).commit(_spec(disabled), episode)
        )
    assert refused.value.code == "POLICY_CAPTURE_DISABLED"
    with pytest.raises(PolicyEvolutionRefused) as missing:
        asyncio.run(
            PolicyCaptureRecorder(engine, blobs).commit(
                _spec(f"polcap:{_hex(5)}"), episode
            )
        )
    assert missing.value.code == "POLICY_CAPABILITY_MISSING"
    assert blobs.blobs == {} and set(engine.records) == before


def test_a_sampler_without_chosen_token_logprobs_fails_closed() -> None:
    sampler, _ = _sampler(_vllm_response(logprobs=False))
    with pytest.raises(PolicyEvolutionRefused) as refused:
        asyncio.run(sampler.sample([{"role": "user", "content": "hi"}], max_tokens=2))
    assert refused.value.code == "POLICY_LOGPROBS_UNSUPPORTED"


def test_top_k_records_are_requested_only_when_configured() -> None:
    response = _vllm_response()
    for entry in response["choices"][0]["logprobs"]["content"]:
        entry["top_logprobs"] = [{"token": "token_id:11", "logprob": -3.0}]

    async def transport(body: dict[str, Any]) -> dict[str, Any]:
        assert body["top_logprobs"] == 1
        return response

    sampler = VllmLogprobSampler(transport, model="policy-base", top_k=1)
    turn = asyncio.run(sampler.sample([], max_tokens=2))
    assert turn.top_k_ids == ((11,), (11,))
    assert sampler.support().top_k == 1


class FakeTrainer:
    """The external gradient substrate: records the job, reports a canned outcome."""

    def __init__(self, outcome: TrainerOutcome) -> None:
        self.outcome = outcome
        self.jobs: list[TrainingJobSpec] = []

    async def run(self, spec: TrainingJobSpec) -> TrainerOutcome:
        self.jobs.append(spec)
        return self.outcome


def _request(capability_id: str) -> PolicyTrainingRequest:
    return PolicyTrainingRequest(
        capability_id=capability_id,
        base_version_id=f"polver:{_hex(6)}",
        capture_ids=(f"polcapture:{_hex(7)}",),
        work_item_id="work:policy-1",
        method={"method": "klpo", "estimator": {"estimator": "sampled_token"}},
        adapter={"adapter": "lora", "rank": 8},
        hyperparameters={"lr": 1e-5, "steps": 10},
        tokenizer_digest=_hex(2),
        checkpoint_digest=_hex(1),
    )


def _succeeded() -> TrainerOutcome:
    return TrainerOutcome(
        status="succeeded",
        trainer_image_digest=_hex(10),
        resources={"wall_ms": 5, "gpu_seconds": 3, "peak_memory_bytes": 7},
        artifact_digest=_hex(12),
        artifact_ref="artifacts:policy-v2",
    )


def test_training_is_refused_before_any_job_when_train_is_off() -> None:
    engine, trainer = FakePolicyEvolution(), FakeTrainer(_succeeded())
    capability_id = engine.put_capability(capture=True, train=False)
    with pytest.raises(PolicyEvolutionRefused) as refused:
        asyncio.run(PolicyTrainingPath(engine, trainer).run(_request(capability_id)))
    assert refused.value.code == "POLICY_TRAIN_DISABLED"
    assert trainer.jobs == []


def test_a_succeeded_run_is_recorded_and_its_adapter_registered() -> None:
    engine, trainer = FakePolicyEvolution(), FakeTrainer(_succeeded())
    substrate = SubstrateTrainer()
    capability_id = engine.put_capability(capture=True, train=True)
    result = asyncio.run(
        PolicyTrainingPath(engine, trainer, substrate).run(_request(capability_id))
    )
    job = trainer.jobs[0]
    assert job.method == "klpo" and job.policy is not None
    assert job.policy.output_destination_ref == "artifacts:policy-new"
    assert substrate.jobs() == [job]
    run = engine.records[result.run.record_id]
    assert isinstance(run, TrainingRun)
    assert run.status.status == "succeeded"
    assert result.version is not None
    version = engine.records[result.version.record_id]
    assert isinstance(version, ModelPolicyVersion)
    assert str(version.adapter_digest) == _hex(12)
    assert isinstance(version.origin, VersionOriginTrained)
    assert version.origin.training_run_id == result.run.record_id


def test_a_failed_run_is_recorded_but_never_becomes_a_version() -> None:
    failed = TrainerOutcome(status="failed", trainer_image_digest=_hex(10))
    engine, trainer = FakePolicyEvolution(), FakeTrainer(failed)
    capability_id = engine.put_capability(capture=True, train=True)
    result = asyncio.run(
        PolicyTrainingPath(engine, trainer).run(_request(capability_id))
    )
    assert result.version is None
    run = engine.records[result.run.record_id]
    assert run.status.status == "failed"
    assert not any(isinstance(r, ModelPolicyVersion) for r in engine.records.values())


def test_policy_jobs_are_digest_bound_and_idempotent() -> None:
    engine, trainer = FakePolicyEvolution(), FakeTrainer(_succeeded())
    substrate = SubstrateTrainer()
    capability_id = engine.put_capability(capture=True, train=True)
    path = PolicyTrainingPath(engine, trainer, substrate)
    first = asyncio.run(path.run(_request(capability_id))).job
    second = asyncio.run(path.run(_request(capability_id))).job
    assert first.job_id == second.job_id and first.job_id.startswith("policy-job-")
    assert first.corpus == [] and first.n_samples == 1
