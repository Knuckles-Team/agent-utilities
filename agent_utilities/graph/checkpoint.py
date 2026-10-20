"""Serialize in-run progress through a host checkpoint port, without a store."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.usage import RunUsage

from agent_utilities.host_ports import RunCheckpointPort

from .state import GraphState, RunPlan


class CheckpointUnavailable(RuntimeError):
    """Checkpoint content cannot safely authorize continued execution."""


class _Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1] = 1
    run_id: str
    admission_identity: str
    plan_digest: str
    phase: Literal["completed", "step_inflight"]
    next_step: int = Field(ge=0)
    messages: list[Any]
    usage: dict[str, int]
    outputs: list[str]
    status: Literal["running", "completed", "stopped"]
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _plan_digest(plan: RunPlan) -> str:
    if plan.toolsets or plan.harness.plan_scaffold != "off":
        raise CheckpointUnavailable(
            "checkpoint requires native tool grants and no scaffold"
        )
    model_name = (
        plan.model
        if isinstance(plan.model, str)
        else getattr(plan.model, "model_name", None)
    )
    if not isinstance(model_name, str) or not model_name:
        raise CheckpointUnavailable("checkpoint requires a stable model identity")
    descriptor = {
        "model": model_name,
        "instructions": plan.instructions,
        "steps": plan.steps,
        "tool_names": plan.tool_names,
        "budget": asdict(plan.budget),
        "stop": asdict(plan.stop),
        "harness": asdict(plan.harness),
    }
    return hashlib.sha256(_canonical(descriptor).encode()).hexdigest()


class RunCheckpoint:
    """Transient serialization state; all durability and revisions belong to host."""

    def __init__(self, port: RunCheckpointPort, state: GraphState, plan: RunPlan):
        self.port, self.state, self.plan = port, state, plan
        self.digest = _plan_digest(plan)
        self.history: list[Any] = []
        self.elapsed = 0.0
        self.started = time.monotonic()
        if not isinstance(port.admission_identity, str) or not port.admission_identity:
            raise CheckpointUnavailable("checkpoint admission identity is unavailable")

    async def restore(self) -> None:
        raw = await self.port.restore()
        if raw is None:
            return
        if not isinstance(raw, str) or len(raw.encode()) > 1_048_576:
            raise CheckpointUnavailable("checkpoint content is invalid")
        try:
            saved = _Snapshot.model_validate_json(raw)
        except ValidationError as exc:
            raise CheckpointUnavailable("checkpoint content is invalid") from exc
        if (saved.run_id, saved.admission_identity, saved.plan_digest) != (
            self.state.run_id,
            self.port.admission_identity,
            self.digest,
        ):
            raise CheckpointUnavailable("checkpoint run or plan binding differs")
        if saved.phase == "step_inflight":
            raise CheckpointUnavailable("interrupted step has uncertain effects")
        max_steps = min(len(self.plan.steps), max(self.plan.stop.max_steps, 0))
        if saved.next_step > max_steps or len(saved.outputs) != saved.next_step:
            raise CheckpointUnavailable("checkpoint step position is invalid")
        if saved.status == "completed" and saved.next_step != max_steps:
            raise CheckpointUnavailable("checkpoint completion position is invalid")
        if saved.status == "stopped" and not (
            saved.outputs
            and any(
                marker in saved.outputs[-1] for marker in self.plan.stop.stop_markers
            )
        ):
            raise CheckpointUnavailable("checkpoint stop state is invalid")
        if saved.next_step and not saved.messages:
            raise CheckpointUnavailable("checkpoint message history is missing")
        expected = {
            "requests",
            "tool_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        }
        if set(saved.usage) != expected or any(
            type(n) is not int or n < 0 for n in saved.usage.values()
        ):
            raise CheckpointUnavailable("checkpoint usage is invalid")
        if (
            saved.usage["total_tokens"]
            != saved.usage["input_tokens"] + saved.usage["output_tokens"]
        ):
            raise CheckpointUnavailable("checkpoint token accounting differs")
        try:
            self.history = ModelMessagesTypeAdapter.validate_python(saved.messages)
        except ValidationError as exc:
            raise CheckpointUnavailable(
                "checkpoint message history is invalid"
            ) from exc
        self.state.step, self.state.outputs = saved.next_step, saved.outputs
        self.state.usage, self.state.status = saved.usage, saved.status
        self.elapsed = saved.elapsed_seconds

    def usage(self) -> RunUsage:
        return RunUsage(
            requests=self.state.usage.get("requests", 0),
            tool_calls=self.state.usage.get("tool_calls", 0),
            input_tokens=self.state.usage.get("input_tokens", 0),
            output_tokens=self.state.usage.get("output_tokens", 0),
        )

    def remaining_timeout(self) -> float:
        return max(
            0.0,
            self.plan.budget.timeout_seconds
            - self.elapsed
            - (time.monotonic() - self.started),
        )

    async def save(self, phase: Literal["completed", "step_inflight"]) -> None:
        usage = {
            key: self.state.usage.get(key, 0)
            for key in (
                "requests",
                "tool_calls",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            )
        }
        status = self.state.status
        if status not in ("running", "completed", "stopped"):
            raise CheckpointUnavailable("checkpoint run state is invalid")
        snapshot = _Snapshot(
            run_id=self.state.run_id,
            admission_identity=self.port.admission_identity,
            plan_digest=self.digest,
            phase=phase,
            next_step=len(self.state.outputs),
            messages=ModelMessagesTypeAdapter.dump_python(self.history, mode="json"),
            usage=usage,
            outputs=self.state.outputs,
            status=status,
            elapsed_seconds=self.elapsed + time.monotonic() - self.started,
        )
        content = _canonical(snapshot.model_dump(mode="json"))
        if len(content.encode()) > 1_048_576:
            raise CheckpointUnavailable("checkpoint exceeds native content limit")
        await self.port.save(content)
