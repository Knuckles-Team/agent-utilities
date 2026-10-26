"""Bounded checkpoint codec for host-owned run transactions.

The harness owns no checkpoint store.  This module validates the host payload,
keeps transient restore state, and asks the bound host port to publish a
canonical snapshot before a model or tool effect and before its completion is
made visible to the run event stream.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.usage import RunUsage

from agent_utilities.host_ports import RunCheckpointPort

from .state import DependencyPlan, GraphState, RunPlan

MAX_CHECKPOINT_BYTES = 1_048_576


class CheckpointUnavailable(RuntimeError):
    """Checkpoint content cannot safely authorize continued execution."""


class _Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    run_id: str
    admission_identity: str
    plan_digest: str
    phase: Literal["completed", "step_inflight"]
    inflight_tasks: list[str]
    next_step: int = Field(ge=0)
    messages: list[Any]
    task_messages: dict[str, list[Any]]
    completed_tasks: list[str]
    dependency_outputs: dict[str, Any]
    usage: dict[str, int]
    outputs: list[str]
    status: Literal["running", "completed", "stopped"]
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointUnavailable(
            "checkpoint content is not JSON-serializable"
        ) from exc


def _model_name(model: Any) -> str:
    value = model if isinstance(model, str) else getattr(model, "model_name", None)
    if not isinstance(value, str) or not value.strip():
        raise CheckpointUnavailable("checkpoint requires a stable model identity")
    return value


def _tool_descriptor(plan: RunPlan) -> list[dict[str, Any]]:
    graph = plan.dependency_plan
    if graph is None:
        return []
    if not isinstance(graph, DependencyPlan):
        raise CheckpointUnavailable("checkpoint dependency plan is invalid")
    tasks: list[dict[str, Any]] = []
    for task in graph.tasks:
        call = None
        if task.call is not None:
            call = {
                "server": task.call.server,
                "tool": task.call.tool,
                "arguments": task.call.arguments,
            }
        tasks.append(
            {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "depends_on": list(task.depends_on),
                "call": call,
            }
        )
    return tasks


def _attachment_descriptor(attachments: list[Any]) -> list[dict[str, Any]]:
    descriptor: list[dict[str, Any]] = []
    for part in attachments:
        if isinstance(part, str):
            raw = part.encode("utf-8")
            media_type = "text/plain"
        elif isinstance(part, BinaryContent) and isinstance(part.data, bytes):
            raw = part.data
            media_type = part.media_type
        else:
            raise CheckpointUnavailable(
                "checkpoint attachment representation is invalid"
            )
        descriptor.append(
            {
                "media_type": media_type,
                "bytes": len(raw),
                "digest": hashlib.sha256(raw).hexdigest(),
            }
        )
    return descriptor


def _plan_digest(plan: RunPlan, attachments: list[Any] | None = None) -> str:
    """Digest the immutable received execution binding, never host secrets."""
    if (
        plan.toolsets
        or plan.harness.plan_scaffold != "off"
        or plan.harness.client_tools
        or plan.client_transport is not None
    ):
        raise CheckpointUnavailable(
            "checkpoint requires native tool grants and no client or scaffold state"
        )
    descriptor = {
        "model": _model_name(plan.model),
        "execution_constraint": plan.execution_constraint,
        "instructions": plan.instructions,
        "steps": list(plan.steps),
        "tool_names": list(plan.tool_names),
        "dependency_plan": {
            "plan_id": plan.dependency_plan.plan_id,
            "output_task_id": plan.dependency_plan.output_task_id,
            "max_concurrency": plan.dependency_plan.max_concurrency,
            "tasks": _tool_descriptor(plan),
        }
        if plan.dependency_plan is not None
        else None,
        "budget": asdict(plan.budget),
        "stop": asdict(plan.stop),
        "harness": {
            "retries": plan.harness.retries,
            "response_format": plan.harness.response_format,
            "model_settings": dict(plan.harness.model_settings),
            "plan_scaffold": plan.harness.plan_scaffold,
        },
        "input_media_types": list(plan.input_media_types),
        "attachments": _attachment_descriptor(attachments or []),
    }
    return hashlib.sha256(_canonical(descriptor).encode("utf-8")).hexdigest()


def _usage_values(usage: dict[str, int]) -> dict[str, int]:
    expected = {
        "requests",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
    if set(usage) != expected or any(
        type(value) is not int or value < 0 for value in usage.values()
    ):
        raise CheckpointUnavailable("checkpoint usage is invalid")
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise CheckpointUnavailable("checkpoint token accounting differs")
    return dict(usage)


def _messages(value: Any) -> list[Any]:
    try:
        return ModelMessagesTypeAdapter.validate_python(value)
    except ValidationError as exc:
        raise CheckpointUnavailable("checkpoint message history is invalid") from exc


class RunCheckpoint:
    """Transient codec state; durability and fencing stay with the host port."""

    def __init__(
        self,
        port: RunCheckpointPort,
        state: GraphState,
        plan: RunPlan,
        attachments: list[Any] | None = None,
    ) -> None:
        self.port, self.state, self.plan = port, state, plan
        self.digest = _plan_digest(plan, attachments)
        self.history: list[Any] = []
        self.task_histories: dict[str, list[Any]] = {}
        self.completed_tasks: set[str] = set()
        self.dependency_outputs: dict[str, Any] = {}
        self.inflight_tasks: set[str] = set()
        self.elapsed = 0.0
        self.started = time.monotonic()
        self.restored = False
        identity = getattr(port, "admission_identity", None)
        if not isinstance(identity, str) or not identity.strip():
            raise CheckpointUnavailable("checkpoint admission identity is unavailable")

    @property
    def admission_identity(self) -> str:
        return self.port.admission_identity

    async def restore(self) -> None:
        raw = await self.port.restore()
        if raw is None:
            return
        if (
            not isinstance(raw, str)
            or not raw
            or len(raw.encode("utf-8")) > MAX_CHECKPOINT_BYTES
        ):
            raise CheckpointUnavailable("checkpoint content is invalid")
        try:
            saved = _Snapshot.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            raise CheckpointUnavailable("checkpoint content is invalid") from exc
        if (saved.run_id, saved.admission_identity, saved.plan_digest) != (
            self.state.run_id,
            self.admission_identity,
            self.digest,
        ):
            raise CheckpointUnavailable("checkpoint run or plan binding differs")
        if saved.phase == "step_inflight" or saved.inflight_tasks:
            raise CheckpointUnavailable("interrupted step has uncertain effects")
        _usage_values(saved.usage)
        # Validate structured child outputs with the same finite JSON codec
        # used on publication; an opaque host object cannot authorize replay.
        try:
            _canonical(saved.dependency_outputs)
        except CheckpointUnavailable:
            raise CheckpointUnavailable(
                "checkpoint dependency outputs are invalid"
            ) from None
        self._validate_shape(saved)
        self.history = _messages(saved.messages)
        self.task_histories = {
            key: _messages(value) for key, value in saved.task_messages.items()
        }
        self.completed_tasks = set(saved.completed_tasks)
        self.dependency_outputs = dict(saved.dependency_outputs)
        self.inflight_tasks = set(saved.inflight_tasks)
        self.state.step = saved.next_step
        self.state.outputs = list(saved.outputs)
        self.state.usage = dict(saved.usage)
        status = saved.status
        if status == "running":
            graph = self.plan.dependency_plan
            if graph is None:
                status = (
                    "completed"
                    if saved.next_step
                    == min(len(self.plan.steps), max(self.plan.stop.max_steps, 0))
                    else status
                )
            elif graph.output_task_id in self.completed_tasks:
                status = "completed"
        self.state.status = status
        self.elapsed = saved.elapsed_seconds
        self.started = time.monotonic()
        self.restored = True

    def _validate_shape(self, saved: _Snapshot) -> None:
        graph = self.plan.dependency_plan
        if graph is None:
            max_steps = min(len(self.plan.steps), max(self.plan.stop.max_steps, 0))
            if saved.completed_tasks or saved.dependency_outputs or saved.task_messages:
                raise CheckpointUnavailable("checkpoint dependency state is invalid")
            if saved.next_step > max_steps or len(saved.outputs) != saved.next_step:
                raise CheckpointUnavailable("checkpoint step position is invalid")
            if saved.status == "completed" and saved.next_step != max_steps:
                raise CheckpointUnavailable("checkpoint completion position is invalid")
            if saved.status == "stopped" and not (
                saved.outputs
                and any(
                    marker in saved.outputs[-1]
                    for marker in self.plan.stop.stop_markers
                )
            ):
                raise CheckpointUnavailable("checkpoint stop state is invalid")
            if saved.next_step and not saved.messages:
                raise CheckpointUnavailable("checkpoint message history is missing")
            return
        nodes = {task.task_id: task for task in graph.tasks}
        completed = set(saved.completed_tasks)
        if (
            len(completed) != len(saved.completed_tasks)
            or not completed <= nodes.keys()
        ):
            raise CheckpointUnavailable("checkpoint dependency task set is invalid")
        if set(saved.dependency_outputs) != completed:
            raise CheckpointUnavailable("checkpoint dependency outputs are incomplete")
        if set(saved.task_messages) - completed:
            raise CheckpointUnavailable("checkpoint dependency history is invalid")
        for key in completed:
            if not set(nodes[key].depends_on) <= completed:
                raise CheckpointUnavailable("checkpoint dependency order is invalid")
        if saved.next_step < len(completed) or saved.next_step > len(nodes):
            raise CheckpointUnavailable(
                "checkpoint dependency step position is invalid"
            )
        if saved.status == "completed" and graph.output_task_id not in completed:
            raise CheckpointUnavailable("checkpoint dependency completion is invalid")
        if saved.status == "stopped" and graph.output_task_id not in completed:
            raise CheckpointUnavailable("checkpoint dependency stop state is invalid")

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

    def mark_inflight(self, task_id: str) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise CheckpointUnavailable("checkpoint task identity is invalid")
        self.inflight_tasks.add(task_id)

    def complete_serial(self, history: list[Any]) -> None:
        """Update the serial history and clear its completed effect marker."""
        self.history = list(history)
        self.inflight_tasks.clear()

    def record_completed(
        self,
        task_id: str,
        output: Any,
        history: list[Any] | None = None,
    ) -> None:
        self.inflight_tasks.discard(task_id)
        self.completed_tasks.add(task_id)
        self.dependency_outputs[task_id] = json.loads(_canonical(output))
        if history is not None:
            self.task_histories[task_id] = list(history)

    async def save(
        self,
        phase: Literal["completed", "step_inflight"],
        *,
        task_id: str | None = None,
        output: Any = None,
        history: list[Any] | None = None,
    ) -> None:
        """Publish one canonical before-effect or completed-step snapshot."""
        if task_id is not None and phase == "step_inflight":
            self.mark_inflight(task_id)
        elif task_id is not None and phase == "completed":
            self.record_completed(task_id, output, history)
        status = self.state.status
        if status not in ("running", "completed", "stopped"):
            raise CheckpointUnavailable("checkpoint run state is invalid")
        if status == "running":
            graph = self.plan.dependency_plan
            if graph is None:
                max_steps = min(len(self.plan.steps), max(self.plan.stop.max_steps, 0))
                if len(self.state.outputs) == max_steps:
                    status = "completed"
            elif graph.output_task_id in self.completed_tasks:
                status = "completed"
        if self.inflight_tasks and phase != "step_inflight":
            # A completed task can be published while another parallel task is
            # active.  Preserve the uncertainty marker in that snapshot.
            phase = "step_inflight"
        usage = _usage_values(
            {
                key: self.state.usage.get(key, 0)
                for key in (
                    "requests",
                    "tool_calls",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                )
            }
        )
        serial_next = len(self.state.outputs)
        next_step = (
            serial_next if self.plan.dependency_plan is None else self.state.step
        )
        snapshot = _Snapshot(
            run_id=self.state.run_id,
            admission_identity=self.admission_identity,
            plan_digest=self.digest,
            phase=phase,
            inflight_tasks=sorted(self.inflight_tasks),
            next_step=next_step,
            messages=ModelMessagesTypeAdapter.dump_python(self.history, mode="json"),
            task_messages={
                key: ModelMessagesTypeAdapter.dump_python(value, mode="json")
                for key, value in self.task_histories.items()
            },
            completed_tasks=sorted(self.completed_tasks),
            dependency_outputs=self.dependency_outputs,
            usage=usage,
            outputs=list(self.state.outputs),
            status=status,
            elapsed_seconds=self.elapsed + time.monotonic() - self.started,
        )
        content = _canonical(snapshot.model_dump(mode="json"))
        if len(content.encode("utf-8")) > MAX_CHECKPOINT_BYTES:
            raise CheckpointUnavailable("checkpoint exceeds native content limit")
        await self.port.save(content)
