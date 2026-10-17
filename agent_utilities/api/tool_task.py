"""Reference-only single-operation worker contract, owned by AU-CONTROL-R035.8."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolTaskRefused(RuntimeError):
    """A refusal at a pre-execution validation boundary.

    Once execute is entered, every exception remains uncertain regardless of
    its type; only a bound definitive result can resolve that execution.
    """


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ToolTaskRequest(_ClosedModel):
    """Immutable owner record; the carrier signs its content-addressed reference.

    Parameters are canonical JSON object data, not an executable prompt. The
    authority reference must resolve to the original caller at execution time.
    Neither the record nor the worker process identity confers permission.
    """

    version: Literal[1] = 1
    operation_id: str = Field(min_length=1, max_length=256)
    parameters_json: str = Field(min_length=2, max_length=65_536)
    principal: str = Field(min_length=1, max_length=256)
    tenant: str = Field(min_length=1, max_length=256)
    session_id: str = Field(min_length=1, max_length=256)
    authority_ref: str = Field(min_length=1, max_length=512)
    policy_revision: str = Field(min_length=1, max_length=256)
    registry_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def canonical_parameters(self) -> Self:
        value = json.loads(self.parameters_json)
        if not isinstance(value, dict):
            raise ValueError("tool parameters must be an object")
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if self.parameters_json != canonical:
            raise ValueError("tool parameters must be canonical JSON")
        if len(self.parameters_json.encode("utf-8")) > 65_536:
            raise ValueError("tool parameters exceed byte limit")
        return self

    @property
    def request_ref(self) -> str:
        encoded = json.dumps(
            self.model_dump(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return "tool-task:sha256:" + hashlib.sha256(encoded).hexdigest()


class ToolTaskResult(_ClosedModel):
    """An owner's bound outcome, not a free-form assertion of task success."""

    job_id: str = Field(min_length=1, max_length=256)
    request_ref: str = Field(pattern=r"^tool-task:sha256:[0-9a-f]{64}$")
    principal: str = Field(min_length=1, max_length=256)
    outcome: Literal["completed", "failed", "refused", "indeterminate"]
    receipt_ref: str = Field(min_length=1, max_length=512)


class ToolTaskDispatcher(Protocol):
    """Server-owned resolution and dispatch; never implemented by queue payloads.

    Resolve only the tenant/session-bound immutable record. Execute rechecks the
    original caller, policy, registry and permission through authority_ref before
    effects. job_id is the durable idempotency key: an uncertain previous attempt
    must not repeat an effect. Persist the full immutable ToolTaskResult receipt
    before returning it, and retrieve it by job_id on redelivery. Reconciliation
    may produce a new definitive receipt without repeating an uncertain effect.
    Indeterminate results leave the WorkItem nonterminal and unacknowledged.
    Call require_current immediately before effects;
    fence loss must propagate, never become a successful receipt.
    """

    def resolve(
        self, request_ref: str, *, tenant: str, session_id: str
    ) -> ToolTaskRequest: ...

    def execute(
        self,
        request: ToolTaskRequest,
        *,
        job_id: str,
        require_current: Callable[[], None],
    ) -> ToolTaskResult: ...
