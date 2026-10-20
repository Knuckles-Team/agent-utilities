"""Observe actual in-run model requests without choosing or storing anything.

Client elapsed time includes transport and provider waiting; it is not provider
queue time, TTFT, or generation time. None of those are inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

MAX_REQUEST_SAMPLES = 128
MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (perf_counter_ns() - started_ns) // 1_000_000)


@dataclass
class RunMetrics:
    """Bounded observations of this execution, never a cross-run accumulator."""

    started_ns: int = field(default_factory=perf_counter_ns)
    requests: list[dict[str, Any]] = field(default_factory=list)
    observed_requests: int = 0

    def record(
        self, started_ns: int, response: ModelResponse | None, *, model: WrapperModel
    ) -> None:
        self.observed_requests += 1
        if len(self.requests) >= MAX_REQUEST_SAMPLES:
            return
        sample: dict[str, Any] = {
            "sequence": self.observed_requests,
            "source": "client_monotonic_clock",
            "client_request_elapsed_ms": _elapsed_ms(started_ns),
            "outcome": "completed" if response is not None else "failed",
            "model_name": (response.model_name if response else None)
            or model.model_name,
            "provider_name": (response.provider_name if response else None)
            or model.system,
        }
        if response is not None:
            if response.provider_response_id:
                sample["provider_response_id"] = response.provider_response_id
            for name in ("input_tokens", "output_tokens"):
                value = getattr(response.usage, name, None)
                if type(value) is int and 0 <= value <= MAX_SAFE_INTEGER:
                    sample[name] = value
        self.requests.append(sample)

    def snapshot(self) -> dict[str, Any]:
        return {
            "source": "client_monotonic_clock",
            "run_elapsed_ms": _elapsed_ms(self.started_ns),
            "observed_model_requests": self.observed_requests,
            "coverage": (
                "complete"
                if len(self.requests) == self.observed_requests
                else "partial"
            ),
            "model_requests": [dict(sample) for sample in self.requests],
        }


class MeasuredModel(WrapperModel):
    """Measure the existing nonstreaming model call, preserving its semantics."""

    def __init__(self, wrapped: Any, metrics: RunMetrics):
        super().__init__(wrapped)
        self.metrics = metrics

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        started = perf_counter_ns()
        response: ModelResponse | None = None
        try:
            response = await self.wrapped.request(
                messages, model_settings, model_request_parameters
            )
            return response
        finally:
            self.metrics.record(started, response, model=self)
