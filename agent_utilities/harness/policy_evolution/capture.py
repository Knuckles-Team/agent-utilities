"""Capture-first open-weight policy evolution: the AU capture half (EH-347).

AU captures; EG records (EH-346). This module extends the provider capability
surface with what an open-weight sampler reports about the tokens it chose,
assembles one episode's token ids / policy mask / frozen chosen-token
log-probabilities, stores those arrays in EG Blob CAS, and commits the
generated ``PolicyCapture`` record. It keeps no local replay database and
never infers trainability from a provider, model or host name: capture runs
only under an attested EG capability whose ``capture`` control is on.

The sampler seam is :class:`LogprobSampler`; :class:`VllmLogprobSampler` is the
OpenAI-compatible vLLM implementation. Its HTTP transport is injected, so
nothing here opens a connection on its own.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from epistemic_graph.generated.policy_evolution import (
    ArrayEncoding,
    HeldBlobRef,
    LogprobSupport,
    OpenWeightPolicyCapability,
    PolicyCapture,
    PolicyRecordReceipt,
    PolicyRecordView,
)
from epistemic_graph.policy_evolution import PolicyEvolutionRefused

__all__ = [
    "BlobStore",
    "CaptureSpec",
    "EpisodeTokens",
    "LogprobSampler",
    "PolicyCaptureRecorder",
    "PolicyRecords",
    "SampledTurn",
    "VllmLogprobSampler",
    "attested_capability",
]

Transport = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
_TOKEN_ID_PREFIX = "token_id:"


class BlobStore(Protocol):
    """EG Blob CAS as the EG client exposes it (``client.blob``)."""

    async def store(self, data: bytes) -> str: ...


class PolicyRecords(Protocol):
    """The generated ``PolicyEvolutionClient`` surface AU consumes."""

    async def get(self, record_id: str) -> PolicyRecordView | None: ...

    async def commit_capture(self, capture: PolicyCapture) -> PolicyRecordReceipt: ...


@dataclass(frozen=True)
class SampledTurn:
    """One sampled completion with its frozen sampler facts.

    ``logprobs[i]`` is the sampler's log-probability of ``token_ids[i]`` at
    sampling time; it is never recomputed later. ``top_k_ids`` holds, per
    sampled token, the alternative token ids the sampler reported (empty when
    Top-K records were not requested).
    """

    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    top_k_ids: tuple[tuple[int, ...], ...] = ()


class LogprobSampler(Protocol):
    """A provider that can report chosen-token log-probabilities."""

    def support(self) -> LogprobSupport: ...

    async def sample(
        self, messages: Sequence[dict[str, Any]], *, max_tokens: int
    ) -> SampledTurn: ...


def _unsupported(detail: str) -> PolicyEvolutionRefused:
    return PolicyEvolutionRefused("POLICY_LOGPROBS_UNSUPPORTED", detail)


def _alternative_ids(entry: dict[str, Any]) -> tuple[int, ...]:
    ids = []
    for alternative in entry.get("top_logprobs") or ():
        token = str(alternative.get("token", ""))
        if not token.startswith(_TOKEN_ID_PREFIX):
            raise _unsupported("top-k records must carry token ids")
        ids.append(int(token.removeprefix(_TOKEN_ID_PREFIX)))
    return tuple(ids)


class VllmLogprobSampler:
    """OpenAI-compatible vLLM sampler that returns token ids and log-probs.

    Requests ``logprobs`` (and ``top_logprobs`` only when ``top_k > 0``) plus
    vLLM's ``return_token_ids`` / ``return_tokens_as_token_ids``. A response
    missing any of them fails closed with ``POLICY_LOGPROBS_UNSUPPORTED``.
    """

    def __init__(self, transport: Transport, *, model: str, top_k: int = 0) -> None:
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        self._transport = transport
        self._model = model
        self._top_k = top_k

    def support(self) -> LogprobSupport:
        return LogprobSupport(chosen_token=True, top_k=self._top_k)

    def request(
        self, messages: Sequence[dict[str, Any]], *, max_tokens: int
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "logprobs": True,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        }
        if self._top_k:
            body["top_logprobs"] = self._top_k
        return body

    async def sample(
        self, messages: Sequence[dict[str, Any]], *, max_tokens: int
    ) -> SampledTurn:
        response = await self._transport(self.request(messages, max_tokens=max_tokens))
        return self.parse(response)

    def parse(self, response: dict[str, Any]) -> SampledTurn:
        choices = response.get("choices") or [{}]
        choice = choices[0]
        token_ids = choice.get("token_ids")
        prompt_ids = response.get("prompt_token_ids")
        content = (choice.get("logprobs") or {}).get("content")
        if token_ids is None or prompt_ids is None or content is None:
            raise _unsupported("response lacks token ids or chosen-token log-probs")
        if len(content) != len(token_ids):
            raise _unsupported("log-probs do not align with sampled tokens")
        top_k_ids = tuple(_alternative_ids(entry) for entry in content)
        return SampledTurn(
            prompt_token_ids=tuple(int(token) for token in prompt_ids),
            token_ids=tuple(int(token) for token in token_ids),
            logprobs=tuple(float(entry["logprob"]) for entry in content),
            top_k_ids=top_k_ids if self._top_k else (),
        )

    async def probe(self) -> tuple[LogprobSupport, str]:
        """Attest the endpoint: one tiny sample must carry every sampler fact.

        Returns the support the endpoint actually demonstrated and the digest of
        the probe evidence, for the capability record graph-os attests.
        """
        turn = await self.sample([{"role": "user", "content": "ok"}], max_tokens=1)
        evidence = json.dumps(
            {
                "model": self._model,
                "top_k": self._top_k,
                "tokens": list(turn.token_ids),
                "logprobs": list(turn.logprobs),
            },
            sort_keys=True,
        )
        return self.support(), hashlib.sha256(evidence.encode()).hexdigest()


@dataclass
class EpisodeTokens:
    """One episode's token stream: context tokens masked, policy tokens kept.

    Prompt, padding and tool-output tokens enter through :meth:`add_context`
    and are masked from the policy loss; sampled tokens enter through
    :meth:`add_turn` together with their frozen log-probabilities.
    """

    token_ids: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)
    log_q: list[float] = field(default_factory=list)
    top_k_ids: list[int] = field(default_factory=list)

    def add_context(self, token_ids: Sequence[int]) -> None:
        self.token_ids.extend(token_ids)
        self.mask.extend(0 for _ in token_ids)

    def add_turn(self, turn: SampledTurn) -> None:
        self.add_context(turn.prompt_token_ids)
        self.token_ids.extend(turn.token_ids)
        self.mask.extend(1 for _ in turn.token_ids)
        self.log_q.extend(turn.logprobs)
        for alternatives in turn.top_k_ids:
            self.top_k_ids.extend(alternatives)

    @property
    def policy_token_count(self) -> int:
        return len(self.log_q)


@dataclass(frozen=True)
class CaptureSpec:
    """Everything a capture binds besides its token arrays."""

    capability_id: str
    sampler_version_id: str
    trajectory_id: str
    trajectory_steps: int
    completion: str
    purpose: str
    captured_at_ms: int
    reward: dict[str, Any] | None = None
    trace_fidelity: str = "full"


async def attested_capability(
    records: PolicyRecords, capability_id: str, control: str
) -> OpenWeightPolicyCapability:
    """The EG-attested capability, refusing unless ``control`` is enabled.

    ``control`` is ``capture``, ``train`` or ``promote``. The refusal is the
    same typed ``POLICY_*`` refusal the engine answers, raised before any
    side effect.
    """
    view = await records.get(capability_id)
    record = getattr(getattr(view, "record", None), "record", None)
    if not isinstance(record, OpenWeightPolicyCapability):
        raise PolicyEvolutionRefused("POLICY_CAPABILITY_MISSING", capability_id)
    if not getattr(record.controls, control).enabled:
        raise PolicyEvolutionRefused(f"POLICY_{control.upper()}_DISABLED")
    return record


class PolicyCaptureRecorder:
    """Store an episode's arrays in Blob CAS and commit its ``PolicyCapture``."""

    def __init__(self, records: PolicyRecords, blobs: BlobStore) -> None:
        self._records = records
        self._blobs = blobs

    async def _held(self, data: bytes, encoding: str, elements: int) -> HeldBlobRef:
        digest = await self._blobs.store(data)
        return HeldBlobRef(
            digest=digest,
            length=len(data),
            encoding=ArrayEncoding(encoding),
            elements=elements,
        )

    async def _arrays(self, episode: EpisodeTokens) -> dict[str, HeldBlobRef]:
        count = len(episode.token_ids)
        policy = episode.policy_token_count
        arrays = {
            "token_ids": await self._held(
                struct.pack(f"<{count}I", *episode.token_ids), "u32_le", count
            ),
            "log_q": await self._held(
                struct.pack(f"<{policy}f", *episode.log_q), "f32_le", policy
            ),
            "action_mask": await self._held(bytes(episode.mask), "u8_mask", count),
        }
        if episode.top_k_ids:
            width = len(episode.top_k_ids)
            arrays["sampler_top_k"] = await self._held(
                struct.pack(f"<{width}I", *episode.top_k_ids), "u32_le", width
            )
        return arrays

    async def commit(
        self, spec: CaptureSpec, episode: EpisodeTokens
    ) -> PolicyRecordReceipt:
        """Refuse unless capture is attested-on; then store and commit.

        Nothing is uploaded when the capability refuses, so a disabled capture
        leaves no blob and no record behind.
        """
        await attested_capability(self._records, spec.capability_id, "capture")
        capture = PolicyCapture.model_validate(
            {
                "capability_id": spec.capability_id,
                "sampler_version_id": spec.sampler_version_id,
                "trajectory_id": spec.trajectory_id,
                "trajectory_steps": spec.trajectory_steps,
                "completion": spec.completion,
                "token_count": len(episode.token_ids),
                "policy_token_count": episode.policy_token_count,
                "reward": spec.reward,
                "purpose": spec.purpose,
                "trace_fidelity": spec.trace_fidelity,
                "captured_at_ms": spec.captured_at_ms,
                **await self._arrays(episode),
            }
        )
        return await self._records.commit_capture(capture)
