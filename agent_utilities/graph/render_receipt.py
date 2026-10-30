"""Run-loop consumer of the engine prompt render receipt.

The engine renders a prompt and returns a render receipt
(``MEDULLA-TYPED-PACKS-R002.9`` and ``R002.10``).  The receipt binds the
prompt, schema, variables, bytes, revisions and request scope.  This module
is the named consumer: the run loop calls :func:`check_render_receipt` before
the first model transport call and the call is refused, with a typed code and
zero model requests, unless the receipt is valid.

The harness holds no store.  Current revisions are read once per check from
the engine client port the run plan carries (``RenderReceiptCheck.revisions``);
there is no second lookup path and no cache.  The receipt fields below follow
the engine row text; the engine publishes no generated client type for the
receipt yet.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

RefusalCode = Literal[
    "receipt_invalid",
    "receipt_stale",
    "receipt_substituted",
    "receipt_wrong_scope",
    "consumer_undefined",
    "revisions_unavailable",
]

DISPATCH_RULES = frozenset({"one_request_per_valid_receipt"})
"""Every dispatch rule the consumer may apply; an unknown rule is rejected."""

CALL_SITE = "agent_utilities.graph.executor.iterate_run"
"""The run-loop function that calls the consumer before model transport."""

_DIGEST = re.compile(r"[0-9a-f]{64}")


class RenderReceiptRefused(Exception):
    """The receipt did not pass; no model request was made."""

    def __init__(self, code: RefusalCode) -> None:
        super().__init__(f"render_receipt_refused: {code}")
        self.code: RefusalCode = code


@dataclass(frozen=True, slots=True)
class RenderReceipt:
    """What the engine bound when it rendered one prompt."""

    prompt_ref: str
    prompt_revision: str
    schema_revision: str
    variables_digest: str
    bytes_digest: str
    scope: str

    def as_dict(self) -> dict[str, str]:
        return {
            "prompt_ref": self.prompt_ref,
            "prompt_revision": self.prompt_revision,
            "schema_revision": self.schema_revision,
            "variables_digest": self.variables_digest,
            "bytes_digest": self.bytes_digest,
            "scope": self.scope,
        }


@dataclass(frozen=True, slots=True)
class CurrentRevisions:
    """The engine's current prompt and schema revisions for one prompt."""

    prompt_revision: str
    schema_revision: str


RevisionReader = Callable[[str], Awaitable[CurrentRevisions]]
"""``(prompt_ref) -> CurrentRevisions`` through the plan's engine client port."""


def rendered_bytes_digest(text: str) -> str:
    """Digest of the exact prompt bytes the run sends to the model."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RenderReceiptConsumer:
    """The consumer definition: its call site and the dispatch rule it applies."""

    call_site: str = CALL_SITE
    dispatch_rule: str = "one_request_per_valid_receipt"

    def __post_init__(self) -> None:
        if self.dispatch_rule not in DISPATCH_RULES:
            raise ValueError(f"unknown dispatch rule: {self.dispatch_rule!r}")
        if type(self.call_site) is not str or not self.call_site.strip():
            raise ValueError("call_site must be a nonempty string")

    def as_dict(self) -> dict[str, str]:
        return {"call_site": self.call_site, "dispatch_rule": self.dispatch_rule}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RenderReceiptConsumer:
        return cls(call_site=value["call_site"], dispatch_rule=value["dispatch_rule"])

    def location_evidence(self, source_snapshot: str) -> dict[str, str]:
        """Name the consumer and its call site at the run's source snapshot."""
        if type(source_snapshot) is not str or not source_snapshot.strip():
            raise ValueError("source_snapshot must be a nonempty string")
        return {**self.as_dict(), "source_snapshot": source_snapshot}


@dataclass(frozen=True, slots=True)
class RenderReceiptCheck:
    """Host-supplied receipt check carried on the run plan.

    ``expected_scope`` is the host's authenticated request scope; ``revisions``
    is the engine client port.  ``consumer`` is None only when the host failed
    to define one, which refuses rather than passing partially.
    """

    receipt: RenderReceipt
    expected_scope: str
    revisions: RevisionReader = field(repr=False, compare=False)
    source_snapshot: str
    consumer: RenderReceiptConsumer | None = field(
        default_factory=RenderReceiptConsumer
    )


def _well_formed(receipt: Any) -> bool:
    if not isinstance(receipt, RenderReceipt):
        return False
    texts = (
        receipt.prompt_ref,
        receipt.prompt_revision,
        receipt.schema_revision,
        receipt.scope,
    )
    return all(type(v) is str and v.strip() for v in texts) and all(
        type(v) is str and _DIGEST.fullmatch(v)
        for v in (receipt.variables_digest, receipt.bytes_digest)
    )


async def check_render_receipt(
    check: RenderReceiptCheck, rendered_prompt: str
) -> dict[str, str]:
    """Validate the receipt before any model transport call.

    Returns the location evidence for the run on success; raises
    :class:`RenderReceiptRefused` with a typed code otherwise.
    """
    if check.consumer is None:
        raise RenderReceiptRefused("consumer_undefined")
    receipt = check.receipt
    if not _well_formed(receipt):
        raise RenderReceiptRefused("receipt_invalid")
    if receipt.scope != check.expected_scope:
        raise RenderReceiptRefused("receipt_wrong_scope")
    if receipt.bytes_digest != rendered_bytes_digest(rendered_prompt):
        raise RenderReceiptRefused("receipt_substituted")
    try:
        current = await check.revisions(receipt.prompt_ref)
    except Exception:  # noqa: BLE001 - fail closed on any port failure
        raise RenderReceiptRefused("revisions_unavailable") from None
    if (
        current.prompt_revision != receipt.prompt_revision
        or current.schema_revision != receipt.schema_revision
    ):
        raise RenderReceiptRefused("receipt_stale")
    return check.consumer.location_evidence(check.source_snapshot)


__all__ = [
    "CALL_SITE",
    "DISPATCH_RULES",
    "CurrentRevisions",
    "RefusalCode",
    "RenderReceipt",
    "RenderReceiptCheck",
    "RenderReceiptConsumer",
    "RenderReceiptRefused",
    "check_render_receipt",
    "rendered_bytes_digest",
]
