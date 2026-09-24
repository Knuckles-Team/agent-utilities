"""Wire builders for EG's ``DecisionLog.retrieval`` ops (EH-394..EH-397).

Pure functions: every learned signal is computed and held by EG, over its own
decision log; AU only attests what its runs returned and cited, asks for what
EG learned, and moves EG's governed pointers with EG's receipts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

#: The widest query vector EG accepts in an outcome.
MAX_QUERY_DIMENSIONS = 4096
#: ``Q16``: ``x`` travels as ``round(x * 2^16)``.
Q16_ONE = 1 << 16
#: Every record window: from the epoch to the end of time.
ALL_TIME = {"from_ms": 0, "to_ms": 2**63 - 1}


def retrieval_op(tenant: str, action: str, **fields: Any) -> dict[str, Any]:
    """One ``DecisionLog.retrieval`` op."""
    return {
        "op": "retrieval",
        "tenant_id": tenant,
        "retrieval": {"action": action, **fields},
    }


def q16(vector: Sequence[float]) -> list[int]:
    """A vector on ``Q16`` (bounded to what EG accepts)."""
    if len(vector) > MAX_QUERY_DIMENSIONS:
        raise ValueError(f"a query vector is at most {MAX_QUERY_DIMENSIONS} wide")
    return [round(float(x) * Q16_ONE) for x in vector]


def space_identity(model_name: str, dimensions: int) -> str:
    """AU's identity of an embedding space: the model that produced it and its
    width. A fine-tuned or upgraded model is a new name, hence a new space."""
    body = json.dumps(
        {"dimensions": int(dimensions), "model": model_name}, sort_keys=True
    )
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def outcome_op(
    tenant: str,
    record_id: str,
    returned: Iterable[tuple[str, str | None]],
    cited: Iterable[str],
    *,
    query: Mapping[str, Any] | None = None,
    path: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attest what a committed retrieval-plan run returned (rank order, with
    each unit's content class) and which returned units the answer cited."""
    returned_ids = [(str(i), c) for i, c in returned]
    kept = {i for i, _ in returned_ids}
    outcome = {
        "record_id": record_id,
        "returned": [{"evidence_id": i, "content_class": c} for i, c in returned_ids],
        "cited": [c for c in dict.fromkeys(str(c) for c in cited) if c in kept],
        "query": None if query is None else dict(query),
        "path": None if path is None else dict(path),
    }
    return retrieval_op(tenant, "record_outcome", outcome=outcome)


def paths_op(tenant: str, task_class: str, composed_digest: str) -> dict[str, Any]:
    request = {
        "task_class": task_class,
        "composed_digest": composed_digest,
        "policy_version": None,
        "window": dict(ALL_TIME),
    }
    return retrieval_op(tenant, "paths", request=request)


def usage_op(tenant: str, window: Mapping[str, int] | None = None) -> dict[str, Any]:
    return retrieval_op(tenant, "usage", window=dict(window or ALL_TIME))


def result_of(answer: Any, kind: str) -> Mapping[str, Any]:
    """The body of a ``RetrievalResult`` of ``kind``; ``ValueError`` otherwise."""
    body = getattr(answer, "payload", answer)
    if not isinstance(body, Mapping) or body.get("result") != kind:
        raise ValueError(f"EG answered {body!r}, not a {kind} result")
    return body


__all__ = [
    "ALL_TIME",
    "MAX_QUERY_DIMENSIONS",
    "Q16_ONE",
    "outcome_op",
    "paths_op",
    "q16",
    "result_of",
    "retrieval_op",
    "space_identity",
    "usage_op",
]
