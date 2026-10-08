"""Wire builders for EG's retrieval learning (AU-CONTEXT-R001, AU-CONTEXT-R002).

EG holds everything learned, over its own decision log. AU WRITES only through
the one ``DecisionLog.learn`` op -- attesting a run's outcome, asking EG to fit
or evaluate, moving a governed pointer with EG's receipt -- and READS what was
learned through the decision views' reserved SQL relations
(``decision_retrieval_outcomes``, ``decision_hard_negatives``,
``decision_class_usage``, ``decision_proven_paths``, ``decision_pointers``).
Pure functions: no I/O here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import msgpack

#: The widest query vector EG accepts in an outcome.
MAX_QUERY_DIMENSIONS = 4096
#: ``Q16``: ``x`` travels as ``round(x * 2^16)``.
Q16_ONE = 1 << 16
#: Every record window: from the epoch to the end of time.
ALL_TIME = {"from_ms": 0, "to_ms": 2**63 - 1}


def learn_op(tenant: str, write: str, **fields: Any) -> dict[str, Any]:
    """The one ``DecisionLog.learn`` write."""
    return {"op": "learn", "tenant_id": tenant, "write": {"write": write, **fields}}


def adapter_pointer(graph: str) -> dict[str, Any]:
    return {"pointer": "adapter", "graph": graph}


def generation_pointer(logical: str) -> dict[str, Any]:
    return {"pointer": "generation", "logical": logical}


def activate(target: str, receipt_digest: str) -> dict[str, Any]:
    return {"movement": "activate", "target": target, "receipt_digest": receipt_digest}


ROLLBACK: dict[str, Any] = {"movement": "rollback"}


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
    return learn_op(tenant, "record_outcome", outcome=outcome)


def literal(value: str) -> str:
    """A SQL string literal (quotes doubled)."""
    return "'" + str(value).replace("'", "''") + "'"


def recorded(answer: Any, kind: str) -> Mapping[str, Any]:
    """The body of a ``LearningRecorded`` of ``kind``; ``ValueError`` otherwise."""
    body = getattr(answer, "payload", answer)
    if not isinstance(body, Mapping) or body.get("recorded") != kind:
        raise ValueError(f"EG answered {body!r}, not a recorded {kind}")
    return body


def rows_of(answer: Any) -> list[dict[str, Any]]:
    """An engine SQL answer (``{"columns", "rows"}``) as row dicts; a row may
    arrive as its own msgpack blob."""
    body = getattr(answer, "payload", answer) or {}
    columns = [str(c) for c in body.get("columns") or []]
    out = []
    for row in body.get("rows") or []:
        cells = (
            msgpack.unpackb(row, raw=False)
            if isinstance(row, bytes | bytearray)
            else row
        )
        out.append(dict(zip(columns, cells, strict=False)))
    return out


__all__ = [
    "ALL_TIME",
    "MAX_QUERY_DIMENSIONS",
    "Q16_ONE",
    "ROLLBACK",
    "activate",
    "adapter_pointer",
    "generation_pointer",
    "learn_op",
    "literal",
    "outcome_op",
    "q16",
    "recorded",
    "rows_of",
    "space_identity",
]
