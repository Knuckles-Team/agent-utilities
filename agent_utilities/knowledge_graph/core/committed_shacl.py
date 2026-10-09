"""One way to reach EG's committed-GraphSchema SHACL validator.

epistemic-graph owns the composed GraphSchema and is the only SHACL
interpreter (43197d7c6). AU renders the data graph and asks EG to validate it
against the committed schema. AU never reads, parses or sends a shapes
document on this path. The report carries the digest of the exact schema
snapshot that was used (``GraphComputeEngine._require_committed_shacl_receipt``).

Callers hold different handles: a ``GraphComputeEngine``, an
``IntelligenceGraphEngine`` that wraps one as ``graph_compute``, or a view that
exposes it as ``graph``. :func:`committed_shacl_authority` resolves all three,
so no caller probes for the method on the wrong object and silently fails
closed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CommittedShaclUnavailable",
    "committed_shacl_authority",
    "shacl_violation_summary",
    "validate_committed",
]

_AUTHORITY_METHOD = "shacl_validate_committed"
_WRAPPED_COMPUTE_ATTRS = ("graph_compute", "graph", "compute")
_DETAIL_FIELDS = ("focus_node", "path", "source_shape", "message")


class CommittedShaclUnavailable(RuntimeError):
    """No committed-GraphSchema SHACL validator is reachable from this handle."""


def committed_shacl_authority(engine: Any) -> Any | None:
    """Return the object that exposes ``shacl_validate_committed``, or ``None``."""
    if engine is None:
        return None
    if callable(getattr(engine, _AUTHORITY_METHOD, None)):
        return engine
    for attr in _WRAPPED_COMPUTE_ATTRS:
        inner = getattr(engine, attr, None)
        if inner is not None and callable(getattr(inner, _AUTHORITY_METHOD, None)):
            return inner
    return None


def validate_committed(engine: Any, data_graph: str) -> Any:
    """Validate Turtle ``data_graph`` against EG's committed composed GraphSchema.

    Raises :class:`CommittedShaclUnavailable` when ``engine`` reaches no
    committed validator. Errors raised by the validator itself propagate.
    """
    authority = committed_shacl_authority(engine)
    if authority is None:
        raise CommittedShaclUnavailable(
            "committed EG GraphSchema SHACL validation is unavailable"
        )
    return authority.shacl_validate_committed(data_graph)


def _result_detail(result: Any) -> str:
    return " ".join(
        str(value)
        for value in (getattr(result, name, None) for name in _DETAIL_FIELDS)
        if value
    )


def shacl_violation_summary(report: Any, *, limit: int = 5) -> str:
    """Summarize a non-conforming typed SHACL report as a short, bounded string."""
    results = list(getattr(report, "results", None) or [])
    if not results:
        return "no violation detail reported"
    seen: list[str] = []
    for result in results:
        detail = _result_detail(result)
        if detail and detail not in seen:
            seen.append(detail)
        if len(seen) >= limit:
            break
    extra = len(results) - len(seen)
    return "; ".join(seen) + (f" (+{extra} more)" if extra > 0 else "")
