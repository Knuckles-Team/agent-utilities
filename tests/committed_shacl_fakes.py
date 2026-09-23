"""Test double for EG's committed-GraphSchema SHACL validator (EH-385).

Connector admission, promotion and governance checks call
``shacl_validate_committed(data_graph)`` on whatever handle
``committed_shacl.committed_shacl_authority`` resolves. They never send a
shapes document. Assign an instance as that attribute on a compute or client
double:

    compute.shacl_validate_committed = CommittedShaclValidator()
"""

from __future__ import annotations

from typing import Any

from epistemic_graph.generated.rdf_report import (
    ShaclSeverity,
    ShaclValidationReport,
    ShaclValidationResult,
)

_COMPOSED_DIGEST = "sha256:" + "0" * 64


def shacl_result(**fields: Any) -> ShaclValidationResult:
    """One generated ``ShaclValidationResult``; unset required fields get defaults."""
    defaults: dict[str, Any] = {
        "constraint_component": "sh:MinCountConstraintComponent",
        "focus_node": "node/test",
        "severity": ShaclSeverity.VIOLATION,
        "source_shape": "shape/test",
    }
    return ShaclValidationResult(**{**defaults, **fields})


def shacl_report(
    *, conforms: bool = True, results: tuple[ShaclValidationResult, ...] = ()
) -> ShaclValidationReport:
    """A generated ``ShaclValidationReport`` bound to one committed schema."""
    return ShaclValidationReport(
        conforms=conforms,
        results=list(results),
        composed_digest=_COMPOSED_DIGEST,
        schema_digests=[_COMPOSED_DIGEST],
    )


class CommittedShaclValidator:
    """Callable ``shacl_validate_committed`` that records each data graph.

    Returns ``reports`` in order and then repeats the last one. With no
    reports it always returns a conforming report.
    """

    def __init__(self, *reports: Any) -> None:
        self.reports: list[Any] = list(reports) or [shacl_report()]
        self.validations: list[str] = []

    def __call__(self, data_graph: str) -> Any:
        self.validations.append(data_graph)
        if len(self.reports) > 1:
            return self.reports.pop(0)
        return self.reports[0]
