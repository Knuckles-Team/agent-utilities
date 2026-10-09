"""AU-SEMANTIC-R021.3: typed declaration model for connector-certification
declared semantic coverage.

Split from AU-SEMANTIC-R021 ("AU's hand-written RDF/OWL/SHACL emitters move
behind EG pack compilation"): this slice ships the typed coverage model and
the refusal for incomplete declared coverage, parsed once instead of by ad
hoc regex sniffing inline. Removing the declared-semantic-validation step
from ``connector_certification.py`` entirely and routing manifest-to-ontology
compilation through EG pack compilation land in a later slice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TARGET_CLASS_RE = re.compile(r"\bsh:targetClass\s+:([A-Za-z_][A-Za-z0-9_-]{0,127})\b")
_PATH_RE = re.compile(r"\bsh:path\s+:([A-Za-z_][A-Za-z0-9_-]{0,127})\b")

REQUIRED_PROVENANCE_PATHS = frozenset(
    {
        "sourceRecordRef",
        "tenantReference",
        "accessPolicyReference",
        "provenanceReference",
    }
)


class DeclaredSemanticCoverageError(RuntimeError):
    """Raised when a connector's declared SHACL shapes do not cover its
    declared record types or the required provenance paths."""


@dataclass(frozen=True)
class DeclaredSemanticCoverage:
    """Typed view of the SHACL target classes and property paths a
    connector declares, parsed once from its signed shapes text."""

    target_classes: frozenset[str]
    paths: frozenset[str]

    @classmethod
    def from_shapes_text(cls, shapes_text: str) -> DeclaredSemanticCoverage:
        return cls(
            target_classes=frozenset(_TARGET_CLASS_RE.findall(shapes_text)),
            paths=frozenset(_PATH_RE.findall(shapes_text)),
        )

    def require_covers(self, declared_record_types: frozenset[str]) -> None:
        """Refuse when the declared shapes do not cover every declared
        record type or the required provenance paths."""

        covers_types = declared_record_types.issubset(self.target_classes)
        covers_paths = REQUIRED_PROVENANCE_PATHS.issubset(self.paths)
        if not (covers_types and covers_paths):
            raise DeclaredSemanticCoverageError(
                "declared semantic coverage is incomplete"
            )
