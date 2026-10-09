"""The typed ``SchemaDriftReport`` (AU-SEC-R004): observation-class evidence.

A report names the source, the two shape digests compared, every classified
change, the verdict, and the held delta (how many records, and the digest of
their ids) -- never the records themselves. It is recorded in the knowledge
graph as a ``SchemaDriftReport`` node whose ``epistemic_class`` is
``observation``: it states what a source delivered, not a belief about why.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .classify import DriftChange

logger = logging.getLogger(__name__)

#: Node label of a recorded report.
REPORT_LABEL = "SchemaDriftReport"
_REPORT_DOMAIN = "au/schema-drift-report/v1"
_DELTA_DOMAIN = "au/schema-drift-delta/v1"


class Verdict(StrEnum):
    """What happened to the drained delta."""

    #: The observation matches the approved contract.
    NO_DRIFT = "no_drift"
    #: Drift, but every class is one the declared policy lets this source absorb.
    CONTINUE = "continue"
    #: Drift the policy does not cover: held, checkpoint not advanced.
    QUARANTINE = "quarantine"


def delta_digest(record_ids: Iterable[str]) -> str:
    """Order-independent digest of the held delta's source ids."""
    joined = "\0".join(sorted(record_ids))
    return hashlib.sha256(f"{_DELTA_DOMAIN}\0{joined}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SchemaDriftReport:
    """One classified comparison of a drained delta against its contract."""

    source: str
    approved_digest: str
    observed_digest: str
    changes: tuple[DriftChange, ...]
    verdict: Verdict
    records_held: int
    delta_digest: str

    def body(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "approved_digest": self.approved_digest,
            "observed_digest": self.observed_digest,
            "changes": [change.to_json() for change in self.changes],
            "verdict": self.verdict.value,
            "records_held": self.records_held,
            "delta_digest": self.delta_digest,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.body(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{_REPORT_DOMAIN}\0{payload}".encode()).hexdigest()

    @property
    def report_id(self) -> str:
        return f"schema-drift:{self.source}:{self.digest[:16]}"

    def summary(self) -> dict[str, Any]:
        """The compact form a sync result carries."""
        return {
            "report_id": self.report_id,
            "verdict": self.verdict.value,
            "changes": [change.to_json() for change in self.changes],
            "records_held": self.records_held,
        }

    def node_properties(self) -> dict[str, Any]:
        """The recorded node: observation-class evidence of the comparison."""
        return {
            "name": f"schema drift in {self.source}",
            "epistemic_class": "observation",
            "source": self.source,
            "verdict": self.verdict.value,
            "report_digest": self.digest,
            "approved_digest": self.approved_digest,
            "observed_digest": self.observed_digest,
            "records_held": self.records_held,
            "delta_digest": self.delta_digest,
            "changes": json.dumps([c.to_json() for c in self.changes], sort_keys=True),
        }


def record_report(engine: Any, report: SchemaDriftReport) -> str | None:
    """Persist ``report`` as a ``SchemaDriftReport`` node; ``None`` if refused."""
    try:
        engine.add_node(
            report.report_id, REPORT_LABEL, properties=report.node_properties()
        )
    except Exception as exc:
        logger.warning(
            "schema drift report %s was not recorded: %s", report.report_id, exc
        )
        return None
    return report.report_id


__all__ = [
    "REPORT_LABEL",
    "SchemaDriftReport",
    "Verdict",
    "delta_digest",
    "record_report",
]
