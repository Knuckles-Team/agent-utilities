"""OKF round-trip write-back sink — KG-distilled skill-graph → OKF bundle on disk.

CONCEPT:AU-ECO.connector.okf-roundtrip-sync — the KG→OKF push arm of the
bidirectional round-trip, wired onto the EXISTING ``graph_writeback`` surface (no
new verb, so surface-parity is preserved). It reuses the unified fail-closed,
dry-run-first write-back machinery: ``dry_run`` previews the exact
create/update/conflict/delete plan (the mdcode Catalog-Snapshot diff), and a live
push (gated on ``OKF_ENABLE_WRITE``) materializes the OKF bundle + rewrites the
``.catalog.state`` checksum file, aborting fail-fast on an interim conflict.

``creations`` (or ``enrichments``) items each carry one push job::

    {"skill_dir": "/path/to/skill-graph",
     "catalog_dir": "/path/to/okf-bundle",
     "allow_delete": false, "force": false}
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)


class OkfSink:
    domain = "okf"
    enable_flag = "OKF_ENABLE_WRITE"

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        from agent_utilities.knowledge_graph.distillation.okf_bundle import (
            OkfConflictError,
            OkfRoundTripSync,
        )

        result = WritebackResult(target=self.domain)
        jobs = list(ops.get("creations") or []) + list(ops.get("enrichments") or [])
        for job in jobs:
            skill_dir = job.get("skill_dir") or job.get("node")
            catalog_dir = job.get("catalog_dir")
            if not skill_dir or not catalog_dir:
                result.skipped += 1
                continue
            sync = OkfRoundTripSync(skill_dir, catalog_dir)
            allow_delete = bool(job.get("allow_delete"))
            force = bool(job.get("force"))
            try:
                plan = sync.push(
                    dry_run=dry_run, allow_delete=allow_delete, force=force
                )
            except OkfConflictError as exc:
                result.errors += 1
                result.proposals.append(
                    {
                        "op": "okf_push",
                        "catalog_dir": catalog_dir,
                        "status": "conflict",
                        "conflicts": exc.conflicts,
                    }
                )
                continue
            if dry_run or plan.get("status") == "conflict":
                result.proposals.append({"op": "okf_push", **plan})
                continue
            result.created += len(plan.get("creates", []))
            result.enriched += len(plan.get("updates", []))
            result.retired += len(plan.get("deletes", []))
        return result


register_sink(OkfSink())
