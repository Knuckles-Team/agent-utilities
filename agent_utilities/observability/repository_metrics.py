"""CONCEPT:AU-KG.audit.repository-job-provenance-bridge — RMDD-19 metric catalog.

The repository-job Prometheus instruments themselves live in
``observability.gateway_metrics`` (this codebase's single metrics module —
every domain's instruments live there, from gateway requests to KV-cache
tiering to KG ingest lag; RMDD-19 follows that existing chokepoint rather
than starting a second metrics module).

This module is the **deterministic test/documentation registry** for just
the repository-job subset: a name -> instrument mapping plus the exact
bounded label set each instrument was declared with, so a test can assert
"every repository metric has a bounded label set, and none of them is a job
id / path / credential" without importing ``prometheus_client`` internals or
hand-copying the label tuples out of ``gateway_metrics``.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.observability import gateway_metrics

REPOSITORY_METRIC_REGISTRY: dict[str, Any] = {
    "job_queue_depth": gateway_metrics.REPOSITORY_JOB_QUEUE_DEPTH,
    "job_queue_age": gateway_metrics.REPOSITORY_JOB_QUEUE_AGE,
    "capacity_reservations": gateway_metrics.REPOSITORY_CAPACITY_RESERVATIONS,
    "build_cache_ops": gateway_metrics.REPOSITORY_BUILD_CACHE_OPS,
    "validation_duration": gateway_metrics.REPOSITORY_VALIDATION_DURATION,
    "bisection_runs": gateway_metrics.REPOSITORY_BISECTION_RUNS,
    "lane_count": gateway_metrics.REPOSITORY_LANE_COUNT,
    "lane_disk_bytes": gateway_metrics.REPOSITORY_LANE_DISK_BYTES,
    "lane_age": gateway_metrics.REPOSITORY_LANE_AGE,
    "worker_health": gateway_metrics.REPOSITORY_WORKER_HEALTH,
    "job_retries": gateway_metrics.REPOSITORY_JOB_RETRIES,
    "landing_drift": gateway_metrics.REPOSITORY_LANDING_DRIFT,
    "release_dag_stage_duration": gateway_metrics.REPOSITORY_RELEASE_DAG_STAGE_DURATION,
}

# The exact label tuple each instrument in the registry above was declared
# with in gateway_metrics.py. Kept side-by-side (rather than introspected
# from the live Prometheus/no-op object, which does not expose its
# labelnames uniformly across both code paths) so a test has one place to
# assert bounded cardinality and the absence of an identifying label
# (job id, path, credential) by name.
REPOSITORY_METRIC_LABELS: dict[str, tuple[str, ...]] = {
    "job_queue_depth": ("repo", "priority_class"),
    "job_queue_age": ("repo",),
    "capacity_reservations": ("resource_class", "host_alias"),
    "build_cache_ops": ("repo", "outcome"),
    "validation_duration": ("repo", "stage"),
    "bisection_runs": ("repo", "outcome"),
    "lane_count": ("repo",),
    "lane_disk_bytes": ("repo",),
    "lane_age": ("repo",),
    "worker_health": ("host_alias",),
    "job_retries": ("repo", "retry_class"),
    "landing_drift": ("repo", "drift_kind"),
    "release_dag_stage_duration": ("repo", "stage"),
}

# Labels that would identify a specific job/path/credential rather than a
# bounded class — a metric label may never use one of these names. Checked
# by the RMDD-19 test suite against REPOSITORY_METRIC_LABELS.
FORBIDDEN_LABEL_NAMES: frozenset[str] = frozenset(
    {
        "job_id",
        "work_item_id",
        "workitem_id",
        "path",
        "file_path",
        "worktree_path",
        "credential",
        "token",
        "secret",
        "password",
        "hostname",
        "host",
        "ip",
        "ip_address",
    }
)

__all__ = [
    "REPOSITORY_METRIC_REGISTRY",
    "REPOSITORY_METRIC_LABELS",
    "FORBIDDEN_LABEL_NAMES",
]
