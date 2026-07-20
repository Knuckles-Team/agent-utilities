#!/usr/bin/python
"""Collect one aggregate, privacy-safe certification sample from Prometheus."""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from agent_utilities.core.transport_security import resolve_configured_tls_profile

_QUERIES = {
    "gatewayP99Seconds": "graphos:gateway_request_p99_seconds:5m",
    "engineP99Seconds": "graphos:engine_request_p99_seconds:5m",
    "gatewayErrorRatio": "graphos:gateway_error_ratio:5m",
    "dispatchQueueDepth": "graphos:dispatch_queue_depth",
    "ingestConsumerLag": "graphos:ingest_consumer_lag",
    "analyticsJobsReady": "graphos:analytics_jobs_ready",
    "reachableEngineMembers": (
        'kube_statefulset_status_replicas_ready{namespace="graphos-cell",'
        'statefulset="epistemic-graph-raft"}'
    ),
    "walAppendDroppedFiveMinutes": "sum(increase(epistemic_graph_wal_append_dropped_total[5m]))",
    "checkpointAgeSeconds": "time() - max(epistemic_graph_checkpoint_last_success_timestamp_seconds)",
    "podRestartsFiveMinutes": 'sum(increase(kube_pod_container_status_restarts_total{namespace=~"graphos-control|graphos-cell"}[5m]))',
}
_MAX_PROMETHEUS_RESPONSE_BYTES = 262_144


def _bounded_json_response(response: Any) -> dict[str, Any]:
    """Decode one Prometheus response without retaining an unbounded body."""

    response.raise_for_status()
    raw_length = str(response.headers.get("Content-Length", "")).strip()
    if raw_length:
        try:
            content_length = int(raw_length)
            if not 0 <= content_length <= _MAX_PROMETHEUS_RESPONSE_BYTES:
                raise RuntimeError("Prometheus response exceeded its size boundary")
        except ValueError as exc:
            raise RuntimeError("Prometheus response length is invalid") from exc
    payload = bytearray()
    for chunk in response.iter_content(chunk_size=65_536):
        if not isinstance(chunk, bytes):
            raise RuntimeError("Prometheus response chunk is invalid")
        if len(payload) + len(chunk) > _MAX_PROMETHEUS_RESPONSE_BYTES:
            raise RuntimeError("Prometheus response exceeded its size boundary")
        payload.extend(chunk)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Prometheus response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Prometheus response is not a JSON object")
    return value


def _headers(config: Any) -> dict[str, str]:
    reference = str(config.cert_prometheus_bearer_token_ref or "").strip()
    if not reference:
        return {}
    from agent_utilities.security.cli_secrets import resolve_runtime_secret_reference

    token = str(resolve_runtime_secret_reference(reference) or "").strip()
    if (
        not token
        or len(token.encode("utf-8")) > 16_384
        or any(character in token for character in "\r\n")
    ):
        raise RuntimeError("Prometheus bearer-token reference is invalid")
    return {"Authorization": f"Bearer {token}"}


def _query(
    session: requests.Session,
    base_url: str,
    query: str,
    headers: dict[str, str],
) -> float:
    response = session.get(
        base_url.rstrip("/") + "/api/v1/query",
        params={"query": query},
        headers=headers,
        timeout=20,
        stream=True,
    )
    try:
        payload = _bounded_json_response(response)
    finally:
        response.close()
    result = payload.get("data", {}).get("result", [])
    if payload.get("status") != "success" or len(result) != 1:
        raise RuntimeError("Prometheus query did not return one aggregate sample")
    return float(result[0]["value"][1])


def main(*, config: Any | None = None) -> int:
    try:
        if config is None:
            from agent_utilities.core.config import AgentConfig

            config = AgentConfig()
        base_url = str(config.cert_prometheus_url or "").strip()
        if not base_url:
            raise RuntimeError("certification Prometheus URL is absent")
        headers = _headers(config)
        trust = resolve_configured_tls_profile(
            "certification-prometheus",
            profile_name=config.cert_prometheus_tls_profile,
            profile_ref=config.cert_prometheus_tls_profile_ref,
            config=config,
        )
        try:
            with requests.Session() as session:
                trust.configure_requests_session(session)
                values = {
                    name: _query(session, base_url, query, headers)
                    for name, query in _QUERIES.items()
                }
        finally:
            trust.cleanup()
        sample = {"timestampUnix": int(time.time()), "values": values}
    except Exception as exc:  # noqa: BLE001 - never expose endpoint/body in output
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, "sample": sample}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
