#!/usr/bin/python
from __future__ import annotations

"""Cross-layer error diagnosis over the live deployment (CONCEPT:KG-2.297).

The ``troubleshoot`` provider of the context plane (KG-2.136): answers *"this
agent run failed / this service is unreachable / this container keeps crashing —
what happened and how do I trace it?"* by (1) pulling whatever the KG already
knows about the symptom — the run's :RunTrace + :ToolCall provenance (KG-2.296),
recent errored runs — and (2) emitting the **layered troubleshooting playbook**:
the exact tool to reach for at every layer of the stack, app-trace → container
log → system log → host reachability → cross-cutting observability.

It is deliberately *anti-sprawl*: it builds no new log store. It composes the KG
reads it can do itself with a precise map onto the EXISTING fleet tools
(``cm__*`` container-manager, ``sm__*`` systems-manager, ``tm__*``
tunnel-manager, the ``lgtm__*`` Grafana surface, ``graph_observe``) so the
operator/agent runs the right next call instead of guessing.

Pure best-effort Cypher reads (never raises) so a degraded backend still yields
the playbook even when the trace can't be fetched.

Intents (all return the full ladder, but bias the synthesis):
``run`` (a failed agent/delegation run — default), ``service`` (an endpoint is
unreachable — host-vs-service), ``health`` (general posture).
"""

from typing import Any

from agent_utilities.knowledge_graph.retrieval.context_plane import read_rows

VALID_INTENTS = ("run", "service", "health")

# Symptom keyword → the stack layer most likely implicated. The provider always
# emits the whole ladder; this only orders/biases the narrative.
_LAYER_HINTS: dict[str, tuple[str, ...]] = {
    "app_trace": (
        "run",
        "agent",
        "trace",
        "delegation",
        "tool call",
        "toolcall",
        "wrong answer",
        "ungrounded",
        "hallucinat",
        "execute_agent",
    ),
    "container": (
        "crash",
        "restart",
        "exit",
        "oom",
        "137",
        "crashloop",
        "crash-loop",
        "container",
        "image",
        "compose",
        "swarm",
    ),
    "system": (
        "journal",
        "journald",
        "systemd",
        "kernel",
        "disk",
        "cpu",
        "load",
        "memory pressure",
        "oom-killer",
        "out of memory",
        "service unit",
    ),
    "host": (
        "unreachable",
        "502",
        "timeout",
        "connection refused",
        "no route",
        "session terminated",
        "down",
        ".arpa",
        "host",
    ),
    "cross_cutting": (
        "latency",
        "slow",
        "grafana",
        "loki",
        "tempo",
        "metrics",
        "p95",
        "throughput",
        "across",
    ),
}


def _trace_id(node_id: str) -> str:
    """Normalize a run/trace identifier to the ``trace:<run_id>`` node id."""
    nid = (node_id or "").strip()
    if not nid:
        return ""
    if nid.startswith("trace:"):
        return nid
    if nid.startswith("toolcall:"):
        return ""
    return f"trace:{nid}"


def _classify(query: str, intent: str) -> list[str]:
    """Order the stack layers by relevance to the symptom text."""
    low = (query or "").lower()
    scored: list[tuple[int, str]] = []
    for layer, hints in _LAYER_HINTS.items():
        scored.append((sum(1 for h in hints if h in low), layer))
    ranked = [layer for n, layer in sorted(scored, key=lambda x: -x[0]) if n > 0]
    # Intent seeds the lead layer when the text is ambiguous.
    seed = {"run": "app_trace", "service": "host", "health": "app_trace"}.get(
        intent, "app_trace"
    )
    if seed not in ranked:
        ranked.insert(0, seed)
    # Always present the full ladder; ranked layers lead.
    full = ["app_trace", "container", "system", "host", "cross_cutting"]
    return ranked + [layer for layer in full if layer not in ranked]


# The per-layer tool map — the load-bearing knowledge. Each entry: which signal
# lives at that layer and the EXACT existing tool to read it (no new logging).
_PLAYBOOK: dict[str, dict[str, str]] = {
    "app_trace": {
        "question": "What did the agent/delegation actually do, and where did it fail?",
        "signal": ":RunTrace + :ToolCall provenance (KG-2.296) + the always-on "
        "Trace/Span/Generation subgraph (OS-5.68).",
        "tool": "graph_query the run: MATCH (t:RunTrace {id:'trace:<run_id>'})"
        "-[:MADE_TOOL_CALL]->(tc:ToolCall) RETURN tc.tool_name, tc.status, "
        "tc.error, tc.result_preview ORDER BY tc.sequence — or graph_observe "
        "action=trace_rootcause for the span/generation analysis.",
    },
    "container": {
        "question": "Is the container crashing, and why (exit code)?",
        "signal": "Container state + stdout/stderr on the host running the service.",
        "tool": "cm__container_operations action=logs host=<alias> (Docker-over-SSH "
        "via the inventory) — or on the swarm manager: docker service ps "
        "<svc> --no-trunc + docker service logs <svc>. Exit 137 = OOM-killed, "
        "143 = SIGTERM, 0-but-restarting = healthcheck failing.",
    },
    "system": {
        "question": "Is the HOST OS the cause (OOM-killer, disk full, unit failed)?",
        "signal": "journald / systemd units / kernel ring buffer / disk + load.",
        "tool": "sm__query_system_logs (journald, filter by unit/priority), "
        "sm__list_services (find a failed unit), sm__get_process_details "
        "(a runaway PID), sm__storage_health (disk). The dmesg OOM-killer line "
        "explains an exit-137 seen at the container layer.",
    },
    "host": {
        "question": "Is the host DOWN, or is only the service down? (the decisive split)",
        "signal": "Raw SSH reachability vs container presence on the host.",
        "tool": "Resolve the host from the inventory (cm__list_hosts / tm__* — "
        "an .arpa edge 502 means SSH the ACTUAL upstream, not the edge). Then "
        "tm__remote / ssh <host>: 'No route to host' or timeout = HOST DOWN "
        "(operator/infra — nothing to restart service-side); connects but the "
        "container is stopped/absent/crash-looping = SERVICE DOWN (restart it, "
        "then read the container layer for the crash cause).",
    },
    "cross_cutting": {
        "question": "Is this a fleet-wide pattern (latency, error spike, saturation)?",
        "signal": "Prometheus /metrics (OS-5.23) + the LGTM stack (Loki logs, "
        "Tempo traces, Mimir/Grafana metrics).",
        "tool": "lgtm__grafana (dashboards/Loki/Tempo queries) + lgtm__alertmanager "
        "for firing alerts; the gateway /metrics series (ENGINE_BREAKER_STATE, "
        "KG_INGEST_QUEUE_DEPTH, MCP_CHILD_BREAKER_STATE, DISPATCH_QUEUE_DEPTH) "
        "show breaker/queue/child health. graph_analyze action=explain "
        "target='ops:health' reads the live :Task lane/queue state (KG-2.137).",
    },
}


def diagnose_symptom(
    engine: Any,
    *,
    query: str = "",
    intent: str = "run",
    node_id: str = "",
    top_k: int = 10,
    **_opts: Any,
) -> dict[str, Any]:
    """Diagnose a deployment symptom: pull the trace it can + emit the playbook."""
    intent = (intent or "run").strip().lower()
    if intent not in VALID_INTENTS:
        # The plane passes intent='how' by default — infer from the text.
        intent = "service" if any(
            h in (query or "").lower() for h in _LAYER_HINTS["host"]
        ) else "run"
    limit = max(1, min(50, int(top_k)))

    citations: list[dict[str, Any]] = []
    used: list[str] = ["troubleshoot_playbook"]

    # (1) App-trace layer — pull the named run's provenance if we have an id.
    trace_id = _trace_id(node_id)
    trace_row: dict[str, Any] = {}
    tool_calls: list[dict[str, Any]] = []
    if trace_id:
        rows = read_rows(
            engine,
            "MATCH (t:RunTrace {id: $tid}) RETURN t.id AS id, "
            "t.agent_name AS agent_name, t.status AS status, t.error AS error, "
            "t.duration_ms AS duration_ms",
            {"tid": trace_id},
        )
        trace_row = rows[0] if rows else {}
        tool_calls = read_rows(
            engine,
            "MATCH (t:RunTrace {id: $tid})-[:MADE_TOOL_CALL]->(tc:ToolCall) "
            "RETURN tc.tool_name AS tool_name, tc.status AS status, "
            "tc.error AS error, tc.sequence AS sequence ORDER BY tc.sequence",
            {"tid": trace_id},
        )
        if trace_row:
            citations.append({"type": "run_trace", **trace_row})
            used.append("run_trace")
        for tc in tool_calls:
            citations.append({"type": "tool_call", **tc})
        if tool_calls:
            used.append("tool_call_chain")

    # (2) Otherwise surface the recent errored runs to triage from.
    failed_runs: list[dict[str, Any]] = []
    if not trace_id:
        failed_runs = read_rows(
            engine,
            "MATCH (t:RunTrace) WHERE t.status IN ['failed','error'] "
            "RETURN t.id AS id, t.agent_name AS agent_name, t.error AS error "
            "ORDER BY t.timestamp DESC LIMIT $k",
            {"k": limit},
        )
        for r in failed_runs:
            citations.append({"type": "failed_run", **r})
        if failed_runs:
            used.append("failed_runs")

    layers = _classify(query, intent)
    answer = _synthesize(query, intent, trace_id, trace_row, tool_calls, failed_runs, layers)

    return {
        "status": "ok",
        "domain": "troubleshoot",
        "intent": intent,
        "query": query,
        "answer": answer,
        "citations": citations,
        "sections": {
            "trace": [trace_row] if trace_row else [],
            "tool_calls": tool_calls,
            "failed_runs": failed_runs,
            "playbook": [
                {"layer": layer, **_PLAYBOOK[layer]} for layer in layers
            ],
        },
        "capability_id": f"troubleshoot:{intent}:{layers[0]}",
        "used_primitives": used,
    }


def _synthesize(
    query: str,
    intent: str,
    trace_id: str,
    trace_row: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    failed_runs: list[dict[str, Any]],
    layers: list[str],
) -> str:
    out: list[str] = []
    if trace_id:
        if trace_row:
            failed_tc = next(
                (tc for tc in tool_calls if str(tc.get("status")) == "error"), None
            )
            head = (
                f"Run {trace_row.get('id')} ({trace_row.get('agent_name')}): "
                f"status={trace_row.get('status')}"
            )
            if trace_row.get("error"):
                head += f", error={str(trace_row.get('error'))[:160]}"
            if failed_tc:
                head += (
                    f"; first failing tool call: {failed_tc.get('tool_name')} "
                    f"(seq {failed_tc.get('sequence')}) — {str(failed_tc.get('error'))[:120]}"
                )
            out.append(head + ".")
        else:
            out.append(
                f"No :RunTrace found for '{trace_id}'. If the run is recent, the "
                "area may be uningested or the run id is wrong — list recent runs "
                "with graph_query MATCH (t:RunTrace) RETURN t.id ORDER BY "
                "t.timestamp DESC."
            )
    elif failed_runs:
        out.append(
            f"{len(failed_runs)} recent errored run(s); start with "
            f"{failed_runs[0].get('id')} ({failed_runs[0].get('agent_name')}): "
            f"{str(failed_runs[0].get('error'))[:160]}."
        )
    else:
        out.append(
            "No specific run supplied and no recent errored :RunTrace in the KG — "
            "diagnosing from the symptom text. Pass node_id=<run_id> to pull a "
            "specific run's :ToolCall chain."
        )

    # The decisive host-vs-service reminder leads when the symptom is reachability.
    if "host" in layers[:2]:
        out.append(
            "DECIDE host-vs-service FIRST: SSH the upstream host (resolve via the "
            "inventory; an .arpa 502 means the edge is up but the upstream isn't). "
            "'No route to host' = HOST down (operator/infra). Connects but the "
            "container is stopped/crash-looping = SERVICE down (restart + read "
            "container logs; exit 137 = OOM)."
        )

    out.append(
        "Trace every layer until grounded — "
        + " → ".join(
            f"{layer}: {_PLAYBOOK[layer]['tool'].split('.')[0].split(' — ')[0]}"
            for layer in layers
        )
        + ". Full per-layer tool map in sections.playbook."
    )
    return " ".join(out)
