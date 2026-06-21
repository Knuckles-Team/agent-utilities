#!/usr/bin/python
"""Wire-level load harness for the AgentBus (CONCEPT:ECO-4.87).

Benchmarks a *live* graph-os hub through its REST twin ``/graph/bus`` (the same path a
remote session uses), so it measures the real cross-host send/receive cost — not an
in-process shortcut. Registers N virtual participants, drives a send workload at a chosen
fan-out and concurrency, and reports send/receive latency percentiles + throughput. It then
prints the modeled expectation from ``docs/scaling/capacity_model.py`` so measured-vs-modeled
is visible in one place.

This is the load generator the capacity model lacked; run it against a hub to record real
constants. It needs no agent-utilities runtime — only ``httpx`` — so it can run from any host
that can reach the hub.

Examples
--------
    # direct messaging, 50 senders, 5k messages
    python scripts/bench_bus.py --base-url http://graphos.arpa:8000 --token "$JWT" \
        --participants 50 --messages 5000 --concurrency 16

    # topic fan-out to 20 subscribers
    python scripts/bench_bus.py --participants 21 --messages 2000 --fanout 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx


async def _call(client: httpx.AsyncClient, base: str, body: dict) -> dict:
    resp = await client.post(f"{base}/graph/bus", json=body, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    # The generic REST adapter wraps the tool's JSON string under a "result" key.
    inner = data.get("result", data)
    return json.loads(inner) if isinstance(inner, str) else inner


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(q * len(s)))
    return s[idx] * 1000.0  # ms


async def run(args: argparse.Namespace) -> dict:
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    ids = [f"bench-{i}" for i in range(args.participants)]
    topic = "bench-topic" if args.fanout > 1 else ""

    async with httpx.AsyncClient(headers=headers) as client:

        async def post(body: dict) -> dict:
            return await _call(client, args.base_url, body)

        # 1) register all participants
        await asyncio.gather(
            *(
                post({"action": "register", "agent_id": a, "provider": "bench"})
                for a in ids
            )
        )
        # 2) subscribe receivers to the topic for fan-out runs
        if topic:
            receivers = ids[1 : 1 + args.fanout]
            await asyncio.gather(
                *(
                    post({"action": "subscribe", "agent_id": a, "topic": topic})
                    for a in receivers
                )
            )

        # 3) drive the send workload, bounded by --concurrency
        sem = asyncio.Semaphore(args.concurrency)
        send_latencies: list[float] = []
        sender = ids[0]

        async def one_send(n: int) -> None:
            body = {"action": "send", "sender": sender, "payload": f"m{n}"}
            if topic:
                body["topic"] = topic
            else:
                body["to"] = ids[1 + (n % max(1, args.participants - 1))]
            async with sem:
                t0 = time.perf_counter()
                await post(body)
                send_latencies.append(time.perf_counter() - t0)

        wall0 = time.perf_counter()
        await asyncio.gather(*(one_send(n) for n in range(args.messages)))
        wall = time.perf_counter() - wall0

        # 4) measure receive latency for one receiver draining its mailbox
        drain_target = (
            (ids[1] if topic else ids[1]) if args.participants > 1 else ids[0]
        )
        recv_latencies: list[float] = []
        cursor = 0
        for _ in range(args.recv_samples):
            t0 = time.perf_counter()
            out = await post(
                {"action": "receive", "agent_id": drain_target, "since": cursor}
            )
            recv_latencies.append(time.perf_counter() - t0)
            cursor = out.get("cursor", cursor)

    delivered = args.messages * (args.fanout if topic else 1)
    return {
        "config": {
            "base_url": args.base_url,
            "participants": args.participants,
            "messages": args.messages,
            "fanout": args.fanout,
            "concurrency": args.concurrency,
        },
        "throughput_msgs_per_sec": round(args.messages / wall, 1) if wall else 0.0,
        "delivered_total": delivered,
        "send_latency_ms": {
            "p50": round(_pct(send_latencies, 0.50), 2),
            "p95": round(_pct(send_latencies, 0.95), 2),
            "p99": round(_pct(send_latencies, 0.99), 2),
        },
        "receive_latency_ms": {
            "p50": round(_pct(recv_latencies, 0.50), 2),
            "p95": round(_pct(recv_latencies, 0.95), 2),
            "p99": round(_pct(recv_latencies, 0.99), 2),
        },
        "wall_seconds": round(wall, 3),
    }


def _modeled(args: argparse.Namespace, measured_rate: float) -> dict:
    """Print the capacity-model expectation alongside the measured rate."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1] / "docs" / "scaling" / "capacity_model.py"
    )
    spec = importlib.util.spec_from_file_location("capacity_model", path)
    assert spec and spec.loader
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    plan = cm.bus_plan_for(
        args.participants, measured_rate, avg_recipients=max(1, args.fanout)
    )
    return {
        "modeled_msgs_per_sec_per_connection": plan.messages_per_sec_per_connection,
        "modeled_engine_connections_for_measured_rate": plan.engine_connections,
        "modeled_hubs_for_participants": plan.hubs,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="AgentBus wire-level load harness (ECO-4.87)"
    )
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument(
        "--token", default="", help="Bearer JWT for a served (cross-host) hub."
    )
    p.add_argument("--participants", type=int, default=20)
    p.add_argument("--messages", type=int, default=1000)
    p.add_argument(
        "--fanout", type=int, default=1, help=">1 → topic fan-out to N subscribers."
    )
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--recv-samples", type=int, default=50)
    args = p.parse_args()

    report = asyncio.run(run(args))
    report["modeled"] = _modeled(args, report["throughput_msgs_per_sec"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
