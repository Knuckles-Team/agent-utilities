"""Correlate read-call stalls with concurrent WRITE activity on the global ServerState lock."""

import asyncio
import re
import statistics
import time
import urllib.request


def snap():
    d = (
        urllib.request.urlopen("http://127.0.0.1:9101/metrics", timeout=10)
        .read()
        .decode()
    )
    s, c = {}, {}
    for l in d.splitlines():
        if l.startswith("#"):
            continue
        m = re.match(
            r'epistemic_graph_request_duration_seconds_(sum|count)\{op="(.*?)"\} (.*)',
            l,
        )
        if m:
            (s if m.group(1) == "sum" else c)[m.group(2)] = float(m.group(3))
    return s, c


WRITE_OPS = {
    "BatchUpdate",
    "ClaimWorkItem",
    "DeferWorkItem",
    "ApplyChangeEnvelope",
    "CommitWorkItemResult",
    "RenewWorkItemLease",
}


async def main():
    from agent_utilities.knowledge_graph.core.session import set_session
    from agent_utilities.mcp.kg_server import _mint_process_session
    from agent_utilities.security.brain_context import set_actor

    s = await asyncio.to_thread(_mint_process_session, "auto")
    s.engine_verified_context()
    set_actor(s.actor)
    set_session(s)
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    eng = IntelligenceGraphEngine.get_active() or IntelligenceGraphEngine.get_or_create(
        defer_background_start=True
    )
    b = eng.backend

    Q = "MATCH (n) WHERE n.id = 'perf0801-nonexistent' RETURN n LIMIT 1"
    print("warming...", flush=True)
    await asyncio.to_thread(b.execute, Q)

    print(
        "\n=== 100 identical trivial Cypher point-reads, per-call latency ===",
        flush=True,
    )
    s0, c0 = snap()
    t0 = time.time()
    lat = []
    for _ in range(100):
        t = time.monotonic()
        try:
            await asyncio.to_thread(b.execute, Q)
        except Exception as e:
            print("ERR", type(e).__name__, str(e)[:120], flush=True)
            break
        lat.append(time.monotonic() - t)
    el = time.time() - t0
    s1, c1 = snap()

    ls = sorted(lat)
    n = len(lat)
    fast = [x for x in lat if x < 0.05]
    slow = [x for x in lat if x >= 0.05]
    print(
        f"n={n} wall={el:.1f}s  min={ls[0] * 1000:.2f}ms p50={ls[n // 2] * 1000:.2f}ms "
        f"p90={ls[int(n * 0.9)] * 1000:.1f}ms p99={ls[int(n * 0.99)] * 1000:.1f}ms max={ls[-1] * 1000:.1f}ms",
        flush=True,
    )
    if fast:
        print(
            f"  FAST(<50ms) : {len(fast):3d} calls  mean {statistics.mean(fast) * 1000:8.3f}ms",
            flush=True,
        )
    if slow:
        print(
            f"  STALL(>=50ms): {len(slow):3d} calls  mean {statistics.mean(slow):8.3f}s  total {sum(slow):.1f}s",
            flush=True,
        )
        print(
            f"  => {100 * len(slow) / n:.0f}% of calls stalled; stalls = {sum(slow) / sum(lat) * 100:.0f}% of ALL elapsed time",
            flush=True,
        )
        if fast:
            print(
                f"  => a stalled call is {statistics.mean(slow) / statistics.mean(fast):,.0f}x a clean one",
                flush=True,
            )
    else:
        print(
            "  STALL: NONE — engine was quiet; reads are genuinely sub-50ms", flush=True
        )

    print("\n=== engine ops that ran CONCURRENTLY in that same window ===", flush=True)
    rows = []
    for op in c1:
        dc = c1[op] - c0.get(op, 0)
        ds = s1[op] - s0.get(op, 0)
        if dc > 0:
            rows.append((ds, dc, op))
    wtime = 0.0
    for ds, dc, op in sorted(rows, reverse=True):
        tag = "WRITE" if op in WRITE_OPS else "read "
        if op in WRITE_OPS:
            wtime += ds
        print(
            f"  {tag} {op:26s} {dc:6.0f} calls  {ds:8.1f}s  mean {ds / dc:7.3f}s",
            flush=True,
        )
    print(
        f"\n  WRITE-op seconds in window: {wtime:.1f}s over {el:.1f}s wall = "
        f"{100 * wtime / el:.0f}% of wall time had a write in flight",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
