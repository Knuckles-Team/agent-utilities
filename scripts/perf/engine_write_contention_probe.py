"""180s: sample read latency every 2s and correlate with the concurrent write rate."""

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
    await asyncio.to_thread(b.execute, Q)

    print("t_s  read_ms  write_s_in_bucket  write_calls", flush=True)
    buckets = []
    s_prev, c_prev = snap()
    t_start = time.time()
    for i in range(90):
        t = time.monotonic()
        try:
            await asyncio.to_thread(b.execute, Q)
        except Exception as e:
            print("ERR", type(e).__name__, str(e)[:100], flush=True)
            break
        rd = time.monotonic() - t
        await asyncio.sleep(1.0)
        s_now, c_now = snap()
        wsec = sum(s_now.get(o, 0) - s_prev.get(o, 0) for o in WRITE_OPS)
        wcnt = sum(c_now.get(o, 0) - c_prev.get(o, 0) for o in WRITE_OPS)
        s_prev, c_prev = s_now, c_now
        buckets.append((rd, wsec, wcnt))
        print(
            f"{time.time() - t_start:5.0f} {rd * 1000:8.1f} {wsec:8.2f} {wcnt:6.0f}",
            flush=True,
        )

    print("\n=== CORRELATION ===", flush=True)
    quiet = [r for r, w, c in buckets if c == 0]
    busy = [r for r, w, c in buckets if c > 0]
    if quiet:
        print(
            f"  reads with NO concurrent write : n={len(quiet):3d} mean={statistics.mean(quiet) * 1000:8.1f}ms median={statistics.median(quiet) * 1000:8.1f}ms max={max(quiet) * 1000:8.1f}ms",
            flush=True,
        )
    if busy:
        print(
            f"  reads WITH concurrent write    : n={len(busy):3d} mean={statistics.mean(busy) * 1000:8.1f}ms median={statistics.median(busy) * 1000:8.1f}ms max={max(busy) * 1000:8.1f}ms",
            flush=True,
        )
    if quiet and busy:
        print(
            f"  => concurrent writes multiply read latency by {statistics.mean(busy) / statistics.mean(quiet):.1f}x",
            flush=True,
        )
    tw = sum(w for _, w, _ in buckets)
    el = time.time() - t_start
    print(
        f"  write-op seconds over window: {tw:.1f}s / {el:.0f}s wall = {100 * tw / el:.0f}% write-in-flight duty cycle",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
