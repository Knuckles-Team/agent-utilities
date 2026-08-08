"""lane-perf-0801: (A) is the semantic index populated? (B) is the latency a per-call stall?"""

import asyncio, time, statistics, json


async def main():
    from agent_utilities.mcp.kg_server import _mint_process_session
    from agent_utilities.knowledge_graph.core.session import set_session
    from agent_utilities.security.brain_context import set_actor

    s = await asyncio.to_thread(_mint_process_session, "auto")
    s.engine_verified_context()
    set_actor(s.actor)
    set_session(s)
    print(f"identity ok tenant={s.tenant} graph={s.graph!r}", flush=True)

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    eng = IntelligenceGraphEngine.get_active() or IntelligenceGraphEngine.get_or_create(
        defer_background_start=True
    )
    b = eng.backend
    print("backend:", type(b).__name__, flush=True)

    # ---------- A. index population ----------
    print("\n=== A. INDEX POPULATION ===", flush=True)
    for q, label in [
        ("MATCH (n) RETURN count(n) AS c", "total nodes"),
        (
            "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) AS c",
            "w/ embedding",
        ),
        ("MATCH (n) WHERE n.text IS NOT NULL RETURN count(n) AS c", "w/ text"),
    ]:
        t = time.monotonic()
        try:
            print(
                f"  {label}: {b.execute(q)}  [{time.monotonic() - t:.2f}s]", flush=True
            )
        except Exception as e:
            print(
                f"  {label}: ERR {type(e).__name__}: {str(e)[:200]}  [{time.monotonic() - t:.2f}s]",
                flush=True,
            )

    print("\n=== A2. SEMANTIC SEARCH quality ===", flush=True)
    for text in [
        "List recent ServiceNow incidents.",
        "servicenow incident",
        "knowledge graph engine",
    ]:
        t = time.monotonic()
        try:
            res = await asyncio.to_thread(b.semantic_search, text, 10)
            el = time.monotonic() - t
            n = len(res) if res is not None else 0
            print(f"  {text!r} -> {n} hits [{el:.2f}s]", flush=True)
            for x in (res or [])[:4]:
                print("      ", str(x)[:170], flush=True)
        except Exception as e:
            print(
                f"  {text!r} ERR {type(e).__name__}: {str(e)[:250]} [{time.monotonic() - t:.2f}s]",
                flush=True,
            )

    # ---------- B. per-call stall ----------
    print("\n=== B. PER-CALL STALL (same trivial op, 60x) ===", flush=True)
    cli = getattr(b, "client", None)
    gname = getattr(b, "graph_name", None) or s.graph
    if cli is None:
        print("  no .client on backend; skipping", flush=True)
    else:
        for opname, fn in [
            ("nodes.has", lambda: cli.nodes.has("definitely-does-not-exist-perf0801")),
            ("health", lambda: cli.health())
            if hasattr(cli, "health")
            else ("skip", None),
        ]:
            if fn is None:
                continue
            lat = []
            for _ in range(60):
                t = time.monotonic()
                try:
                    await fn()
                except Exception as e:
                    print(
                        f"  {opname} ERR {type(e).__name__}: {str(e)[:120]}", flush=True
                    )
                    break
                lat.append(time.monotonic() - t)
            if not lat:
                continue
            lat_s = sorted(lat)
            fast = [x for x in lat if x < 0.05]
            slow = [x for x in lat if x >= 0.05]
            print(
                f"  {opname}: n={len(lat)} min={lat_s[0] * 1000:.2f}ms "
                f"p50={lat_s[len(lat_s) // 2] * 1000:.2f}ms p90={lat_s[int(len(lat_s) * 0.9)] * 1000:.2f}ms "
                f"max={lat_s[-1] * 1000:.2f}ms",
                flush=True,
            )
            print(
                f"    FAST(<50ms): {len(fast)} calls, mean {statistics.mean(fast) * 1000:.2f}ms"
                if fast
                else "    FAST: none",
                flush=True,
            )
            print(
                f"    SLOW(>=50ms): {len(slow)} calls, mean {statistics.mean(slow):.2f}s, total {sum(slow):.1f}s"
                if slow
                else "    SLOW: none",
                flush=True,
            )
            print(
                f"    => {100 * len(slow) / len(lat):.0f}% of calls stalled; stalls are {sum(slow) / sum(lat) * 100:.0f}% of all time",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
