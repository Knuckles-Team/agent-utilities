"""What ACTUALLY opens the engine circuit breaker? Capture the real exception chain."""
import asyncio, time, traceback

def chain(e):
    out=[]; cur=e; seen=set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append(f"{type(cur).__module__}.{type(cur).__name__}: {str(cur)[:300]}")
        out.append(f"    MRO: {[c.__name__ for c in type(cur).__mro__[:6]]}")
        cur = cur.__cause__ or cur.__context__
    return "\n  caused by ".join(out)

async def main():
    from agent_utilities.mcp.kg_server import _mint_process_session
    from agent_utilities.knowledge_graph.core.session import set_session
    from agent_utilities.security.brain_context import set_actor
    s = await asyncio.to_thread(_mint_process_session, "auto")
    s.engine_verified_context(); set_actor(s.actor); set_session(s)

    from agent_utilities.knowledge_graph.core import engine_breaker as eb
    print("=== breaker registry state BEFORE ===", flush=True)
    for ep, br in eb._breakers.items():
        print(f"  {ep!r}: state={br._state} failures={br._failures} threshold={br.threshold} cooldown={br.cooldown} enabled={br.enabled}", flush=True)
    if not eb._breakers: print("  (no breakers constructed yet in this process)", flush=True)

    from agent_utilities.core.config import config
    print(f"  config: threshold={config.engine_breaker_threshold} cooldown={config.engine_breaker_cooldown}", flush=True)
    import os
    for k in ("GRAPH_SERVICE_RPC_TIMEOUT","GRAPH_SERVICE_CONNECT_TIMEOUT","GRAPH_SERVICE_HEAVY_RPC_TIMEOUT","GRAPH_SERVICE_WRITE_TIMEOUT","ENGINE_BREAKER_THRESHOLD"):
        print(f"  env {k}={os.environ.get(k, '(unset -> default)')}", flush=True)

    # Exercise the REAL bounded grounding path the NL planner uses
    from agent_utilities.core import contextual_model as cm
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    msgs=[ModelRequest(parts=[UserPromptPart(content="List recent ServiceNow incidents.")])]
    print(f"\n=== bounded compile (_CONTEXT_COMPILE_TIMEOUT_S={cm._CONTEXT_COMPILE_TIMEOUT_S}) x5 ===", flush=True)
    fn = getattr(cm, "_compiled_evidence_and_bundle_bounded", None)
    print("bounded fn:", fn, flush=True)
    for i in range(5):
        t=time.monotonic()
        try:
            if asyncio.iscoroutinefunction(fn): r = await fn(msgs, "qwen/qwen3.6-27b")
            else: r = await asyncio.to_thread(fn, msgs, "qwen/qwen3.6-27b")
            print(f"  [{i}] OK in {time.monotonic()-t:.2f}s", flush=True)
        except Exception as e:
            print(f"  [{i}] RAISED after {time.monotonic()-t:.2f}s:\n  {chain(e)}", flush=True)
    print("\n=== breaker registry state AFTER ===", flush=True)
    for ep, br in eb._breakers.items():
        print(f"  {ep!r}: state={br._state} failures={br._failures}", flush=True)

asyncio.run(main())
