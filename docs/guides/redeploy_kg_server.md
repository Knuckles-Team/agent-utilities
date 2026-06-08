# Redeploying the KG Server (exposing new MCP actions)

> When new `graph_orchestrate` / `graph_analyze` actions are added (e.g. the
> `assimilate` action), the **running** kg server must restart to expose them.
> `graph_orchestrate(action="assimilate")` returning *"Unknown orchestration action"*
> means the live build predates the code. CONCEPT:KG-2.7 / OS-5.5.

## No rebuild needed — the package is editable

`agent-utilities` is installed **editable** in the venv
(`/home/apps/workspace/.venv/.../agent_utilities` → the repo). Code changes are
already on the import path; redeploy = **restart the processes**, no `pip install`.

## The two live processes

| Process | Command (verbatim from the live host) |
|---|---|
| Compute daemon | `epistemic-graph-server --socket-path /tmp/epistemic-graph.sock` |
| Gateway/KG daemon (serves MCP) | `python -m agent_utilities.gateway.daemon` with the env below |

Live env (from the running daemon):

```bash
GRAPH_BACKEND=tiered GRAPH_BACKEND_L1=epistemic_graph \
GRAPH_DB_URI="postgresql://postgres:postgres@pggraph.arpa:5432/pggraph" \
GRAPH_SERVICE_SOCKET=/tmp/epistemic-graph.sock \
AGENT_UTILITIES_CONFIG_DIR=/home/genius/.config/agent-utilities \
WORKSPACE_PATH=/home/apps/workspace KG_FILE_WATCH=0 KG_EMBED_BACKFILL_BATCH=1000
```

## Restart procedure

1. **Pre-flight** — tests/gates green for the new code (this session's VUs).
2. **Graceful stop** of the gateway daemon (keep `epistemic-graph-server` running so
   the graph state persists — the gateway reconnects to the same socket):
   ```bash
   pkill -f "agent_utilities.gateway.daemon"
   ```
3. **Restart** with the same env (reuses the live socket; backfill off for a fast
   start):
   ```bash
   cd /home/apps/workspace/agent-packages/agent-utilities
   GRAPH_BACKEND=tiered GRAPH_BACKEND_L1=epistemic_graph \
   GRAPH_DB_URI="postgresql://postgres:postgres@pggraph.arpa:5432/pggraph" \
   GRAPH_SERVICE_SOCKET=/tmp/epistemic-graph.sock \
   AGENT_UTILITIES_CONFIG_DIR=/home/genius/.config/agent-utilities \
   WORKSPACE_PATH=/home/apps/workspace KG_FILE_WATCH=0 KG_EMBED_BACKFILL_BATCH=1000 \
   nohup /home/apps/workspace/.venv/bin/python -m agent_utilities.gateway.daemon \
     > /home/apps/workspace/reports/host-daemon.log 2>&1 &
   ```
4. **Verify the new action is live** (via the MCP multiplexer or directly):
   ```
   graph_orchestrate(action="assimilate")        # no longer "Unknown action"
   graph_orchestrate(action="assimilate", task="synthesize")
   ```
   Expect a JSON report (`skipped`/`duplicates_superseded`/`open_gaps`/…).
5. **Enable the autonomous loop** (optional) — restart with the golden-loop + breadth
   envs so the daemon tick runs the assimilation pipeline each cycle:
   ```bash
   KG_GOLDEN_LOOP=1 KG_GOLDEN_BREADTH=1 \
   KG_BREADTH_LIBRARY_ROOTS=/home/apps/workspace/open-source-libraries \
   KG_BREADTH_REPO_ROOTS=/home/apps/workspace/agent-packages
   ```

## Notes
- Restarting the gateway is **reversible** and does not drop graph state (that lives
  in `epistemic-graph-server` + pggraph). Only in-flight gateway requests are
  interrupted.
- If running in Portainer/swarm instead of bare host, redeploy the stack/service
  rather than `pkill` (the image picks up the editable mount on restart).
- Monitoring: after restart, each cycle persists an `EvolutionCycle`
  (`orchestration_cycle`) node — query `c.error_count` / `c.stage_ms` to confirm
  health.
