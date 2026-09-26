# Recipe — Split-storage engine (fast-storage flavor)

A deployment **flavor** that places the `epistemic-graph` engine on the fleet's
**fastest-disk node** (NVMe/SSD) and connects graph-os + the host daemon to it over **TCP**,
instead of co-locating everything on one host. Config-only — no code fork.

## When to use it
The durable write path is **fsync-bound**. In the recorded benchmark, HDD fsync
p99 near 460 ms built a writer backlog while NVMe was near 3.2 ms, a roughly
143× tail-latency cut. If your fastest disk and your most cores/RAM are on **different** hosts (common in a
heterogeneous homelab/cluster), this flavor lets the engine sit on the fast disk while compute
stays where the cores are. It is also the unit that later composes into a **sharded cluster** —
each shard-engine on its own fast disk, clients routing by tenant.

## How it works
- The engine binds a **TCP listener** (`--tcp-addr`, published `:9100`) in addition to its local UDS.
- graph-os/host resolve the engine via
  `GRAPH_SERVICE_ENDPOINTS=tls://<engine-host>:9100`. A configured endpoint list
  is connect-only; empty selects the packaged local lifecycle.
- The persist dir lives on the fast-storage node's NVMe (`ENGINE_PERSIST`), the only thing that
  must be on the fast disk. The source tree / compute can stay elsewhere.

## Config surface
All knobs live in `services/epistemic-graph/flavors/split-storage.env` (parameterized into both
composes; defaults reproduce the original single-host config, so the flavor is **reversible** —
unset and redeploy):

| Var | Side | Meaning |
|---|---|---|
| `SERVER` | engine | placement node label (the fast-storage host) |
| `ENGINE_PERSIST` | engine | persist dir on the NVMe/SSD |
| `ENGINE_BIN` | engine (dev) | host-built binary path on that node (prod bakes it into the image) |
| `ENGINE_TCP_ADDR` | engine | `0.0.0.0:9100` — enable the TCP listener |
| `GRAPH_SERVICE_ENDPOINTS` | clients | `tls://<engine-host>:9100` for the host daemon + MCP dial |

## Dev vs prod
- **Dev (source-mounted, `compose.dev.yml`):** the engine binary is bind-mounted from a host path
  (`ENGINE_BIN`) on the fast-storage node — `cargo build` + redeploy is live.
- **Prod (pre-built image):** the binary is **baked into the registry image**
  (`${CONTAINER_REGISTRY}/agent-utilities`), so `ENGINE_BIN` is unused — drop the binary bind-mount, set the
  image tag, keep `SERVER` / `ENGINE_PERSIST` / `ENGINE_TCP_ADDR` / `GRAPH_SERVICE_ENDPOINTS`. Same
  flavor, image-based. (Genesis can offer this as a placement option in `single-node-prod` /
  `enterprise` profiles: "put the engine on the fastest-disk node.")

## Cutover (one-time, ~1–2 min downtime, reversible)
1. Stage the engine binary on the fast node (`ENGINE_BIN`); create `ENGINE_PERSIST` on its NVMe.
2. Stop the engine (`docker service scale epistemic-graph_epistemic-graph=0`).
3. `rsync` the `*.redb` + `.annidx` from the old persist dir → the NVMe `ENGINE_PERSIST`.
4. `source split-storage.env`; `docker stack deploy` epistemic-graph (now on the fast node, TCP).
5. `docker stack deploy` graph-os (clients pick up `GRAPH_SERVICE_ENDPOINTS`).
6. Verify: engine healthy on the fast node, `graph_search` returns, ingestion drains. Revert =
   unset the vars + redeploy (old data is untouched on the original node).

## Security
The engine and clients share `GRAPH_SERVICE_AUTH_SECRET`; every request carries
the current verified authority envelope. Routable TCP must use a configured
TLS/mTLS profile and must not be exposed as plaintext, even on a trusted LAN.
