# Day-0 / Swarm Troubleshooting Runbook

Hard-won runbook from real incidents recovering the Heaven Homestead swarm. Symptom →
diagnosis → fix. Topology: manager **R820 (10.0.0.13)**; workers R510(.10) R710(.11)
RW710(.12) GR1080(.16) GB10(.18); DNS = Technitium macvlan **10.0.0.199**; ingress = Caddy
on R820 (host-mode 80/443); ~76 stacks, GitOps-bound to `http://gitlab.arpa`.

> Access notes: passwordless `sudo` works on **R820 only** (NOT R710/R510 — those need the
> operator). Drive everything over the **tunnel-manager MCP** (`tun__tm_remote run_command`,
> host=IP, user=genius, id_file=~/.ssh/id_rsa) — always pass a short `timeout`. During any DNS
> outage, the Portainer/Caddy/Technitium MCPs (which target `*.arpa`) go dark; only tunnel-manager
> (IP-based) survives.

---

## 1. Swarm reports "no leader" / quorum lost

**Symptom:** `docker node ls` → `rpc error: ... swarm does not have a leader`.
**Diagnose:** `docker info | grep -A3 'Manager Addresses'` and `docker node inspect self --format '{{.ManagerStatus.Addr}}'`. Check whether a listed manager IP still exists: `ip -4 addr show | grep <ip>`.
**Root cause seen:** the manager (R820) was renumbered (`.136`→`.13`); raft still advertised the dead `.136`, so quorum couldn't form.
**Fix (surgical, keeps services running):**
```bash
# On the surviving manager ONLY (never two nodes):
docker swarm init --force-new-cluster --advertise-addr 10.0.0.13
```
This restores a single-manager leader with all services intact.

## 2. force-new-cluster did NOT change the advertise address

**Symptom:** after force-new-cluster, `docker node inspect self --format '{{.ManagerStatus.Addr}}'` still shows the OLD/dead IP; workers can't reconnect (stay `Down`); `docker info` on a worker shows `Manager Addresses: <dead ip>`.
**Cause:** Docker cannot rebind a manager's advertise-addr in place; force-new-cluster preserves it.
**Fixes:**
- **Zero-downtime:** re-add the old IP as an alias so the advertised address resolves to the manager again:
  ```bash
  sudo ip addr add 10.0.0.136/8 dev eno1   # NOT reboot-persistent — persist via netplan/systemd
  ```
- **Clean (downtime):** full swarm re-init with `--advertise-addr 10.0.0.13` (only way to truly drop the old IP).

## 3. Workers `Down` after a manager change

**Fix:** on each worker, `docker swarm leave --force` then rejoin with a FRESH token:
```bash
# manager: docker swarm join-token -q worker   (token rotates on re-init!)
docker swarm leave --force; sleep 3
docker swarm join --token <FRESH> 10.0.0.13:2377
```
Workers connect via the listen addr (`0.0.0.0:2377`), so `10.0.0.13:2377` works even if the advertise addr is wrong. After rejoin, remove stale `Down` duplicates and relabel:
```bash
docker node ls --format '{{.ID}}|{{.Status}}' | awk -F'|' '$2=="Down"{print $1}' | xargs -r docker node rm --force
for n in $(docker node ls --format '{{.Hostname}}'); do docker node update --label-add name=$n $n; done
```

## 4. `*.arpa` resolves to a dead IP

**Diagnose:** `td__zones export_zone` (zone=arpa). Look for the **wildcard `*`** record and explicit overrides.
**Fix:** repoint via `td__zones update_record` (params: domain, zone, type=A, ipAddress=OLD, newIpAddress=NEW). The wildcard `*.arpa` covers every name without an explicit record — fixing it fixes most services at once.

---

## 5. ⚠️ DO NOT do an in-place `caddy` overlay subnet change on a live swarm

**This caused a multi-hour partial outage. Change shared-overlay subnets ONLY via a fresh network on a re-init/maintenance window — never by mass detach/reattach of a live network.**

**What happens if you try** (`docker service update --network-rm/--network-add` across all attached services):
- `docker network rm caddy` **fails** ("has active endpoints") even after the manager shows 0 endpoints — a swarm stale-endpoint race across nodes; retries don't help.
- The mass detach/reattach **corrupts the overlay's service-VIP load-balancer (IPVS) state**: ~N services end up running-but-unreachable via Caddy (502).

**The 502 signature (and how to tell it's the LB/VIP, not the app or the network):**
```bash
CC=$(docker ps --format '{{.Names}}' | grep -i caddy | head -1)
docker exec $CC ping -c2 <svc_vip>        # L3 works (0% loss)  → overlay routing OK
docker exec $CC wget -S -O /dev/null http://<svc_vip>:<port>/   # TCP to VIP TIMES OUT
# On the backend's node: app is healthy locally:
docker exec <task> wget -qO- -S http://127.0.0.1:<port>/        # HTTP/1.1 200 OK
cat /sys/class/net/eth*/mtu                                     # caddy iface MTU == network MTU (rule out MTU)
```
App healthy + ping OK + TCP-to-VIP times out + MTU correct = **stale overlay VIP/IPVS state**.

**What does NOT fix it** (all tried and failed): Caddy restart; `docker service update --force <svc>`; docker restart on the manager AND backend nodes; node leave/rejoin + relabel; per-service serial `--network-rm`/`--network-add`.

**What DOES fix it (deterministic):**
- **Recreate the `caddy` network fresh** — clears all VIP/IPVS state. Requires `docker network rm caddy` to succeed (only reliable after all attached services are gone, e.g. during a stack-wide redeploy or swarm re-init).
- **Swarm teardown + re-init** + GitOps redeploy (guaranteed; biggest outage).

**Prereq before any teardown:** verify the GitOps redeploy path works FIRST — 73/76 stacks redeploy from `http://gitlab.arpa` with auth. If Portainer git-redeploy errors (500), a teardown is unrecoverable. Confirm `gitlab.arpa` git endpoint reachable: `curl -s -o /dev/null -w '%{http_code}' -L http://gitlab.arpa/<group>/<repo>.git/info/refs?service=git-upload-pack` → expect `401` (reachable, needs auth).

---

## 6. Caddy ingress reachability quick test
```bash
for h in portainer gitlab grafana <svc>; do
  printf "%-14s " "$h.arpa"
  curl -s -m6 -o /dev/null -w "%{http_code}\n" -H "Host: $h.arpa" http://10.0.0.13/
done   # 2xx/3xx/401/403 = reachable; 502 = Caddy up but upstream unreachable; 000 = ingress down
```

## 7. tunnel-manager MCP
Always pass a short `timeout` (e.g. 10). The hardened client (bounded connect/banner/auth
timeouts, retry+backoff, keepalive, bounded exit-status) fails fast on dead hosts instead of hanging.

---

## 8. Validate an MCP connector the RIGHT way (tool-level, not just reachable)

`initialize` succeeding only proves the server is **up + auth passes** — it hides bugs that
only fire on a real tool call (eunomia middleware, missing module, bad URL). Validate with a
**full host-side MCP session** carrying the multiplexer's A0 token, with short timeouts (never
hang the session):

```bash
# 1) mint the A0 token (client_credentials) exactly like the multiplexer does
TOK=$(curl -s -m10 -X POST "http://keycloak.arpa/realms/master/protocol/openid-connect/token" \
  -d grant_type=client_credentials -d client_id=mcp-multiplexer -d client_secret="$SECRET" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
# 2) initialize -> capture mcp-session-id -> notifications/initialized -> tools/list -> tools/call
#    Accept: application/json, text/event-stream ; Authorization: Bearer $TOK
#    A 401 unauth + 200 with token = jwt enforced + A0 works.
```

Sweep the whole fleet from `mcp_config*.json` URLs this way (8s timeout each) to get a real
up/down + per-tool matrix. **Through the live multiplexer**, prefer a non-disruptive
`multiplexer_status` first; mux tool calls can hang 300s on a stale child session (see §12).

---

## 9. Tool calls fail with `'FunctionTool' object has no attribute 'enabled'` (eunomia ↔ fastmcp 3.x)

**Signature:** auth + `initialize` + `tools/list` all succeed, but every `tools/call` on a
**eunomia-enforced** server returns `Internal error: 'FunctionTool' object has no attribute
'enabled'`. Servers WITHOUT `EUNOMIA_TYPE` work fine.

**Cause:** `eunomia-mcp` (≤0.3.10) gates each call on `component.enabled` — a **fastmcp 2.x**
attribute. **fastmcp 3.x removed it** (every registered component is live), so the access raises.
This silently breaks the *entire* jwt+eunomia rollout once images carry fastmcp 3.x.

**Fix (in agent-utilities):** `apply_fastmcp_enabled_compat()` in
`agent_utilities/mcp/eunomia_principal.py` sets `FastMCPComponent.enabled=True` (3.x semantics);
applied at import (embedded/JWT path) and from the server-factory eunomia block (remote path).
Reaches a deployed server only after an agent-utilities release + image rebuild (or, for
editable services that `pip install` at start, a restart). **Validate every eunomia service at
the tool-call level (§8) after flipping `AUTH_TYPE=jwt`/`EUNOMIA_TYPE=remote`.**

---

## 10. Deployed connector ignores its env (inert-env pattern)

A deployed `-mcp` container only receives env the **compose actually passes**. Setting a var in
the Portainer **stack Env** does nothing unless the compose `environment:` interpolates it
(`- MY_VAR=${MY_VAR}`). Symptom: connector behaves as if a token/URL is unset though the stack
Env shows it. **Fix:** add both — the value in stack Env AND `- VAR=${VAR}` in the compose, then
redeploy. Keep `HOST`/`PORT`/`TRANSPORT` as the deployment sets them; don't let dev `.env`
values override. (Connector URL/token var NAMES vary: e.g. caddy reads `CADDY_URL`, technitium
`TECHNITIUM_DNS_URL`, erpnext `ERPNEXT_URL`, dockerhub `DOCKER_HUB_TOKEN`/`DOCKER_HUB_USER`.)

---

## 11. Service crash-loops → 502 from a healthcheck PORT mismatch

**Signature:** swarm tasks cycle `running`→`complete`/`shutdown` every few minutes; logs show a
clean `Application startup complete` immediately followed by `Shutting down`; `*.arpa` = 502.
**Cause:** the compose **healthcheck probes the wrong port** (e.g. `localhost:8026/health` while
the server listens on `PORT=8000`) → healthcheck never passes → swarm kills + restarts.
**Fix:** align the healthcheck to the container's `PORT`. Prefer a port-only check that doesn't
depend on a route existing on older images:
`python3 -c "import socket; socket.create_connection(('localhost', 8000), timeout=5)"`
(use `/health` only when the image ships agent-utilities ≥ the B1 build that adds the route).
Generators must derive the healthcheck port from `PORT`, not a per-service offset.

---

## 12. Multiplexer tool call hangs ~300s after you restarted a child container

**Signature:** a tool call through the multiplexer hangs to its `call_timeout` (300s) and
returns `MCPChildCallTimeoutError ... did not answer`, even though the child is healthy when
probed directly (§8 returns 200 in ~10ms). **Cause:** the live mux holds a **stale streamable-
HTTP session** to a child you just redeployed; the child dropped the session, the mux waits
forever to reuse it. **Fix:** `/mcp reconnect` (re-spawns the mux: clears stale sessions, reloads
A0 env from `~/.claude.json`, and mounts any newly-added children). In **dynamic** mode
`multiplexer_status` shows only the **mounted** subset (~a dozen), NOT all configured children —
that is expected, not a missing-config bug. After restarting containers, reconnect before
re-validating through the mux.

---

## 13. New connector won't deploy / appear in the fleet

- **Only a `compose.dev.yml`, no Portainer stack** → the service was never deployed. Create the
  swarm stack (POST `/api/stacks/create/swarm/string`, `endpointId=3`) from a compose mirroring a
  working sibling (baked image + `/src` bind + `PYTHONPATH=/src` + `node.labels.name==RW710`).
- **Image name ≠ service name.** Some packages publish under the API name, not `<svc>-mcp` (e.g.
  `agents/dockerhub-api` → image `knucklessg1/dockerhub-api:latest`, console script
  `dockerhub-mcp`). Check the package's own `docker/*.compose.yml` for the real `image:`.
- **Add it to `mcp_config*.json`** (and reconnect) so the multiplexer mounts it.
- **Connector needs its working data mounted.** Tools that operate on host state need the data
  bind-mounted into the editable container: repository-manager needs the workspace
  (`/home/apps/workspace` + `WORKSPACE_PATH`); container-manager/tunnel-manager need
  `~/.config/agent-utilities/inventory.yaml` + `~/.ssh`. A missing **dependency module**
  (e.g. container-manager importing `tunnel_manager`) is a packaging fix — add the dep + rebuild;
  a volume mount alone won't satisfy an absent import.
