# arr-stack VPN hardening — gluetun namespace + fail-closed kill-switch

How the **arr-suite** (Step 11, tier **T6**, node **R510**) must be deployed so torrent/indexer
traffic egresses **only** through the VPN and is **blocked** (not leaked) whenever the tunnel is
down. This is the source-of-truth recipe; the live config lives in the `arr-stack` service repo
(`compose.vpn-hardened.yml` + `AGENTS.md`).

## Why (incident 2026-06-16)
An ISP notice was traced to qBittorrent egressing via the **host public IP**. Root cause was the
old **route-override** design: a `lhns/vpn-gateway:/elevate` sidecar rewrote the `vpn` overlay's
default route to a launcher-run gluetun. That is **not fail-closed**:
- **Startup race** — before the sidecar rewrites the route, containers default-route via Swarm's
  `docker_gwbridge` → host public IP; qBittorrent auto-resumes torrents and announces immediately.
- **No app-level kill-switch** — gluetun's firewall only protects traffic *inside its own
  namespace*; the arr containers live in separate namespaces, so if gluetun drops they fall back
  to the gwbridge and leak.
- **DNS leak** — arr containers used Docker's embedded resolver, not the VPN DNS.

Empirically: qBittorrent exit IP == host IP, default route via `docker_gwbridge`, gluetun firewall
irrelevant.

## The hard constraint
**Docker Swarm does NOT support `network_mode: service:`/`container:`.** That is the entire reason
the fragile route-override existed. A real kill-switch requires every protected app to share
gluetun's network namespace — which means running the arr-suite as a **standalone `docker compose`
on R510**, not as a Swarm stack. All arr services are already pinned to R510, so Swarm scheduling
buys nothing here.

## Target design (fail-closed)
- **gluetun** runs directly (qmcgaw/gluetun, `cap_add: NET_ADMIN`, `devices: /dev/net/tun`),
  single-homed on the compose default bridge for a **deterministic WAN route** (do NOT attach it to
  multiple overlays — multi-homing reintroduces the default-route ambiguity that caused the leak).
- `FIREWALL=on` — the **fail-closed kill-switch**: no egress unless the tunnel is up.
- `FIREWALL_INPUT_PORTS=<all WebUI ports>` and `FIREWALL_OUTBOUND_SUBNETS=<LAN CIDR>,<swarm overlay CIDRs>`
  (take the LAN CIDR from the operator's network / `inventory.yaml`; overlay CIDRs from the
  networking contract) keep the WebUIs reachable and LAN/overlay management working even when the
  tunnel is down.
- `DNS_ADDRESS=<VPN DNS>` kills the DNS leak.
- **Every other arr service** uses `network_mode: "service:gluetun"` (no own `networks:`/`ports:`),
  `depends_on: { gluetun: { condition: service_started } }` (NOT `service_healthy` — else the apps
  never start while the tunnel is down and you lose management access).
- **gluetun publishes every WebUI port** on its node's host; **Caddy** reverse-proxies each `*.arpa`
  host to `<arr-stack-node-ip>:<port>` — resolve that node's IP from `inventory.yaml` (the node the
  arr-stack is pinned to) rather than hardcoding — replacing the old `arr-stack_<svc>:<port>`
  swarm-service upstreams.
- Drop `add-vpn-gateway` and the `gluetun-launcher` — obsolete under namespace sharing.
- In the shared namespace the apps reach each other on `localhost`; the VPN DNS won't resolve
  internal `*.arpa`, so cross-app URLs (e.g. unpackerr → sonarr/radarr) must use `http://localhost:<port>`.

## NordVPN credentials
- **OpenVPN** (works today): `VPN_SERVICE_PROVIDER=nordvpn`, `VPN_TYPE=openvpn`, `OPENVPN_USER`/
  `OPENVPN_PASSWORD` = NordVPN **service credentials** (dashboard → Manual setup → Service
  credentials — NOT the account login). `SERVER_COUNTRIES` must be a **literal** (`United States`);
  the old `.env` used a swarm-launcher space hack `United@_@States` which gluetun rejects (crash on
  "country is not valid").
- **WireGuard / NordLynx** (more reliable; optional upgrade): `VPN_TYPE=wireguard` +
  `WIREGUARD_PRIVATE_KEY=<NordLynx key>`. The key comes from NordVPN (access token via the API, or
  `nordvpn`/`wg show nordlynx private-key` on a logged-in machine) — a self-hosted **wg-easy**
  server is unrelated and cannot supply it. Deploy gluetun in a mode that *stays up* (OpenVPN, or a
  valid WG key) — a WireGuard container with a missing/invalid key exits and takes the
  namespace-sharing apps down with it.

## arr MCP gotchas (Step A3/A5 fleet)
- **arr-mcp** must carry every service key: `BAZARR_API_KEY`, `CHAPTARR_API_KEY`, `SEERR_API_KEY`
  are easy to leave blank (they were). Pull keys from each app's config on R510
  (`config.xml`/`config.yaml`/`settings.json`).
- **qbittorrent-mcp**: WebUI auth is **username/password only** (no API key). Point
  `QBITTORRENT_HOST` at the reachable proxy host (`qbittorrent.arpa`), not `127.0.0.1`.
- **Editable au skew (the fleet-wide one):** an MCP whose `/src` imports a symbol newer than the
  image's installed `agent-utilities` (e.g. `run_blocking`) crash-loops with `ImportError`. Fix:
  mount the local au source `-v …/agent-utilities:/au:ro` and set `PYTHONPATH=/au:/src`.
- **chaptarr** runs as uid **99:100** with auth=Basic on **port 8789**; keep its `/config` owned by
  99:100 or it crash-loops on the PID-file write.

## Intra-namespace wiring (REQUIRED — collapsing to one netns breaks every swarm-name ref)
Once all services share gluetun's namespace, the swarm service names (`sonarr`, `qbittorrent`,
`flaresolverr`, `jellyfin`, …) stop resolving (NordVPN DNS can't see them). Re-point every
cross-reference:
- **Intra-stack → `localhost`**: *arr download client → `localhost:8080`; prowlarr Applications
  `baseUrl`+`prowlarrUrl` → `http://localhost:<port>`; prowlarr FlareSolverr proxy →
  `http://localhost:8191`; seerr Sonarr/Radarr → `localhost`; unpackerr → `http://localhost:<port>`.
- **External-but-internal dep → LAN IP** (never `*.arpa`): seerr→Jellyfin uses the Jellyfin node's
  LAN IP (from `inventory.yaml`)`:8096`,
  and Jellyfin (separate stack) must **publish host port 8096** so namespaced apps reach it by IP
  (gluetun `FIREWALL_OUTBOUND_SUBNETS` permits the LAN hop; the app's external traffic still
  egresses via VPN). Apps that need *many* internal services (e.g. homarr) should stay OUT of the VPN.

## ⭐ Bind the BitTorrent client to the VPN interface (`tun0`)
Set qBittorrent → Network Interface = **`tun0`** (`current_network_interface=tun0`). **Mandatory** —
without it, libtorrent's UDP (DHT/UDP-trackers/peers) takes the default `eth0` route and gluetun's
`OUTPUT DROP` kill-switch rejects it (**"Operation not permitted"**, DHT 0 nodes, torrents
`stalledDL`/0 peers) while HTTPS trackers still work — a confusing half-broken state. Binding to
`tun0` routes all client traffic through the tunnel (`-A OUTPUT -o tun0 -j ACCEPT`).

## Download path mapping
qBittorrent saves to `/downloads/complete` but the *arr apps mount the same downloads volume at
`/data`. Add a Remote Path Mapping in each *arr: host=download-client host (`localhost`),
remote=`/downloads/`, local=`/data/` — else *"directory does not appear to exist inside the
container."*

## qBittorrent-agent (MCP) gotchas (supersedes the qbittorrent-mcp note above)
- Reads **`QBITTORRENT_URL`** (e.g. `http://qbittorrent.arpa`), NOT `QBITTORRENT_HOST`/`PORT`.
- qBittorrent **5.x** endpoints: `torrents/start`|`stop` (not `resume`|`pause`).
- Tool return annotations must be `-> Any` (not `-> dict`) or non-dict results trip FastMCP's
  `structured_content must be a dict`.

## Verify
Resolve `$NODE` (the arr-stack node) and `$USER` from `inventory.yaml` — never hardcode:
```
ssh $USER@$NODE 'docker exec gluetun wget -qO- http://localhost:8000/v1/publicip/ip'  # VPN IP
ssh $USER@$NODE 'docker exec gluetun wget -qO- ifconfig.me'   # must NOT equal host public IP
# kill-switch: stop gluetun -> dependent apps have NO network (fail-closed), recover on start
# downloads working: docker exec <qbit> ... transfer/info -> connection_status=connected, dht_nodes>0
```
