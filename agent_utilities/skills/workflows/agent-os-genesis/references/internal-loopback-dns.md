# Internal-loopback DNS for `.arpa` — in-cluster clients hit the in-cluster Service

> **Principle.** An in-cluster client that resolves `<name>.arpa` must get the **in-cluster
> Service ClusterIP**, never the external edge VIP. This keeps all pod-to-pod traffic
> **internal** — off the ingress, with no extra TLS-termination/auth hop and no dependence on
> the edge staying up during a migration. It is the DNS-layer generalization of the
> `OIDC_TOKEN_URL`-pinning trick in [`graph-os-fleet-gateway-auth.md`](graph-os-fleet-gateway-auth.md):
> instead of pinning one URL per service to the in-cluster IdP Service, CoreDNS makes **every**
> eligible `.arpa` name resolve internally, so the stable `<name>.arpa` names in
> `MCP_CONFIG` / `MCP_HTTP_ALLOWED_PRIVATE_HOSTS` / connector configs "just work" without the
> edge in the path.

## The problem it fixes

Technitium (`10.0.0.199`) is authoritative for `.arpa` and answers **every** `.arpa` name with
the ingress VIP `10.0.0.240` (the `rke2-ingress-nginx` LoadBalancer). CoreDNS (`kube-system`,
`rke2-coredns-rke2-coredns`) forwards the `arpa` zone to Technitium
(`forward arpa 10.0.0.199`). So a **pod** resolving `github-mcp.arpa` gets `10.0.0.240` and
hairpins **through the ingress**:

```
pod ──▶ github-mcp.arpa ──DNS──▶ 10.0.0.240 (ingress VIP) ──▶ ingress-nginx ──▶ github-mcp Service ──▶ pod
        └─ extra hop: kube-proxy DNAT to an ingress pod, Host-routing, (for TLS hosts) TLS
           termination + the "edge 502 during migration" risk
```

The graph-os fleet gateway dials ~66 `*-mcp.arpa` children plus itself this way, so this hairpin
is the hot path for essentially all agent traffic. Internal-loopback DNS collapses it to:

```
pod ──▶ github-mcp.arpa ──DNS(rewrite)──▶ github-mcp.apps.svc.cluster.local ──▶ 100.65.4.63 (ClusterIP) ──▶ pod
```

External clients (browsers) are **unaffected** — they resolve `.arpa` against Technitium
directly and keep getting `10.0.0.240`. CoreDNS rewrites apply **only to in-cluster pods**.

## The eligibility rule (READ before adding a name) — port/scheme must match

DNS can only map a **name → IP**. It cannot change the **port or scheme** the client dials. So a
name is only safe to rewrite when the **in-cluster Service serves on the same port the client
uses on the `.arpa` name**. Two cases:

- **Plain-HTTP service, HTTP ingress (SAFE to rewrite).** The `*-mcp` fleet is dialed
  `http://<name>-mcp.arpa/mcp` on **port 80**, the in-cluster Service is **port 80**, and the
  `-mcp` ingresses do **not** terminate TLS — so `:80` before (via VIP) and `:80` after (direct
  ClusterIP) are identical. `graph-os.arpa` is the same (Service `:80`, in-cluster clients + its
  own health probe speak HTTP). Rewriting these is transparent.
- **TLS-terminated-at-ingress service (NOT safe to rewrite as-is).** `keycloak.arpa` is dialed
  `https://keycloak.arpa` (**:443**), but the in-cluster keycloak Service is plain **HTTP :8080**
  (no `:443`/`:8443` listener at all — verified: `curl https://<clusterIP>:443` and `:8443` both
  time out; `http://<clusterIP>:8080` returns 200). Pointing `keycloak.arpa` at the ClusterIP
  would make every `https://keycloak.arpa` JWKS/token fetch hit `ClusterIP:443` → **connection
  refused** → **fleet-wide inbound-JWT failure** (the documented "MCP fleet network contract
  break" SPOF). `openbao.arpa` is analogous (ingress maps `:80`→Service `:8200`, so a pod dialing
  `http://openbao.arpa` on `:80` would miss the `:8200`-only Service).

**To make a TLS/other-port service internal you must FIRST reconcile the client, then rewrite:**
either (a) switch the client to the in-cluster scheme+port on the `.arpa` name (e.g. fleet
`FASTMCP_SERVER_AUTH_JWT_JWKS_URI` → `http://keycloak.arpa:8080/...`; note the last-applied
`mcp-common-env` already used `http://keycloak.platform.svc:8080` for exactly this), **or**
(b) give the pod an in-cluster TLS listener (cert-manager cert mounted into keycloak on `:8443` +
a Service `:443/:8443` port) so `https://keycloak.arpa` stays valid against the ClusterIP. Both
are the **complementary config/deploy change**, out of scope for the DNS layer — until one lands,
leave `keycloak.arpa`/`openbao.arpa` on the edge (they still work, and the edge path is still
in-cluster via kube-proxy DNAT).

## The CoreDNS config

**Per-host `rewrite name exact` rules** — one per fleet child
(`<name>-mcp.arpa` → `<name>-mcp.apps.svc.cluster.local`) plus the non-uniform infra names
(`graph-os.arpa` → `graph-os.platform.svc.cluster.local`). **Use `exact`, not a `regex`**
(see the verification lesson below): `rewrite name exact` normalizes the trailing dot and
resolves reliably from a real glibc/`getaddrinfo` workload pod, whereas a `name regex` pattern
proved **fragile** under that resolver's trailing-dot + `ndots:5` search-domain semantics — it
passed `nslookup` but returned `gaierror` from the actual graph-os pod. The ~66 fleet rules are
**generated** from the live fleet (`kubectl -n apps get svc -o name | grep -- -mcp`) so a new
`-mcp` service just needs a regen, not a hand-edit. Rules run in the CoreDNS `rewrite` plugin,
which executes **before** `kubernetes` and `forward`, so the rewritten name is answered
authoritatively by the `kubernetes` plugin from the real ClusterIP (no hardcoded IPs — Service
IP changes tracked automatically).

### Before (original Corefile)

```
.:53 {
    errors
    health { lameduck 10s }
    ready
    kubernetes  cluster.local  cluster.local in-addr.arpa ip6.arpa {
        pods insecure
        fallthrough in-addr.arpa ip6.arpa
        ttl 30
    }
    prometheus  0.0.0.0:9153
    forward  arpa 10.0.0.199
    forward  . /etc/resolv.conf
    cache  30
    loop
    reload
    loadbalance
}
```

### After (added lines only — everything else byte-for-byte unchanged)

```
    ready
    # internal-loopback DNS: in-cluster <name>.arpa -> in-cluster Service ClusterIP.
    # EXACT rules (not regex) — trailing-dot-safe from a real getaddrinfo pod under ndots:5.
    # The 66 fleet lines are generated: kubectl -n apps get svc -o name | grep -- -mcp
    rewrite stop name exact ansible-tower-mcp.arpa ansible-tower-mcp.apps.svc.cluster.local
    rewrite stop name exact archimate-mcp.arpa     archimate-mcp.apps.svc.cluster.local
    # … one line per *-mcp Service in `apps` (66 total) …
    rewrite stop name exact github-mcp.arpa        github-mcp.apps.svc.cluster.local
    rewrite stop name exact wger-mcp.arpa          wger-mcp.apps.svc.cluster.local
    rewrite stop name exact graph-os.arpa          graph-os.platform.svc.cluster.local
    kubernetes  cluster.local  cluster.local in-addr.arpa ip6.arpa {
```

Why it is correct and safe:

- **`exact` is trailing-dot-safe — this is the whole reason it's exact, not regex.**
  `rewrite name exact` normalizes both sides to the FQDN form, so the pod's **absolute** query
  `github-mcp.arpa.` matches and rewrites, while the `ndots:5` **search-expanded** probes
  (`github-mcp.arpa.platform.svc.cluster.local.`, …) do **not** match and correctly fall through
  to `kubernetes` (NXDOMAIN) — so `getaddrinfo` lands on the absolute name and resolves to the
  ClusterIP. A `name regex` pattern (`(.*)-mcp\.arpa`) did **not** full-match the trailing-dot
  qname from a real pod → it fell through to `forward arpa` → `gaierror`, even though `nslookup`
  (absolute name, no search/ndots) made the regex look fine.
- **Reverse DNS untouched.** No rule references `in-addr.arpa`/`ip6.arpa`; those hit the
  `kubernetes` plugin (with `fallthrough`) then `forward arpa` exactly as before.
- **`exact` auto-restores the answer owner name**, so a strict client sees
  `Name: github-mcp.arpa → <ClusterIP>` (no separate `answer name` rule needed, unlike regex).
- **`stop`** halts rule evaluation after a match; match-sets are disjoint, so ordering isn't
  load-bearing, but `stop` is explicit and cheap.
- **Everything else is preserved:** `keycloak.arpa`, `openbao.arpa`, and all non-listed app names
  (`jellyfin.arpa`, `gitlab.arpa`, …) fall through to `forward arpa 10.0.0.199` → Technitium →
  `10.0.0.240` exactly as before; `.svc.cluster.local`, external `.`, and reverse DNS are intact.

> **⚠️ Verify from a REAL workload pod with `getaddrinfo`, not `nslookup`/`dig`.** `nslookup` and
> `dig` query the **absolute** name and skip glibc's `search` + `ndots:5` expansion and
> trailing-dot handling — so they can report success while the actual application resolver
> (`getaddrinfo`, what every Python/Go/curl workload uses) fails. That gap is exactly how the
> first `name regex` cut passed a `netshoot nslookup` check but returned `gaierror` from the
> graph-os pod. The acceptance test must be, from the real pod:
> ```bash
> GP=$(kubectl -n platform get pod -l app=graph-os -o jsonpath='{.items[0].metadata.name}')
> kubectl -n platform exec "$GP" -c graph-os -- python3 -c \
>   "import socket; print(socket.getaddrinfo('github-mcp.arpa',80)[0][4][0])"   # → 100.65.4.63
> ```

## Host → in-cluster Service map

Every `*-mcp` Service lives in `apps` and its name equals the host minus `.arpa`, so one
generated **`exact`** rule per child covers the whole fleet (host `==` Service name, ns `apps`,
port `80`). Infra names are non-uniform and get their own exact rule.

| `.arpa` name | in-cluster Service | ns | Service port | rewrite | status |
|---|---|---|---|---|---|
| `<name>-mcp.arpa` (all 66: `github-mcp`, `gitlab-mcp`, `keycloak-mcp`, `openbao-mcp`, `repository-manager-mcp`, `container-manager-mcp`, … `technitium-dns-mcp`) | `<name>-mcp` | `apps` | 80 (http) | **exact** (one per host, generated) → `<name>-mcp.apps.svc.cluster.local` | ✅ applied |
| `graph-os.arpa` | `graph-os` | `platform` | 80 (http) | **exact** → `graph-os.platform.svc.cluster.local` | ✅ applied |
| `keycloak.arpa` (IdP) | `keycloak` | `platform` | **8080 (http-only; ingress terminates :443)** | exact → `keycloak.platform.svc.cluster.local` | ⏸ **deferred** — needs client→in-cluster scheme/port first (see eligibility rule) |
| `openbao.arpa` | `openbao` | `platform` | **8200 (ingress maps :80→:8200)** | exact → `openbao.platform.svc.cluster.local` | ⏸ **deferred** — port mismatch; graph-os already uses `http://openbao.platform.svc:8200` directly |

> The `-mcp → apps` convention holds because **every** `*-mcp` Service lives in `apps` and its
> name equals the host minus `.arpa`. Infra services live in `platform` and are non-uniform, so
> they are exact rules. Ordinary app hostnames whose name ≠ Service name (`au.arpa`→`agent-webui`,
> `firefly.arpa`→`firefly-iii`, `ciso.arpa`→`ciso-assistant`, …) are **not** part of pod-to-pod
> agent traffic and are intentionally left on the edge; add an exact rule per service only if a
> real in-cluster caller needs one (and only if it passes the port/scheme eligibility rule).

## SPOF-safe apply + rollback (live cluster)

> ⚠️ **Cluster DNS is a SPOF.** Technitium (`10.0.0.199`) is the sole upstream for every node
> **and** CoreDNS; a CoreDNS **pod restart** once deadlocked the cluster. Therefore change the
> Corefile via the **ConfigMap + graceful `reload`** (the `reload` plugin is in the Corefile —
> CoreDNS re-reads the mounted file and reloads **in place, no restart**). **Never**
> `kubectl rollout restart` CoreDNS, and never change the Corefile through the HelmChartConfig on
> a live cluster (its pod template carries a `checksum/config` annotation, so a Helm-driven
> Corefile change **rolls the CoreDNS pods** — the deadlock risk).

```bash
# 0) Back up the current ConfigMap (this IS the rollback artifact)
kubectl -n kube-system get cm rke2-coredns-rke2-coredns -o yaml > /tmp/coredns-ORIGINAL.yaml

# 1) Apply the rewrite via a merge patch (YAML block scalar → no escaping of \  or {1})
#    patch.yaml holds:  data:\n  Corefile: |-\n    <the full After Corefile>
kubectl -n kube-system patch cm rke2-coredns-rke2-coredns --type merge --patch-file patch.yaml

# 2) Wait for reload — NO restart. kubelet syncs the mounted file (~60-90s), then the reload
#    plugin picks it up (~30s). Confirm from CoreDNS logs (safe: an invalid Corefile is REJECTED
#    and the running config is KEPT — reload never crashes a healthy CoreDNS):
kubectl -n kube-system logs -l k8s-app=kube-dns --tail=20 | grep -E "Reloading|reload|error"
#    → "[INFO] Reloading" / "Running configuration SHA512 = …" / "[INFO] Reloading complete"

# 3) Verify from a THROWAWAY pod (both replicas load-balance the Service, so poll until converged)
kubectl run dnsprobe --image=nicolaka/netshoot --restart=Never -n default --command -- sleep 3600
kubectl -n default exec dnsprobe -- getent hosts github-mcp.arpa   # → <ClusterIP>, NOT 10.0.0.240
kubectl -n default exec dnsprobe -- getent hosts one.one.one.one   # → external IP (still works)
kubectl -n default exec dnsprobe -- getent hosts graph-os.platform.svc.cluster.local  # → ClusterIP
kubectl -n default exec dnsprobe -- nslookup 100.65.0.1            # → kubernetes.default… (reverse ok)
kubectl -n default delete pod dnsprobe

# ROLLBACK (any doubt): re-apply the backup; reload reverts in place, still no restart.
kubectl -n kube-system apply -f /tmp/coredns-ORIGINAL.yaml
```

**Verify BEFORE and AFTER, for all three name classes** — a fleet `.arpa`, an external name
(`one.one.one.one`), and a `*.svc.cluster.local` — to prove you broke neither external nor
cluster-internal resolution, and do the fleet `.arpa` check with **`getaddrinfo` from a real
workload pod** (see the callout above), not just `nslookup`. Add the reverse-DNS check
(`nslookup <a-cluster-IP>`) since the rewrite rules sit in front of the reverse zone.

## Genesis (fresh environment) — make it durable from day 0

On a **greenfield** cluster there is no live traffic to protect, so bake the rewrite into the
**RKE2 `rke2-coredns` HelmChartConfig** (`kube-system`) as structured `servers[].plugins` — this
is the durable source of truth (a manual ConfigMap edit is reverted on the next Helm reconcile /
node reboot). Insert the per-host `rewrite` plugins **before** `kubernetes` (generate the fleet
entries from the live Services — do not hand-maintain 66 lines):

```yaml
# HelmChartConfig/rke2-coredns  (spec.valuesContent, under the existing servers[0].plugins list)
# Generate the fleet block:
#   kubectl -n apps get svc -o name | grep -- -mcp | sed 's|service/||' | while read s; do
#     printf '      - name: rewrite\n        parameters: stop name exact %s.arpa %s.apps.svc.cluster.local\n' "$s" "$s"
#   done
      - name: ready
      - name: rewrite
        parameters: stop name exact github-mcp.arpa github-mcp.apps.svc.cluster.local
      # … one EXACT rule per *-mcp Service in `apps` (66 total) …
      - name: rewrite
        parameters: stop name exact wger-mcp.arpa wger-mcp.apps.svc.cluster.local
      - name: rewrite
        parameters: stop name exact graph-os.arpa graph-os.platform.svc.cluster.local
      - name: kubernetes
        parameters: cluster.local in-addr.arpa ip6.arpa
        configBlock: |-
          pods insecure
          fallthrough in-addr.arpa ip6.arpa
          ttl 30
```

> **EXACT, not regex** (same lesson as the live config): `rewrite name exact` is trailing-dot-safe
> from a real `getaddrinfo` workload pod under `ndots:5`; a `name regex` cut passed `nslookup` but
> `gaierror`'d from the graph-os pod.

Genesis ordering: do this in the **DNS/ingress step**, right after Technitium is repointed to the
ingress VIP and the CoreDNS `forward arpa <technitium>` is in place — so the moment the fleet
comes up, its `<name>-mcp.arpa` / `graph-os.arpa` traffic is internal. For a fresh environment,
also set the fleet's IdP endpoints to the **in-cluster scheme+port** from the start
(`FASTMCP_SERVER_AUTH_JWT_JWKS_URI = http://keycloak.arpa:8080/...` **or** stand keycloak up with
an in-cluster TLS listener), which makes `keycloak.arpa`/`openbao.arpa` eligible for the same
exact-rule treatment — completing "every `.arpa` is internal".

## Relationship to the graph-os self-dial code fix

This is the **DNS half** of keeping graph-os traffic internal. The **code half** (graph-os not
HTTP-dialing *itself* over the network — using the in-process path instead) is separate and
complementary. Even before that lands, this rewrite already improves the self-path: a graph-os
HTTP self-dial to `graph-os.arpa` now resolves to the ClusterIP and stays in-cluster instead of
hairpinning the edge.
