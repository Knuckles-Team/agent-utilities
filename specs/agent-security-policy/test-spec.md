# Test contract — AU-SEC-01

| Test | Requirements | Fixture / action | Expected result |
|---|---|---|---|
| AUTH-1 | 01/04 | Valid non-admin tenant JWT against local JWKS; read/write at gateway | Only declared capability executes; exact projected claims and audit. |
| AUTH-2 | 01/04 | Forged, expired, wrong-audience, unknown-class and cross-tenant claims | 401/403; no session or graph call. |
| LOCAL-1 | 02 | Packaged `tiny` stdio with no external identity; advance fake clock | Private session works then expires/renews; key/token never persisted. |
| LOCAL-2 | 02 | Network listener, external endpoint, or configured invalid identity | Local mint unavailable; no admin fallback. |
| CTRL-1 | 03 | Backend refuses `__control__` view | Typed unavailable; no content-graph write. |
| LEASE-1 | 05/06 | Requester and distinct approver, scoped lease, tool invocation | Active lease satisfies only named action/resource; audit includes trace. |
| LEASE-2 | 05/06 | Self-approval, agent issued grant, replay, revoked/expired lease, RBAC deny | Every protected action denied; approval cannot override RBAC. |
| EVOLVE-1 | 07 | Bounded tightening with evidence and current revision | Governed activation plus auditable old/new digest. |
| EVOLVE-2 | 07 | Loosening without approval, stale revision, absent bounds, policy outage | Old profile remains active; typed denial/conflict. |
| CACHE-1 | 08 | Invalidation, dropped event and learned high/low change rates | No value outlives configured TTL; learned rate never extends TTL. |
| DRIFT-1 | 09 | Compatible/incompatible connector schema with SDK fixture | Incompatible report quarantines; checkpoint unchanged until disposition. |

Unit tests run from a fresh clone without a live environment. Contract tests may start disposable local services. A live-path claim requires a provisioned engine/gateway and exact revision. The required quality evidence is `scripts/check_complexity_staged.py` (CCCC: no new cyclomatic >10 or cognitive >15 and no regression), `scripts/check_dupehound.py` (configured 0.85 similarity, 40 tokens), and the `clone-jscpd-diff` hook from `.config/pre-commit.yaml` (jscpd 5.0.16, 50 tokens, 5 lines), plus relevant Ruff, mypy, liveness, privacy, and repository all-files/hosted CI. The current gate configuration is authoritative if these commands change. KISS review must show reuse of current identity, policy, decision, and cache seams and removal of superseded paths. Record failed/unavailable gates accurately.
