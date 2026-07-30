# Graph-OS fleet gateway authentication

Graph-OS accepts a user/workload identity and may call downstream MCP servers with a
service or delegated identity. Treat those as separate trust decisions.

## Inbound

Validate issuer, audience, signature/JWKS rotation, time bounds, tenant, subject,
roles/scopes, policy version, and transport TLS. Static tokens are suitable only for
bounded deployments and must still be referenced from a secret provider.

## Outbound

Prefer workload identity or client credentials scoped to the destination. Use a
cluster-internal token endpoint/service route where appropriate, but validate that
its issuer and certificate semantics match external verification. A failed token
mint must fail closed and appear as an attributed trace/error; never silently call a
protected child without authorization.

Bind every tool call to:

- authenticated caller and tenant;
- selected server/tool and trusted registration;
- allow-list and consent/approval decision;
- validated arguments and idempotency key;
- downstream identity and permission;
- redacted result/error and correlated trace.

Do not trust tool annotations received from an MCP server to grant permission.

## Validation

Test valid, expired, wrong-audience, wrong-tenant, revoked/rotated, missing, and
insufficient-scope identities. Verify that caches include the tenant/policy boundary
and cannot return another caller’s response.
