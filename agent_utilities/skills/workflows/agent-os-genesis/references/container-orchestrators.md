# Compose, Swarm, and Podman

Resolve one logical service model, then render it for the selected runtime. Do not
pretend the runtimes have identical security or scheduling semantics.

## Common service contract

Every service declares an immutable image, command, non-secret environment,
secret-reference source, healthcheck, restart policy, resource bounds, ports,
networks, persistent volumes, dependencies, and rollback artifact.

Keep these networks distinct where possible:

- public ingress;
- internal application;
- restricted data;
- controlled egress.

Persistent engine data must have one writer unless the selected engine topology
explicitly provides distributed consensus.

## Docker Compose

Use for one durable host. Validate with `docker compose config --quiet`, then inspect
the fully rendered model for unresolved values and secret leakage. Use secrets or
runtime-mounted reference files instead of inline environment values. Capture the
prior image digest and config before `up -d`.

## Docker Swarm

Use for an existing multi-host Docker estate when Kubernetes is not desired. Before
initializing or joining, validate advertise/listen addresses, quorum, time sync,
firewall ports, overlay CIDRs, MTU, labels, and registry trust. Use replicated jobs or
services only where semantics permit; constrain stateful writers to compatible
storage. Store secrets in Swarm secrets or an external provider.

Never replace an overlay network in place without a staged migration. Preserve
manager quorum and test manager loss.

## Podman

Prefer rootless Podman for development and low-privilege single-user services.
Production system services normally use rootful Podman with Quadlet/systemd unless
the operator has designed rootless lingering, subuid/subgid, privileged-port, and
storage behavior. Validate generated units and reboot persistence.

Podman Compose compatibility is not proof of Docker Compose parity. Test health,
secrets, networking, volume labels/SELinux, and restart semantics on the selected
version.

## Gates

- offline render/config validation;
- no floating image tags in production;
- no plaintext credentials;
- healthcheck and dependency failure tests;
- resource and disk pressure behavior;
- restart/reboot persistence;
- external ingress and internal isolation;
- backup/restore;
- idempotent second apply;
- rollback to the prior exact image/config.
