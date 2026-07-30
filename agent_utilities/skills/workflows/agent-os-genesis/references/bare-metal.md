# Bare-metal and systemd

Bare metal is a supported production target, not shorthand for an interactive shell.

## Layout

- dedicated unprivileged service account;
- immutable release or virtual environment outside the source checkout;
- configuration under the platform configuration hierarchy;
- persistent engine state under the platform state hierarchy;
- logs through journald/OpenTelemetry rather than ad-hoc files;
- secrets resolved at runtime from protected credentials or a secret provider;
- a writable temporary/cache directory separate from the release.

Use platform-native paths rather than embedding one user’s home directory.

## Unit requirements

Generate one unit per independently managed process. Include:

- explicit `User`, `Group`, `WorkingDirectory`, and executable path;
- `After`/`Wants` only for real dependencies;
- bounded restart backoff and start timeout;
- graceful stop timeout long enough for an engine checkpoint;
- resource accounting and file descriptor limits;
- `NoNewPrivileges`, private temporary storage, protected system/home paths, and a
  minimal writable-path allow-list;
- environment/config files that contain references, not secret values;
- readiness exposed through the application health endpoint.

Validate with `systemd-analyze verify` and run under the final service identity.

## Host preparation

Check architecture, supported Python/runtime versions, filesystem and free space,
IOPS, memory pressure, cgroups, NTP, DNS, CA trust, firewall, outbound provider
access, and backup destination. Do not install packages or change firewall rules
outside the approved plan.

## Upgrade and recovery

Install releases side by side, stop accepting work, checkpoint, switch an atomic
current-release reference, start, and verify. Preserve the prior release and data
snapshot until the user-visible path passes. Roll back binaries/config separately
from data migrations; never downgrade a data format without a supported migration.
