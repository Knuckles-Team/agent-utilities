#!/usr/bin/env bash
# Schema-validate compose files with inert placeholder values.
# See the docker-compose-check hook comment in .pre-commit-config.yaml.
set -euo pipefail

env_file="docker/.env.validation"
if [[ ! -f "$env_file" ]]; then
  echo "missing $env_file (needed to interpolate required vars)" >&2
  exit 1
fi

status=0
for file in "$@"; do
  if ! output=$(docker compose --env-file "$env_file" -f "$file" config 2>&1 >/dev/null); then
    echo "ERROR: $file" >&2
    echo "$output" | sed 's/^/  /' >&2
    status=1
  fi
done
exit "$status"
