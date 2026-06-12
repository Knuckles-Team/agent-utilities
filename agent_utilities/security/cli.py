#!/usr/bin/env python
"""Secret Manager CLI.

CONCEPT:OS-5.1

Provides a command-line interface to manage secrets using the agent_utilities
SecretsClient. Supports setting, getting, deleting, and listing keys across
all backends (SQLite, Vault, InMemory).
"""

import argparse
import sys

from agent_utilities.core.config import setting
from agent_utilities.security.secrets_client import SecretsConfig, create_secrets_client


def main() -> None:
    """Entry point for the secret-manager CLI."""
    parser = argparse.ArgumentParser(description="Agent Utilities Secret Manager")
    parser.add_argument(
        "--backend",
        help="Backend to use (sqlite, vault, inmemory). Overrides SECRETS_BACKEND env var.",
    )
    parser.add_argument(
        "--sqlite-path",
        help="Path for sqlite backend. Overrides SECRETS_SQLITE_PATH env var.",
    )
    parser.add_argument(
        "--vault-url",
        help="URL for vault backend. Overrides SECRETS_VAULT_URL env var.",
    )
    parser.add_argument(
        "--vault-auth",
        help="Vault auth method: 'oidc', 'approle', 'token', 'kubernetes', 'auto'. Overrides VAULT_AUTH_METHOD.",
    )
    parser.add_argument(
        "--vault-auth-mount",
        help="Vault auth method mount path (e.g. 'jwt', 'oidc', 'my-okta-auth'). Overrides VAULT_AUTH_MOUNT.",
    )
    parser.add_argument(
        "--vault-role",
        help="Vault role for OIDC/JWT or Kubernetes login. Overrides VAULT_ROLE.",
    )
    parser.add_argument(
        "--vault-path-prefix",
        help="Path prefix within the KV v2 mount (e.g. 'agents/mcp/'). Overrides VAULT_PATH_PREFIX.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Set command
    set_parser = subparsers.add_parser("set", help="Set a secret")
    set_parser.add_argument("key", help="Key name (e.g., gitlab/token)")
    set_parser.add_argument("value", help="Secret value")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get a secret")
    get_parser.add_argument("key", help="Key name")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a secret")
    delete_parser.add_argument("key", help="Key name")

    # List command
    subparsers.add_parser("list", help="List all keys")

    args = parser.parse_args()

    # Build config overrides. We only pass values explicitly set on the CLI
    # so that the defaults (which read from os.environ) can take over.
    config_kwargs = {}
    if args.backend:
        config_kwargs["backend"] = args.backend
    elif setting("SECRETS_BACKEND"):
        config_kwargs["backend"] = setting("SECRETS_BACKEND")

    if args.sqlite_path:
        config_kwargs["sqlite_path"] = args.sqlite_path
    elif setting("SECRETS_SQLITE_PATH"):
        config_kwargs["sqlite_path"] = setting("SECRETS_SQLITE_PATH")

    if args.vault_url:
        config_kwargs["vault_url"] = args.vault_url
    elif setting("SECRETS_VAULT_URL"):
        config_kwargs["vault_url"] = setting("SECRETS_VAULT_URL")

    if getattr(args, "vault_auth", None):
        config_kwargs["vault_auth_method"] = args.vault_auth
    elif setting("VAULT_AUTH_METHOD"):
        config_kwargs["vault_auth_method"] = setting("VAULT_AUTH_METHOD")

    if getattr(args, "vault_auth_mount", None):
        config_kwargs["vault_auth_mount"] = args.vault_auth_mount
    elif setting("VAULT_AUTH_MOUNT"):
        config_kwargs["vault_auth_mount"] = setting("VAULT_AUTH_MOUNT")

    if getattr(args, "vault_role", None):
        config_kwargs["vault_role"] = args.vault_role
    elif setting("VAULT_ROLE"):
        config_kwargs["vault_role"] = setting("VAULT_ROLE")

    if getattr(args, "vault_path_prefix", None):
        config_kwargs["vault_path_prefix"] = args.vault_path_prefix
    elif setting("VAULT_PATH_PREFIX"):
        config_kwargs["vault_path_prefix"] = setting("VAULT_PATH_PREFIX")

    if config_kwargs:
        config = SecretsConfig(**config_kwargs)  # type: ignore[arg-type]
    else:
        config = None

    client = create_secrets_client(config)

    if args.command == "set":
        client.set(args.key, args.value)
        print(
            f"Successfully set secret: {args.key} in {client.backend.__class__.__name__}"
        )
    elif args.command == "get":
        val = client.get(args.key)
        if val is None:
            print(f"Secret not found: {args.key}", file=sys.stderr)
            sys.exit(1)
        print(val)
    elif args.command == "delete":
        success = client.delete(args.key)
        if success:
            print(
                f"Successfully deleted secret: {args.key} from {client.backend.__class__.__name__}"
            )
        else:
            print(f"Secret not found: {args.key}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "list":
        keys = client.list_keys()
        if not keys:
            print(f"No secrets stored in {client.backend.__class__.__name__}.")
        else:
            for k in keys:
                print(k)


if __name__ == "__main__":
    main()
