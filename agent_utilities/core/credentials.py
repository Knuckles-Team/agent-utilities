"""CONCEPT:AU-ORCH.adapter.three-tier-credential-resolution — Four-tier credential resolution
(secret_ref > env > file > none).

Assimilated from open-design's ``media-config`` resolution order: a provider's API key/base-url is
resolved from the process environment first (CI/Docker/systemd), then a JSON config file (packaged or
GUI-persisted installs), then ``None``. No secrets in argv/logs; no separate vault required (though a
vault can populate the env tier).

CONCEPT:AU-ORCH.adapter.openai-catalog-verification adds a HIGHER-precedence tier: a provider whose
AgentConfig carries a ``<provider>_api_key_ref`` field (see ``_REF_CONFIG_FIELDS``) resolves through
the SAME ``env://``/``vault://``/``secret://`` runtime-reference convention already used elsewhere in
this repo (e.g. ``ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF``), so an OpenBao-backed key never has to be
a literal in the environment or the media-config file. Resolution is best-effort: an unresolvable
reference falls through to the env/file/none tiers rather than raising.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# Canonical env var names per provider (first match wins). Mirrors agent-utilities' existing aliases.
_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY", "AGENT_UTILITIES_OPENAI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "AGENT_UTILITIES_ANTHROPIC_API_KEY"),
    "azure": ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "ollama": ("OLLAMA_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
}

_BASE_URL_KEYS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_BASE_URL",),
    "ollama": ("OLLAMA_BASE_URL",),
    "deepseek": ("DEEPSEEK_BASE_URL",),
    "vllm": ("VLLM_BASE_URL",),
}

# Provider -> the AgentConfig field holding its runtime secret reference
# (CONCEPT:AU-ORCH.adapter.openai-catalog-verification). Only OpenAI is wired today (the task at hand); add a
# provider here as a real second reference field is configured, following the SAME
# ``<provider>_api_key_ref`` naming as ``openai_api_key_ref``.
_REF_CONFIG_FIELDS: dict[str, str] = {
    "openai": "openai_api_key_ref",
}


@dataclass(slots=True)
class ProviderCredentials:
    """Resolved credentials for one provider."""

    provider: str
    api_key: str | None = None
    base_url: str | None = None
    source: str = "none"  # which tier won: "secret_ref" | "env" | "file" | "none"


class CredentialResolver:
    """Resolve provider credentials with the precedence env > file > none.

    ``config_path`` defaults to ``$AGENT_UTILITIES_CONFIG_DIR/media-config.json`` (or
    ``~/.config/agent-utilities/media-config.json``); the file maps ``{provider: {api_key, base_url}}``.
    """

    def __init__(
        self,
        *,
        env: dict[str, str] | None = None,
        config_path: str | Path | None = None,
    ):
        self._env = env if env is not None else dict(os.environ)
        self._config_path = (
            Path(config_path) if config_path else self._default_config_path()
        )
        self._file_cache: dict[str, dict] | None = None

    @staticmethod
    def _default_config_path() -> Path:
        base = setting("AGENT_UTILITIES_CONFIG_DIR")
        root = Path(base) if base else Path.home() / ".config" / "agent-utilities"
        return root / "media-config.json"

    def _file(self) -> dict[str, dict]:
        if self._file_cache is None:
            try:
                self._file_cache = (
                    json.loads(self._config_path.read_text())
                    if self._config_path.exists()
                    else {}
                )
            except (OSError, json.JSONDecodeError):
                logger.debug(
                    "credential config unreadable at %s",
                    self._config_path,
                    exc_info=True,
                )
                self._file_cache = {}
        return self._file_cache

    def _from_env(self, keys: tuple[str, ...]) -> str | None:
        for k in keys:
            v = self._env.get(k)
            if v:
                return v
        return None

    def _from_secret_ref(self, provider: str) -> str | None:
        """Resolve ``provider``'s configured runtime secret reference, if any.

        Returns ``None`` (never raises) when no ref field is configured for this
        provider, none is set, or it fails to resolve — callers fall through to the
        env/file/none tiers exactly as if this tier did not exist.
        """
        ref_field = _REF_CONFIG_FIELDS.get(provider)
        if not ref_field:
            return None
        try:
            from agent_utilities.core.config import config as _cfg

            reference = getattr(_cfg, ref_field, None)
        except Exception:
            return None
        if not reference:
            return None
        try:
            from agent_utilities.security.cli_secrets import (
                resolve_runtime_secret_reference,
            )

            return resolve_runtime_secret_reference(reference)
        except Exception:
            logger.debug(
                "credential secret reference unresolved for provider=%s", provider
            )
            return None

    def resolve(self, provider: str) -> ProviderCredentials:
        """Return resolved credentials for ``provider`` (secret_ref > env > file > none)."""
        p = provider.lower()
        ref_key = self._from_secret_ref(p)
        if ref_key:
            return ProviderCredentials(p, api_key=ref_key, source="secret_ref")
        env_key = self._from_env(_ENV_KEYS.get(p, ()))
        env_url = self._from_env(_BASE_URL_KEYS.get(p, ()))
        if env_key or env_url:
            return ProviderCredentials(
                p, api_key=env_key, base_url=env_url, source="env"
            )
        entry = self._file().get(p, {})
        if entry.get("api_key") or entry.get("base_url"):
            return ProviderCredentials(
                p,
                api_key=entry.get("api_key"),
                base_url=entry.get("base_url"),
                source="file",
            )
        return ProviderCredentials(p, source="none")
