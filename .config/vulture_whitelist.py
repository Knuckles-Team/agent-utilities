"""Vulture whitelist.

Names vulture would otherwise report as unused but are load-bearing: they are
part of a fixed, framework-mandated override signature, so they cannot be
renamed (e.g. with a leading underscore) or dropped without breaking the
runtime call. See AGENTS.md "Quality Bar".
"""

# agent_utilities/core/config.py: AgentConfig.settings_customise_sources overrides
# pydantic_settings.BaseSettings.settings_customise_sources, which pydantic-settings
# calls as cls.settings_customise_sources(cls, init_settings=..., env_settings=...,
# dotenv_settings=..., file_secret_settings=...) (see pydantic_settings/main.py).
# The override only needs init_settings/env_settings, but settings_cls,
# dotenv_settings, and file_secret_settings must stay in the signature under
# these exact names to match the keyword-argument call.
settings_cls: object = None
dotenv_settings: object = None
file_secret_settings: object = None
_ = (settings_cls, dotenv_settings, file_secret_settings)
