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

# agent_utilities/control_plane/{agents,connectors}/repository.py: the keyset
# cursor Protocol methods AgentRepository.cursor_for / ConnectorRepository
# .cursor_for. Their parameter names are the contract -- each one names a real
# field on the corresponding cursor model (AgentKeysetCursor.after_agent_id at
# agents/models.py:709, ConnectorKeysetCursor.after_server_id at
# connectors/models.py:583, both typed Identifier), so they cannot be renamed or
# underscore-prefixed without breaking the keyset pagination contract.
# Vulture reports them only because a Protocol body is a docstring and the
# durable adapters that will implement these are still unlanded (tracked as
# Case B in the Wire-First disposition), so no implementation binds the names yet.
after_agent_id: object = None
after_version_id: object = None
after_server_id: object = None
_ = (after_agent_id, after_version_id, after_server_id)
