"""Retired legacy executable-plugin adapter.

The historical adapter recursively imported arbitrary Python from a process
working-directory and turned JSON ``command`` fields into host subprocesses.
That is remote-code execution, not a plugin boundary. Governed plugin bundles,
MCP servers, and skills are the supported extension surfaces; this compatibility
class remains only so old imports fail closed during migration.
"""

from __future__ import annotations

from agent_utilities.core.registry.service_adapter import ServiceDescriptor


class ExternalPluginAdapter:
    """Compatibility facade that never imports or executes host plugin files."""

    @staticmethod
    def load_plugins_from_directory(_plugin_dir: str) -> list[ServiceDescriptor]:
        """Return no plugins; executable directory discovery is retired."""
        return []

    @staticmethod
    def _load_python_plugin(_filepath: str) -> list[ServiceDescriptor]:
        """Reject legacy Python plugins without importing their source."""
        return []

    @staticmethod
    def _load_json_plugin(_filepath: str) -> list[ServiceDescriptor]:
        """Reject legacy command plugins without starting a subprocess."""
        return []
