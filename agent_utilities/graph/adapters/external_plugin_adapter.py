"""External Plugin Adapter — CONCEPT:ORCH-1.4

Provides backward compatibility for legacy plugins, allowing them
to be natively loaded into the agent-utilities ServiceRegistry.
"""

import importlib.util
import json
import logging
import os
import sys

from agent_utilities.graph.service_registry import ServiceDescriptor

logger = logging.getLogger(__name__)


class ExternalPluginAdapter:
    """Loads external legacy plugins and returns ServiceDescriptors."""

    @staticmethod
    def load_plugins_from_directory(plugin_dir: str) -> list[ServiceDescriptor]:
        """Scan a directory for legacy plugins and convert them."""
        descriptors: list[ServiceDescriptor] = []
        if not os.path.exists(plugin_dir):
            return descriptors

        for root, _, files in os.walk(plugin_dir):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    desc = ExternalPluginAdapter._load_python_plugin(
                        os.path.join(root, file)
                    )
                    if desc:
                        descriptors.extend(desc)
                elif file.endswith(".json"):
                    desc = ExternalPluginAdapter._load_json_plugin(
                        os.path.join(root, file)
                    )
                    if desc:
                        descriptors.extend(desc)

        return descriptors

    @staticmethod
    def _load_python_plugin(filepath: str) -> list[ServiceDescriptor]:
        """Load a Python-based legacy plugin."""
        descriptors: list[ServiceDescriptor] = []
        module_name = f"legacy_plugin_{os.path.splitext(os.path.basename(filepath))[0]}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if not spec or not spec.loader:
                return descriptors
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Legacy plugins typically define a 'Plugin' class or 'register' function
            # Let's look for any class that has an 'execute' or 'run' method
            for attr_name in dir(module):
                if attr_name.startswith("__"):
                    continue
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and (
                    hasattr(attr, "execute") or hasattr(attr, "run")
                ):
                    capability_name = getattr(attr, "capability", attr_name.lower())
                    descriptors.append(
                        ServiceDescriptor(
                            module_path=module_name,
                            function_name=attr_name,
                            capability=f"legacy_{capability_name}",
                            domain="external",
                            layer="plugin",
                            description=getattr(attr, "__doc__", "")
                            or f"Legacy Plugin {attr_name}",
                        )
                    )
        except Exception as e:
            logger.warning(
                "Failed to load legacy python plugin from %s: %s", filepath, e
            )

        return descriptors

    @staticmethod
    def _load_json_plugin(filepath: str) -> list[ServiceDescriptor]:
        """Load a JSON-based legacy plugin."""
        descriptors: list[ServiceDescriptor] = []
        try:
            with open(filepath) as f:
                data = json.load(f)

            # A simple JSON plugin might just define mcp_servers or webhooks
            if "name" in data and "command" in data:
                # We can register it as an external service
                capability_name = data["name"].lower().replace(" ", "_")

                # We'll create a dummy class that runs the command
                class JsonPluginRunner:
                    def __init__(self, **kwargs):
                        self.config = data

                    def execute(self, task: str) -> str:
                        import subprocess

                        args = self.config.get("args", [])
                        cmd = [self.config["command"]] + args + [task]
                        try:
                            result = subprocess.run(
                                cmd, capture_output=True, text=True, check=True
                            )
                            return result.stdout
                        except subprocess.CalledProcessError as e:
                            return f"Plugin error: {e.stderr}"

                # Inject the dynamic class into this module namespace so it can be retrieved
                import sys

                module = sys.modules[__name__]
                class_name = f"JsonPlugin_{capability_name}"
                setattr(module, class_name, JsonPluginRunner)

                descriptors.append(
                    ServiceDescriptor(
                        module_path=__name__,
                        function_name=class_name,
                        capability=f"legacy_{capability_name}",
                        domain="external",
                        layer="plugin",
                        description=data.get(
                            "description", f"JSON Plugin {data['name']}"
                        ),
                    )
                )
        except Exception as e:
            logger.warning("Failed to load legacy JSON plugin from %s: %s", filepath, e)

        return descriptors
