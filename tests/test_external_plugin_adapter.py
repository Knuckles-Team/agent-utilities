import os
import tempfile
import json
from agent_utilities.graph.adapters.external_plugin_adapter import ExternalPluginAdapter
from agent_utilities.graph.service_registry import ServiceDescriptor


def test_load_python_plugin():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock python plugin
        py_plugin_path = os.path.join(tmp_dir, "test_plugin.py")
        with open(py_plugin_path, "w") as f:
            f.write(
                "class MyLegacyPlugin:\n"
                "    capability = 'test_py_cap'\n"
                "    def execute(self, task: str):\n"
                "        return 'done'\n"
            )

        # Load the plugin
        descriptors = ExternalPluginAdapter.load_plugins_from_directory(tmp_dir)

        assert len(descriptors) == 1
        desc = descriptors[0]
        assert desc.capability == "legacy_test_py_cap"
        assert desc.function_name == "MyLegacyPlugin"
        assert desc.domain == "external"


def test_load_json_plugin():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock JSON plugin
        json_plugin_path = os.path.join(tmp_dir, "test_plugin.json")
        with open(json_plugin_path, "w") as f:
            json.dump({"name": "Test Json Cap", "command": "echo", "args": ["-n"]}, f)

        # Load the plugin
        descriptors = ExternalPluginAdapter.load_plugins_from_directory(tmp_dir)

        assert len(descriptors) == 1
        desc = descriptors[0]
        assert desc.capability == "legacy_test_json_cap"
        assert desc.function_name == "JsonPlugin_test_json_cap"
        assert desc.domain == "external"
