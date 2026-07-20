import json
import os
import tempfile

from agent_utilities.graph.adapters.external_plugin_adapter import ExternalPluginAdapter


def test_legacy_python_plugin_fails_closed():
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

        descriptors = ExternalPluginAdapter.load_plugins_from_directory(tmp_dir)

        assert descriptors == []


def test_legacy_command_plugin_fails_closed():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock JSON plugin
        json_plugin_path = os.path.join(tmp_dir, "test_plugin.json")
        with open(json_plugin_path, "w") as f:
            json.dump({"name": "Test Json Cap", "command": "echo", "args": ["-n"]}, f)

        descriptors = ExternalPluginAdapter.load_plugins_from_directory(tmp_dir)

        assert descriptors == []
