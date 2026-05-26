import pytest
from pathlib import Path
from agent_utilities.mcp.multiplexer import get_server_prefix, clean_tool_name

def test_get_server_prefix_hosts():
    # Test standard systems-manager hosts
    assert get_server_prefix("systems-manager-mcp-r510") == "sys_r510"
    assert get_server_prefix("systems-manager-mcp-rw710") == "sys_rw710"
    assert get_server_prefix("systems-manager-mcp-gr1080") == "sys_gr1080"

    # Test standard container-manager hosts
    assert get_server_prefix("container-manager-mcp-r510") == "cnt_r510"
    assert get_server_prefix("container-manager-mcp-gr1080") == "cnt_gr1080"

    # Test other standard nicknamable servers
    assert get_server_prefix("graph-os") == "kg"
    assert get_server_prefix("repository-manager-mcp") == "rep"

    # Test fallback
    assert get_server_prefix("some-random-mcp-server") == "some"

def test_clean_tool_name_prefixing():
    # Verify that clean_tool_name applies prefixes correctly without collisions and within length budgets
    prefix = get_server_prefix("systems-manager-mcp-r510")
    assert prefix == "sys_r510"

    cleaned = clean_tool_name(prefix, "systems-manager-mcp-r510", "systems_manager_mcp_run_command")
    # Prefix (sys_r510) + "__" + stripped tool name (run_command)
    assert cleaned == "sys_r510__run_command"


def test_multiplexer_tool_filtering():
    import fnmatch
    # Mocking the filtering logic of multiplexer
    tools = [
        "cm_image_operations",
        "cm_volume_operations",
        "cm_compose_operations",
        "trace_port_namespace",
    ]

    # Whitelist only image and volume
    enabled_tools = ["*image*", "*volume*"]
    disabled_tools = []

    filtered = []
    for t in tools:
        if enabled_tools is not None:
            matched = any(fnmatch.fnmatch(t, pat) for pat in enabled_tools)
            if not matched:
                continue
        if disabled_tools:
            matched_disabled = any(fnmatch.fnmatch(t, pat) for pat in disabled_tools)
            if matched_disabled:
                continue
        filtered.append(t)

    assert filtered == ["cm_image_operations", "cm_volume_operations"]

    # Test blacklist only
    enabled_tools_2 = None
    disabled_tools_2 = ["*compose*"]

    filtered_2 = []
    for t in tools:
        if enabled_tools_2 is not None:
            matched = any(fnmatch.fnmatch(t, pat) for pat in enabled_tools_2)
            if not matched:
                continue
        if disabled_tools_2:
            matched_disabled = any(fnmatch.fnmatch(t, pat) for pat in disabled_tools_2)
            if matched_disabled:
                continue
        filtered_2.append(t)

    assert filtered_2 == ["cm_image_operations", "cm_volume_operations", "trace_port_namespace"]
