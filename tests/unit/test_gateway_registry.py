"""Tests for agent_utilities.gateway.registry — migrated from service-dashboard-core."""

import pytest

from agent_utilities.gateway.registry import Registry, get_registry
from agent_utilities.gateway.models import ServiceCategory


def test_registry_singleton():
    reg1 = get_registry()
    reg2 = get_registry()
    assert reg1 is reg2


def test_list_all_known():
    reg = Registry()
    known = reg.list_all_known()
    assert "portainer" in known
    assert "uptime_kuma" in known
    assert "technitium" in known


def test_get_invalid_widget():
    reg = Registry()
    widget = reg.get_widget("non_existent_widget")
    assert widget is None
