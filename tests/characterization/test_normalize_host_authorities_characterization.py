"""Characterize ``normalize_host_authorities`` before its complexity refactor.

The assertions in this file pin the exact validation and rendering behavior of
``agent_utilities.security.http_boundary.normalize_host_authorities`` as it
exists on the lane's starting main.  Keep this file byte-identical in the
paired refactor commit.
"""

from __future__ import annotations

import ipaddress

import pytest

from agent_utilities.security.http_boundary import normalize_host_authorities


def test_accepts_exact_host_forms_and_normalizes_them() -> None:
    authorities = (
        value
        for value in (
            "Service.Example.",
            "127.0.0.1:8443",
            "[2001:db8::1]:9443",
            "bücher.example",
            "service.example:443",
            "service.example",
            "service.example",
        )
    )

    assert normalize_host_authorities(authorities) == frozenset(
        {
            "service.example",
            "127.0.0.1:8443",
            "[2001:db8::1]:9443",
            "xn--bcher-kva.example",
            "service.example:443",
        }
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "host allowlist must contain exact authorities"),
        ("service/example", "host allowlist must contain exact authorities"),
        ("service\\example", "host allowlist must contain exact authorities"),
        ("user@service.example", "host allowlist must contain exact authorities"),
        ("service.example/path", "host allowlist must contain exact authorities"),
        ("service.example?query", "host allowlist must contain exact authorities"),
        ("service.example#fragment", "host allowlist must contain exact authorities"),
        ("service.example:bad", "host allowlist must contain exact authorities"),
        ("service.example.", ""),
        ("-service.example", "host allowlist contains an invalid name"),
        ("service-.example", "host allowlist contains an invalid name"),
        ("service..example", "host allowlist contains an invalid name"),
    ],
)
def test_rejects_invalid_authority_forms(value: str, message: str) -> None:
    if not message:
        assert normalize_host_authorities([value]) == frozenset({"service.example"})
        return

    with pytest.raises(ValueError, match=message):
        normalize_host_authorities([value])


def test_rejects_malformed_brackets_and_unbracketed_ipv6() -> None:
    for value in ("[2001:db8::1", "2001:db8::1", "[service.example]"):
        with pytest.raises(ValueError, match="host allowlist must contain exact"):
            normalize_host_authorities([value])


def test_rejects_empty_and_oversized_allowlists() -> None:
    with pytest.raises(ValueError, match="host allowlist must contain 1..256"):
        normalize_host_authorities([])

    authorities = [f"service-{index}.example" for index in range(257)]
    with pytest.raises(ValueError, match="host allowlist must contain 1..256"):
        normalize_host_authorities(authorities)


def test_ipaddress_rendering_matches_compressed_forms() -> None:
    ipv4 = ipaddress.ip_address("192.0.2.1")
    ipv6 = ipaddress.ip_address("2001:0db8:0:0:0:0:0:1")

    assert normalize_host_authorities([str(ipv4), f"[{ipv6}]"]) == {
        "192.0.2.1",
        "[2001:db8::1]",
    }
