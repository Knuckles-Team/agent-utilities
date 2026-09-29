"""Configuration type names must not depend on the running Python release."""

from __future__ import annotations

from typing import Literal, Optional, Union

from agent_utilities.core.config_admin import _type_name


def test_union_type_names_are_spelled_out_on_every_python() -> None:
    assert _type_name(str | None) == "str | None"
    assert _type_name(Optional[int]) == "int | None"  # noqa: UP045
    assert _type_name(Union[int, str]) == "int | str"  # noqa: UP007
    assert _type_name(Literal["a"]) == "Literal"
    assert _type_name(list[str]) == "list"
    assert _type_name(bool) == "bool"
