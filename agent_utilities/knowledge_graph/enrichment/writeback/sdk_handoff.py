"""AU-SEMANTIC-R017.1: typed handoff contract for vendor-enrichment writeback.

Split from AU-SEMANTIC-R017 ("Vendor enrichment extraction and writeback move
to the SDK and EG"): this slice ships the typed request model and the
refusal that AU holds no local vendor-writeback authority. Relocating the
extractor modules under ``enrichment/extractors/**`` to SDK manifest presets
and wiring the real call through the SDK ``writeback/`` module and EG's
served ``WriteBack`` method land in later slices.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class VendorWritebackAuthorityError(RuntimeError):
    """Raised when vendor-enrichment writeback is attempted through AU.

    AU no longer performs vendor writeback itself; callers must route the
    request through the agent-connector-sdk ``writeback/`` module and EG's
    served ``WriteBack`` method instead.
    """


@dataclass(frozen=True)
class VendorWritebackRequest:
    """Typed shape of a vendor-enrichment writeback call.

    This model exists so callers construct a well-typed payload before
    handing it to the SDK ``writeback`` module / EG ``WriteBack``, rather
    than passing ad hoc dicts to local AU writeback code.
    """

    connector: str
    source_instance: str
    record_type: str
    payload: Mapping[str, object]


def execute_local_vendor_writeback(request: VendorWritebackRequest) -> None:
    """Refuse: AU holds no local vendor-writeback authority.

    Callers must route ``request`` through the SDK's ``writeback/`` module
    and EG's served ``WriteBack`` method instead of calling this path.
    """

    raise VendorWritebackAuthorityError(
        "AU-SEMANTIC-R017: vendor-enrichment writeback has no local AU "
        f"authority for connector {request.connector!r}; route via "
        "agent-connector-sdk writeback/ and EG WriteBack."
    )
