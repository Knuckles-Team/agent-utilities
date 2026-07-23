import base64
import binascii
import logging
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic_ai import BinaryContent

from agent_utilities.core.config import config, setting

from ..models import ModelDefinition, ModelRegistry
from .models import ReloadableApp

logger = logging.getLogger(__name__)


def setup_server_file_logging(workspace: str | None = None) -> str | None:
    """Configure a file handler for the root logger to capture all server logs."""
    from agent_utilities.core.workspace import WORKSPACE_DIR

    ws = workspace or WORKSPACE_DIR or "."
    root = Path(ws).expanduser().resolve()
    log_dir = (root / ".agent_data" / "logs").resolve()
    try:
        log_dir.relative_to(root)
    except ValueError as exc:
        raise PermissionError("Server log directory escapes the workspace") from exc
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "server.log"
    if log_file.is_symlink():
        raise PermissionError("Symbolic-link log targets are not permitted")
    log_file.touch(mode=0o600, exist_ok=True)
    try:
        log_file.chmod(0o600)
    except OSError:
        pass

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    logger.info("Server file logging enabled")
    return str(log_file)


def _sniff_image_media_type(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def process_parts(parts: list[dict[str, Any]]) -> list[Any]:
    """Process incoming message parts from the Agent UI."""
    if not isinstance(parts, list) or len(parts) > 64:
        raise HTTPException(status_code=422, detail="invalid message parts")
    processed: list[Any] = []
    from pydantic_ai.messages import TextPart

    for part in parts:
        if not isinstance(part, dict) or len(part) > 16:
            raise HTTPException(status_code=422, detail="invalid message part")
        if "text" in part:
            text = part["text"]
            if not isinstance(text, str):
                raise HTTPException(status_code=422, detail="invalid text part")
            if len(text.encode("utf-8")) > config.max_upload_size:
                raise HTTPException(status_code=413, detail="text part too large")
            processed.append(TextPart(text))
        elif "image" in part or "binary" in part:
            img_data = part.get("image") or part.get("binary")
            if not img_data:
                continue
            if isinstance(img_data, str) and img_data.startswith("data:"):
                _, img_data = img_data.split(",", 1)

            if isinstance(img_data, str):
                try:
                    raw_bytes = base64.b64decode(img_data, validate=True)
                except (binascii.Error, ValueError, TypeError):
                    logger.warning("Upload rejected: invalid base64 encoding")
                    continue
            else:
                raw_bytes = img_data
            if not isinstance(raw_bytes, (bytes, bytearray)):
                logger.warning("Upload rejected: image content must be bytes")
                continue
            raw_bytes = bytes(raw_bytes)

            if len(raw_bytes) > config.max_upload_size:
                raise HTTPException(status_code=413, detail="binary part too large")

            media_type = _sniff_image_media_type(raw_bytes)
            if media_type is None:
                logger.warning("Upload rejected: unsupported image content")
                continue

            # Binary content remains request-scoped. Persisting uploads here
            # would create an unnecessary retention and symlink-race surface.
            processed.append(BinaryContent(data=raw_bytes, media_type=media_type))
    return processed


def inject_reload_app(app, reload_app: ReloadableApp):
    """Recursively inject a ReloadableApp reference into FastAPI state."""
    app.state.reload_app = reload_app
    if hasattr(app, "routes"):
        for route in app.routes:
            if hasattr(route, "app") and hasattr(route.app, "state"):
                inject_reload_app(route.app, reload_app)


def _build_model_from_registry(
    registry: ModelRegistry | None, model_id: str | None
) -> Any | None:
    """Resolve model_id against registry and build a pydantic-ai Model."""
    if not model_id or registry is None or not getattr(registry, "models", None):
        return None
    definition = registry.get_by_id(model_id)
    if definition is None:
        logger.debug(
            "Requested model id '%s' not found in registry; using default.", model_id
        )
        return None
    try:
        from agent_utilities.core.model_factory import create_model

        api_key = setting(definition.api_key_env) if definition.api_key_env else None
        # CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle — graph-os's registry-driven
        # model path historically only carried a static api_key_env; a definition configured with
        # an oauth2 client-credentials block instead mints/renews its own bearer transparently.
        return create_model(
            provider=definition.provider,
            model_id=definition.model_id,
            base_url=definition.base_url,
            api_key=api_key,
            oauth2=definition.oauth2,
        )
    except Exception as exc:
        logger.warning(
            "Failed to build override model; using fallback (exception_type=%s)",
            type(exc).__name__,
        )
        return None


def resolve_model_registry(
    *,
    registry: ModelRegistry | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> ModelRegistry:
    """Resolve the active model registry."""
    if registry is not None:
        return registry

    cfg_path = config.model_registry_path
    if cfg_path:
        p = Path(cfg_path)
        if p.is_file():
            try:
                return ModelRegistry.load_from_file(p)
            except Exception as exc:
                logger.error(
                    "Failed to load model registry configuration (exception_type=%s)",
                    type(exc).__name__,
                )

    if model_id:
        _id = f"{provider}:{model_id}" if provider else model_id
        return ModelRegistry(
            models=[
                ModelDefinition(
                    id=_id,
                    name=model_id,
                    provider=provider or "openai",
                    model_id=model_id,
                    base_url=base_url,
                    api_key_env=api_key_env,
                    tier="medium",
                    is_default=True,
                )
            ]
        )

    return ModelRegistry()
