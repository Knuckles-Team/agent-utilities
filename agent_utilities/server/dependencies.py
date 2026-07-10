import base64
import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic_ai import BinaryContent

from agent_utilities.core.config import config, setting

from ..models import ModelDefinition, ModelRegistry
from .models import ReloadableApp

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Legacy security dependency — delegates to the unified auth module.

    Kept for backward compatibility.  New code should use
    ``auth.verify_credentials`` instead.

    CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
    """
    if not config.enable_api_auth and not config.auth_jwt_jwks_uri:
        return
    if config.enable_api_auth and config.agent_api_key:
        if api_key != config.agent_api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Could not validate credentials",
            )


def setup_server_file_logging(workspace: str | None = None) -> str | None:
    """Configure a file handler for the root logger to capture all server logs."""
    from agent_utilities.core.workspace import WORKSPACE_DIR

    ws = workspace or WORKSPACE_DIR or "."
    log_dir = Path(ws) / ".agent_data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "server.log"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    logger.info(f"Server logs redirected to: {log_file}")
    return str(log_file)


async def process_parts(parts: list[dict[str, Any]]) -> list[Any]:
    """Process incoming message parts from the Agent UI."""
    processed: list[Any] = []
    from pydantic_ai.messages import TextPart

    for part in parts:
        if "text" in part:
            processed.append(TextPart(part["text"]))
        elif "image" in part or "binary" in part:
            img_data = part.get("image") or part.get("binary")
            if not img_data:
                continue
            media_type = part.get("media_type", "image/png")
            if isinstance(img_data, str) and img_data.startswith("data:"):
                _, img_data = img_data.split(",", 1)

            if isinstance(img_data, str):
                raw_bytes = base64.b64decode(img_data)
            else:
                raw_bytes = img_data

            if len(raw_bytes) > config.max_upload_size:
                logger.warning(
                    f"Upload rejected: size {len(raw_bytes)} exceeds limit {config.max_upload_size}"
                )
                continue

            try:
                from uuid import uuid4

                from agent_utilities.core.workspace import WORKSPACE_DIR

                img_dir = Path(WORKSPACE_DIR or ".") / ".agent_data" / "images"
                img_dir.mkdir(parents=True, exist_ok=True)

                img_filename = f"{uuid4().hex}.{media_type.split('/')[-1]}"
                img_path = img_dir / img_filename
                img_path.write_bytes(raw_bytes)
                logger.debug(f"Saved uploaded image to: {img_path}")
            except Exception as e:
                logger.warning(f"Failed to save image to disk: {e}")

            processed.append(BinaryContent(data=raw_bytes, media_type=media_type))
    return processed


def get_http_client(
    ssl_verify: bool = True, timeout: float = 300.0
) -> httpx.AsyncClient | None:
    """Create a configured HTTPX AsyncClient for internal requests.

    Returns ``None`` when TLS verification is on (callers fall back to their
    library's default client); only an explicit ``ssl_verify=False`` opt-out
    yields an insecure client — built via the canonical factory, whose
    default stays ``verify=True``.
    """
    from agent_utilities.core.http_client import create_async_http_client

    if not ssl_verify:
        return create_async_http_client(verify=False, timeout=timeout)  # nosec B501
    return None


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
    except Exception as e:
        logger.warning(
            "Failed to build override model for '%s'; falling back: %s", model_id, e
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
            except Exception as e:
                logger.error(
                    "Failed to load model registry config from %s: %s", cfg_path, e
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
