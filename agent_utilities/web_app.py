from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic_ai import Agent
from starlette.middleware.cors import CORSMiddleware

from .web_templates import DASHBOARD_HTML
from .agent_utilities import (
    load_workspace_file,
    write_md_file,
    CORE_FILES,
    get_workspace_path,
    load_skills_from_directory,
    get_skills_path,
    reload_cron_tasks,
)
from . import agent_utilities

logger = logging.getLogger(__name__)


def create_enhanced_web_app(
    agent: Agent, name: str = "Agent", description: str = "AI Agent", emoji: str = "🤖"
) -> FastAPI:
    app = FastAPI(title=f"{name} Dashboard")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Define enhanced routes first to avoid being shadowed by the /api mount
    @app.on_event("startup")
    async def startup_event():
        try:
            from .agent_utilities import initialize_workspace

            initialize_workspace()
            await reload_cron_tasks()
        except Exception as e:
            logging.error(f"Failed to initialize dashboard on startup: {e}")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        from . import __version__

        return DASHBOARD_HTML.format(
            agent_name=name,
            agent_emoji=emoji,
            agent_description=description,
            agent_version=__version__,
        )

    @app.get("/api/enhanced/info")
    async def get_info():
        return {
            "name": name,
            "description": description,
            "emoji": emoji,
        }

    @app.get("/api/enhanced/files")
    async def list_files():
        workspace_dir = get_workspace_path("")
        config_basenames = [f for f in CORE_FILES.values()]

        # List all files in workspace
        all_files = [f.name for f in workspace_dir.iterdir() if f.is_file()]

        # Filter config files
        config_files = [
            f for f in all_files if f in config_basenames or f == "mcp_config.json"
        ]

        # Filter generated files (ignore system/hidden files)
        generated = [
            f
            for f in all_files
            if f not in config_basenames
            and f != "CRON_LOG.md"
            and f != "mcp_config.json"
            and not f.startswith(".")
            and f != "chats"  # Should be a directory anyway
        ]

        return {"config": sorted(config_files), "generated": sorted(generated)}

    @app.get("/api/enhanced/files/{filename}")
    async def get_file(filename: str):
        content = load_workspace_file(filename)
        return {"content": content}

    @app.put("/api/enhanced/files/{filename}")
    async def update_file(filename: str, data: Dict[str, str]):
        if not filename.endswith(".md"):
            raise HTTPException(status_code=400, detail="Only .md files allowed")
        write_md_file(filename, data.get("content", ""))
        return {"status": "success"}

    @app.get("/api/enhanced/skills")
    async def list_skills():
        all_skills = []
        if skills_path := get_skills_path():
            all_skills.extend(load_skills_from_directory(skills_path))

        # Also include universal skills if we can find them
        try:
            # 1. Try direct import
            try:
                from universal_skills.skill_utilities import get_universal_skills_path

                for upath in get_universal_skills_path():
                    all_skills.extend(load_skills_from_directory(upath))
            except (ImportError, ModuleNotFoundError):
                # 2. Try sibling directory if in source tree
                current_dir = Path(__file__).parent.resolve()
                # We are in agent_utilities/
                workspace_root = current_dir.parent.parent  # agent-packages/
                universal_skills_dir = (
                    workspace_root / "universal-skills" / "universal_skills" / "skills"
                )
                if universal_skills_dir.is_dir():
                    all_skills.extend(
                        load_skills_from_directory(str(universal_skills_dir))
                    )
        except Exception as e:
            logging.error(f"Error loading universal skills: {e}")

        import os
        
        # Deduplicate by ID
        seen = set()
        deduped = []
        for s in all_skills:
            sid = s.get('id')
            if sid and sid not in seen:
                deduped.append(s)
                seen.add(sid)

        result = []
        for s in deduped:
            sid = s.get("id")
            # Default to true unless explicitly disabled
            env_var = f"ENABLE_{sid.upper().replace('-', '_')}"
            enabled = os.environ.get(env_var, "true").lower() != "false"
            
            result.append({
                "id": sid,
                "name": s.get("name"),
                "description": s.get("description"),
                "version": s.get("version", "0.1.0"),
                "enabled": enabled
            })
            
        return result

    @app.post("/api/enhanced/skills/{skill_id}/toggle")
    async def toggle_skill(skill_id: str):
        import os
        env_var = f"ENABLE_{skill_id.upper().replace('-', '_')}"
        current = os.environ.get(env_var, "true").lower() != "false"
        new_state = not current
        os.environ[env_var] = "true" if new_state else "false"
        
        return {"status": "success", "skill_id": skill_id, "enabled": new_state}

    @app.post("/api/enhanced/reload")
    async def reload_agent(request: Request):
        try:
            from .agent_utilities import initialize_workspace
            initialize_workspace()
            
            # Find the reloadable app wrapper in the state
            # In FastAPI, request.app is the current sub-app
            # We injected reload_app recursively, so it should be there.
            reloadable = getattr(request.app.state, "reload_app", None)
            if not reloadable:
                raise HTTPException(status_code=501, detail="Reloadable wrapper not found in app state")

            reloadable.reload()
            return {"status": "success", "message": "Agent reloaded successfully"}
        except Exception as e:
            logger.error(f"Reload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/enhanced/cron/calendar")
    async def get_cron_calendar():
        res = []
        now = datetime.now()
        # Use agent_utilities.tasks to avoid stale reference
        for t in agent_utilities.tasks:
            next_run = t.last_run + timedelta(minutes=t.interval_minutes)
            res.append(
                {
                    "id": t.id,
                    "name": t.name,
                    "interval_min": t.interval_minutes,
                    "last_run": t.last_run.strftime("%Y-%m-%d %H:%M"),
                    "next_approx": next_run.strftime("%Y-%m-%d %H:%M"),
                    "prompt": t.prompt[:100] + ("..." if len(t.prompt) > 100 else ""),
                    "active": t.active,
                }
            )
        return res

    @app.post("/api/enhanced/upload")
    async def upload_file(file: UploadFile = File(...)):
        workspace_dir = get_workspace_path("")
        file_path = workspace_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename}

    @app.get("/api/enhanced/download/{filename}")
    async def download_file(filename: str):
        workspace_dir = get_workspace_path("")
        file_path = workspace_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path=file_path, filename=filename)

    @app.get("/api/enhanced/chats")
    async def list_chats():
        chats_dir = get_workspace_path("chats")
        if not chats_dir.exists():
            chats_dir.mkdir(parents=True, exist_ok=True)

        chat_files = list(chats_dir.glob("*.json"))
        # Sort by mtime descending
        chat_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        res = []
        for cf in chat_files[:50]:  # limit to 50
            try:
                with open(cf, "r") as f:
                    data = json.load(f)
                    res.append(
                        {
                            "id": cf.stem,
                            "title": data.get("title", "Untitled Chat"),
                            "last_updated": datetime.fromtimestamp(
                                cf.stat().st_mtime
                            ).strftime("%Y-%m-%d %H:%M"),
                            "message_count": len(data.get("messages", [])),
                        }
                    )
            except Exception as e:
                logger.error(f"Error loading chat {cf}: {e}")
        return res

    @app.get("/api/enhanced/chats/{chat_id}")
    async def get_chat(chat_id: str):
        chats_dir = get_workspace_path("chats")
        chat_file = chats_dir / f"{chat_id}.json"
        if not chat_file.exists():
            raise HTTPException(status_code=404, detail="Chat not found")

        with open(chat_file, "r") as f:
            return json.load(f)

    @app.post("/api/enhanced/chats")
    async def save_chat(data: Dict[str, Any]):
        chats_dir = get_workspace_path("chats")
        if not chats_dir.exists():
            chats_dir.mkdir(parents=True, exist_ok=True)

        chat_id = data.get("id")
        # Ensure we don't treat "null" string or empty as a valid ID
        if not chat_id or chat_id == "null":
            chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        chat_file = chats_dir / f"{chat_id}.json"

        # Extract title from first user message if not provided
        title = data.get("title")
        messages = data.get("messages", [])
        if not title and messages:
            for m in messages:
                if m.get("role") == "user":
                    content = m.get("content", "")
                    if not content and m.get("parts"):
                        for p in m.get("parts", []):
                            if p.get("type") == "text":
                                content = p.get("text", "")
                                break
                    title = content[:50] + "..." if len(content) > 50 else content
                    break

        save_data = {
            "id": chat_id,
            "title": title or "Untitled Chat",
            "messages": messages,
            "updated_at": datetime.now().isoformat(),
        }

        with open(chat_file, "w") as f:
            json.dump(save_data, f, indent=2)

        return {"status": "success", "id": chat_id, "title": save_data["title"]}

    @app.put("/api/enhanced/chats/{chat_id}/title")
    async def update_chat_title(chat_id: str, data: Dict[str, Any]):
        chats_dir = get_workspace_path("chats")
        chat_file = chats_dir / f"{chat_id}.json"

        if not chat_file.exists():
            return {"status": "error", "message": "Chat not found"}

        new_title = data.get("title")
        if not new_title:
            return {"status": "error", "message": "Title is required"}

        with open(chat_file, "r") as f:
            chat_data = json.load(f)

        chat_data["title"] = new_title
        chat_data["updated_at"] = datetime.now().isoformat()

        with open(chat_file, "w") as f:
            json.dump(chat_data, f, indent=2)

        return {"status": "success", "title": new_title}

    # Finally mount the Pydantic AI API at /api
    from pydantic_ai.ui._web.api import create_api_app

    api_app = create_api_app(agent=agent)
    app.mount("/api", api_app)

    return app
