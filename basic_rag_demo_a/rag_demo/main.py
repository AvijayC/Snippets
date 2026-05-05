from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import AppSettings, load_settings, project_root
from .runtime import DemoRuntime


runtime = DemoRuntime(project_root())


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    runtime.initialize()
    yield


app = FastAPI(title="Basic RAG Demo A", lifespan=lifespan)
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class CreateChatRequest(BaseModel):
    title: str = "New chat"


class SendMessageRequest(BaseModel):
    content: str


class ConfigPatchRequest(BaseModel):
    patch: dict[str, Any]


class ConfigReplaceRequest(BaseModel):
    config: dict[str, Any]


class ConfigTestRequest(BaseModel):
    config: dict[str, Any]


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "app_name": runtime.config.app_name,
        "hooks": runtime.hooks.status(),
        "tools": len(runtime.tools.all()),
    }


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return runtime.config.public_dict()


@app.patch("/api/config")
async def patch_config(request: ConfigPatchRequest) -> dict[str, Any]:
    patch = _sanitize_config_input(request.patch)
    config = runtime.update_config(patch)
    return config.public_dict()


@app.put("/api/config")
async def replace_config(request: ConfigReplaceRequest) -> dict[str, Any]:
    config_data = _sanitize_config_input(request.config)
    config = runtime.replace_config(config_data)
    return config.public_dict()


@app.post("/api/config/reload")
async def reload_config_file() -> dict[str, Any]:
    config = load_settings(project_root() / "config" / "default_config.json")
    runtime.replace_config(config.model_dump(exclude={"api": {"api_key"}}))
    return runtime.config.public_dict()


@app.post("/api/config/test-api")
async def test_config_api(request: ConfigTestRequest) -> dict[str, Any]:
    config_data = _sanitize_config_input(request.config)
    try:
        candidate = AppSettings.model_validate(config_data)
    except Exception as exc:
        return {"ok": False, "stage": "config_validation", "error": str(exc)}
    return await runtime.test_api_connection(candidate)


@app.get("/api/tools")
async def tools() -> dict[str, Any]:
    return {"tools": runtime.tool_status()}


@app.get("/api/models")
async def models(refresh: bool = False) -> dict[str, Any]:
    try:
        model_ids = await runtime.list_endpoint_models(refresh=refresh)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"models": model_ids, "base_url": runtime.config.api.base_url}


@app.get("/api/token-usage")
async def token_usage(window_minutes: int = 10, average_minutes: int = 2) -> dict[str, Any]:
    return runtime.token_usage_metrics(window_minutes=window_minutes, average_minutes=average_minutes)


@app.post("/api/docs/reindex")
async def reindex_docs() -> dict[str, Any]:
    return runtime.rag.reindex(use_chroma=True)


@app.post("/api/docs/reload")
async def reload_docs_and_embeddings() -> dict[str, Any]:
    return runtime.rag.reindex(use_chroma=True)


@app.get("/api/docs")
async def docs() -> dict[str, Any]:
    return {"docs": runtime.state.list_ingested_docs()}


@app.get("/api/chats")
async def list_chats() -> dict[str, Any]:
    chats = runtime.state.list_chats()
    if not chats:
        chats = [runtime.state.create_chat()]
    return {"chats": chats}


@app.post("/api/chats")
async def create_chat(request: CreateChatRequest) -> dict[str, Any]:
    return {"chat": runtime.state.create_chat(request.title)}


@app.delete("/api/chats")
async def delete_all_chats() -> dict[str, Any]:
    result = runtime.state.clear_chat_history()
    chat = runtime.state.create_chat()
    runtime.debug_event("chat_history_deleted", result, chat_id=chat["id"])
    return {"ok": True, "deleted": result, "chat": chat}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str) -> dict[str, Any]:
    chat = runtime.state.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"chat": chat, "messages": runtime.state.get_messages(chat_id)}


@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, request: SendMessageRequest) -> dict[str, Any]:
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Message content is empty")
    if runtime.state.get_chat(chat_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        result = await runtime.send_message(chat_id, request.content)
    except Exception as exc:
        runtime.debug_event("run_failed", {"error": str(exc)}, chat_id=chat_id)
        message = runtime.state.add_message(
            chat_id,
            "assistant",
            f"Error: {exc}",
            metadata={"error": True},
        )
        return {"message": message, "error": str(exc)}
    return result


@app.get("/api/chats/{chat_id}/debug")
async def chat_debug(chat_id: str, limit: int = 500) -> dict[str, Any]:
    return {"events": runtime.state.get_debug_events(chat_id=chat_id, limit=limit)}


@app.get("/api/debug")
async def global_debug(limit: int = 500) -> dict[str, Any]:
    return {"events": runtime.state.get_debug_events(limit=limit)}


def _sanitize_config_input(data: dict[str, Any]) -> dict[str, Any]:
    copy = json.loads(json.dumps(data))
    api = copy.get("api")
    if isinstance(api, dict):
        if api.get("api_key") in {"********", "", "(local placeholder)"}:
            api.pop("api_key", None)
        api.pop("api_key_source", None)
    return copy


def main() -> None:
    import uvicorn

    uvicorn.run("rag_demo.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
