from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from .agent import AgentRunner
from .config import AppSettings, load_settings, merge_settings, project_root, resolve_project_path
from .hooks import HookManager
from .rag import RagIndex, register_rag_tools
from .sample_data import ensure_sample_assets
from .sql_tools import register_sql_tools
from .state import AppState, utc_now
from .tooling import ToolContext, ToolRegistry


class DemoRuntime:
    def __init__(self, root: Path | None = None):
        self.project_root = root or project_root()
        self.config: AppSettings = load_settings(self.project_root / "config" / "default_config.json")
        ensure_sample_assets(self.project_root)
        self.state = AppState(resolve_project_path(self.config.data_dir, self.project_root) / "app_state.sqlite")
        latest = self.state.latest_config()
        if latest:
            self.config = AppSettings.model_validate(latest)
        self.hooks = HookManager(self.config.hooks.module, self.project_root)
        self.tools = ToolRegistry()
        self.rag = RagIndex(self)
        self.agent_runner = AgentRunner(self, self.tools)
        self._model_cache_key: str | None = None
        self._model_cache: list[str] | None = None
        self._run_token_usage: dict[str, dict[str, int]] = {}
        self._run_subagent_trace: dict[str, list[dict[str, Any]]] = {}
        self.rebuild_tools()

    def initialize(self) -> dict[str, Any]:
        ensure_sample_assets(self.project_root)
        result = self.rag.reindex(use_chroma=self.config.rag.chroma_on_startup)
        self.debug_event("runtime_initialized", {"rag": result, "hooks": self.hooks.status()})
        return result

    def rebuild_tools(self) -> None:
        registry = ToolRegistry()
        register_rag_tools(registry)
        register_sql_tools(registry)
        self.hooks.register_tools(registry)
        self.tools = registry
        self.agent_runner.tools = registry

    def update_config(self, patch: dict[str, Any]) -> AppSettings:
        self.config = merge_settings(self.config, patch)
        if self.config.hooks.reload_on_config_update:
            self.hooks = HookManager(self.config.hooks.module, self.project_root)
        self.rebuild_tools()
        self.state.save_config(self.config.model_dump(exclude={"api": {"api_key"}}))
        self.debug_event("config_updated", {"config": self.config.public_dict(), "hooks": self.hooks.status()})
        return self.config

    def replace_config(self, data: dict[str, Any]) -> AppSettings:
        self.config = AppSettings.model_validate(data)
        if self.config.hooks.reload_on_config_update:
            self.hooks = HookManager(self.config.hooks.module, self.project_root)
        self.rebuild_tools()
        self.state.save_config(self.config.model_dump(exclude={"api": {"api_key"}}))
        self.debug_event("config_replaced", {"config": self.config.public_dict(), "hooks": self.hooks.status()})
        return self.config

    async def send_message(self, chat_id: str, content: str) -> dict[str, Any]:
        return await self.agent_runner.run_chat(chat_id, content)

    def start_token_usage(self, run_id: str) -> None:
        self._run_token_usage[run_id] = {}

    def start_subagent_trace(self, run_id: str) -> None:
        self._run_subagent_trace[run_id] = []

    def record_subagent_started(
        self,
        run_id: str | None,
        chat_id: str | None,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        item = {
            "id": trace.get("id") or f"subagent_{uuid.uuid4().hex[:12]}",
            "status": "running",
            "started_at": utc_now(),
            **trace,
        }
        if run_id:
            self._run_subagent_trace.setdefault(run_id, []).append(item)
        self.debug_event("subagent_started", item, chat_id=chat_id, run_id=run_id)
        return item

    def record_subagent_completed(
        self,
        run_id: str | None,
        chat_id: str | None,
        trace_id: str,
        patch: dict[str, Any],
    ) -> None:
        item = None
        if run_id:
            for candidate in self._run_subagent_trace.setdefault(run_id, []):
                if candidate.get("id") == trace_id:
                    item = candidate
                    break
        if item is None:
            item = {"id": trace_id}
            if run_id:
                self._run_subagent_trace.setdefault(run_id, []).append(item)
        item.update({"completed_at": utc_now(), **patch})
        event_type = "subagent_completed" if item.get("status") == "completed" else "subagent_finished"
        self.debug_event(event_type, item, chat_id=chat_id, run_id=run_id)

    def record_subagent_skipped(
        self,
        run_id: str | None,
        chat_id: str | None,
        trace: dict[str, Any],
    ) -> None:
        item = {
            "id": trace.get("id") or f"subagent_{uuid.uuid4().hex[:12]}",
            "status": "skipped",
            "started_at": utc_now(),
            "completed_at": utc_now(),
            **trace,
        }
        if run_id:
            self._run_subagent_trace.setdefault(run_id, []).append(item)
        self.debug_event("subagent_skipped", item, chat_id=chat_id, run_id=run_id)

    def consume_subagent_trace(self, run_id: str) -> list[dict[str, Any]]:
        traces = self._run_subagent_trace.pop(run_id, [])
        return [{key: value for key, value in item.items() if not key.startswith("_")} for item in traces]

    def record_token_usage(self, run_id: str | None, usage: dict[str, int]) -> None:
        if not run_id or not usage:
            return
        bucket = self._run_token_usage.setdefault(run_id, {})
        for key, value in usage.items():
            if isinstance(value, int):
                bucket[key] = bucket.get(key, 0) + value

    def consume_token_usage(self, run_id: str) -> dict[str, int]:
        return self._run_token_usage.pop(run_id, {})

    async def summarize_chunks(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        parent_ctx: ToolContext,
        summarizer_prompt: str = "",
    ) -> list[dict[str, Any]]:
        agent_config = self.config.agent("doc_summarizer")
        summarizer_prompt = " ".join(str(summarizer_prompt or "").split())[:1200]
        if not self.config.api.resolved_api_key():
            for chunk in chunks:
                self.record_subagent_skipped(
                    parent_ctx.run_id,
                    parent_ctx.chat_id,
                    {
                        "agent": "doc_summarizer",
                        "kind": "document_summary",
                        "task": "Summarize retrieved document chunk",
                        "query": query,
                        "model": agent_config.model,
                        "reasoning_effort": agent_config.reasoning_effort,
                        "source_id": chunk.get("id"),
                        "source_title": chunk.get("title"),
                        "source_path": chunk.get("source_path"),
                        "chunk_index": chunk.get("chunk_index"),
                        "summarizer_prompt": summarizer_prompt,
                        "tools_enabled": [],
                        "tool_calls": [],
                        "error": "No API key configured; returned truncated chunk instead of subagent summary.",
                    },
                )
            return [
                {
                    "ok": False,
                    "id": chunk.get("id"),
                    "title": chunk.get("title"),
                    "source_path": chunk.get("source_path"),
                    "summary": chunk.get("text", "")[:900],
                    "error": "No API key configured; returned truncated chunk instead of subagent summary.",
                    "summarizer_prompt": summarizer_prompt,
                }
                for chunk in chunks
            ]
        semaphore = asyncio.Semaphore(3)

        async def run_one(chunk: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                started = time.perf_counter()
                trace = self.record_subagent_started(
                    parent_ctx.run_id,
                    parent_ctx.chat_id,
                    {
                        "agent": "doc_summarizer",
                        "kind": "document_summary",
                        "task": "Summarize retrieved document chunk",
                        "query": query,
                        "model": agent_config.model,
                        "reasoning_effort": agent_config.reasoning_effort,
                        "source_id": chunk.get("id"),
                        "source_title": chunk.get("title"),
                        "source_path": chunk.get("source_path"),
                        "chunk_index": chunk.get("chunk_index"),
                        "summarizer_prompt": summarizer_prompt,
                        "input_chars": len(chunk.get("text", "")),
                        "tools_enabled": [],
                        "tool_calls": [],
                    },
                )
                result = await self.agent_runner.summarize_chunk(
                    query,
                    chunk,
                    parent_ctx,
                    summarizer_prompt=summarizer_prompt,
                )
                status = "completed" if result.get("ok") else "failed"
                self.record_subagent_completed(
                    parent_ctx.run_id,
                    parent_ctx.chat_id,
                    trace["id"],
                    {
                        "status": status,
                        "duration_ms": round((time.perf_counter() - started) * 1000),
                        "summary": result.get("summary") or "",
                        "error": result.get("error"),
                        "summarizer_prompt": result.get("summarizer_prompt") or summarizer_prompt,
                        "token_usage": result.get("token_usage") or {},
                        "tools_enabled": result.get("tools_enabled") or [],
                        "tool_calls": result.get("tool_calls") or [],
                        "conversation": result.get("conversation") or [],
                    },
                )
                return result

        summaries = await asyncio.gather(*(run_one(chunk) for chunk in chunks))
        self.debug_event(
            "subagent_summaries_completed",
            {
                "query": query,
                "summarizer_prompt": summarizer_prompt,
                "count": len(summaries),
                "errors": [item["error"] for item in summaries if item.get("error")],
            },
            chat_id=parent_ctx.chat_id,
            run_id=parent_ctx.run_id,
        )
        return list(summaries)

    def debug_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        chat_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.state.add_debug_event(event_type, self.redact(payload), chat_id=chat_id, run_id=run_id)

    def redact(self, payload: Any) -> Any:
        secret = self.config.api.resolved_api_key()
        return _redact(payload, secret)

    def tool_status(self) -> list[dict[str, Any]]:
        enabled = set(self.config.agent("main").enabled_tools)
        items = []
        for spec in self.tools.all():
            available = spec.always_enabled or spec.name in enabled
            if spec.requires_raw_data and not self.config.database.raw_data_enabled:
                available = False
            items.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "always_enabled": spec.always_enabled,
                    "requires_raw_data": spec.requires_raw_data,
                    "available": available,
                    "schema": spec.openai_schema(),
                }
            )
        return items

    async def list_endpoint_models(self, refresh: bool = False) -> list[str]:
        cache_key = self.config.api.base_url.rstrip("/")
        if not refresh and self._model_cache_key == cache_key and self._model_cache is not None:
            return self._model_cache
        headers = {}
        api_key = self.config.api.resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{cache_key}/models", headers=headers)
            response.raise_for_status()
            payload = response.json()
        models = _extract_model_ids(payload)
        self._model_cache_key = cache_key
        self._model_cache = models
        return models

    async def test_api_connection(self, candidate_config: AppSettings) -> dict[str, Any]:
        api = candidate_config.api
        agent_config = candidate_config.agent("main")
        base_url = api.base_url.rstrip("/")
        api_key = api.resolved_api_key()
        result: dict[str, Any] = {
            "ok": False,
            "base_url": base_url,
            "api_key_env": api.api_key_env,
            "api_key_source": api.api_key_source(),
            "configured_model": agent_config.model,
            "resolved_model": agent_config.model,
            "model_available": False,
            "models": [],
            "elapsed_ms": 0,
            "status_code": None,
            "error": None,
        }
        if not api_key:
            result["error"] = f"Missing API key. Set {api.api_key_env} or provide a runtime key."
            self.debug_event("config_api_test", result)
            return result
        headers = {"Authorization": f"Bearer {api_key}"}
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{base_url}/models", headers=headers)
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            result["status_code"] = response.status_code
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            result["error"] = "API test timed out after 5 seconds."
            self.debug_event("config_api_test", result)
            return result
        except Exception as exc:
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            result["error"] = str(exc)
            self.debug_event("config_api_test", result)
            return result

        model_ids = _extract_model_ids(payload)
        resolved_model = _resolve_model_name_from_ids(agent_config.model, model_ids, api.resolve_model_aliases)
        result.update(
            {
                "ok": True,
                "models": model_ids[:50],
                "model_count": len(model_ids),
                "resolved_model": resolved_model,
                "model_available": resolved_model in model_ids if model_ids else False,
            }
        )
        if model_ids and resolved_model not in model_ids:
            result["ok"] = False
            result["error"] = f"Endpoint responded, but model '{agent_config.model}' was not found."
        self.debug_event("config_api_test", result)
        return result

    def token_usage_metrics(self, window_minutes: int = 10, average_minutes: int = 2) -> dict[str, Any]:
        now = datetime.now(UTC)
        window_minutes = max(1, min(int(window_minutes), 60))
        average_minutes = max(1, min(int(average_minutes), window_minutes))
        since = (now - timedelta(minutes=window_minutes)).isoformat()
        events = self.state.get_debug_events_since(since, event_type="api_response", limit=20000)
        calls = [_usage_call_from_event(event) for event in events]
        calls = [call for call in calls if call is not None]
        window_start = now - timedelta(minutes=window_minutes)
        average_start = now - timedelta(minutes=average_minutes)
        window_calls = [call for call in calls if call["created_at"] >= window_start]
        average_calls = [call for call in window_calls if call["created_at"] >= average_start]
        totals = _sum_usage(call["usage"] for call in window_calls)
        average_totals = _sum_usage(call["usage"] for call in average_calls)
        projected = {
            key: round((value / average_minutes) * window_minutes)
            for key, value in average_totals.items()
            if isinstance(value, int)
        }
        buckets = _usage_buckets(window_calls, now, window_minutes=window_minutes, bucket_seconds=30)
        limits = {
            "prompt_tokens": self.config.api.input_token_limit_per_10m,
            "completion_tokens": self.config.api.output_token_limit_per_10m,
            "total_tokens": self.config.api.total_token_limit_per_10m
            or _optional_sum(
                self.config.api.input_token_limit_per_10m,
                self.config.api.output_token_limit_per_10m,
            ),
        }
        percentages = {
            key: round((totals.get(key, 0) / limit) * 100, 1)
            for key, limit in limits.items()
            if limit
        }
        projected_percentages = {
            key: round((projected.get(key, 0) / limit) * 100, 1)
            for key, limit in limits.items()
            if limit
        }
        return {
            "window_minutes": window_minutes,
            "average_minutes": average_minutes,
            "bucket_seconds": 30,
            "started_at": window_start.isoformat(),
            "ended_at": now.isoformat(),
            "totals": totals,
            "moving_average_window_totals": average_totals,
            "projected_per_10m_from_average": projected,
            "limits_per_10m": limits,
            "percentages_of_limit": percentages,
            "projected_percentages_of_limit": projected_percentages,
            "call_count": len(window_calls),
            "agents": _usage_by_agent(window_calls),
            "buckets": buckets,
        }

    async def resolve_model_name(
        self,
        requested_model: str,
        chat_id: str | None = None,
        run_id: str | None = None,
    ) -> str:
        if not self.config.api.resolve_model_aliases:
            return requested_model
        try:
            model_ids = await self.list_endpoint_models()
        except Exception as exc:
            self.debug_event(
                "model_discovery_failed",
                {"base_url": self.config.api.base_url, "error": str(exc), "requested_model": requested_model},
                chat_id=chat_id,
                run_id=run_id,
            )
            return requested_model
        if requested_model in model_ids:
            return requested_model
        resolved = _resolve_model_name_from_ids(requested_model, model_ids, self.config.api.resolve_model_aliases)
        if resolved != requested_model:
            self.debug_event(
                "model_alias_resolved",
                {"requested_model": requested_model, "resolved_model": resolved},
                chat_id=chat_id,
                run_id=run_id,
            )
            return resolved
        self.debug_event(
            "model_alias_not_found",
            {"requested_model": requested_model, "available_models": model_ids},
            chat_id=chat_id,
            run_id=run_id,
        )
        return requested_model


def _extract_model_ids(payload: dict[str, Any]) -> list[str]:
    return [item["id"] for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]


def _resolve_model_name_from_ids(requested_model: str, model_ids: list[str], resolve_aliases: bool = True) -> str:
    if not resolve_aliases or requested_model in model_ids:
        return requested_model
    candidates = [f"openai/{requested_model}"]
    candidates.extend(model_id for model_id in model_ids if model_id.rsplit("/", 1)[-1] == requested_model)
    for candidate in candidates:
        if candidate in model_ids:
            return candidate
    return requested_model


def _usage_call_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    payload = event.get("payload") or {}
    usage = _normalize_usage(payload.get("usage") or {})
    if not usage:
        return None
    try:
        created_at = datetime.fromisoformat(event["created_at"])
    except Exception:
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return {
        "created_at": created_at.astimezone(UTC),
        "agent": payload.get("agent") or "unknown",
        "usage": usage,
    }


def _normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    prompt = _int_value(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion = _int_value(usage.get("completion_tokens") or usage.get("output_tokens"))
    total = _int_value(usage.get("total_tokens")) or prompt + completion
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    reasoning = _int_value(details.get("reasoning_tokens") if isinstance(details, dict) else None)
    result = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    if reasoning:
        result["reasoning_tokens"] = reasoning
    return {key: value for key, value in result.items() if value}


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _sum_usage(usages: Any) -> dict[str, int]:
    totals: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
        totals.setdefault(key, 0)
    return totals


def _usage_buckets(
    calls: list[dict[str, Any]],
    now: datetime,
    window_minutes: int,
    bucket_seconds: int,
) -> list[dict[str, Any]]:
    bucket_count = max(1, int((window_minutes * 60) / bucket_seconds))
    start = now - timedelta(minutes=window_minutes)
    buckets = []
    for index in range(bucket_count):
        bucket_start = start + timedelta(seconds=index * bucket_seconds)
        buckets.append(
            {
                "start": bucket_start.isoformat(),
                "end": (bucket_start + timedelta(seconds=bucket_seconds)).isoformat(),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "call_count": 0,
            }
        )
    for call in calls:
        offset_seconds = (call["created_at"] - start).total_seconds()
        index = int(offset_seconds // bucket_seconds)
        if index < 0 or index >= bucket_count:
            continue
        bucket = buckets[index]
        bucket["call_count"] += 1
        for key, value in call["usage"].items():
            if isinstance(value, int):
                bucket[key] = bucket.get(key, 0) + value
    return buckets


def _usage_by_agent(calls: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    agents: dict[str, list[dict[str, int]]] = {}
    for call in calls:
        agents.setdefault(call["agent"], []).append(call["usage"])
    return {agent: _sum_usage(usages) for agent, usages in agents.items()}


def _optional_sum(*values: int | None) -> int | None:
    numeric = [value for value in values if isinstance(value, int)]
    return sum(numeric) if numeric else None


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, dict):
        return {key: _redact(_redact_key(key, item, secret), secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, str):
        text = value
        if secret:
            text = text.replace(secret, "********")
        return text
    return value


def _redact_key(key: str, value: Any, secret: str) -> Any:
    lowered = key.lower()
    if lowered.endswith("_tokens") or lowered.endswith("_tokens_details") or lowered in {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
    }:
        return value
    if any(part in lowered for part in ("api_key", "authorization", "password", "secret", "token")):
        return "********" if value else value
    return value
