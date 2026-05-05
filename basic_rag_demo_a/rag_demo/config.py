from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ApiSettings(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    model_env: str = "OPENAI_MODEL"
    timeout_seconds: int = 120
    input_token_limit_per_10m: int | None = 450000
    output_token_limit_per_10m: int | None = 80000
    total_token_limit_per_10m: int | None = None
    api_key: str | None = Field(default=None, exclude=True)
    allow_local_placeholder_key: bool = True
    local_placeholder_api_key: str = Field(default="lm-studio", exclude=True)
    resolve_model_aliases: bool = True

    model_config = ConfigDict(extra="forbid")

    def resolved_api_key(self) -> str:
        explicit = self.api_key or os.environ.get(self.api_key_env, "")
        if explicit:
            return explicit
        if self.allow_local_placeholder_key and self.is_local_base_url():
            return self.local_placeholder_api_key
        return ""

    def api_key_source(self) -> str:
        if self.api_key:
            return "runtime_config"
        if os.environ.get(self.api_key_env, ""):
            return f"env:{self.api_key_env}"
        if self.allow_local_placeholder_key and self.is_local_base_url():
            return "local_placeholder"
        return "missing"

    def has_explicit_api_key(self) -> bool:
        return bool(self.api_key or os.environ.get(self.api_key_env, ""))

    def is_local_base_url(self) -> bool:
        host = urlparse(self.base_url).hostname
        return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    def redacted(self) -> dict[str, Any]:
        data = self.model_dump()
        if self.has_explicit_api_key():
            data["api_key"] = "********"
        elif self.allow_local_placeholder_key and self.is_local_base_url():
            data["api_key"] = "(local placeholder)"
        else:
            data["api_key"] = ""
        data["api_key_source"] = self.api_key_source()
        data["local_placeholder_api_key"] = "********"
        return data


class AgentSettings(BaseModel):
    name: str
    model: str = "gpt-oss-120b"
    temperature: float = 0.1
    reasoning_effort: str | None = "high"
    context_window: int = 32768
    max_output_tokens: int = 2048
    min_iterations: int = 1
    max_iterations: int = 6
    min_tool_calls: int = 0
    max_tool_calls: int = 8
    max_tool_calls_per_iteration: int | None = 4
    enabled_tools: list[str] = Field(default_factory=list)
    system_instruction: str = ""
    system_prompt: str = ""

    model_config = ConfigDict(extra="forbid")

    def effective_system_instruction(self) -> str:
        return self.system_instruction or self.system_prompt


class DatabaseSettings(BaseModel):
    path: str = Field(default="data/sample_diet.db", validation_alias=AliasChoices("path", "sqlite_path"))
    raw_data_enabled: bool = False
    default_limit: int = 50
    max_limit: int = 200
    allow_sqlglot_validation: bool = True

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @property
    def sqlite_path(self) -> str:
        return self.path

    @sqlite_path.setter
    def sqlite_path(self, value: str) -> None:
        self.path = value


class RagSettings(BaseModel):
    docs_path: str = "sample_docs"
    chroma_path: str = "data/chroma"
    collection_name: str = "diet_demo_docs"
    chunk_chars: int = 2800
    chunk_overlap_chars: int = 350
    top_k: int = 8
    min_score: float = 0.0
    max_tool_top_k: int = 20
    summarize_top_k: int = 3
    summarize_min_relative_score: float = 0.7
    use_subagent_summaries: bool = True
    fallback_keyword_search: bool = True
    chroma_on_startup: bool = False
    coverage_mode: str = "auto"
    coverage_candidate_k: int = 8
    coverage_wave_size: int = 3
    coverage_max_chunks: int = 4
    coverage_subagent_max_chunks: int = 3
    coverage_min_candidate_score: float = 0.01
    coverage_skip_zero_score_candidates: bool = True
    coverage_reference_max_chunks: int = 4
    coverage_reference_max_depth: int = 4
    coverage_stop_after_empty_waves: int = 1
    coverage_min_new_facts: int = 1
    coverage_follow_references: bool = True

    model_config = ConfigDict(extra="forbid")


class HooksSettings(BaseModel):
    module: str = ""
    reload_on_config_update: bool = True

    model_config = ConfigDict(extra="forbid")


class DebugSettings(BaseModel):
    store_api_payloads: bool = True
    store_tool_payloads: bool = True
    redact_secrets: bool = True
    max_events_per_chat: int = 500

    model_config = ConfigDict(extra="forbid")


class AppSettings(BaseModel):
    app_name: str = "Basic RAG Demo A"
    data_dir: str = "data"
    sample_docs_dir: str = "sample_docs"
    api: ApiSettings = Field(default_factory=ApiSettings)
    agents: dict[str, AgentSettings]
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    hooks: HooksSettings = Field(default_factory=HooksSettings)
    debug: DebugSettings = Field(default_factory=DebugSettings)

    model_config = ConfigDict(extra="forbid")

    def agent(self, name: str) -> AgentSettings:
        if name in self.agents:
            return self.agents[name]
        raise KeyError(f"Unknown agent config: {name}")

    def public_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["api"] = self.api.redacted()
        return data


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_project_path(path_value: str | Path, root: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root or project_root()) / path


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_settings(config_path: Path | None = None) -> AppSettings:
    root = project_root()
    load_env_file(root / ".env")
    path = config_path or root / "config" / "default_config.json"
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    settings = AppSettings.model_validate(raw)
    base_url = os.environ.get(settings.api.base_url_env)
    if base_url:
        settings.api.base_url = base_url
    model_name = os.environ.get(settings.api.model_env)
    if model_name and "main" in settings.agents:
        settings.agents["main"].model = model_name
    return settings


def merge_settings(current: AppSettings, patch: dict[str, Any]) -> AppSettings:
    data = current.model_dump()
    data = _deep_merge(data, patch)
    if "api" in patch and "api_key" in patch["api"]:
        data.setdefault("api", {})["api_key"] = patch["api"]["api_key"]
    return AppSettings.model_validate(data)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
