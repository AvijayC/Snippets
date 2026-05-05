from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolResult(BaseModel):
    ok: bool
    data: Any = None
    summary: str = ""
    debug_messages: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


ToolHandler = Callable[["ToolContext", dict[str, Any]], ToolResult | dict[str, Any] | Awaitable[ToolResult | dict[str, Any]]]


@dataclass
class ToolContext:
    runtime: Any
    chat_id: str | None = None
    run_id: str | None = None
    debug_messages: list[str] = field(default_factory=list)

    @property
    def config(self) -> Any:
        return self.runtime.config

    @property
    def state(self) -> Any:
        return self.runtime.state

    @property
    def project_root(self) -> Any:
        return self.runtime.project_root

    def debug(self, message: str) -> None:
        self.debug_messages.append(message)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    requires_raw_data: bool = False
    always_enabled: bool = False

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def openai_schemas(self, enabled_names: list[str]) -> list[dict[str, Any]]:
        enabled = set(enabled_names)
        schemas = []
        for spec in self.all():
            if spec.always_enabled or spec.name in enabled:
                schemas.append(spec.openai_schema())
        return schemas


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def normalize_tool_result(value: Any) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, dict):
        return ToolResult.model_validate(value)
    return ToolResult(ok=True, data=value, summary=str(value))
