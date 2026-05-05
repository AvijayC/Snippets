from __future__ import annotations

from typing import Any

from .tooling import ToolContext, ToolResult, ToolSpec


def validate_sql_query(context: ToolContext, sql: str) -> dict[str, Any]:
    """Example override that delegates to the default validator and annotates debug output."""
    from .sql_tools import validate_select_sql

    result = validate_select_sql(context, sql).model_dump()
    result.setdefault("debug_messages", []).append("Validated by example_hooks.validate_sql_query.")
    return result


def register_tools(registry: Any) -> None:
    registry.register(
        ToolSpec(
            name="demo_echo",
            description="Echo a string through a custom hook-provided tool.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=demo_echo,
        )
    )


def demo_echo(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    text = str(args.get("text", ""))
    context.debug(f"demo_echo received {len(text)} characters.")
    return ToolResult(
        ok=True,
        data={"text": text},
        summary=f"Echoed: {text}",
        debug_messages=["Custom hook tool ran successfully."],
    )
