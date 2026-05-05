from __future__ import annotations

from rag_demo.example_hooks import register_tools
from rag_demo.tooling import ToolRegistry


def test_example_hook_registers_tool() -> None:
    registry = ToolRegistry()

    register_tools(registry)

    assert registry.get("demo_echo") is not None
