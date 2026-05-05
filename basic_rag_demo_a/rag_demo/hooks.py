from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .tooling import ToolRegistry


class HookManager:
    def __init__(self, module_name: str = "", project_root: Path | None = None):
        self.module_name = module_name.strip()
        self.project_root = project_root
        self.module: ModuleType | None = None
        self.error: str | None = None
        if self.module_name:
            self.load()

    def load(self) -> None:
        self.module = None
        self.error = None
        if not self.module_name:
            return
        try:
            if self.project_root:
                root = str(self.project_root)
                if root not in sys.path:
                    sys.path.insert(0, root)
            if self.module_name in sys.modules:
                self.module = importlib.reload(sys.modules[self.module_name])
            else:
                self.module = importlib.import_module(self.module_name)
        except Exception as exc:
            self.error = str(exc)

    def register_tools(self, registry: ToolRegistry) -> None:
        if self.module is None:
            return
        register = getattr(self.module, "register_tools", None)
        if callable(register):
            register(registry)

    def call(self, name: str, context: Any, **kwargs: Any) -> Any:
        if self.module is None:
            return None
        func = getattr(self.module, name, None)
        if not callable(func):
            return None
        return func(context, **kwargs)

    def status(self) -> dict[str, Any]:
        return {
            "module": self.module_name,
            "loaded": self.module is not None,
            "error": self.error,
        }
