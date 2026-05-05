from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rag_demo.config import AppSettings, load_settings, project_root
from rag_demo.hooks import HookManager
from rag_demo.sample_data import ensure_sample_database
from rag_demo.sql_tools import run_sql_query_tool, validate_select_sql
from rag_demo.tooling import ToolContext


def make_context(tmp_path: Path, raw_data_enabled: bool = False) -> ToolContext:
    settings: AppSettings = load_settings(project_root() / "config" / "default_config.json")
    db_path = tmp_path / "data" / "sample.db"
    ensure_sample_database(db_path)
    settings.database.path = str(db_path)
    settings.database.raw_data_enabled = raw_data_enabled
    runtime = SimpleNamespace(
        config=settings,
        project_root=tmp_path,
        hooks=HookManager(""),
    )
    return ToolContext(runtime=runtime)


def test_validate_select_query_accepts_valid_sql(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)

    result = validate_select_sql(ctx, "select name, calories from foods order by calories desc")

    assert result.ok is True
    assert result.data["dry_run"] == "limit_0"


def test_validate_select_query_rejects_write_sql(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)

    result = validate_select_sql(ctx, "delete from foods")

    assert result.ok is False
    assert "SELECT" in result.summary or "blocked" in result.summary


async def test_run_query_respects_raw_data_toggle(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, raw_data_enabled=False)

    result = await run_sql_query_tool(ctx, {"sql": "select name from foods", "limit": 3})

    assert result.ok is False
    assert "disabled" in result.summary.lower()


async def test_run_query_returns_limited_rows_when_enabled(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, raw_data_enabled=True)

    result = await run_sql_query_tool(ctx, {"sql": "select name from foods order by id", "limit": 2})

    assert result.ok is True
    assert result.data["row_count"] == 2
    assert result.data["rows"][0]["name"] == "oatmeal"
