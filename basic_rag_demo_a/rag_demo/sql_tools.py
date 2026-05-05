from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .tooling import ToolContext, ToolRegistry, ToolResult, ToolSpec, maybe_await, normalize_tool_result

try:
    import sqlglot
    from sqlglot import expressions as exp
except Exception:  # pragma: no cover - exercised when optional dependency is absent
    sqlglot = None
    exp = None


BLOCKED_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|create|alter|replace|truncate|attach|detach|vacuum|pragma|reindex)\b",
    re.IGNORECASE,
)


def register_sql_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="list_database_schema",
            description="List the database tables and columns available for querying.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=list_database_schema_tool,
        )
    )
    registry.register(
        ToolSpec(
            name="validate_sql_query",
            description="Validate a SELECT-style database query without returning rows.",
            parameters={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SELECT-style database query to validate.",
                    }
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
            handler=validate_sql_query_tool,
            always_enabled=True,
        )
    )
    registry.register(
        ToolSpec(
            name="run_sql_query",
            description="Run a validated read-only SELECT query and return limited rows when raw data access is enabled.",
            parameters={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SELECT-style database query.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows to return.",
                        "minimum": 1,
                    },
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
            handler=run_sql_query_tool,
            requires_raw_data=True,
        )
    )


def db_path_from_context(ctx: ToolContext) -> Path:
    return resolve_project_path(ctx.config.database.path, ctx.project_root)


def read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


async def list_database_schema_tool(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    del args
    path = db_path_from_context(ctx)
    if not path.exists():
        return ToolResult(ok=False, error=f"Database not found: {path}", summary="Database is missing.")
    tables: list[dict[str, Any]] = []
    with read_only_connection(path) as conn:
        table_rows = conn.execute(
            "select name from sqlite_master where type = 'table' and name not like 'sqlite_%' order by name"
        ).fetchall()
        for row in table_rows:
            columns = conn.execute(f"pragma table_info({_quote_identifier(row['name'])})").fetchall()
            tables.append(
                {
                    "name": row["name"],
                    "columns": [
                        {
                            "name": column["name"],
                            "type": column["type"],
                            "notnull": bool(column["notnull"]),
                            "primary_key": bool(column["pk"]),
                        }
                        for column in columns
                    ],
                }
            )
    return ToolResult(
        ok=True,
        data={"tables": tables},
        summary=f"Found {len(tables)} tables in the database.",
        debug_messages=[f"Schema read from {path}"],
    )


async def validate_sql_query_tool(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    sql = str(args.get("sql", ""))
    hook_result = await _call_hook(ctx, "validate_sql_query", sql=sql)
    if hook_result is not None:
        return hook_result
    return validate_select_sql(ctx, sql)


async def run_sql_query_tool(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not ctx.config.database.raw_data_enabled:
        return ToolResult(
            ok=False,
            error="Raw data access is disabled.",
            summary="Raw SQL data access is disabled in the current configuration.",
            warnings=["Enable database.raw_data_enabled in the Config tab to allow row-returning queries."],
        )
    sql = str(args.get("sql", ""))
    limit = int(args.get("limit") or ctx.config.database.default_limit)
    limit = max(1, min(limit, ctx.config.database.max_limit))
    hook_result = await _call_hook(ctx, "run_sql_query", sql=sql, limit=limit)
    if hook_result is not None:
        return hook_result
    validation = validate_select_sql(ctx, sql)
    if not validation.ok:
        return validation
    path = db_path_from_context(ctx)
    wrapped = _wrap_query(sql, limit=limit)
    with read_only_connection(path) as conn:
        rows = conn.execute(wrapped).fetchall()
    data = [dict(row) for row in rows]
    return ToolResult(
        ok=True,
        data={"rows": data, "row_count": len(data), "limit": limit},
        summary=f"Returned {len(data)} rows.",
        debug_messages=[f"Executed read-only query against {path}", f"Applied row limit {limit}."],
    )


def validate_select_sql(ctx: ToolContext, sql: str) -> ToolResult:
    stripped = sql.strip().rstrip(";")
    debug_messages = []
    if not stripped:
        return ToolResult(ok=False, error="SQL query is empty.", summary="The SQL query is empty.")
    if ";" in stripped:
        return ToolResult(
            ok=False,
            error="Multiple SQL statements are not allowed.",
            summary="Only one SELECT-style statement is allowed.",
        )
    if BLOCKED_SQL_RE.search(stripped):
        return ToolResult(
            ok=False,
            error="Only SELECT-style read-only SQL is allowed.",
            summary="The query contains a blocked SQL keyword.",
        )
    if ctx.config.database.allow_sqlglot_validation:
        parsed_result = _validate_with_sqlglot(stripped)
        if parsed_result is not None:
            ok, message = parsed_result
            debug_messages.append(message)
            if not ok:
                return ToolResult(ok=False, error=message, summary="SQL parser validation failed.")
    path = db_path_from_context(ctx)
    if not path.exists():
        return ToolResult(ok=False, error=f"Database not found: {path}", summary="Database is missing.")
    try:
        with read_only_connection(path) as conn:
            conn.execute(_wrap_query(stripped, limit=0)).fetchall()
    except sqlite3.Error as exc:
        return ToolResult(
            ok=False,
            error=str(exc),
            summary="Database dry-run validation failed.",
            debug_messages=debug_messages,
        )
    return ToolResult(
        ok=True,
        data={"sql": stripped, "dry_run": "limit_0"},
        summary="SQL query validated successfully.",
        debug_messages=debug_messages + ["Database dry run completed with LIMIT 0."],
    )


def _validate_with_sqlglot(sql: str) -> tuple[bool, str] | None:
    if sqlglot is None or exp is None:
        return None
    try:
        expressions = sqlglot.parse(sql, read="sqlite")
    except Exception as exc:
        return False, f"sqlglot parse error: {exc}"
    if len(expressions) != 1:
        return False, "sqlglot rejected multiple statements."
    tree = expressions[0]
    blocked = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Command,
    )
    if any(isinstance(node, blocked) for node in tree.walk()):
        return False, "sqlglot found a non-read-only SQL expression."
    if not isinstance(tree, (exp.Select, exp.Union, exp.With, exp.Subquery)):
        return False, f"sqlglot parsed statement as {tree.__class__.__name__}, not a SELECT-style query."
    return True, "sqlglot parser validation passed."


def _wrap_query(sql: str, limit: int) -> str:
    cleaned = sql.strip().rstrip(";")
    return f"select * from ({cleaned}) as _rag_demo_query limit {int(limit)}"


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


async def _call_hook(ctx: ToolContext, name: str, **kwargs: Any) -> ToolResult | None:
    hooks = getattr(ctx.runtime, "hooks", None)
    if hooks is None:
        return None
    value = await maybe_await(hooks.call(name, ctx, **kwargs))
    if value is None:
        return None
    return normalize_tool_result(value)
