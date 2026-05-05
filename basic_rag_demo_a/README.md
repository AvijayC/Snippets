# Basic RAG Demo A

A portable FastAPI demo for an OpenAI-compatible RAG and database-query agent.

The default target model is `gpt-oss-120b` with a 32k context budget. Model
names, base URLs, API keys, reasoning effort, tool limits, prompts, retrieval
settings, and agent settings are runtime configurable from the UI.

## What It Demonstrates

- Plain HTML chat UI with multiple conversations.
- Tools tab for live enable/disable controls for model-accessible tools.
- Debug tab with tool calls, API history, retrieval details, token usage, and errors.
- Config tab with reloadable runtime settings and JSON syntax highlighting.
- Configurable `system_instruction` per agent, with legacy `system_prompt` fallback.
- Chat-visible reasoning summaries and agent loop traces based on observable tool and loop behavior.
- ChromaDB-backed local document retrieval with Markdown, HTML, PDF, and DOCX ingestion.
- Database query validation and optional raw-data access through swappable functions.
- OpenAI Chat Completions tool schemas plus fallback parsing for printed JSON tool calls.
- Python hook points for replacing database access, validation, tools, and document behavior.

## Quick Start

```bash
cd /Users/avijaychakravorti/Development/Snippets/basic_rag_demo_a
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env with your OpenAI-compatible endpoint details
uvicorn rag_demo.main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

The app creates the bundled sample database and sample docs on startup. Startup
uses the keyword fallback index so the UI appears immediately; click **Reload
docs + embeddings** in the Config tab to build the ChromaDB index with the
local default embedding model. You can also regenerate sample assets manually:

```bash
python -m rag_demo.sample_data
```

## Endpoint Compatibility

The runtime uses the OpenAI-compatible Chat Completions API:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=api_key, base_url=base_url)
await client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
```

Set these environment variables or update the Config tab:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

The key env var name is configurable. Set `api.api_key_env` to any environment
variable name, then make that variable available to the server process. For
example:

```bash
export MY_RAG_API_KEY=sk-...
```

```json
{
  "api": {
    "api_key_env": "MY_RAG_API_KEY"
  }
}
```

You can also put the same `MY_RAG_API_KEY=...` entry in `.env` before starting
the server.

For LM Studio on `http://127.0.0.1:1234`, use:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:1234/v1
export OPENAI_API_KEY=lm-studio
export OPENAI_MODEL=openai/gpt-oss-120b
```

For local endpoints such as LM Studio, the app uses a harmless `lm-studio`
placeholder key if no explicit API key is configured. It also tries to resolve
model aliases from `/v1/models`, so `gpt-oss-120b` can map to LM Studio's
`openai/gpt-oss-120b` model ID.

## Runtime Config

Edit the JSON in the Config tab and click **Save config**. The server applies
the new settings without restarting.

When the runtime config JSON or Runtime API key field changes, the Config tab
waits 5 seconds after the last edit and then tests the edited config against
the configured `/models` endpoint without saving it. Use **Test now** for an
immediate check.

The Config tab also has model dropdowns for the main agent and document
summarizer. They are populated from the configured endpoint's `/models`
response. Selecting a likely reasoning model, including GPT OSS models, sets
that agent's `reasoning_effort` to `high`.

Important fields:

- `agents.main.system_instruction`: primary behavior instructions for the main agent.
- `agents.main.reasoning_effort`: defaults to `high`; if an endpoint rejects it, the app retries without this parameter.
- `agents.main.min_iterations` / `max_iterations`: controls how many model loop passes are required or allowed.
- `agents.main.min_tool_calls` / `max_tool_calls`: controls total tool calls per chat turn.
- `agents.main.max_tool_calls_per_iteration`: caps parallel tool calls in one model loop. Set it to `1` if you want strict one-tool-call-at-a-time behavior.
- `agents.main.enabled_tools`: tools exposed to the main agent, excluding tools marked `always_enabled`.
- `database.raw_data_enabled`: toggles whether row-returning query tools are exposed.
- `rag.docs_path`: folder scanned for docs when indexing.
- `rag.top_k`: default number of document chunks retrieved.
- `rag.max_tool_top_k`: upper bound the model can request in a `retrieve_docs` tool call.
- `rag.min_score`: default score floor. Keep this low for recall; raise it to make retrieval stricter.
- `rag.use_subagent_summaries`: enables document summarizer subagents after retrieval.
- `rag.summarize_top_k`: number of retrieved chunks sent to summarizer subagents.
- `rag.coverage_*`: controls wave-based retrieval coverage when the model needs distinct facts across many matches.
- `hooks.module`: Python module path for custom validation, query execution, and custom tools.

The chat header shows token usage plus the latest main-agent context window
usage: estimated input tokens, input budget, configured context window, and any
history messages dropped to fit the budget.

It also shows all model token usage over the last 10 minutes, including
subagent calls. The small bar chart uses 30-second buckets, and the 2-minute
moving average projects current usage onto a 10-minute window. Tune the
comparison limits with `api.input_token_limit_per_10m`,
`api.output_token_limit_per_10m`, and `api.total_token_limit_per_10m`.

## Retrieval Tuning

Retrieval can be tuned globally through config and per call by the main model.
The `retrieve_docs` tool accepts:

- `query`: search text. The model can retry with different keywords.
- `top_k`: how many chunks to retrieve, capped by `rag.max_tool_top_k`.
- `min_score`: score floor from `0.0` to `1.0`; `0.0` is broadest.
- `search_mode`: `auto`, `keyword`, or `chroma`.
- `include_terms`: terms that must appear in returned chunks.
- `exclude_terms`: terms that must not appear in returned chunks.
- `summarizer_prompt`: extra focus guidance appended to the base
  `doc_summarizer` prompt. The main model should use this often to tell
  summarizers what details to preserve.
- `coverage_mode`: `on`, `off`, or `auto`. Use `on` when later matches may
  add distinct facts.
- `coverage_goal`: tells coverage subagents what new facts, caveats,
  exceptions, or conflicts to look for.
- `candidate_k` and `max_chunks`: broaden the initial candidate pool and cap
  how many chunks coverage can examine.
- `follow_references`: lets coverage follow obvious named source references
  found in retrieved chunks.

The default system instruction tells the main model to retry retrieval with
broader terms, a larger `top_k`, `min_score: 0`, or keyword mode when the first
result set is empty or too narrow, and to include `summarizer_prompt` when
subagent summaries would help. Retrieval details shown under each assistant
message include the exact settings and summarizer focus the model used.

For literal search, the `keyword_search_docs` tool scans documentation chunks
with ranked keyword or fuzzy-keyword matching. It accepts `query`, `keywords`,
`fuzziness`, `top_n`, and `match_mode`. Use low fuzziness for exact phrases and
higher fuzziness for misspellings or approximate terms.

For named references, `lookup_doc_source` opens chunks by source title, source
path, guide, runbook, appendix, or document name. The system instruction tells
the main agent to use it when retrieved text says to check another source.

Coverage mode summarizes candidates in waves. Each wave receives the known
facts found so far and asks subagents to preserve only materially new facts,
duplicates, conflicts, or caveats. The tool returns a compact coverage report
with candidates examined, distinct facts, reference follow-ups, waves, and stop
reason. Retrieval details in the chat UI show the same coverage report.

## Subagent Usage

The built-in subagents are document summarizers. They are normally used when the
main model calls `retrieve_docs` and `rag.use_subagent_summaries` is `true`.
The retrieval tool sends up to `rag.summarize_top_k` chunks to the
`doc_summarizer` agent, which compresses each chunk before the main agent sees
the tool result.

Subagent usage is controlled by:

- `rag.use_subagent_summaries`: global on/off switch.
- `rag.summarize_top_k`: maximum chunks summarized per retrieval call.
- `agents.doc_summarizer.model`: model used for summarization.
- `agents.doc_summarizer.reasoning_effort`, `max_output_tokens`, and prompt fields.

The main model does not directly spawn these summarizers. It decides whether and
how to call `retrieve_docs`; the retrieval tool then fans out summarizer
subagents according to config. The chat UI shows a **Subagents** dropdown with
agent name, source chunk, status, model, reasoning effort, tools enabled, tool
calls, duration, token usage, summary, and errors.

## Document Reloads

Click **Reload docs + embeddings** in the Config tab to reread every supported
file in `rag.docs_path`, prune missing document records, and rebuild the vector
collection from the current folder contents.

## Tool Safety

`validate_sql_query` is always available. It accepts SELECT-style SQL only and
performs two checks by default:

1. Parse and policy-check with `sqlglot`.
2. Execute a dry run by wrapping the query with `LIMIT 0`.

`run_sql_query` is only exposed when `database.raw_data_enabled` is `true`.
When disabled, the tool reports that raw data access is unavailable.

The Tools tab is the quickest way to toggle model-accessible tools live. Tools
marked as always enabled stay locked on, and row-returning tools remain
unavailable until raw row access is enabled.

The default adapter is intentionally small and local. For a real database or
warehouse, replace validation and row retrieval with hooks that call your own
client, permissions layer, and audit logic.

## Custom Tools And Database Functions

Set `hooks.module` in the Config tab or `config/default_config.json` to an
importable Python module, for example:

```json
{
  "hooks": {
    "module": "my_project.rag_hooks",
    "reload_on_config_update": true
  }
}
```

The module can replace the default query functions:

```python
from typing import Any
from rag_demo.tooling import ToolContext


def validate_sql_query(context: ToolContext, sql: str) -> dict[str, Any]:
    # Return None to fall back to the built-in validator.
    # Return a ToolResult-shaped dict to override validation.
    return {
        "ok": True,
        "data": {"sql": sql, "dry_run": "limit_0"},
        "summary": "Query validated by the external adapter.",
        "debug_messages": ["Adapter validation succeeded."],
        "warnings": [],
        "error": None,
        "metadata": {"adapter": "external"},
    }


async def run_sql_query(context: ToolContext, sql: str, limit: int = 50) -> dict[str, Any]:
    rows = []  # Call your database client here.
    return {
        "ok": True,
        "data": {"rows": rows, "row_count": len(rows), "limit": limit},
        "summary": f"Returned {len(rows)} rows.",
        "debug_messages": ["External adapter query completed."],
        "warnings": [],
        "error": None,
        "metadata": {"adapter": "external"},
    }
```

The module can also add new OpenAI-compatible tools:

```python
from typing import Any
from rag_demo.tooling import ToolContext, ToolResult, ToolSpec


def register_tools(registry: Any) -> None:
    registry.register(
        ToolSpec(
            name="lookup_metric_definition",
            description="Look up the definition for a business metric.",
            parameters={
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Metric name to look up."
                    }
                },
                "required": ["metric_name"],
                "additionalProperties": False,
            },
            handler=lookup_metric_definition,
        )
    )


def lookup_metric_definition(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    metric_name = str(args.get("metric_name", ""))
    context.debug(f"lookup_metric_definition received metric={metric_name}")
    return ToolResult(
        ok=True,
        data={"metric_name": metric_name, "definition": "Demo definition."},
        summary=f"Found definition for {metric_name}.",
        debug_messages=["Custom metric tool completed."],
        warnings=[],
        metadata={"source": "metrics_catalog"},
    )
```

Then add the tool name to `agents.main.enabled_tools`:

```json
{
  "agents": {
    "main": {
      "enabled_tools": [
        "retrieve_docs",
        "keyword_search_docs",
        "list_database_schema",
        "validate_sql_query",
        "run_sql_query",
        "lookup_metric_definition"
      ]
    }
  }
}
```

Every tool handler receives:

- `context`: includes `config`, `state`, `project_root`, `chat_id`, `run_id`, and `context.debug(message)`.
- `args`: parsed JSON arguments matching the tool's `parameters` schema.

Every tool handler should return a `ToolResult` or a dict with this envelope:

```json
{
  "ok": true,
  "data": {},
  "summary": "Short user-facing result",
  "debug_messages": ["Detailed debug line"],
  "warnings": [],
  "error": null,
  "metadata": {}
}
```

Use `summary` for the model-readable result, `data` for structured facts or
rows, `debug_messages` for the Debug tab, `warnings` for recoverable issues,
and `metadata` for adapter-specific traces.

See `rag_demo/example_hooks.py` for a complete example.

## Documentation Handling

Documents are loaded from `rag.docs_path`. Supported extensions are `.md`,
`.markdown`, `.html`, `.htm`, `.pdf`, `.docx`, and `.txt`.

Startup scans the folder and uses keyword retrieval immediately. Chroma indexing
is built when `rag.chroma_on_startup` is `true` or when you click **Reload docs
+ embeddings**. If you add, remove, or edit docs while the app is running, click
**Reload docs + embeddings** to refresh the retrieval index.

## Tests

```bash
pytest
```

The tests use the sample data generator and mock the model client where needed,
so they do not need a live model endpoint.
