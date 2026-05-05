from types import SimpleNamespace

from rag_demo.agent import (
    AgentRunner,
    build_reasoning_summary,
    extract_citations,
    extract_retrieval_detail,
    merge_citations,
    merge_token_usage,
    normalize_inline_citations,
    tool_messages_to_trace,
)


def test_extract_document_citations_from_retrieve_docs_result() -> None:
    citations = extract_citations(
        "retrieve_docs",
        {
            "ok": True,
            "data": {
                "chunks": [
                    {
                        "title": "Hydration Guide",
                        "source_path": "sample_docs/hydration_guide.html",
                        "chunk_index": 0,
                        "score": 0.9,
                        "text": "The sample goal is 72 ounces per day.",
                    }
                ]
            },
            "summary": "Retrieved docs.",
        },
    )

    assert citations[0]["kind"] == "document"
    assert citations[0]["title"] == "Hydration Guide"
    assert "72 ounces" in citations[0]["snippet"]


def test_extract_database_citation_from_sql_tool_result() -> None:
    citations = extract_citations(
        "validate_sql_query",
        {"ok": True, "summary": "SQL query validated successfully.", "data": {}},
    )

    assert citations == [
        {
            "kind": "database",
            "title": "Sample diet database",
            "source_path": "configured database",
            "snippet": "SQL query validated successfully.",
        }
    ]


def test_merge_citations_deduplicates_sources() -> None:
    citation = {
        "kind": "document",
        "title": "Meal Notes",
        "source_path": "sample_docs/meal_notes.md",
        "chunk_index": 0,
    }

    assert merge_citations([citation], [citation]) == [citation]


def test_normalize_inline_citations_replaces_generic_source_markers() -> None:
    content = "The sample hydration goal is 72 ounces per day【source】. A second fact 【citation】."
    citations = [{"title": "Hydration Guide"}, {"title": "Meal Notes"}]

    normalized = normalize_inline_citations(content, citations)

    assert normalized == "The sample hydration goal is 72 ounces per day [1]. A second fact [2]."


def test_normalize_inline_citations_replaces_numeric_and_title_markers() -> None:
    content = (
        "Goal text【1】. "
        "Line marker【1\u2020L13-L16】. "
        "Meal text 【Meal Notes】."
    )
    citations = [{"title": "Hydration Guide"}, {"title": "Meal Notes"}]

    normalized = normalize_inline_citations(content, citations)

    assert normalized == "Goal text [1]. Line marker [1]. Meal text [2]."


def test_merge_token_usage_adds_counts() -> None:
    target = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    merge_token_usage(target, {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})

    assert target == {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20}


def test_tool_messages_to_trace_summarizes_tool_result() -> None:
    trace = tool_messages_to_trace(
        [
            {
                "role": "tool",
                "name": "validate_sql_query",
                "content": '{"ok": true, "summary": "SQL query validated successfully.", "error": null}',
            }
        ]
    )

    assert trace == [
        {
            "name": "validate_sql_query",
            "ok": True,
            "summary": "SQL query validated successfully.",
            "error": None,
        }
    ]


def test_build_reasoning_summary_uses_visible_loop_data() -> None:
    summary = build_reasoning_summary(
        loop_trace=[
            {
                "iteration": 1,
                "assistant_content": "",
                "tool_calls": [{"name": "retrieve_docs"}],
                "tool_results": [{"name": "retrieve_docs", "ok": True, "summary": "Retrieved docs."}],
                "notices": [],
            }
        ],
        citations=[{"title": "Hydration Guide"}],
        iterations=1,
        tool_calls=1,
        subagent_trace=[{"status": "completed"}, {"status": "failed"}],
    )

    assert "Ran 1 model iteration and executed 1 tool call." in summary
    assert "Iteration 1: requested tool(s): retrieve_docs." in summary
    assert "Tool retrieve_docs succeeded: Retrieved docs." in summary
    assert "Evidence came from: Hydration Guide." in summary
    assert "Subagents ran: 1 completed, 1 failed." in summary


def test_extract_retrieval_detail_compacts_chunks_and_summaries() -> None:
    detail = extract_retrieval_detail(
        {
            "ok": True,
            "summary": "Retrieved 1 chunk.",
            "warnings": ["fallback search used"],
            "data": {
                "query": "hydration",
                "source": "keyword",
                "settings": {
                    "top_k": 4,
                    "min_score": 0.1,
                    "search_mode": "keyword",
                    "include_terms": ["ounces"],
                    "exclude_terms": [],
                },
                "summarizer_prompt": "Keep targets and units.",
                "unfiltered_count": 2,
                "chunks": [
                    {
                        "id": "doc1:0",
                        "title": "Hydration Guide",
                        "source_path": "sample_docs/hydration_guide.html",
                        "chunk_index": 0,
                        "score": 0.7,
                        "text": "Hydration details\nwith spacing.",
                    }
                ],
                "summaries": [
                    {
                        "id": "doc1:0",
                        "title": "Hydration Guide",
                        "source_path": "sample_docs/hydration_guide.html",
                        "ok": True,
                        "summary": "Relevant hydration details.",
                    }
                ],
            },
        }
    )

    assert detail["query"] == "hydration"
    assert detail["source"] == "keyword"
    assert detail["settings"]["top_k"] == 4
    assert detail["unfiltered_count"] == 2
    assert detail["summarizer_prompt"] == "Keep targets and units."
    assert detail["warnings"] == ["fallback search used"]
    assert detail["chunks"][0]["snippet"] == "Hydration details with spacing."
    assert detail["summaries"][0]["summary"] == "Relevant hydration details."


async def test_summarize_chunk_appends_tool_summarizer_prompt() -> None:
    settings = SimpleNamespace(
        agent=lambda name: SimpleNamespace(
            name=name,
            effective_system_instruction=lambda: "Base summarizer instruction.",
        )
    )
    runtime = SimpleNamespace(config=settings)
    runner = AgentRunner(runtime, tools=SimpleNamespace())
    captured = {}

    async def fake_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return {
            "choices": [{"message": {"content": "summary"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        }

    runner._chat_completion = fake_completion
    result = await runner.summarize_chunk(
        "hydration?",
        {
            "id": "doc1:0",
            "title": "Hydration Guide",
            "source_path": "sample_docs/hydration_guide.html",
            "text": "The sample goal is 72 ounces per day.",
        },
        SimpleNamespace(chat_id="chat", run_id="run"),
        summarizer_prompt="Keep hydration targets and units.",
    )

    user_prompt = captured["messages"][1]["content"]
    assert "Additional summarization guidance from the main agent" in user_prompt
    assert "Keep hydration targets and units." in user_prompt
    assert result["summarizer_prompt"] == "Keep hydration targets and units."
    assert result["token_usage"]["total_tokens"] == 15
    assert [message["role"] for message in result["conversation"]] == ["system", "user", "assistant"]
    assert result["conversation"][2]["content"] == "summary"
