from rag_demo.agent import parse_printed_tool_calls


def test_parse_single_printed_tool_call() -> None:
    calls = parse_printed_tool_calls(
        '{"name": "validate_sql_query", "arguments": {"sql": "select * from foods"}}'
    )

    assert len(calls) == 1
    assert calls[0]["name"] == "validate_sql_query"
    assert "select * from foods" in calls[0]["arguments"]
    assert calls[0]["source"] == "printed_json"


def test_parse_openai_like_printed_tool_calls() -> None:
    calls = parse_printed_tool_calls(
        """
        ```json
        {
          "tool_calls": [
            {
              "id": "call_1",
              "function": {
                "name": "retrieve_docs",
                "arguments": {"query": "hydration goal"}
              }
            }
          ]
        }
        ```
        """
    )

    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["name"] == "retrieve_docs"


def test_parse_ignores_non_json_text() -> None:
    assert parse_printed_tool_calls("I can answer directly.") == []
