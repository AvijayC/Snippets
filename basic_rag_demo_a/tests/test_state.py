from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime, timedelta

from rag_demo.state import AppState


def test_upsert_ingested_doc_replaces_stale_hash_for_same_path(tmp_path: Path) -> None:
    state = AppState(tmp_path / "state.db")

    state.upsert_ingested_doc("old", "docs/a.md", "old-hash", 1.0, 1)
    state.upsert_ingested_doc("new", "docs/a.md", "new-hash", 2.0, 1)

    docs = state.list_ingested_docs()
    assert len(docs) == 1
    assert docs[0]["id"] == "new"
    assert docs[0]["content_hash"] == "new-hash"


def test_prune_ingested_docs_removes_missing_paths(tmp_path: Path) -> None:
    state = AppState(tmp_path / "state.db")

    state.upsert_ingested_doc("a", "docs/a.md", "hash-a", 1.0, 1)
    state.upsert_ingested_doc("b", "docs/b.md", "hash-b", 1.0, 1)

    removed = state.prune_ingested_docs(["docs/a.md"])

    assert removed == 1
    assert [doc["path"] for doc in state.list_ingested_docs()] == ["docs/a.md"]


def test_clear_chat_history_removes_chats_messages_and_debug(tmp_path: Path) -> None:
    state = AppState(tmp_path / "state.db")
    chat = state.create_chat()
    state.add_message(chat["id"], "user", "hello")
    state.add_debug_event("test_event", {"ok": True}, chat_id=chat["id"])

    result = state.clear_chat_history()

    assert result == {"chats_deleted": 1, "messages_deleted": 1, "debug_events_deleted": 1}
    assert state.list_chats() == []
    assert state.get_debug_events() == []


def test_get_debug_events_since_filters_event_type(tmp_path: Path) -> None:
    state = AppState(tmp_path / "state.db")
    since = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    state.add_debug_event("api_response", {"usage": {"total_tokens": 10}})
    state.add_debug_event("tool_call_completed", {"ok": True})

    events = state.get_debug_events_since(since, event_type="api_response")

    assert len(events) == 1
    assert events[0]["event_type"] == "api_response"
    assert events[0]["payload"]["usage"]["total_tokens"] == 10
