from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class AppState:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(
                """
                create table if not exists chats (
                    id text primary key,
                    title text not null,
                    created_at text not null,
                    updated_at text not null
                );
                create table if not exists messages (
                    id text primary key,
                    chat_id text not null references chats(id) on delete cascade,
                    role text not null,
                    content text not null,
                    metadata_json text not null default '{}',
                    created_at text not null
                );
                create index if not exists idx_messages_chat_created
                    on messages(chat_id, created_at);
                create table if not exists debug_events (
                    id integer primary key autoincrement,
                    chat_id text,
                    run_id text,
                    event_type text not null,
                    payload_json text not null,
                    created_at text not null
                );
                create index if not exists idx_debug_chat_id
                    on debug_events(chat_id, id);
                create table if not exists config_snapshots (
                    id integer primary key autoincrement,
                    config_json text not null,
                    created_at text not null
                );
                create table if not exists ingested_docs (
                    id text primary key,
                    path text not null,
                    content_hash text not null,
                    mtime real not null,
                    chunks integer not null,
                    metadata_json text not null default '{}',
                    indexed_at text not null
                );
                """
            )

    def create_chat(self, title: str = "New chat") -> dict[str, Any]:
        chat_id = uuid.uuid4().hex
        now = utc_now()
        with self._lock, self.connect() as conn:
            conn.execute(
                "insert into chats (id, title, created_at, updated_at) values (?, ?, ?, ?)",
                (chat_id, title, now, now),
            )
        return {"id": chat_id, "title": title, "created_at": now, "updated_at": now}

    def list_chats(self) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            rows = conn.execute(
                "select id, title, created_at, updated_at from chats order by updated_at desc"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                "select id, title, created_at, updated_at from chats where id = ?",
                (chat_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_chat_title(self, chat_id: str, title: str) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                "update chats set title = ?, updated_at = ? where id = ?",
                (title[:120] or "New chat", utc_now(), chat_id),
            )

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = uuid.uuid4().hex
        now = utc_now()
        payload = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                insert into messages (id, chat_id, role, content, metadata_json, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (message_id, chat_id, role, content, payload, now),
            )
            conn.execute("update chats set updated_at = ? where id = ?", (now, chat_id))
        return {
            "id": message_id,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": now,
        }

    def get_messages(self, chat_id: str) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            rows = conn.execute(
                """
                select id, chat_id, role, content, metadata_json, created_at
                from messages where chat_id = ? order by created_at asc
                """,
                (chat_id,),
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            messages.append(item)
        return messages

    def clear_chat_history(self) -> dict[str, int]:
        with self._lock, self.connect() as conn:
            message_count = conn.execute("select count(*) from messages").fetchone()[0]
            chat_count = conn.execute("select count(*) from chats").fetchone()[0]
            debug_count = conn.execute("select count(*) from debug_events").fetchone()[0]
            conn.execute("delete from messages")
            conn.execute("delete from chats")
            conn.execute("delete from debug_events")
        return {
            "chats_deleted": int(chat_count),
            "messages_deleted": int(message_count),
            "debug_events_deleted": int(debug_count),
        }

    def add_debug_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        chat_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                insert into debug_events (chat_id, run_id, event_type, payload_json, created_at)
                values (?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    run_id,
                    event_type,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    utc_now(),
                ),
            )

    def get_debug_events(
        self,
        chat_id: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if chat_id:
            sql = """
                select id, chat_id, run_id, event_type, payload_json, created_at
                from debug_events where chat_id = ? order by id desc limit ?
            """
            params: tuple[Any, ...] = (chat_id, limit)
        else:
            sql = """
                select id, chat_id, run_id, event_type, payload_json, created_at
                from debug_events order by id desc limit ?
            """
            params = (limit,)
        with self._lock, self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        events = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
            events.append(item)
        events.reverse()
        return events

    def get_debug_events_since(
        self,
        since: str,
        event_type: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        clauses = ["created_at >= ?"]
        params: list[Any] = [since]
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        params.append(limit)
        sql = f"""
            select id, chat_id, run_id, event_type, payload_json, created_at
            from debug_events
            where {' and '.join(clauses)}
            order by id asc
            limit ?
        """
        with self._lock, self.connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        events = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
            events.append(item)
        return events

    def save_config(self, config: dict[str, Any]) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(
                "insert into config_snapshots (config_json, created_at) values (?, ?)",
                (json.dumps(config, ensure_ascii=False, default=str), utc_now()),
            )

    def latest_config(self) -> dict[str, Any] | None:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                "select config_json from config_snapshots order by id desc limit 1"
            ).fetchone()
        return json.loads(row["config_json"]) if row else None

    def upsert_ingested_doc(
        self,
        doc_id: str,
        path: str,
        content_hash: str,
        mtime: float,
        chunks: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self.connect() as conn:
            conn.execute("delete from ingested_docs where path = ? and id <> ?", (path, doc_id))
            conn.execute(
                """
                insert into ingested_docs
                    (id, path, content_hash, mtime, chunks, metadata_json, indexed_at)
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                    path = excluded.path,
                    content_hash = excluded.content_hash,
                    mtime = excluded.mtime,
                    chunks = excluded.chunks,
                    metadata_json = excluded.metadata_json,
                    indexed_at = excluded.indexed_at
                """,
                (
                    doc_id,
                    path,
                    content_hash,
                    mtime,
                    chunks,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    utc_now(),
                ),
            )

    def prune_ingested_docs(self, active_paths: list[str]) -> int:
        with self._lock, self.connect() as conn:
            if not active_paths:
                cursor = conn.execute("delete from ingested_docs")
            else:
                placeholders = ", ".join("?" for _ in active_paths)
                cursor = conn.execute(
                    f"delete from ingested_docs where path not in ({placeholders})",
                    tuple(active_paths),
                )
            return cursor.rowcount

    def list_ingested_docs(self) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            rows = conn.execute(
                """
                select id, path, content_hash, mtime, chunks, metadata_json, indexed_at
                from ingested_docs order by path
                """
            ).fetchall()
        docs = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            docs.append(item)
        return docs
