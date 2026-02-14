from __future__ import annotations

import hashlib
import json
import logging
import uuid
import zipfile
from pathlib import Path
from typing import Any

from src.parsers.base import BaseParser

logger = logging.getLogger(__name__)


_CLAUDE_FILENAME_HINTS = ("claude", "conversation", "conversations", "chat", "chats")


class ClaudeParser(BaseParser):
    def parse(self, file_path: str) -> list[dict[str, Any]]:
        return self.parse_file(file_path)

    def parse_file(self, file_path: str) -> list[dict[str, Any]]:
        path = Path(file_path)
        sources = self._load_sources(path)

        parsed: list[dict[str, Any]] = []
        schema_counts: dict[str, int] = {}

        for source_name, payload in sources:
            conversations, schema_branch = _extract_conversation_candidates(payload)
            if not conversations:
                continue
            schema_counts[schema_branch] = schema_counts.get(schema_branch, 0) + len(conversations)
            for idx, conversation in enumerate(conversations):
                parsed_conv = _parse_conversation(conversation, source_name, idx)
                if parsed_conv is not None:
                    parsed.append(parsed_conv)

        if not parsed:
            raise ValueError(
                "No parseable Claude conversations found. Supported inputs are JSON, ZIP, or directory containing Claude-style JSON."
            )

        total_messages = sum(len(conv.get("messages") or []) for conv in parsed)
        logger.info(
            "Claude parser extracted %s conversations and %s messages from %s (schema branches: %s)",
            len(parsed),
            total_messages,
            file_path,
            schema_counts,
        )
        return parsed

    def _load_sources(self, path: Path) -> list[tuple[str, Any]]:
        if not path.exists():
            raise ValueError(f"Input path does not exist: {path}")

        if path.is_file() and path.suffix.lower() == ".json":
            return [(path.name, _read_json_file(path))]

        if path.is_file() and path.suffix.lower() == ".zip":
            return _load_json_from_zip(path)

        if path.is_dir():
            files = sorted([p for p in path.rglob("*.json") if p.is_file()])
            if not files:
                raise ValueError(f"No JSON files found in directory: {path}")
            return [(str(p.relative_to(path)), _read_json_file(p)) for p in files]

        raise ValueError(f"Unsupported Claude input type for path: {path}")


def _read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_from_zip(path: Path) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    with zipfile.ZipFile(path, "r") as zf:
        json_names = [name for name in zf.namelist() if name.lower().endswith(".json")]
        if not json_names:
            raise ValueError(f"No JSON files found in ZIP archive: {path}")

        json_names.sort(key=_zip_json_priority, reverse=True)
        for name in json_names:
            with zf.open(name) as f:
                out.append((name, json.load(f)))
    return out


def _zip_json_priority(name: str) -> tuple[int, int]:
    lower = name.lower()
    name_hint_score = sum(1 for hint in _CLAUDE_FILENAME_HINTS if hint in lower)
    return (name_hint_score, -len(lower))


def _extract_conversation_candidates(payload: Any) -> tuple[list[dict[str, Any]], str]:
    if isinstance(payload, dict):
        for key in ("conversations", "chats", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)], f"container:{key}"

        if "messages" in payload and isinstance(payload.get("messages"), list):
            return [payload], "single:messages"

        if "chat_messages" in payload and isinstance(payload.get("chat_messages"), list):
            return [payload], "single:chat_messages"

    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            sample = payload[0]
            if _looks_like_message(sample):
                return [{"messages": payload}], "list:messages"
            return [item for item in payload if isinstance(item, dict)], "list:conversations"
        return [], "list:unknown"

    return [], "unknown"


def _parse_conversation(conversation: dict[str, Any], source_name: str, index: int) -> dict[str, Any] | None:
    messages_raw = _extract_messages(conversation)
    parsed_messages: list[dict[str, Any]] = []

    conv_id = _as_str(
        conversation.get("id")
        or conversation.get("uuid")
        or conversation.get("conversation_id")
        or conversation.get("chat_id")
        or conversation.get("thread_id")
    )
    if not conv_id:
        conv_id = _deterministic_id(f"claude-conversation:{source_name}:{index}")

    for msg_index, message in enumerate(messages_raw):
        parsed = _parse_message(message, conv_id, msg_index)
        if parsed is not None:
            parsed_messages.append(parsed)

    if not parsed_messages:
        return None

    title = _as_str(conversation.get("title") or conversation.get("name") or conversation.get("chat_name"))
    if not title:
        title = f"Claude Conversation {conv_id[:8]}"

    created_at = conversation.get("created_at") or conversation.get("createdAt") or conversation.get("created")
    updated_at = conversation.get("updated_at") or conversation.get("updatedAt") or conversation.get("updated")
    if not created_at and parsed_messages:
        created_at = parsed_messages[0].get("timestamp")
    if not updated_at and parsed_messages:
        updated_at = parsed_messages[-1].get("timestamp")

    return {
        "id": conv_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at or created_at,
        "messages": parsed_messages,
        "raw_metadata": {
            "source_file": source_name,
            "schema_keys": sorted(conversation.keys()),
        },
    }


def _extract_messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("messages", "chat_messages", "entries", "turns", "items"):
        value = conversation.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _parse_message(message: dict[str, Any], conversation_id: str, index: int) -> dict[str, Any] | None:
    text = _extract_text(message)
    if not text:
        return None

    msg_id = _as_str(message.get("id") or message.get("uuid") or message.get("message_id"))
    if not msg_id:
        stable_seed = f"{conversation_id}:{index}:{text[:120]}"
        msg_id = _deterministic_id(stable_seed)

    role_raw = (
        message.get("role")
        or message.get("speaker")
        or message.get("sender")
        or (message.get("author") or {}).get("role")
        or (message.get("author") or {}).get("type")
        or "user"
    )

    timestamp = (
        message.get("timestamp")
        or message.get("created_at")
        or message.get("createdAt")
        or message.get("created")
        or message.get("time")
    )

    metadata = dict(message)
    metadata.pop("content", None)
    metadata.pop("text", None)
    metadata.pop("message", None)

    return {
        "id": msg_id,
        "role": _map_role(role_raw),
        "timestamp": timestamp,
        "text": text,
        "parent_message_id": _as_str(message.get("parent_message_id") or message.get("parent") or message.get("parentId")),
        "token_count": message.get("token_count"),
        "raw_metadata": metadata,
    }


def _map_role(role: Any) -> str:
    normalized = _as_str(role).lower()
    if normalized in ("assistant", "claude", "model", "ai", "bot"):
        return "assistant"
    if normalized in ("system", "developer"):
        return "system"
    if normalized in ("tool", "function"):
        return "tool"
    return "user"


def _extract_text(message: dict[str, Any]) -> str:
    for key in ("content", "text", "message", "body", "value"):
        if key in message:
            text = _flatten_text(message.get(key))
            if text:
                return text
    return ""


def _flatten_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _flatten_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        if "text" in content:
            return _flatten_text(content.get("text"))
        if "content" in content:
            return _flatten_text(content.get("content"))
        if "parts" in content:
            return _flatten_text(content.get("parts"))
        if content.get("type") == "text":
            return _flatten_text(content.get("value"))
    return ""


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _deterministic_id(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _looks_like_message(item: dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    if "role" in item and any(k in item for k in ("content", "text", "message")):
        return True
    return False
