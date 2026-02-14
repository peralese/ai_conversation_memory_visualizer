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

_GEMINI_FILENAME_HINTS = ("gemini", "bard", "conversations", "my_activity", "takeout")
_GEMINI_PATH_HINTS = ("takeout", "my activity", "gemini")


class GeminiParser(BaseParser):
    def parse(self, file_path: str) -> list[dict[str, Any]]:
        return self.parse_file(file_path)

    def parse_file(self, file_path: str) -> list[dict[str, Any]]:
        return parse_gemini_export(file_path)


def parse_gemini_export(file_path: str) -> list[dict[str, Any]]:
    path = Path(file_path)
    sources = _load_sources(path)
    logger.info("Gemini parser inspecting %s JSON source files for %s", len(sources), file_path)

    parsed: list[dict[str, Any]] = []
    schema_counts: dict[str, int] = {}
    skipped_records = 0
    tried_files: list[str] = []

    for source_name, payload in sources:
        tried_files.append(source_name)
        candidates, schema_branch = _discover_candidates(payload)
        if not candidates:
            logger.info("Gemini parser skipped %s (no candidate conversations for branch %s)", source_name, schema_branch)
            continue

        extracted_for_file = 0
        for idx, candidate in enumerate(candidates):
            conversation = _parse_candidate(candidate, source_name, idx)
            if conversation is None:
                skipped_records += 1
                continue
            parsed.append(conversation)
            extracted_for_file += 1

        if extracted_for_file > 0:
            schema_counts[schema_branch] = schema_counts.get(schema_branch, 0) + extracted_for_file

    if not parsed:
        raise ValueError(
            "No parseable Gemini data found. Provide a Gemini JSON, Takeout ZIP, or extracted directory with Gemini activity files."
        )

    conversations_count = len(parsed)
    messages_count = sum(len(conv.get("messages") or []) for conv in parsed)
    logger.info(
        "Gemini parser extracted %s conversations and %s messages from %s (files tried: %s, schema branches: %s, skipped records: %s)",
        conversations_count,
        messages_count,
        file_path,
        tried_files,
        schema_counts,
        skipped_records,
    )
    return parsed


def _load_sources(path: Path) -> list[tuple[str, Any]]:
    if not path.exists():
        raise ValueError(f"Input path does not exist: {path}")

    if path.is_file() and path.suffix.lower() == ".json":
        return [(path.name, _read_json_file(path))]

    if path.is_file() and path.suffix.lower() == ".zip":
        return _load_json_from_zip(path)

    if path.is_dir():
        json_files = sorted([p for p in path.rglob("*.json") if p.is_file()])
        if not json_files:
            raise ValueError(f"No JSON files found in directory: {path}")
        json_files.sort(key=lambda p: _source_priority(str(p)), reverse=True)
        return [(str(p.relative_to(path)), _read_json_file(p)) for p in json_files]

    raise ValueError(f"Unsupported Gemini input type for path: {path}")


def _read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_from_zip(path: Path) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    with zipfile.ZipFile(path, "r") as zf:
        json_names = [name for name in zf.namelist() if name.lower().endswith(".json")]
        if not json_names:
            raise ValueError(f"No JSON files found in ZIP archive: {path}")
        json_names.sort(key=_source_priority, reverse=True)
        for name in json_names:
            with zf.open(name) as f:
                try:
                    out.append((name, json.load(f)))
                except Exception:
                    continue
    return out


def _source_priority(name: str) -> tuple[int, int]:
    lowered = name.lower()
    filename_score = sum(1 for hint in _GEMINI_FILENAME_HINTS if hint in lowered)
    path_score = sum(1 for hint in _GEMINI_PATH_HINTS if hint in lowered)
    return (filename_score + path_score, -len(lowered))


def _discover_candidates(payload: Any) -> tuple[list[dict[str, Any]], str]:
    if isinstance(payload, dict):
        for key in ("conversations", "items", "events", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                if key in ("events", "items") and value and _looks_like_activity_record(value[0]):
                    return [{"events": value, "source_container": key}], f"container:{key}:events"
                return [v for v in value if isinstance(v, dict)], f"container:{key}"
        if isinstance(payload.get("messages"), list) or isinstance(payload.get("turns"), list):
            return [payload], "single:conversation"
        if _looks_like_activity_record(payload):
            return [{"events": [payload], "source_container": "single_event"}], "single:event"

    if isinstance(payload, list):
        if not payload:
            return [], "list:empty"
        if all(isinstance(v, dict) for v in payload):
            sample = payload[0]
            if _looks_like_activity_record(sample):
                return [{"events": payload, "source_container": "list"}], "list:events"
            if _looks_like_conversation_record(sample):
                return [v for v in payload if isinstance(v, dict)], "list:conversations"
            if _looks_like_message_record(sample):
                return [{"messages": payload}], "list:messages"
        return [], "list:unknown"

    return [], "unknown"


def _parse_candidate(candidate: dict[str, Any], source_name: str, index: int) -> dict[str, Any] | None:
    if isinstance(candidate.get("events"), list):
        return _parse_event_activity_conversation(candidate.get("events"), source_name, index)
    return _parse_structured_conversation(candidate, source_name, index)


def _parse_structured_conversation(record: dict[str, Any], source_name: str, index: int) -> dict[str, Any] | None:
    message_records = _extract_message_records(record)
    messages: list[dict[str, Any]] = []
    fallback_ts: Any = None
    for msg_index, message in enumerate(message_records):
        parsed_message = _parse_message(message, msg_index, record)
        if parsed_message is None:
            continue
        messages.append(parsed_message)
        if fallback_ts is None:
            fallback_ts = parsed_message.get("timestamp")

    if not messages:
        return None

    explicit_id = _as_str(record.get("id") or record.get("conversation_id") or record.get("thread_id") or record.get("name"))
    title = _as_str(record.get("title") or record.get("name") or record.get("header"))
    created_at = (
        record.get("created_at")
        or record.get("create_time")
        or record.get("createTime")
        or record.get("created")
        or record.get("time")
        or fallback_ts
    )
    updated_at = (
        record.get("updated_at")
        or record.get("update_time")
        or record.get("updateTime")
        or record.get("lastModified")
        or messages[-1].get("timestamp")
        or created_at
    )
    conv_id = explicit_id or _deterministic_id(f"gemini:{source_name}:{title}:{created_at}:{index}")
    if not title:
        title = f"Gemini Conversation {conv_id[:8]}"

    return {
        "id": conv_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": messages,
        "raw_metadata": {
            "source_file": source_name,
            "schema_keys": sorted(record.keys()),
        },
    }


def _parse_event_activity_conversation(events: Any, source_name: str, index: int) -> dict[str, Any] | None:
    if not isinstance(events, list):
        return None

    messages: list[dict[str, Any]] = []
    created_at: Any = None
    updated_at: Any = None
    first_title = ""

    for event_idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        first_title = first_title or _as_str(event.get("title") or event.get("header"))
        event_ts = event.get("time") or event.get("timestamp") or event.get("time_usec") or event.get("timeUsec")
        if created_at is None and event_ts is not None:
            created_at = event_ts
        if event_ts is not None:
            updated_at = event_ts

        messages.extend(_messages_from_event(event, event_idx))

    if not messages:
        return None

    conv_seed = f"gemini-activity:{source_name}:{first_title}:{created_at}:{index}"
    conv_id = _deterministic_id(conv_seed)
    title = first_title or f"Gemini Conversation {conv_id[:8]}"

    return {
        "id": conv_id,
        "title": title,
        "created_at": created_at or messages[0].get("timestamp"),
        "updated_at": updated_at or messages[-1].get("timestamp"),
        "messages": messages,
        "raw_metadata": {
            "source_file": source_name,
            "schema_keys": ["events"],
            "event_count": len(events),
        },
    }


def _extract_message_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("messages", "turns", "chatMessages", "entries", "items"):
        value = record.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _parse_message(message: dict[str, Any], index: int, record: dict[str, Any]) -> dict[str, Any] | None:
    text = _extract_text(message)
    if not text:
        return None
    role_raw = (
        message.get("role")
        or message.get("author")
        or message.get("sender")
        or message.get("participant")
        or (message.get("author") or {}).get("role")
        or "user"
    )
    timestamp = (
        message.get("timestamp")
        or message.get("timestampMs")
        or message.get("time")
        or message.get("create_time")
        or message.get("created_at")
        or message.get("createdAt")
    )
    msg_id = _as_str(message.get("id") or message.get("message_id") or message.get("uuid"))
    if not msg_id:
        msg_id = _deterministic_id(f"gemini-msg:{record.get('id')}:{index}:{text[:120]}")

    metadata = dict(message)
    for key in ("content", "text", "parts", "message", "response", "prompt"):
        metadata.pop(key, None)

    return {
        "id": msg_id,
        "role": _map_role(role_raw),
        "timestamp": timestamp,
        "text": text,
        "parent_message_id": _as_str(message.get("parent_message_id") or message.get("parent") or message.get("parentId")),
        "token_count": message.get("token_count"),
        "raw_metadata": metadata,
    }


def _messages_from_event(event: dict[str, Any], event_idx: int) -> list[dict[str, Any]]:
    timestamp = event.get("time") or event.get("timestamp") or event.get("time_usec") or event.get("timeUsec")
    out: list[dict[str, Any]] = []

    prompt_obj = event.get("prompt") or event.get("user_query") or event.get("query") or event.get("input")
    response_obj = event.get("response") or event.get("model_response") or event.get("output") or event.get("answer")
    standalone_text = _extract_text(event)

    prompt_text = _flatten_text(prompt_obj)
    response_text = _flatten_text(response_obj)
    if prompt_text:
        prompt_id = _deterministic_id(f"gemini-event:{event_idx}:prompt:{prompt_text[:120]}")
        out.append(
            {
                "id": prompt_id,
                "role": "user",
                "timestamp": timestamp,
                "text": prompt_text,
                "parent_message_id": None,
                "token_count": None,
                "raw_metadata": {"event_id": event.get("id"), "event_type": event.get("type"), "source": "activity_prompt"},
            }
        )

    if response_text:
        response_id = _deterministic_id(f"gemini-event:{event_idx}:response:{response_text[:120]}")
        parent_id = out[-1]["id"] if out else None
        out.append(
            {
                "id": response_id,
                "role": "assistant",
                "timestamp": timestamp,
                "text": response_text,
                "parent_message_id": parent_id,
                "token_count": None,
                "raw_metadata": {"event_id": event.get("id"), "event_type": event.get("type"), "source": "activity_response"},
            }
        )

    if not out and standalone_text:
        msg_id = _deterministic_id(f"gemini-event:{event_idx}:standalone:{standalone_text[:120]}")
        out.append(
            {
                "id": msg_id,
                "role": "user",
                "timestamp": timestamp,
                "text": standalone_text,
                "parent_message_id": None,
                "token_count": None,
                "raw_metadata": {"event_id": event.get("id"), "event_type": event.get("type"), "source": "activity_standalone"},
            }
        )
    return out


def _extract_text(message: dict[str, Any]) -> str:
    for key in ("content", "text", "message", "parts", "prompt", "response", "value"):
        if key in message:
            extracted = _flatten_text(message.get(key))
            if extracted:
                return extracted
    return ""


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [part for part in (_flatten_text(v) for v in value) if part]
        return "\n\n".join(parts).strip()
    if isinstance(value, dict):
        if "text" in value:
            return _flatten_text(value.get("text"))
        if "content" in value:
            return _flatten_text(value.get("content"))
        if "parts" in value:
            return _flatten_text(value.get("parts"))
        if value.get("type") == "text":
            return _flatten_text(value.get("value"))
    return ""


def _map_role(role_value: Any) -> str:
    role = _as_str(role_value).lower()
    if role in ("assistant", "model", "gemini", "bard", "bot"):
        return "assistant"
    if role in ("system", "developer"):
        return "system"
    if role in ("tool", "function"):
        return "tool"
    return "user"


def _looks_like_activity_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    key_text = " ".join(str(k) for k in record.keys()).lower()
    value_text = " ".join(str(v) for v in [record.get("title"), record.get("header"), record.get("product"), record.get("products")]).lower()
    return (
        any(key in record for key in ("time", "timestamp", "time_usec", "timeUsec"))
        and ("gemini" in value_text or "bard" in value_text or "my activity" in value_text or "prompt" in key_text)
    )


def _looks_like_conversation_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    return isinstance(record.get("messages"), list) or isinstance(record.get("turns"), list)


def _looks_like_message_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    role = _as_str(record.get("role") or record.get("author") or record.get("sender")).lower()
    return role in ("assistant", "model", "gemini", "bard", "user", "human") and any(
        k in record for k in ("text", "content", "message", "parts")
    )


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _deterministic_id(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))
