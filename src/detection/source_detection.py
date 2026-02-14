from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from src.models import SourceType

_GEMINI_FILENAME_HINTS = ("gemini", "bard", "my_activity")
_GEMINI_PATH_HINTS = ("takeout", "my activity", "gemini")


def detect_source(file_path: str) -> SourceType:
    path = Path(file_path)
    name = path.name.lower()

    if path.is_dir():
        return _detect_from_directory(path)

    if path.suffix.lower() == ".zip":
        return _detect_from_zip(path)

    if "chatgpt" in name:
        return SourceType.CHATGPT
    if "claude" in name:
        return SourceType.CLAUDE
    if "gemini" in name or "bard" in name:
        return SourceType.GEMINI

    try:
        payload = _load_json(path)
    except Exception:
        return SourceType.UNKNOWN

    return _detect_by_structure(payload)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _detect_from_directory(path: Path) -> SourceType:
    json_files = sorted([p for p in path.rglob("*.json") if p.is_file()])
    json_files.sort(key=lambda p: _gemini_path_score(str(p)), reverse=True)
    for json_file in json_files:
        detected = detect_source(str(json_file))
        if detected != SourceType.UNKNOWN:
            return detected
    return SourceType.UNKNOWN


def _detect_from_zip(path: Path) -> SourceType:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                lower = name.lower()
                if not lower.endswith(".json"):
                    continue
                if "claude" in lower:
                    return SourceType.CLAUDE
                if "chatgpt" in lower:
                    return SourceType.CHATGPT
                if _gemini_path_score(lower) > 0:
                    return SourceType.GEMINI

                with zf.open(name) as f:
                    try:
                        payload = json.load(f)
                    except Exception:
                        continue
                detected = _detect_by_structure(payload)
                if detected != SourceType.UNKNOWN:
                    return detected
    except Exception:
        return SourceType.UNKNOWN
    return SourceType.UNKNOWN


def _detect_by_structure(payload: Any) -> SourceType:
    if isinstance(payload, list) and payload:
        sample = payload[0]
        if isinstance(sample, dict):
            if "mapping" in sample and "title" in sample:
                return SourceType.CHATGPT
            if "chat_messages" in sample or sample.get("source") == "claude":
                return SourceType.CLAUDE
            if "messages" in sample and _looks_like_claude_messages(sample.get("messages")):
                return SourceType.CLAUDE
            if "messages" in sample and sample.get("platform") == "gemini":
                return SourceType.GEMINI
            if _looks_like_claude_message(sample):
                return SourceType.CLAUDE
            if _looks_like_gemini_message(sample):
                return SourceType.GEMINI
            if _looks_like_gemini_activity_record(sample):
                return SourceType.GEMINI

    if isinstance(payload, dict):
        keys = set(payload.keys())
        if {"conversations", "mapping"}.issubset(keys):
            return SourceType.CHATGPT
        if "claude_conversations" in keys:
            return SourceType.CLAUDE
        if "gemini_threads" in keys:
            return SourceType.GEMINI
        for container_key in ("conversations", "chats", "data"):
            value = payload.get(container_key)
            if isinstance(value, list) and value:
                sample = value[0]
                if isinstance(sample, dict):
                    if "mapping" in sample:
                        return SourceType.CHATGPT
                    if "chat_messages" in sample or "claude" in str(sample.get("model", "")).lower():
                        return SourceType.CLAUDE
                    if "messages" in sample and _looks_like_claude_messages(sample.get("messages")):
                        return SourceType.CLAUDE
                    if _looks_like_gemini_conversation(sample):
                        return SourceType.GEMINI
        if "messages" in keys and _looks_like_claude_messages(payload.get("messages")):
            return SourceType.CLAUDE
        if _looks_like_gemini_conversation(payload):
            return SourceType.GEMINI
        if "events" in keys and _looks_like_gemini_events(payload.get("events")):
            return SourceType.GEMINI

    return SourceType.UNKNOWN


def _looks_like_claude_messages(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    sample = value[0]
    return isinstance(sample, dict) and _looks_like_claude_message(sample)


def _looks_like_claude_message(message: dict[str, Any]) -> bool:
    if not isinstance(message, dict):
        return False
    role = str(message.get("role") or message.get("sender") or "").lower()
    if role in ("assistant", "claude", "human", "user"):
        return any(k in message for k in ("content", "text", "message"))
    return False


def _looks_like_gemini_message(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    role = str(message.get("role") or message.get("author") or message.get("sender") or "").lower()
    if role in ("model", "assistant", "gemini", "bard", "user", "human"):
        return any(k in message for k in ("content", "text", "parts", "message"))
    return False


def _looks_like_gemini_conversation(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    if isinstance(record.get("turns"), list) or (
        isinstance(record.get("messages"), list) and _looks_like_gemini_messages(record.get("messages"))
    ):
        return True
    model_hint = str(record.get("model") or record.get("app") or record.get("platform") or "").lower()
    if "gemini" in model_hint or "bard" in model_hint:
        return True
    return False


def _looks_like_gemini_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    sample = messages[0]
    return _looks_like_gemini_message(sample)


def _looks_like_gemini_events(events: Any) -> bool:
    if not isinstance(events, list) or not events:
        return False
    sample = events[0]
    return _looks_like_gemini_activity_record(sample)


def _looks_like_gemini_activity_record(sample: Any) -> bool:
    if not isinstance(sample, dict):
        return False
    title = str(sample.get("title") or sample.get("header") or "").lower()
    product = str(sample.get("product") or sample.get("products") or "").lower()
    has_time = any(k in sample for k in ("time", "timestamp", "time_usec", "timeUsec"))
    return has_time and (
        "gemini" in title
        or "bard" in title
        or "gemini" in product
        or "bard" in product
        or "my activity" in title
        or "prompt" in sample
    )


def _gemini_path_score(path_name: str) -> int:
    lowered = path_name.lower()
    score = sum(1 for hint in _GEMINI_FILENAME_HINTS if hint in lowered)
    score += sum(1 for hint in _GEMINI_PATH_HINTS if hint in lowered)
    return score
