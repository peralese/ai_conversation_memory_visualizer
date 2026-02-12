from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models import SourceType


def detect_source(file_path: str) -> SourceType:
    path = Path(file_path)
    name = path.name.lower()

    if "chatgpt" in name or "conversations" in name:
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


def _detect_by_structure(payload: Any) -> SourceType:
    if isinstance(payload, list) and payload:
        sample = payload[0]
        if isinstance(sample, dict):
            if "mapping" in sample and "title" in sample:
                return SourceType.CHATGPT
            if "chat_messages" in sample or sample.get("source") == "claude":
                return SourceType.CLAUDE
            if "messages" in sample and sample.get("platform") == "gemini":
                return SourceType.GEMINI

    if isinstance(payload, dict):
        keys = set(payload.keys())
        if {"conversations", "mapping"}.issubset(keys):
            return SourceType.CHATGPT
        if "claude_conversations" in keys:
            return SourceType.CLAUDE
        if "gemini_threads" in keys:
            return SourceType.GEMINI

    return SourceType.UNKNOWN
