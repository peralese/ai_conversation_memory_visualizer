from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.parsers.base import BaseParser


class ChatGPTParser(BaseParser):
    def parse_file(self, file_path: str) -> list[dict[str, Any]]:
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "conversations" in data:
            conversations = data["conversations"]
        elif isinstance(data, list):
            conversations = data
        else:
            raise ValueError("Unsupported ChatGPT export format")

        out: list[dict[str, Any]] = []
        for conv in conversations:
            if not isinstance(conv, dict):
                continue
            out.append(conv)
        return out


def flatten_content_parts(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join([flatten_content_parts(part) for part in content if flatten_content_parts(part)]).strip()
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            values = [flatten_content_parts(p) for p in parts]
            return "\n".join([v for v in values if v]).strip()
        if "text" in content:
            return flatten_content_parts(content.get("text"))
    return ""
