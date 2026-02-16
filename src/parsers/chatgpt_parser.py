from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from src.parsers.base import BaseParser

_CHATGPT_FILENAME_HINTS = ("chatgpt", "conversation", "conversations")


class ChatGPTParser(BaseParser):
    def parse_file(self, file_path: str) -> list[dict[str, Any]]:
        path = Path(file_path)
        sources = self._load_sources(path)

        out: list[dict[str, Any]] = []
        for _, payload in sources:
            conversations = _extract_conversations(payload)
            for conv in conversations:
                if isinstance(conv, dict):
                    out.append(conv)

        if not out:
            raise ValueError(
                "No parseable ChatGPT conversations found. Supported inputs are JSON, ZIP, or directory containing ChatGPT-style JSON."
            )
        return out

    def _load_sources(self, path: Path) -> list[tuple[str, Any]]:
        if not path.exists():
            raise ValueError(f"Input path does not exist: {path}")

        if path.is_file() and path.suffix.lower() == ".json":
            return [(path.name, _read_json_file(path))]

        if path.is_file() and path.suffix.lower() == ".zip":
            return _load_json_from_zip(path)

        if path.is_dir():
            files = sorted([p for p in path.rglob("*.json") if p.is_file()])
            files.sort(key=lambda p: _source_priority(str(p)), reverse=True)
            if not files:
                raise ValueError(f"No JSON files found in directory: {path}")
            return [(str(p.relative_to(path)), _read_json_file(p)) for p in files]

        raise ValueError(f"Unsupported ChatGPT input type for path: {path}")


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
    score = sum(1 for hint in _CHATGPT_FILENAME_HINTS if hint in lowered)
    return (score, -len(lowered))


def _extract_conversations(payload: Any) -> list[Any]:
    if isinstance(payload, dict) and isinstance(payload.get("conversations"), list):
        return payload["conversations"]
    if isinstance(payload, list):
        return payload
    return []


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
