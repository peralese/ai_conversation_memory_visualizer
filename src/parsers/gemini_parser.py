from __future__ import annotations

from typing import Any

from src.parsers.base import BaseParser


class GeminiParser(BaseParser):
    def parse_file(self, file_path: str) -> list[dict[str, Any]]:
        # TODO: Implement Gemini JSON parsing heuristics.
        return []
