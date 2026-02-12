from __future__ import annotations

from src.models import SourceType
from src.parsers.base import BaseParser
from src.parsers.chatgpt_parser import ChatGPTParser
from src.parsers.claude_parser import ClaudeParser
from src.parsers.gemini_parser import GeminiParser


def parser_for_source(source: SourceType) -> BaseParser:
    if source == SourceType.CHATGPT:
        return ChatGPTParser()
    if source == SourceType.CLAUDE:
        return ClaudeParser()
    if source == SourceType.GEMINI:
        return GeminiParser()
    raise ValueError(f"No parser implemented for source={source}")
