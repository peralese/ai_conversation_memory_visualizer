import zipfile
from pathlib import Path

import pytest

from src.parsers.claude_parser import ClaudeParser


def test_claude_parser_single_conversation_fixture():
    parser = ClaudeParser()
    conversations = parser.parse_file("tests/fixtures/claude/single_conversation.json")

    assert len(conversations) == 1
    conv = conversations[0]
    assert conv["id"] == "claude_conv_1"
    assert len(conv["messages"]) == 2
    assert conv["messages"][0]["role"] == "user"
    assert conv["messages"][1]["role"] == "assistant"
    assert conv["messages"][1]["text"] == "1) Calendar for focused work\n2) Async standup summarizer"
    assert conv["messages"][0]["timestamp"] == "2025-01-10T12:00:10Z"


def test_claude_parser_bulk_zip_fixture(tmp_path):
    zip_path = tmp_path / "claude_bulk.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write("tests/fixtures/claude/bulk_export.json", arcname="export/bulk_export.json")

    parser = ClaudeParser()
    conversations = parser.parse_file(str(zip_path))

    assert len(conversations) == 1
    assert len(conversations[0]["messages"]) == 2
    assert conversations[0]["messages"][1]["text"] == "Define goals\nCreate timeline"


def test_claude_parser_directory_fixture(tmp_path):
    fixture_dir = tmp_path / "claude_dir"
    fixture_dir.mkdir()
    (fixture_dir / "single.json").write_text(
        Path("tests/fixtures/claude/single_conversation.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    parser = ClaudeParser()
    conversations = parser.parse_file(str(fixture_dir))

    assert len(conversations) == 1
    assert conversations[0]["id"] == "claude_conv_1"


def test_claude_parser_fails_gracefully_for_non_claude_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{\"foo\": \"bar\"}", encoding="utf-8")

    parser = ClaudeParser()
    with pytest.raises(ValueError, match="No parseable Claude conversations found"):
        parser.parse_file(str(bad))
