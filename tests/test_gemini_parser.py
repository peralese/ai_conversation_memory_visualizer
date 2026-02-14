import zipfile
from pathlib import Path

import pytest

from src.parsers.gemini_parser import GeminiParser


def test_gemini_parser_conversations_fixture():
    parser = GeminiParser()
    path = "tests/fixtures/gemini/Takeout/My Activity/Gemini/Conversations.json"
    conversations = parser.parse_file(path)

    assert len(conversations) == 1
    conversation = conversations[0]
    assert conversation["id"] == "gemini_conv_1"
    assert len(conversation["messages"]) == 2
    assert conversation["messages"][0]["role"] == "user"
    assert conversation["messages"][1]["role"] == "assistant"
    assert conversation["messages"][1]["text"] == "Day 1: Pike Place Market\n\nDay 2: Museum + ferry ride"
    assert conversation["messages"][0]["timestamp"] == 1736690400000


def test_gemini_parser_activity_fixture():
    parser = GeminiParser()
    path = "tests/fixtures/gemini/Takeout/My Activity/Gemini/My_Activity.json"
    conversations = parser.parse_file(path)

    assert len(conversations) == 1
    conversation = conversations[0]
    assert len(conversation["messages"]) == 2
    assert conversation["messages"][0]["role"] == "user"
    assert conversation["messages"][1]["role"] == "assistant"
    assert conversation["messages"][1]["parent_message_id"] == conversation["messages"][0]["id"]
    assert conversation["messages"][1]["text"] == "Summary line 1\n\nSummary line 2"


def test_gemini_parser_zip_takeout(tmp_path):
    zip_path = tmp_path / "gemini_takeout.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(
            "tests/fixtures/gemini/Takeout/My Activity/Gemini/Conversations.json",
            arcname="Takeout/My Activity/Gemini/Conversations.json",
        )
        zf.write(
            "tests/fixtures/gemini/Takeout/My Activity/Gemini/My_Activity.json",
            arcname="Takeout/My Activity/Gemini/My_Activity.json",
        )

    parser = GeminiParser()
    conversations = parser.parse_file(str(zip_path))
    assert len(conversations) >= 1


def test_gemini_parser_directory_takeout(tmp_path):
    root = tmp_path / "Takeout" / "My Activity" / "Gemini"
    root.mkdir(parents=True)
    content = Path("tests/fixtures/gemini/Takeout/My Activity/Gemini/Conversations.json").read_text(encoding="utf-8")
    (root / "Conversations.json").write_text(content, encoding="utf-8")

    parser = GeminiParser()
    conversations = parser.parse_file(str(tmp_path))
    assert len(conversations) == 1


def test_gemini_parser_fails_gracefully_for_non_gemini_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{\"foo\": \"bar\"}", encoding="utf-8")

    parser = GeminiParser()
    with pytest.raises(ValueError, match="No parseable Gemini data found"):
        parser.parse_file(str(bad))
