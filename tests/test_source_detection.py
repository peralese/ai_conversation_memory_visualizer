from src.detection.source_detection import detect_source
from src.models import SourceType


def test_detect_chatgpt_by_filename():
    assert detect_source("fixtures/chatgpt_export_sample.json") == SourceType.CHATGPT


def test_detect_unknown_for_invalid_json(tmp_path):
    p = tmp_path / "foo.json"
    p.write_text("not-json", encoding="utf-8")
    assert detect_source(str(p)) == SourceType.UNKNOWN


def test_detect_claude_by_structure(tmp_path):
    p = tmp_path / "export.json"
    p.write_text('[{"source":"claude","chat_messages":[]}]', encoding="utf-8")
    assert detect_source(str(p)) == SourceType.CLAUDE


def test_detect_gemini_by_filename(tmp_path):
    p = tmp_path / "gemini_export.json"
    p.write_text("[]", encoding="utf-8")
    assert detect_source(str(p)) == SourceType.GEMINI
