import zipfile

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


def test_detect_claude_fixture_json():
    assert detect_source("tests/fixtures/claude/single_conversation.json") == SourceType.CLAUDE


def test_detect_claude_from_zip_fixture(tmp_path):
    zip_path = tmp_path / "claude_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write("tests/fixtures/claude/bulk_export.json", arcname="conversations.json")
    assert detect_source(str(zip_path)) == SourceType.CLAUDE


def test_detect_claude_from_directory_fixture(tmp_path):
    p = tmp_path / "exports"
    p.mkdir()
    (p / "single.json").write_text(
        '{"messages":[{"role":"human","content":"hello"},{"role":"assistant","content":"hi"}]}',
        encoding="utf-8",
    )
    assert detect_source(str(p)) == SourceType.CLAUDE


def test_detect_gemini_by_filename(tmp_path):
    p = tmp_path / "gemini_export.json"
    p.write_text("[]", encoding="utf-8")
    assert detect_source(str(p)) == SourceType.GEMINI
