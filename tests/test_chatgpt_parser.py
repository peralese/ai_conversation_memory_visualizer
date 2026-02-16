import zipfile

from src.parsers.chatgpt_parser import ChatGPTParser


def test_chatgpt_parser_reads_list_fixture():
    parser = ChatGPTParser()
    conversations = parser.parse_file("fixtures/chatgpt_export_sample.json")

    assert len(conversations) == 1
    assert conversations[0]["id"] == "conv_1"
    assert "mapping" in conversations[0]


def test_chatgpt_parser_reads_zip_fixture(tmp_path):
    zip_path = tmp_path / "openai_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write("fixtures/chatgpt_export_sample.json", arcname="conversations.json")
        zf.writestr("metadata.json", '{"version":1}')

    parser = ChatGPTParser()
    conversations = parser.parse_file(str(zip_path))

    assert len(conversations) == 1
    assert conversations[0]["id"] == "conv_1"


def test_chatgpt_parser_reads_directory_fixture(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "conversations.json").write_text(
        '[{"id":"conv_123","mapping":{"a":{"message":{"author":{"role":"user"},"content":{"parts":["hello"]}}}}}]',
        encoding="utf-8",
    )

    parser = ChatGPTParser()
    conversations = parser.parse_file(str(export_dir))

    assert len(conversations) == 1
    assert conversations[0]["id"] == "conv_123"
