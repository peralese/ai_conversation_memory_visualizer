from src.parsers.chatgpt_parser import ChatGPTParser


def test_chatgpt_parser_reads_list_fixture():
    parser = ChatGPTParser()
    conversations = parser.parse_file("fixtures/chatgpt_export_sample.json")

    assert len(conversations) == 1
    assert conversations[0]["id"] == "conv_1"
    assert "mapping" in conversations[0]
