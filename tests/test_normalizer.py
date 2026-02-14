import json
from pathlib import Path

from src.models import SourceType
from src.normalize.normalizer import normalize


def test_normalize_chatgpt_bundle():
    payload = json.loads(Path("fixtures/chatgpt_export_sample.json").read_text(encoding="utf-8"))[0]
    bundle = normalize(payload, SourceType.CHATGPT)

    assert bundle.conversation.id == "conv_1"
    assert bundle.conversation.source == SourceType.CHATGPT
    assert len(bundle.messages) == 3
    assert bundle.messages[0].speaker_role.value == "user"
    assert bundle.messages[2].speaker_role.value == "tool"
    assert bundle.messages[1].parent_message_id == "node_1"
    assert bundle.messages[0].timestamp.endswith("+00:00")


def test_normalize_claude_bundle():
    payload = {
        "id": "claude_conv_1",
        "title": "Claude Test",
        "created_at": "2025-01-10T12:00:00Z",
        "updated_at": "2025-01-10T12:02:00Z",
        "messages": [
            {"id": "m1", "role": "user", "timestamp": "2025-01-10T12:00:10Z", "text": "Hi"},
            {"id": "m2", "role": "assistant", "timestamp": "2025-01-10T12:00:20Z", "text": "Hello"},
        ],
    }
    bundle = normalize(payload, SourceType.CLAUDE)

    assert bundle.conversation.id == "claude_conv_1"
    assert bundle.conversation.source == SourceType.CLAUDE
    assert len(bundle.messages) == 2
    assert bundle.messages[0].speaker_role.value == "user"
    assert bundle.messages[1].speaker_role.value == "assistant"
