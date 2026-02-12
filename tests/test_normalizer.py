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
