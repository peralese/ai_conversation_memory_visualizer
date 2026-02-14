from datetime import datetime, timezone

from src.metrics.service import MetricsService
from src.models import CanonicalConversationBundle, Conversation, Message, SourceType, SpeakerRole
from src.storage.repository import SQLiteRepository


def _bundle(conv_id: str, source: SourceType, msg_id: str, text: str) -> CanonicalConversationBundle:
    now = datetime.now(timezone.utc)
    return CanonicalConversationBundle(
        conversation=Conversation(
            id=conv_id,
            source=source,
            title=conv_id,
            created_at=now,
            updated_at=now,
            participants=["user", "assistant"],
            tags=[],
            raw_metadata={},
        ),
        messages=[
            Message(
                id=msg_id,
                conversation_id=conv_id,
                timestamp=now.isoformat(),
                speaker_role=SpeakerRole.USER,
                text=text,
                parent_message_id=None,
                token_count=None,
                raw_metadata={},
            )
        ],
    )


def test_dataset_profile_summary_and_by_source(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "profile.db"))
    repo.upsert_bundle(_bundle("cgpt_1", SourceType.CHATGPT, "m1", "AWS migration cloud architecture"))
    repo.upsert_bundle(_bundle("gem_1", SourceType.GEMINI, "m2", "Claude summary about kubernetes and cloud"))

    profile = MetricsService(repo).dataset_profile(top_n=10)

    assert profile["total_messages"] == 2
    assert profile["total_conversations"] == 2
    assert "by_source" in profile
    assert "CHATGPT" in profile["by_source"]
    assert "GEMINI" in profile["by_source"]
    assert profile["domain_token_pct"]["cloud"] == 100.0
