from datetime import datetime, timezone

from src.clustering.service import ClusteringService
from src.models import CanonicalConversationBundle, Conversation, Message, SourceType, SpeakerRole
from src.storage.repository import SQLiteRepository


def _bundle(conv_id: str, idx: int, text: str) -> CanonicalConversationBundle:
    now = datetime.now(timezone.utc)
    return CanonicalConversationBundle(
        conversation=Conversation(
            id=conv_id,
            source=SourceType.CHATGPT,
            title=f"Conversation {conv_id}",
            created_at=now,
            updated_at=now,
            participants=["user", "assistant"],
            tags=[],
            raw_metadata={},
        ),
        messages=[
            Message(
                id=f"m_{idx}",
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


def test_clustering_basic(tmp_path):
    db = tmp_path / "test.db"
    repo = SQLiteRepository(str(db))

    repo.upsert_bundle(_bundle("c1", 1, "python clustering topic model"))
    repo.upsert_bundle(_bundle("c2", 2, "gardening soil plants watering"))
    repo.save_embedding("m_1", [0.1, 0.2, 0.3, 0.4], "stub")
    repo.save_embedding("m_2", [0.9, 0.8, 0.7, 0.6], "stub")

    result = ClusteringService(repo).cluster_embeddings(k=2)
    assert result["clusters"] == 2
    assert len(repo.list_clusters()) == 2
