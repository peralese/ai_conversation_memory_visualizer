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


def test_clustering_labels_are_unique(tmp_path):
    db = tmp_path / "labels.db"
    repo = SQLiteRepository(str(db))

    same_text = "prompted please create migration risk matrix"
    repo.upsert_bundle(_bundle("c1", 1, same_text))
    repo.upsert_bundle(_bundle("c2", 2, same_text))
    repo.save_embedding("m_1", [0.0, 0.0, 0.0, 0.0], "stub")
    repo.save_embedding("m_2", [1.0, 1.0, 1.0, 1.0], "stub")

    ClusteringService(repo).cluster_embeddings(k=2)
    labels = [c["label"] for c in repo.list_clusters()]
    assert len(labels) == 2
    assert labels[0] != labels[1]


def test_clustering_labels_exclude_domain_stopwords(tmp_path):
    db = tmp_path / "domain_labels.db"
    repo = SQLiteRepository(str(db))

    repo.upsert_bundle(_bundle("c1", 1, "aws migration cloud terraform terraform deployment"))
    repo.upsert_bundle(_bundle("c2", 2, "aws migration cloud kubernetes kubernetes platform"))
    repo.save_embedding("m_1", [0.0, 0.1, 0.0, 0.1], "stub")
    repo.save_embedding("m_2", [1.0, 0.9, 1.0, 0.9], "stub")

    ClusteringService(repo).cluster_embeddings(k=2)
    labels = [c["label"].lower() for c in repo.list_clusters()]
    assert any("terraform" in l or "kubernetes" in l for l in labels)
    assert all("aws" not in l and "migration" not in l and "cloud" not in l for l in labels)


def test_clustering_labels_exclude_application_tokens(tmp_path):
    db = tmp_path / "application_labels.db"
    repo = SQLiteRepository(str(db))

    repo.upsert_bundle(_bundle("c1", 1, "application applications platform roadmap"))
    repo.upsert_bundle(_bundle("c2", 2, "application applications platform strategy"))
    repo.save_embedding("m_1", [0.0, 0.1, 0.0, 0.1], "stub")
    repo.save_embedding("m_2", [1.0, 0.9, 1.0, 0.9], "stub")

    service = ClusteringService(repo)
    service.cluster_embeddings(k=2)
    labels = [c["label"].lower() for c in service.list_clusters()]
    assert all("application" not in l and "applications" not in l for l in labels)


def test_cluster_source_breakdown_counts(tmp_path):
    db = tmp_path / "breakdown.db"
    repo = SQLiteRepository(str(db))

    now = datetime.now(timezone.utc)
    bundles = [
        CanonicalConversationBundle(
            conversation=Conversation(
                id="cgpt",
                source=SourceType.CHATGPT,
                title="cgpt",
                created_at=now,
                updated_at=now,
                participants=["user", "assistant"],
                tags=[],
                raw_metadata={},
            ),
            messages=[
                Message(
                    id="m_cgpt",
                    conversation_id="cgpt",
                    timestamp=now.isoformat(),
                    speaker_role=SpeakerRole.USER,
                    text="strategy roadmap",
                    parent_message_id=None,
                    token_count=None,
                    raw_metadata={},
                )
            ],
        ),
        CanonicalConversationBundle(
            conversation=Conversation(
                id="cclaude",
                source=SourceType.CLAUDE,
                title="cclaude",
                created_at=now,
                updated_at=now,
                participants=["user", "assistant"],
                tags=[],
                raw_metadata={},
            ),
            messages=[
                Message(
                    id="m_claude",
                    conversation_id="cclaude",
                    timestamp=now.isoformat(),
                    speaker_role=SpeakerRole.ASSISTANT,
                    text="strategy roadmap",
                    parent_message_id=None,
                    token_count=None,
                    raw_metadata={},
                )
            ],
        ),
        CanonicalConversationBundle(
            conversation=Conversation(
                id="cgem",
                source=SourceType.GEMINI,
                title="cgem",
                created_at=now,
                updated_at=now,
                participants=["user", "assistant"],
                tags=[],
                raw_metadata={},
            ),
            messages=[
                Message(
                    id="m_gem",
                    conversation_id="cgem",
                    timestamp=now.isoformat(),
                    speaker_role=SpeakerRole.USER,
                    text="strategy roadmap",
                    parent_message_id=None,
                    token_count=None,
                    raw_metadata={},
                )
            ],
        ),
    ]
    for b in bundles:
        repo.upsert_bundle(b)

    repo.save_embedding("m_cgpt", [0.0, 0.0, 0.0, 0.0], "stub")
    repo.save_embedding("m_claude", [0.0, 0.0, 0.0, 0.1], "stub")
    repo.save_embedding("m_gem", [0.1, 0.1, 0.1, 0.1], "stub")

    service = ClusteringService(repo)
    service.cluster_embeddings(k=2)
    clusters = service.list_clusters()
    assert clusters

    detail = service.cluster_detail(int(clusters[0]["cluster_id"]))
    counts = detail["source_breakdown"]["counts"]
    assert set(counts.keys()) == {"CHATGPT", "CLAUDE", "GEMINI"}
    assert sum(counts.values()) == detail["message_count"]
