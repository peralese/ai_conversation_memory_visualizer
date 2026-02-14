from datetime import datetime, timedelta, timezone

from src.clustering.service import ClusteringService
from src.models import CanonicalConversationBundle, Cluster, Conversation, Message, SourceType, SpeakerRole
from src.storage.repository import SQLiteRepository


def _conversation_with_messages(conv_id: str, source: SourceType, message_ids: list[str], prefix: str) -> CanonicalConversationBundle:
    now = datetime.now(timezone.utc)
    messages = []
    for idx, mid in enumerate(message_ids):
        ts = (now + timedelta(minutes=idx)).isoformat()
        messages.append(
            Message(
                id=mid,
                conversation_id=conv_id,
                timestamp=ts,
                speaker_role=SpeakerRole.USER if idx % 2 == 0 else SpeakerRole.ASSISTANT,
                text=f"{prefix} message {idx}",
                parent_message_id=None,
                token_count=None,
                raw_metadata={},
            )
        )
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
        messages=messages,
    )


def test_cluster_context_metrics_are_computed(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "metrics.db"))
    bundle = _conversation_with_messages("c1", SourceType.CHATGPT, ["m1", "m2", "m3"], "hello")
    repo.upsert_bundle(bundle)

    now = datetime.now(timezone.utc)
    clusters = [Cluster(cluster_id=1, label="topic", member_ids=["m1", "m2"], centroid=None, created_at=now)]
    memberships = [(1, "m1"), (1, "m2")]
    topic_events = [
        (1, bundle.messages[0].timestamp, "c1", "m1"),
        (1, bundle.messages[1].timestamp, "c1", "m2"),
    ]
    repo.replace_clusters(clusters, memberships, topic_events)

    detail = ClusteringService(repo).cluster_detail(1, include_subclusters=False)
    assert detail["message_count"] == 2
    assert detail["conversations_count"] == 1
    assert detail["dataset_percentage"] == 66.67
    assert isinstance(detail["average_message_length"], int)


def test_subclusters_created_when_threshold_exceeded(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "subclusters_large.db"))
    message_ids = [f"m{i}" for i in range(120)]
    bundle = _conversation_with_messages("big", SourceType.GEMINI, message_ids, "aws migration platform")
    repo.upsert_bundle(bundle)

    for i, mid in enumerate(message_ids):
        vec = [0.0, 0.0, 0.0, 0.0] if i < 60 else [1.0, 1.0, 1.0, 1.0]
        repo.save_embedding(mid, vec, "stub")

    now = datetime.now(timezone.utc)
    clusters = [Cluster(cluster_id=1, label="aws migration", member_ids=message_ids, centroid=[0.5, 0.5, 0.5, 0.5], created_at=now)]
    memberships = [(1, mid) for mid in message_ids]
    topic_events = [(1, bundle.messages[i].timestamp, "big", mid) for i, mid in enumerate(message_ids)]
    repo.replace_clusters(clusters, memberships, topic_events)

    result = ClusteringService(repo).subclusters_for_cluster(1)
    assert len(result["subclusters"]) >= 1
    assert sum(sc["message_count"] for sc in result["subclusters"]) <= 120


def test_no_subclusters_when_below_threshold(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "subclusters_small.db"))
    small_ids = [f"s{i}" for i in range(20)]
    big_ids = [f"b{i}" for i in range(220)]
    repo.upsert_bundle(_conversation_with_messages("small", SourceType.CLAUDE, small_ids, "small topic"))
    repo.upsert_bundle(_conversation_with_messages("other", SourceType.CHATGPT, big_ids, "other background data"))

    for i, mid in enumerate(small_ids):
        vec = [0.1, 0.1, 0.1, 0.1] if i % 2 == 0 else [0.2, 0.2, 0.2, 0.2]
        repo.save_embedding(mid, vec, "stub")

    now = datetime.now(timezone.utc)
    clusters = [Cluster(cluster_id=1, label="small", member_ids=small_ids, centroid=[0.15, 0.15, 0.15, 0.15], created_at=now)]
    memberships = [(1, mid) for mid in small_ids]
    topic_events = [(1, datetime.now(timezone.utc).isoformat(), "small", mid) for mid in small_ids]
    repo.replace_clusters(clusters, memberships, topic_events)

    result = ClusteringService(repo).subclusters_for_cluster(1)
    assert result["subclusters"] == []
