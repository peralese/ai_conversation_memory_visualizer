from datetime import datetime, timedelta, timezone

from src.metrics.drift_service import DriftService
from src.models import CanonicalConversationBundle, Conversation, Message, SourceType, SpeakerRole
from src.storage.repository import SQLiteRepository


def _bundle(conv_id: str, start: datetime, message_ids: list[str]) -> CanonicalConversationBundle:
    return CanonicalConversationBundle(
        conversation=Conversation(
            id=conv_id,
            source=SourceType.CHATGPT,
            title=conv_id,
            created_at=start,
            updated_at=start,
            participants=["user", "assistant"],
            tags=[],
            raw_metadata={},
        ),
        messages=[
            Message(
                id=mid,
                conversation_id=conv_id,
                timestamp=(start + timedelta(minutes=i)).isoformat(),
                speaker_role=SpeakerRole.USER,
                text=f"text {mid}",
                parent_message_id=None,
                token_count=None,
                raw_metadata={},
            )
            for i, mid in enumerate(message_ids)
        ],
    )


def test_drift_computation_and_sorted_buckets(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "drift.db"))
    w1 = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)   # Monday
    w2 = datetime(2026, 1, 12, 10, 0, tzinfo=timezone.utc)  # next Monday

    repo.upsert_bundle(_bundle("c1", w1, ["m1", "m2", "m3"]))
    repo.upsert_bundle(_bundle("c2", w2, ["m4", "m5", "m6"]))

    # Cluster with 6 msgs across two weeks
    with repo.connection() as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO clusters(cluster_id, label, centroid_json, first_seen, last_seen, message_count, created_at) VALUES (1,'topic',NULL,?,?,6,?)",
            (now, now, now),
        )
        conn.executemany(
            "INSERT INTO cluster_membership(cluster_id, message_id) VALUES (1, ?)",
            [("m1",), ("m2",), ("m3",), ("m4",), ("m5",), ("m6",)],
        )

    # Week 1 centroid [1,0], week 2 centroid [0,1] => cosine distance ~1.0
    repo.save_embedding("m1", [1.0, 0.0], "stub")
    repo.save_embedding("m2", [1.0, 0.0], "stub")
    repo.save_embedding("m3", [1.0, 0.0], "stub")
    repo.save_embedding("m4", [0.0, 1.0], "stub")
    repo.save_embedding("m5", [0.0, 1.0], "stub")
    repo.save_embedding("m6", [0.0, 1.0], "stub")

    service = DriftService(repo)
    service.compute_and_persist(level="cluster")

    summary = service.summary(level="cluster")
    assert summary
    assert summary[0]["cluster_id"] == "1"
    assert summary[0]["cumulative_drift"] == 1.0

    detail = service.detail(level="cluster", cluster_id="1")
    buckets = [r["bucket_start_date"] for r in detail["series"]]
    assert buckets == sorted(buckets)
