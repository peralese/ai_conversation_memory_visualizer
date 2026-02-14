from datetime import datetime, timezone

from src.metrics.specialization_service import ModelSpecializationService
from src.models import CanonicalConversationBundle, Conversation, Message, SourceType, SpeakerRole
from src.storage.repository import SQLiteRepository


def _bundle(conv_id: str, source: SourceType, message_ids: list[str]) -> CanonicalConversationBundle:
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
                id=mid,
                conversation_id=conv_id,
                timestamp=now.isoformat(),
                speaker_role=SpeakerRole.USER,
                text=f"text-{mid}",
                parent_message_id=None,
                token_count=None,
                raw_metadata={},
            )
            for mid in message_ids
        ],
    )


def test_model_specialization_lift_and_dominant_source(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "spec.db"))

    repo.upsert_bundle(_bundle("gpt", SourceType.CHATGPT, [f"g{i}" for i in range(1, 7)]))  # 6
    repo.upsert_bundle(_bundle("claude", SourceType.CLAUDE, [f"c{i}" for i in range(1, 4)]))  # 3
    repo.upsert_bundle(_bundle("gem", SourceType.GEMINI, ["m1"]))  # 1

    # Cluster 1 has 5 msgs: 3 CLAUDE + 2 CHATGPT (dominant lift should be CLAUDE: 0.6 / 0.3 = 2.0)
    now = datetime.now(timezone.utc).isoformat()
    with repo.connection() as conn:
        conn.execute(
            "INSERT INTO clusters(cluster_id, label, centroid_json, first_seen, last_seen, message_count, created_at) VALUES (1,'x',NULL,?,?,5,?)",
            (now, now, now),
        )
        memberships = [(1, "c1"), (1, "c2"), (1, "c3"), (1, "g1"), (1, "g2")]
        conn.executemany("INSERT INTO cluster_membership(cluster_id, message_id) VALUES (?, ?)", memberships)

    out = ModelSpecializationService(repo).compute(level="cluster")
    assert out["baseline"]["CHATGPT"]["count"] == 6
    assert out["baseline"]["CLAUDE"]["count"] == 3
    assert out["baseline"]["GEMINI"]["count"] == 1

    row = out["items"][0]
    assert row["cluster_id"] == 1
    assert row["source_breakdown"]["counts"]["CLAUDE"] == 3
    assert row["source_breakdown"]["percents"]["CLAUDE"] == 60.0
    assert row["lift_by_source"]["CLAUDE"] == 2.0
    assert row["dominant_source"] == "CLAUDE"


def test_specialization_skips_zero_baseline_sources(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "spec_zero_baseline.db"))
    # No CHATGPT data at all.
    repo.upsert_bundle(_bundle("claude", SourceType.CLAUDE, [f"c{i}" for i in range(1, 5)]))
    repo.upsert_bundle(_bundle("gem", SourceType.GEMINI, [f"g{i}" for i in range(1, 3)]))

    now = datetime.now(timezone.utc).isoformat()
    with repo.connection() as conn:
        conn.execute(
            "INSERT INTO clusters(cluster_id, label, centroid_json, first_seen, last_seen, message_count, created_at) VALUES (1,'x',NULL,?,?,4,?)",
            (now, now, now),
        )
        conn.executemany(
            "INSERT INTO cluster_membership(cluster_id, message_id) VALUES (1, ?)",
            [("c1",), ("c2",), ("c3",), ("g1",)],
        )

    out = ModelSpecializationService(repo).compute(level="cluster")
    assert out["baseline_available"]["CHATGPT"] is False
    row = out["items"][0]
    assert "CHATGPT" not in row["lift_by_source"]
    assert row["dominant_source"] in {"CLAUDE", "GEMINI"}
    assert row["dominant_lift"] is not None
