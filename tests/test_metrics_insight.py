from datetime import datetime, timezone

from src.metrics.service import MetricsService
from src.models import CanonicalConversationBundle, Cluster, Conversation, Message, SourceType, SpeakerRole
from src.storage.repository import SQLiteRepository


def _bundle(conv_id: str, msg_id: str, ts: str, text: str = "text") -> CanonicalConversationBundle:
    dt = datetime.fromisoformat(ts)
    return CanonicalConversationBundle(
        conversation=Conversation(
            id=conv_id,
            source=SourceType.GEMINI,
            title=conv_id,
            created_at=dt,
            updated_at=dt,
            participants=["user", "assistant"],
            tags=[],
            raw_metadata={},
        ),
        messages=[
            Message(
                id=msg_id,
                conversation_id=conv_id,
                timestamp=ts,
                speaker_role=SpeakerRole.USER,
                text=text,
                parent_message_id=None,
                token_count=None,
                raw_metadata={},
            )
        ],
    )


def test_topic_evolution_returns_chronological_week_start(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "metrics.db"))
    repo.upsert_bundle(_bundle("c1", "m1", "2026-02-10T10:00:00+00:00"))
    repo.upsert_bundle(_bundle("c2", "m2", "2026-01-20T10:00:00+00:00"))

    now = datetime.now(timezone.utc)
    clusters = [
        Cluster(cluster_id=1, label="alpha", member_ids=["m1"], centroid=None, created_at=now),
        Cluster(cluster_id=2, label="beta", member_ids=["m2"], centroid=None, created_at=now),
    ]
    memberships = [(1, "m1"), (2, "m2")]
    topic_events = [
        (1, "2026-02-10T10:00:00+00:00", "c1", "m1"),
        (2, "2026-01-20T10:00:00+00:00", "c2", "m2"),
    ]
    repo.replace_clusters(clusters, memberships, topic_events)

    rows = MetricsService(repo).topic_evolution(granularity="week")
    week_starts = [row["week_start"] for row in rows]
    assert week_starts == sorted(week_starts)
    assert rows[0]["bucket"].startswith("2026-W")


def test_idea_half_life_returns_none_for_single_active_bucket(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "half_life_single.db"))
    repo.upsert_bundle(_bundle("c1", "m1", "2026-02-10T10:00:00+00:00"))

    now = datetime.now(timezone.utc)
    clusters = [Cluster(cluster_id=1, label="alpha", member_ids=["m1"], centroid=None, created_at=now)]
    memberships = [(1, "m1")]
    topic_events = [(1, "2026-02-10T10:00:00+00:00", "c1", "m1")]
    repo.replace_clusters(clusters, memberships, topic_events)

    out = MetricsService(repo).idea_half_life()
    assert out[0]["half_life_weeks"] is None


def test_idea_half_life_counts_weeks_after_peak(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "half_life_peak.db"))
    for idx, ts in enumerate(
        [
            "2026-01-05T10:00:00+00:00",
            "2026-01-12T10:00:00+00:00",
            "2026-01-12T12:00:00+00:00",
            "2026-01-12T14:00:00+00:00",
            "2026-01-19T10:00:00+00:00",
        ],
        start=1,
    ):
        repo.upsert_bundle(_bundle("c1", f"m{idx}", ts))

    now = datetime.now(timezone.utc)
    clusters = [Cluster(cluster_id=1, label="alpha", member_ids=[f"m{i}" for i in range(1, 6)], centroid=None, created_at=now)]
    memberships = [(1, f"m{i}") for i in range(1, 6)]
    topic_events = [
        (1, "2026-01-05T10:00:00+00:00", "c1", "m1"),
        (1, "2026-01-12T10:00:00+00:00", "c1", "m2"),
        (1, "2026-01-12T12:00:00+00:00", "c1", "m3"),
        (1, "2026-01-12T14:00:00+00:00", "c1", "m4"),
        (1, "2026-01-19T10:00:00+00:00", "c1", "m5"),
    ]
    repo.replace_clusters(clusters, memberships, topic_events)

    out = MetricsService(repo).idea_half_life()
    assert out[0]["peak_weekly_volume"] == 3
    assert out[0]["half_life_weeks"] == 1
