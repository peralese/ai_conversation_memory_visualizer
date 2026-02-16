from datetime import datetime, timedelta, timezone

from src.conversation_pipeline.rollup import RollupConfig, build_conversation_rollups
from src.labeling.evidence_packet import build_cluster_evidence_packet
from src.labeling.gpt_labeler import GPTClusterLabeler, GPTLabelerConfig
from src.models import CanonicalConversationBundle, Conversation, Message, SourceType, SpeakerRole
from src.storage.repository import SQLiteRepository


def _bundle(conv_id: str, messages: list[tuple[str, str, str]]) -> CanonicalConversationBundle:
    now = datetime.now(timezone.utc)
    out_messages = []
    for idx, (mid, role, text) in enumerate(messages):
        out_messages.append(
            Message(
                id=mid,
                conversation_id=conv_id,
                timestamp=(now + timedelta(seconds=idx)).isoformat(),
                speaker_role=SpeakerRole.USER if role == "user" else SpeakerRole.ASSISTANT,
                text=text,
                parent_message_id=None,
                token_count=None,
                raw_metadata={},
            )
        )
    return CanonicalConversationBundle(
        conversation=Conversation(
            id=conv_id,
            source=SourceType.CHATGPT,
            title=f"Title {conv_id}",
            created_at=now,
            updated_at=now,
            participants=["user", "assistant"],
            tags=[],
            raw_metadata={},
        ),
        messages=out_messages,
    )


def test_rollup_text_stable_and_excludes_stopwords(tmp_path):
    db = tmp_path / "conv_rollup.db"
    repo = SQLiteRepository(str(db))

    repo.upsert_bundle(
        _bundle(
            "c1",
            [
                ("m1", "user", "Please run this id data file import path helper"),
                ("m2", "assistant", "Use azure pricing estimator and monthly cost script"),
            ],
        )
    )

    cfg = RollupConfig(config_version="test_v1")
    first = build_conversation_rollups(repo, cfg)
    second = build_conversation_rollups(repo, cfg)

    assert first == ["c1"]
    assert second == ["c1"]

    rollup = repo.get_conversation_rollup("c1")
    assert rollup is not None
    text = str(rollup["rollup_text"])
    assert "id" not in text
    assert "data" not in text
    assert "file" not in text
    assert "azure" in text

    same_hash = str(rollup["rollup_hash"])
    build_conversation_rollups(repo, cfg)
    rollup_again = repo.get_conversation_rollup("c1")
    assert rollup_again is not None
    assert str(rollup_again["rollup_hash"]) == same_hash


def test_evidence_hash_changes_when_snippets_change(tmp_path):
    db = tmp_path / "evidence.db"
    repo = SQLiteRepository(str(db))

    repo.upsert_bundle(_bundle("c1", [("m1", "user", "alpha topic"), ("m2", "assistant", "beta topic")]))
    build_conversation_rollups(repo)

    cid = repo.create_conv_cluster_run("kmeans", '{"k":1}')
    repo.upsert_conv_cluster_members(cid, [{"conversation_id": "c1", "distance": 0.0, "is_representative": True}])

    packet1 = build_cluster_evidence_packet(repo, cid)
    hash1 = packet1["evidence_hash"]

    repo.upsert_conversation_rollup(
        {
            "conversation_id": "c1",
            "source": "CHATGPT",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:10:00+00:00",
            "message_count": 2,
            "user_message_count": 1,
            "assistant_message_count": 1,
            "avg_message_length": 10,
            "top_terms": ["alpha", "beta"],
            "representative_snippets": ["changed snippet content"],
            "rollup_text": "alpha beta changed snippet content",
            "rollup_hash": "changed",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    packet2 = build_cluster_evidence_packet(repo, cid)
    hash2 = packet2["evidence_hash"]
    assert hash1 != hash2


class _FakeLabeler(GPTClusterLabeler):
    def __init__(self, repo, config=None):
        super().__init__(repo, config)
        self.calls = 0

    def _call_openai(self, packet):
        self.calls += 1
        return ({"title": "Test Cluster", "summary": "Test summary", "tags": ["test", "cluster"]}, 10, 5)


def test_gpt_label_caching_prevents_repeated_calls(tmp_path, monkeypatch):
    db = tmp_path / "cache.db"
    repo = SQLiteRepository(str(db))
    cid = repo.create_conv_cluster_run("kmeans", "{}")
    packet = {
        "cluster_id": cid,
        "size": {"conversations": 1, "messages": 2},
        "date_range": {"started_at": None, "ended_at": None},
        "source_breakdown": {"CHATGPT": 1},
        "top_terms": ["azure", "pricing"],
        "representative_conversations": [],
        "evidence_hash": "abc123",
    }

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    labeler = _FakeLabeler(repo, GPTLabelerConfig(max_requests_per_run=10, min_seconds_between_requests=0.0))

    first = labeler.generate_label(cid, packet)
    second = labeler.generate_label(cid, packet)

    assert first["cached"] is False
    assert second["cached"] is True
    assert labeler.calls == 1
