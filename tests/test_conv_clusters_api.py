import sqlite3
from datetime import datetime, timezone

from src.api import main as api_main
from src.models import CanonicalConversationBundle, Conversation, Message, SourceType, SpeakerRole
from src.storage.repository import SQLiteRepository


def _bundle(conv_id: str, source: SourceType, message_id: str, text: str) -> CanonicalConversationBundle:
    now = datetime.now(timezone.utc)
    return CanonicalConversationBundle(
        conversation=Conversation(
            id=conv_id,
            source=source,
            title=f"Title {conv_id}",
            created_at=now,
            updated_at=now,
            participants=["user", "assistant"],
            tags=[],
            raw_metadata={},
        ),
        messages=[
            Message(
                id=message_id,
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


def test_conv_clusters_endpoint_returns_rows_without_labels(monkeypatch, tmp_path):
    db_path = tmp_path / "conv_api.db"
    repo = SQLiteRepository(str(db_path))
    monkeypatch.setattr(api_main, "repo", repo)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    now = datetime.now(timezone.utc).isoformat()
    repo.upsert_bundle(_bundle("c1", SourceType.CHATGPT, "m1", "azure pricing scripts"))
    repo.upsert_bundle(_bundle("c2", SourceType.CLAUDE, "m2", "eleventy print templates"))
    repo.upsert_conversation_rollup(
        {
            "conversation_id": "c1",
            "source": "CHATGPT",
            "started_at": now,
            "ended_at": now,
            "message_count": 1,
            "user_message_count": 1,
            "assistant_message_count": 0,
            "avg_message_length": 20.0,
            "top_terms": ["azure", "pricing"],
            "representative_snippets": ["azure pricing scripts"],
            "rollup_text": "azure pricing scripts",
            "rollup_hash": "h1",
            "updated_at": now,
        }
    )
    repo.upsert_conversation_rollup(
        {
            "conversation_id": "c2",
            "source": "CLAUDE",
            "started_at": now,
            "ended_at": now,
            "message_count": 1,
            "user_message_count": 1,
            "assistant_message_count": 0,
            "avg_message_length": 24.0,
            "top_terms": ["eleventy", "templates"],
            "representative_snippets": ["eleventy print templates"],
            "rollup_text": "eleventy print templates",
            "rollup_hash": "h2",
            "updated_at": now,
        }
    )
    cid = repo.create_conv_cluster_run("kmeans", "{}")
    repo.upsert_conv_cluster_members(
        cid,
        [
            {"conversation_id": "c1", "distance": 0.0, "is_representative": True},
            {"conversation_id": "c2", "distance": 0.1, "is_representative": False},
        ],
    )

    conn = sqlite3.connect(db_path)
    has_conversation_clusters_table = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='conversation_clusters'"
    ).fetchone()[0]
    conn.close()
    assert has_conversation_clusters_table == 0

    rows = api_main.conv_clusters(use_semantic_labels=True, show_legacy_labels=False)
    assert len(rows) == 1
    assert rows[0]["cluster_id"] == cid
    assert rows[0]["conversation_count"] == 2
    assert rows[0]["label_display"]
    assert rows[0]["semantic"]["title"]
    assert rows[0]["source_breakdown"]["CHATGPT"] == 1
    assert rows[0]["source_breakdown"]["CLAUDE"] == 1


def test_pipeline_conversation_endpoint_populates_cluster_tables(monkeypatch, tmp_path):
    db_path = tmp_path / "conv_pipeline.db"
    repo = SQLiteRepository(str(db_path))
    monkeypatch.setattr(api_main, "repo", repo)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    repo.upsert_bundle(_bundle("c1", SourceType.CHATGPT, "m1", "azure monthly pricing script"))
    repo.upsert_bundle(_bundle("c2", SourceType.GEMINI, "m2", "json normalization pipeline"))
    repo.upsert_bundle(_bundle("c3", SourceType.CLAUDE, "m3", "eleventy recipe print layout"))

    payload = api_main.conversation_pipeline_run(dry_run=True, force_recluster=True)
    assert payload["clusters"] > 0
    assert payload["cluster_members"] > 0

    counts = api_main.conv_cluster_debug_counts()
    assert counts["conversations"] == 3
    assert counts["conversation_embeddings"] > 0
    assert counts["conv_clusters"] > 0
    assert counts["conv_cluster_members"] > 0

    rows = api_main.conv_clusters(use_semantic_labels=True, show_legacy_labels=False)
    assert rows
    assert rows[0]["cluster_id"] > 0
    assert "source_breakdown" in rows[0]
