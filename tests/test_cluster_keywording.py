from datetime import datetime, timezone

from src.clustering.keywords import build_distinctive_labels
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


def test_keywording_strips_code_urls_paths_and_hashes():
    text = (
        "id data file path upload messages "
        "```python\ndef run_job():\n  return 42\n``` "
        "see `pip install foo`\n"
        "Traceback (most recent call last):\n"
        "File \"/tmp/app.py\", line 7, in <module>\n"
        "ValueError: boom\n"
        "visit https://localhost:8000/api/import and /home/user/data/export.json "
        "uuid 550e8400-e29b-41d4-a716-446655440000 hash abcdef1234567890fedcba "
        "retrieval ranking embeddings"
    )

    out = build_distinctive_labels({1: [text]}, exclude_domain_stopwords=True, include_debug_for={1})
    result = out[1]

    assert "retrieval" in result.terms
    assert "ranking" in result.terms
    assert "id" not in result.terms
    assert "data" not in result.terms
    assert "file" not in result.terms
    assert "path" not in result.terms

    removed = result.debug["removed_by_rule"]
    assert removed["code"]
    assert removed["url_path"]
    assert removed["hash_uuid"]


def test_cluster_labels_remove_ingestion_noise_when_enabled(tmp_path):
    db = tmp_path / "noise.db"
    repo = SQLiteRepository(str(db))

    repo.upsert_bundle(_bundle("c1", 1, "id data file path json csv upload metrics dashboard latency"))
    repo.upsert_bundle(_bundle("c2", 2, "id data file path json csv upload garden soil compost"))
    repo.save_embedding("m_1", [0.0, 0.0, 0.0, 0.0], "stub")
    repo.save_embedding("m_2", [1.0, 1.0, 1.0, 1.0], "stub")

    service = ClusteringService(repo)
    service.cluster_embeddings(k=2)
    labels = [str(c["label"]).lower() for c in service.list_clusters(exclude_domain_stopwords=True)]

    banned = {"id", "data", "file", "path"}
    for label in labels:
        for token in banned:
            assert token not in label


def test_distinctive_labels_favor_cluster_specific_terms():
    texts_by_cluster = {
        1: [
            "id data file path python pandas dataframe joins",
            "assistant user python pandas indexing dataframe",
        ],
        2: [
            "id data file path gardening soil compost seedlings",
            "assistant user gardening soil watering compost",
        ],
    }

    labels = build_distinctive_labels(texts_by_cluster, exclude_domain_stopwords=True)
    terms_1 = labels[1].terms[:4]
    terms_2 = labels[2].terms[:4]

    assert any(token in terms_1 for token in ("python", "pandas", "dataframe"))
    assert any(token in terms_2 for token in ("gardening", "soil", "compost"))
    assert all(token not in terms_1 for token in ("id", "data", "file", "path"))
    assert all(token not in terms_2 for token in ("id", "data", "file", "path"))
