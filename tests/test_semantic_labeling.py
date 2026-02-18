from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time

from src.labeling.evidence import build_cluster_evidence_packet
from src.labeling.gpt_labeler import SemanticLabeler, SemanticLabelerConfig
from src.labeling.service import SemanticLabelService
from src.models import CanonicalConversationBundle, Cluster, Conversation, Message, SourceType, SpeakerRole
from src.storage.repository import SQLiteRepository


def _bundle(conv_id: str, mid: str, text: str) -> CanonicalConversationBundle:
    now = datetime.now(timezone.utc)
    return CanonicalConversationBundle(
        conversation=Conversation(
            id=conv_id,
            source=SourceType.CHATGPT,
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
                text=text,
                parent_message_id=None,
                token_count=None,
                raw_metadata={},
            )
        ],
    )


def test_evidence_hash_stable_for_cluster_packet(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "evidence_hash.db"))
    now = datetime.now(timezone.utc)
    repo.upsert_bundle(_bundle("c1", "m1", "azure pricing monthly script"))
    repo.replace_clusters(
        [Cluster(cluster_id=1, label="x", member_ids=["m1"], centroid=None, created_at=now)],
        memberships=[(1, "m1")],
        topic_events=[(1, now.isoformat(), "c1", "m1")],
    )
    packet1 = build_cluster_evidence_packet(repo, 1)
    packet2 = build_cluster_evidence_packet(repo, 1)
    assert packet1["evidence_hash"] == packet2["evidence_hash"]


class _FakeLabeler:
    def __init__(self):
        self.calls = 0

    async def label_cluster(self, packet):
        self.calls += 1
        return {"label": "Azure Pricing Scripts", "summary": "summary", "tags": ["azure"], "provider": "heuristic"}

    async def label_conv_cluster(self, packet):
        self.calls += 1
        return {"label": "Conversation Topic", "summary": "summary", "tags": ["topic"], "provider": "heuristic"}


def test_caching_skips_relabel_when_evidence_hash_unchanged(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "cache.db"))
    now = datetime.now(timezone.utc)
    repo.upsert_bundle(_bundle("c1", "m1", "azure pricing monthly script"))
    repo.replace_clusters(
        [Cluster(cluster_id=1, label="x", member_ids=["m1"], centroid=None, created_at=now)],
        memberships=[(1, "m1")],
        topic_events=[(1, now.isoformat(), "c1", "m1")],
    )
    fake = _FakeLabeler()
    service = SemanticLabelService(repo, labeler=fake)

    first = service.label_one_cluster(1, force=False)
    second = service.label_one_cluster(1, force=False)
    assert first["cached"] is False
    assert second["cached"] is True
    assert fake.calls == 1


def test_rate_limit_waits_when_bucket_full(monkeypatch):
    labeler = SemanticLabeler(SemanticLabelerConfig(rpm=2, concurrency=1, dry_run=False))
    labeler._request_times.clear()
    now = time.monotonic()
    labeler._request_times.extend([now, now])
    sleep_calls: list[float] = []

    async def _fake_sleep(seconds):
        sleep_calls.append(float(seconds))
        labeler._request_times.clear()

    monkeypatch.setattr("src.labeling.gpt_labeler.asyncio.sleep", _fake_sleep)

    asyncio.run(labeler._wait_for_rate_slot())
    assert sleep_calls
    assert sleep_calls[0] > 0


def test_repository_semantic_label_crud(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "repo_semantic.db"))
    now = datetime.now(timezone.utc)
    repo.upsert_bundle(_bundle("c1", "m1", "topic one"))
    repo.replace_clusters(
        [Cluster(cluster_id=1, label="x", member_ids=["m1"], centroid=None, created_at=now)],
        memberships=[(1, "m1")],
        topic_events=[(1, now.isoformat(), "c1", "m1")],
    )
    conv_cluster_id = repo.create_conv_cluster_run("kmeans", "{}")
    repo.upsert_conv_cluster_members(
        conv_cluster_id,
        [{"conversation_id": "c1", "distance": 0.0, "is_representative": True}],
    )

    repo.upsert_cluster_semantic_label(
        cluster_id=1,
        label="Cluster One",
        summary="summary",
        tags=["one", "cluster"],
        provider="heuristic",
        evidence_hash="h1",
    )
    row = repo.get_cluster_semantic_label(1)
    assert row is not None
    assert row["label"] == "Cluster One"
    assert row["tags"] == ["one", "cluster"]

    repo.upsert_conv_cluster_semantic_label(
        conv_cluster_id=conv_cluster_id,
        label="Conversation One",
        summary="summary",
        tags=["one"],
        provider="heuristic",
        evidence_hash="h2",
    )
    conv_row = repo.get_conv_cluster_semantic_label(conv_cluster_id)
    assert conv_row is not None
    assert conv_row["label"] == "Conversation One"
    assert conv_row["tags"] == ["one"]

    assert repo.list_clusters_missing_semantic_labels(force=False) == []
    assert repo.list_conv_clusters_missing_semantic_labels(force=False) == []
