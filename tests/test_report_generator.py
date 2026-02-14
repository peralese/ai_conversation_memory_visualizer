from datetime import datetime, timedelta, timezone

from src.models import CanonicalConversationBundle, Cluster, Conversation, Message, SourceType, SpeakerRole
from src.reports.generator import CognitiveSummaryReportGenerator
from src.storage.repository import SQLiteRepository


def _bundle(conv_id: str, source: SourceType, messages: list[tuple[str, str]]) -> CanonicalConversationBundle:
    created = datetime.fromisoformat(messages[0][1])
    return CanonicalConversationBundle(
        conversation=Conversation(
            id=conv_id,
            source=source,
            title=conv_id,
            created_at=created,
            updated_at=created,
            participants=["user", "assistant"],
            tags=[],
            raw_metadata={},
        ),
        messages=[
            Message(
                id=mid,
                conversation_id=conv_id,
                timestamp=ts,
                speaker_role=SpeakerRole.USER if i % 2 == 0 else SpeakerRole.ASSISTANT,
                text=f"message {mid} about architecture decisions",
                parent_message_id=None,
                token_count=None,
                raw_metadata={},
            )
            for i, (mid, ts) in enumerate(messages)
        ],
    )


def test_report_json_structure_and_counts(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "report.db"))
    base = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)

    b1 = _bundle("cgpt", SourceType.CHATGPT, [("m1", base.isoformat()), ("m2", (base + timedelta(days=7)).isoformat())])
    b2 = _bundle("cclaude", SourceType.CLAUDE, [("m3", (base + timedelta(days=1)).isoformat()), ("m4", (base + timedelta(days=8)).isoformat())])
    b3 = _bundle("cgem", SourceType.GEMINI, [("m5", (base + timedelta(days=2)).isoformat()), ("m6", (base + timedelta(days=9)).isoformat())])
    for b in (b1, b2, b3):
        repo.upsert_bundle(b)

    for i, mid in enumerate(["m1", "m2", "m3", "m4", "m5", "m6"]):
        vec = [1.0, 0.0] if i < 3 else [0.0, 1.0]
        repo.save_embedding(mid, vec, "stub")

    now = datetime.now(timezone.utc)
    clusters = [Cluster(cluster_id=1, label="architecture", member_ids=["m1", "m2", "m3", "m4", "m5", "m6"], centroid=[0.5, 0.5], created_at=now)]
    memberships = [(1, mid) for mid in ["m1", "m2", "m3", "m4", "m5", "m6"]]
    topic_events = [(1, ts, conv, mid) for conv, mid, ts in [
        ("cgpt", "m1", b1.messages[0].timestamp),
        ("cgpt", "m2", b1.messages[1].timestamp),
        ("cclaude", "m3", b2.messages[0].timestamp),
        ("cclaude", "m4", b2.messages[1].timestamp),
        ("cgem", "m5", b3.messages[0].timestamp),
        ("cgem", "m6", b3.messages[1].timestamp),
    ]]
    repo.replace_clusters(clusters, memberships, topic_events)

    generator = CognitiveSummaryReportGenerator(repo)
    report = generator.generate_json_report()

    assert report["report_version"] == "1.0"
    assert report["overview"]["total_conversations"] == 3
    assert report["overview"]["total_messages"] == 6
    assert "top_cognitive_modes" in report
    assert "model_specialization_highlights" in report
    assert "drift_insights" in report
    assert "timeline_summary" in report


def test_report_markdown_has_required_sections(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "report_md.db"))
    base = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc).isoformat()
    repo.upsert_bundle(_bundle("c1", SourceType.CHATGPT, [("m1", base)]))

    now = datetime.now(timezone.utc)
    repo.replace_clusters(
        [Cluster(cluster_id=1, label="topic", member_ids=["m1"], centroid=None, created_at=now)],
        [(1, "m1")],
        [(1, base, "c1", "m1")],
    )

    generator = CognitiveSummaryReportGenerator(repo)
    md = generator.generate_markdown_report(generator.generate_json_report())

    assert "## 1) Dataset Overview" in md
    assert "## 2) Top Cognitive Modes" in md
    assert "## 3) Model Specialization Highlights" in md
    assert "## 4) Evolving vs Stable Topics (Drift)" in md
    assert "## 5) Idea Half-Life Insights" in md
    assert "## 6) Timeline Summary" in md


def test_report_notes_missing_baseline_source(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "report_baseline_note.db"))
    base = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc).isoformat()
    repo.upsert_bundle(_bundle("cclaude", SourceType.CLAUDE, [("m1", base)]))
    repo.upsert_bundle(_bundle("cgem", SourceType.GEMINI, [("m2", base)]))

    now = datetime.now(timezone.utc)
    repo.replace_clusters(
        [Cluster(cluster_id=1, label="topic", member_ids=["m1", "m2"], centroid=None, created_at=now)],
        [(1, "m1"), (1, "m2")],
        [(1, base, "cclaude", "m1"), (1, base, "cgem", "m2")],
    )

    generator = CognitiveSummaryReportGenerator(repo)
    report = generator.generate_json_report()
    md = generator.generate_markdown_report(report)

    assert any("Source CHATGPT has no data" in note for note in report["model_specialization_notes"])
    assert "Source CHATGPT has no data and is excluded from specialization calculations." in md
