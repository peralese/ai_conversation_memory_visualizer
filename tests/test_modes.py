from datetime import datetime, timedelta, timezone

import pytest

from src.modes.scoring import score_mode_weights
from src.storage.repository import SQLiteRepository
from src.metrics.modes_service import ModesService


def test_mode_scoring_expected_dominant_mode():
    scored = score_mode_weights(
        top_keywords=["error", "stack trace", "diagnose"],
        label="troubleshoot failing request",
        sample_snippets=["The service is not working and we need to fix the bug quickly."],
    )
    assert scored["dominant_mode"] == "troubleshooting_debugging"
    assert scored["dominant_weight"] is not None


def test_mode_weights_normalize_to_one():
    scored = score_mode_weights(
        top_keywords=["design", "architecture", "blueprint"],
        label="system design approach",
        sample_snippets=["compare tradeoff and evaluate constraints"],
    )
    total = sum(scored["mode_weights"].values())
    assert abs(total - 1.0) <= 0.002


def test_mode_aggregation_matches_weighted_percentages(tmp_path, monkeypatch):
    repo = SQLiteRepository(str(tmp_path / "modes_agg.db"))
    with repo.connection() as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT INTO conversations(id, source, title, created_at, updated_at, participants_json, tags_json, raw_metadata_json) VALUES ('c1','CHATGPT','c1',?,?, '[]','[]','{}')", (now, now))
        conn.execute("INSERT INTO conversations(id, source, title, created_at, updated_at, participants_json, tags_json, raw_metadata_json) VALUES ('c2','CLAUDE','c2',?,?, '[]','[]','{}')", (now, now))
        for i in range(1, 5):
            conn.execute("INSERT INTO messages(id, conversation_id, timestamp, speaker_role, original_text, redacted_text, parent_message_id, token_count, raw_metadata_json) VALUES (?,?,?,?,?,NULL,NULL,NULL,'{}')",
                         (f"m{i}", "c1", now, "user", "x"))
        for i in range(5, 7):
            conn.execute("INSERT INTO messages(id, conversation_id, timestamp, speaker_role, original_text, redacted_text, parent_message_id, token_count, raw_metadata_json) VALUES (?,?,?,?,?,NULL,NULL,NULL,'{}')",
                         (f"m{i}", "c2", now, "assistant", "y"))
        conn.execute("INSERT INTO clusters(cluster_id, label, centroid_json, first_seen, last_seen, message_count, created_at) VALUES (1,'design',NULL,?,?,4,?)", (now, now, now))
        conn.execute("INSERT INTO clusters(cluster_id, label, centroid_json, first_seen, last_seen, message_count, created_at) VALUES (2,'debug',NULL,?,?,2,?)", (now, now, now))
        conn.executemany("INSERT INTO cluster_membership(cluster_id, message_id) VALUES (1,?)", [(f"m{i}",) for i in range(1, 5)])
        conn.executemany("INSERT INTO cluster_membership(cluster_id, message_id) VALUES (2,?)", [(f"m{i}",) for i in range(5, 7)])

    repo.replace_mode_scores(
        "cluster",
        [
            {
                "entity_id": "1",
                "mode_weights": {"design_synthesis": 1.0, "troubleshooting_debugging": 0.0},
                "dominant_mode": "design_synthesis",
                "dominant_weight": 1.0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "entity_id": "2",
                "mode_weights": {"design_synthesis": 0.0, "troubleshooting_debugging": 1.0},
                "dominant_mode": "troubleshooting_debugging",
                "dominant_weight": 1.0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        ],
    )
    monkeypatch.setattr(repo, "mode_scores_stale", lambda level: False)
    data = ModesService(repo).metrics(level="cluster")
    # 4/6 from cluster1 and 2/6 from cluster2
    assert round(data["overall_mode_distribution"]["design_synthesis"], 4) == round(4 / 6, 4)
    assert round(data["overall_mode_distribution"]["troubleshooting_debugging"], 4) == round(2 / 6, 4)


def test_modes_timeline_bucket_sorting(tmp_path, monkeypatch):
    repo = SQLiteRepository(str(tmp_path / "modes_timeline.db"))
    with repo.connection() as conn:
        now = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
        later = now + timedelta(days=7)
        created = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT INTO conversations(id, source, title, created_at, updated_at, participants_json, tags_json, raw_metadata_json) VALUES ('c1','CHATGPT','c1',?,?, '[]','[]','{}')", (created, created))
        conn.execute("INSERT INTO clusters(cluster_id, label, centroid_json, first_seen, last_seen, message_count, created_at) VALUES (1,'design',NULL,?,?,2,?)", (created, created, created))
        conn.execute("INSERT INTO messages(id, conversation_id, timestamp, speaker_role, original_text, redacted_text, parent_message_id, token_count, raw_metadata_json) VALUES ('m1','c1',?,'user','x',NULL,NULL,NULL,'{}')", (later.isoformat(),))
        conn.execute("INSERT INTO messages(id, conversation_id, timestamp, speaker_role, original_text, redacted_text, parent_message_id, token_count, raw_metadata_json) VALUES ('m2','c1',?,'user','x',NULL,NULL,NULL,'{}')", (now.isoformat(),))
        conn.execute("INSERT INTO cluster_membership(cluster_id, message_id) VALUES (1,'m1')")
        conn.execute("INSERT INTO cluster_membership(cluster_id, message_id) VALUES (1,'m2')")

    repo.replace_mode_scores(
        "cluster",
        [
            {
                "entity_id": "1",
                "mode_weights": {"design_synthesis": 1.0},
                "dominant_mode": "design_synthesis",
                "dominant_weight": 1.0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    monkeypatch.setattr(repo, "mode_scores_stale", lambda level: False)
    out = ModesService(repo).timeline(level="cluster", bucket="week")
    buckets = [r["bucket_start"] for r in out["rows"]]
    assert buckets == sorted(buckets)
