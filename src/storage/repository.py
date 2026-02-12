from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.models import CanonicalConversationBundle, Cluster
from src.storage.migrations import MigrationManager


class SQLiteRepository:
    def __init__(self, db_path: str = "memory_viz.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            MigrationManager(conn).migrate()

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_bundle(self, bundle: CanonicalConversationBundle) -> None:
        c = bundle.conversation
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations(id, source, title, created_at, updated_at, participants_json, tags_json, raw_metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  source=excluded.source,
                  title=excluded.title,
                  created_at=excluded.created_at,
                  updated_at=excluded.updated_at,
                  participants_json=excluded.participants_json,
                  tags_json=excluded.tags_json,
                  raw_metadata_json=excluded.raw_metadata_json
                """,
                (
                    c.id,
                    c.source.value,
                    c.title,
                    c.created_at.isoformat(),
                    c.updated_at.isoformat(),
                    json.dumps(c.participants),
                    json.dumps(c.tags),
                    json.dumps(c.raw_metadata),
                ),
            )

            for m in bundle.messages:
                conn.execute(
                    """
                    INSERT INTO messages(id, conversation_id, timestamp, speaker_role, original_text, redacted_text, parent_message_id, token_count, raw_metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      conversation_id=excluded.conversation_id,
                      timestamp=excluded.timestamp,
                      speaker_role=excluded.speaker_role,
                      original_text=excluded.original_text,
                      parent_message_id=excluded.parent_message_id,
                      token_count=excluded.token_count,
                      raw_metadata_json=excluded.raw_metadata_json
                    """,
                    (
                        m.id,
                        m.conversation_id,
                        m.timestamp,
                        m.speaker_role.value,
                        m.text,
                        None,
                        m.parent_message_id,
                        m.token_count,
                        json.dumps(m.raw_metadata),
                    ),
                )

    def list_conversations(self, q: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT id, source, title, created_at, updated_at FROM conversations"
        params: list[Any] = []
        if q:
            sql += " WHERE title LIKE ?"
            params.append(f"%{q}%")
        sql += " ORDER BY updated_at DESC"

        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def list_messages_for_embedding(self, since: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT id, original_text, COALESCE(redacted_text, original_text) AS embedding_text, timestamp FROM messages"
        params: list[Any] = []
        if since:
            sql += " WHERE timestamp >= ?"
            params.append(since)
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def save_redacted_text(self, message_id: str, redacted_text: str) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE messages SET redacted_text = ? WHERE id = ?", (redacted_text, message_id))

    def has_embedding(self, item_id: str) -> bool:
        with self.connection() as conn:
            row = conn.execute("SELECT 1 FROM embeddings WHERE item_id = ?", (item_id,)).fetchone()
            return row is not None

    def save_embedding(self, item_id: str, vector: list[float], model_name: str) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO embeddings(item_id, vector_json, model_name, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                  vector_json=excluded.vector_json,
                  model_name=excluded.model_name,
                  created_at=excluded.created_at
                """,
                (item_id, json.dumps(vector), model_name, datetime.now(timezone.utc).isoformat()),
            )

    def load_embeddings(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT e.item_id, e.vector_json, m.timestamp, m.conversation_id, m.original_text FROM embeddings e JOIN messages m ON m.id = e.item_id"
            ).fetchall()
            out = []
            for row in rows:
                out.append(
                    {
                        "item_id": row["item_id"],
                        "vector": json.loads(row["vector_json"]),
                        "timestamp": row["timestamp"],
                        "conversation_id": row["conversation_id"],
                        "original_text": row["original_text"],
                    }
                )
            return out

    def replace_clusters(
        self,
        clusters: list[Cluster],
        memberships: list[tuple[int, str]],
        topic_events: list[tuple[int, str, str, str]],
    ) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM cluster_membership")
            conn.execute("DELETE FROM topic_events")
            conn.execute("DELETE FROM clusters")

            for c in clusters:
                conn.execute(
                    """
                    INSERT INTO clusters(cluster_id, label, centroid_json, first_seen, last_seen, message_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        c.cluster_id,
                        c.label,
                        json.dumps(c.centroid) if c.centroid is not None else None,
                        None,
                        None,
                        len(c.member_ids),
                        c.created_at.isoformat(),
                    ),
                )

            conn.executemany(
                "INSERT INTO cluster_membership(cluster_id, message_id) VALUES (?, ?)",
                memberships,
            )
            conn.executemany(
                "INSERT INTO topic_events(cluster_id, timestamp, conversation_id, message_id) VALUES (?, ?, ?, ?)",
                topic_events,
            )

            conn.executescript(
                """
                UPDATE clusters
                SET
                  first_seen = (SELECT MIN(timestamp) FROM topic_events te WHERE te.cluster_id = clusters.cluster_id),
                  last_seen = (SELECT MAX(timestamp) FROM topic_events te WHERE te.cluster_id = clusters.cluster_id),
                  message_count = (SELECT COUNT(*) FROM cluster_membership cm WHERE cm.cluster_id = clusters.cluster_id);
                """
            )

    def list_clusters(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT cluster_id, label, first_seen, last_seen, message_count FROM clusters ORDER BY message_count DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def cluster_detail(self, cluster_id: int) -> dict[str, Any]:
        with self.connection() as conn:
            cluster = conn.execute(
                "SELECT cluster_id, label, first_seen, last_seen, message_count FROM clusters WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchone()
            if cluster is None:
                raise ValueError(f"Unknown cluster: {cluster_id}")

            samples = conn.execute(
                """
                SELECT m.id, m.timestamp, m.original_text, m.conversation_id
                FROM cluster_membership cm
                JOIN messages m ON m.id = cm.message_id
                WHERE cm.cluster_id = ?
                ORDER BY m.timestamp DESC
                LIMIT 10
                """,
                (cluster_id,),
            ).fetchall()

            return {
                "cluster": dict(cluster),
                "examples": [dict(r) for r in samples],
            }

    def topic_events(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT cluster_id, timestamp, conversation_id, message_id FROM topic_events ORDER BY timestamp"
            ).fetchall()
            return [dict(r) for r in rows]

    def cluster_keywords_corpus(self, cluster_id: int) -> list[str]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT m.original_text
                FROM cluster_membership cm
                JOIN messages m ON m.id = cm.message_id
                WHERE cm.cluster_id = ?
                """,
                (cluster_id,),
            ).fetchall()
            return [str(r[0]) for r in rows]
