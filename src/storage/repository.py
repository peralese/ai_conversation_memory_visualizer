from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.analysis.text import build_analysis_text
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
            out = []
            for row in rows:
                embedding_text = str(row["embedding_text"] or "")
                out.append(
                    {
                        "id": row["id"],
                        "original_text": row["original_text"],
                        "embedding_text": embedding_text,
                        "analysis_text": build_analysis_text(embedding_text),
                        "timestamp": row["timestamp"],
                    }
                )
            return out

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
                        "analysis_text": build_analysis_text(str(row["original_text"] or "")),
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
                """
                SELECT
                  c.cluster_id,
                  c.label,
                  c.first_seen,
                  c.last_seen,
                  c.message_count,
                  (
                    SELECT substr(replace(m.original_text, char(10), ' '), 1, 180)
                    FROM cluster_membership cm
                    JOIN messages m ON m.id = cm.message_id
                    WHERE cm.cluster_id = c.cluster_id
                    ORDER BY m.timestamp DESC
                    LIMIT 1
                  ) AS preview_snippet
                FROM clusters c
                ORDER BY c.message_count DESC
                """
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

    def cluster_member_records(self, cluster_id: int, limit: int | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT
              cm.cluster_id,
              m.id,
              m.timestamp,
              m.speaker_role,
              m.original_text,
              m.conversation_id,
              c.title AS conversation_title,
              c.source,
              e.vector_json
            FROM cluster_membership cm
            JOIN messages m ON m.id = cm.message_id
            JOIN conversations c ON c.id = m.conversation_id
            LEFT JOIN embeddings e ON e.item_id = m.id
            WHERE cm.cluster_id = ?
            ORDER BY m.timestamp DESC
        """
        params: list[Any] = [cluster_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))

        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                out.append(
                    {
                        "cluster_id": int(row["cluster_id"]),
                        "id": str(row["id"]),
                        "timestamp": str(row["timestamp"]),
                        "speaker_role": str(row["speaker_role"]),
                        "original_text": str(row["original_text"] or ""),
                        "analysis_text": build_analysis_text(str(row["original_text"] or "")),
                        "conversation_id": str(row["conversation_id"]),
                        "conversation_title": str(row["conversation_title"] or ""),
                        "source": str(row["source"]),
                        "vector": json.loads(row["vector_json"]) if row["vector_json"] else None,
                    }
                )
            return out

    def all_cluster_member_records(self) -> list[dict[str, Any]]:
        sql = """
            SELECT
              cm.cluster_id,
              m.id,
              m.timestamp,
              m.speaker_role,
              m.original_text,
              m.conversation_id,
              c.title AS conversation_title,
              c.source,
              e.vector_json
            FROM cluster_membership cm
            JOIN messages m ON m.id = cm.message_id
            JOIN conversations c ON c.id = m.conversation_id
            LEFT JOIN embeddings e ON e.item_id = m.id
            ORDER BY m.timestamp DESC
        """
        with self.connection() as conn:
            rows = conn.execute(sql).fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                out.append(
                    {
                        "cluster_id": int(row["cluster_id"]),
                        "id": str(row["id"]),
                        "timestamp": str(row["timestamp"]),
                        "speaker_role": str(row["speaker_role"]),
                        "original_text": str(row["original_text"] or ""),
                        "analysis_text": build_analysis_text(str(row["original_text"] or "")),
                        "conversation_id": str(row["conversation_id"]),
                        "conversation_title": str(row["conversation_title"] or ""),
                        "source": str(row["source"]),
                        "vector": json.loads(row["vector_json"]) if row["vector_json"] else None,
                    }
                )
            return out

    def get_cluster_row(self, cluster_id: int) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT cluster_id, label, first_seen, last_seen, message_count, centroid_json
                FROM clusters
                WHERE cluster_id = ?
                """,
                (cluster_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown cluster: {cluster_id}")
            out = dict(row)
            if out.get("centroid_json"):
                out["centroid"] = json.loads(out["centroid_json"])
            else:
                out["centroid"] = None
            out.pop("centroid_json", None)
            return out

    def cluster_context_stats(self, cluster_id: int) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT
                  COUNT(*) AS message_count,
                  COUNT(DISTINCT m.conversation_id) AS conversations_count,
                  AVG(LENGTH(m.original_text)) AS avg_length
                FROM cluster_membership cm
                JOIN messages m ON m.id = cm.message_id
                WHERE cm.cluster_id = ?
                """,
                (cluster_id,),
            ).fetchone()
            total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

        message_count = int(row["message_count"] or 0) if row else 0
        conversations_count = int(row["conversations_count"] or 0) if row else 0
        avg_length = int(round(float(row["avg_length"] or 0.0))) if row else 0
        dataset_percentage = round((message_count / max(1, int(total_messages))) * 100.0, 2)
        return {
            "message_count": message_count,
            "conversations_count": conversations_count,
            "average_message_length": avg_length,
            "dataset_percentage": dataset_percentage,
        }

    def clear_subclusters_for_parent(self, parent_cluster_id: int) -> None:
        with self.connection() as conn:
            sub_ids = [
                int(r[0])
                for r in conn.execute("SELECT id FROM subclusters WHERE parent_cluster_id = ?", (parent_cluster_id,)).fetchall()
            ]
            if sub_ids:
                conn.executemany("DELETE FROM subcluster_membership WHERE subcluster_id = ?", [(sid,) for sid in sub_ids])
            conn.execute("DELETE FROM subclusters WHERE parent_cluster_id = ?", (parent_cluster_id,))

    def clear_all_subclusters(self) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM subcluster_membership")
            conn.execute("DELETE FROM subclusters")

    def replace_subclusters_for_parent(self, parent_cluster_id: int, subclusters: list[dict[str, Any]]) -> None:
        with self.connection() as conn:
            existing = [
                int(r[0])
                for r in conn.execute("SELECT id FROM subclusters WHERE parent_cluster_id = ?", (parent_cluster_id,)).fetchall()
            ]
            if existing:
                conn.executemany("DELETE FROM subcluster_membership WHERE subcluster_id = ?", [(sid,) for sid in existing])
            conn.execute("DELETE FROM subclusters WHERE parent_cluster_id = ?", (parent_cluster_id,))

            for sub in subclusters:
                cur = conn.execute(
                    "INSERT INTO subclusters(parent_cluster_id, label, created_at) VALUES (?, ?, ?)",
                    (
                        parent_cluster_id,
                        str(sub.get("label") or "Subtopic"),
                        str(sub.get("created_at") or datetime.now(timezone.utc).isoformat()),
                    ),
                )
                subcluster_id = int(cur.lastrowid)
                message_ids = [str(mid) for mid in (sub.get("message_ids") or [])]
                conn.executemany(
                    "INSERT INTO subcluster_membership(subcluster_id, message_id) VALUES (?, ?)",
                    [(subcluster_id, mid) for mid in message_ids],
                )

    def list_subclusters(self, parent_cluster_id: int) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                  s.id,
                  s.parent_cluster_id,
                  s.label,
                  s.created_at,
                  COUNT(sm.message_id) AS message_count,
                  COUNT(DISTINCT m.conversation_id) AS conversations_count,
                  MIN(m.timestamp) AS first_seen,
                  MAX(m.timestamp) AS last_seen
                FROM subclusters s
                LEFT JOIN subcluster_membership sm ON sm.subcluster_id = s.id
                LEFT JOIN messages m ON m.id = sm.message_id
                WHERE s.parent_cluster_id = ?
                GROUP BY s.id, s.parent_cluster_id, s.label, s.created_at
                ORDER BY message_count DESC, s.id ASC
                """,
                (parent_cluster_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def subcluster_member_records(self, subcluster_id: int, limit: int | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT
              s.id AS subcluster_id,
              m.id,
              m.timestamp,
              m.speaker_role,
              m.original_text,
              m.conversation_id,
              c.title AS conversation_title,
              c.source
            FROM subcluster_membership sm
            JOIN subclusters s ON s.id = sm.subcluster_id
            JOIN messages m ON m.id = sm.message_id
            JOIN conversations c ON c.id = m.conversation_id
            WHERE sm.subcluster_id = ?
            ORDER BY m.timestamp DESC
        """
        params: list[Any] = [subcluster_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def subclusters_stale(self, parent_cluster_id: int) -> bool:
        with self.connection() as conn:
            cluster = conn.execute(
                "SELECT created_at, message_count FROM clusters WHERE cluster_id = ?",
                (parent_cluster_id,),
            ).fetchone()
            if cluster is None:
                return False
            if int(cluster["message_count"] or 0) == 0:
                return False

            existing = conn.execute(
                "SELECT MAX(created_at) AS latest_subcluster_at, COUNT(*) AS sub_count FROM subclusters WHERE parent_cluster_id = ?",
                (parent_cluster_id,),
            ).fetchone()
            if existing is None or int(existing["sub_count"] or 0) == 0:
                return True

            latest_subcluster_at = str(existing["latest_subcluster_at"] or "")
            cluster_created_at = str(cluster["created_at"] or "")
            if cluster_created_at and latest_subcluster_at and latest_subcluster_at < cluster_created_at:
                return True

            embedding_row = conn.execute(
                """
                SELECT MAX(e.created_at) AS latest_embedding_at
                FROM cluster_membership cm
                JOIN embeddings e ON e.item_id = cm.message_id
                WHERE cm.cluster_id = ?
                """,
                (parent_cluster_id,),
            ).fetchone()
            latest_embedding_at = str(embedding_row["latest_embedding_at"] or "") if embedding_row else ""
            if latest_embedding_at and latest_subcluster_at and latest_subcluster_at < latest_embedding_at:
                return True
            return False

    def profile_message_rows(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT m.id, m.original_text, c.source
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                """
            ).fetchall()
            return [
                {
                    "id": str(r["id"]),
                    "source": str(r["source"]),
                    "analysis_text": build_analysis_text(str(r["original_text"] or "")),
                }
                for r in rows
            ]

    def dataset_counts(self) -> dict[str, int]:
        with self.connection() as conn:
            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            return {"messages": int(msg_count), "conversations": int(conv_count)}

    def dataset_time_range(self) -> dict[str, str | None]:
        with self.connection() as conn:
            row = conn.execute("SELECT MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts FROM messages").fetchone()
            return {
                "first_message_at": str(row["first_ts"]) if row and row["first_ts"] else None,
                "last_message_at": str(row["last_ts"]) if row and row["last_ts"] else None,
            }

    def average_message_length(self) -> int:
        with self.connection() as conn:
            row = conn.execute("SELECT AVG(LENGTH(original_text)) AS avg_len FROM messages").fetchone()
            if row is None or row["avg_len"] is None:
                return 0
            return int(round(float(row["avg_len"])))

    def dataset_source_counts(self) -> dict[str, int]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.source, COUNT(*) AS count
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                GROUP BY c.source
                """
            ).fetchall()
            return {str(r["source"]): int(r["count"]) for r in rows}

    def topic_events(self, source: str | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT te.cluster_id, te.timestamp, te.conversation_id, te.message_id, c.source
            FROM topic_events te
            JOIN conversations c ON c.id = te.conversation_id
        """
        params: list[Any] = []
        if source and source.upper() != "ALL":
            sql += " WHERE c.source = ?"
            params.append(source.upper())
        sql += " ORDER BY te.timestamp"
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def all_subclusters(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                  s.id,
                  s.parent_cluster_id,
                  s.label,
                  s.created_at,
                  COUNT(sm.message_id) AS message_count,
                  COUNT(DISTINCT m.conversation_id) AS conversations_count
                FROM subclusters s
                LEFT JOIN subcluster_membership sm ON sm.subcluster_id = s.id
                LEFT JOIN messages m ON m.id = sm.message_id
                GROUP BY s.id, s.parent_cluster_id, s.label, s.created_at
                ORDER BY message_count DESC, s.id ASC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def cluster_source_counts(self, cluster_id: int) -> dict[str, int]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.source, COUNT(*) AS count
                FROM cluster_membership cm
                JOIN messages m ON m.id = cm.message_id
                JOIN conversations c ON c.id = m.conversation_id
                WHERE cm.cluster_id = ?
                GROUP BY c.source
                """,
                (cluster_id,),
            ).fetchall()
            return {str(r["source"]): int(r["count"]) for r in rows}

    def subcluster_source_counts(self, subcluster_id: int) -> dict[str, int]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.source, COUNT(*) AS count
                FROM subcluster_membership sm
                JOIN messages m ON m.id = sm.message_id
                JOIN conversations c ON c.id = m.conversation_id
                WHERE sm.subcluster_id = ?
                GROUP BY c.source
                """,
                (subcluster_id,),
            ).fetchall()
            return {str(r["source"]): int(r["count"]) for r in rows}

    def cluster_embedding_points(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT cm.cluster_id, m.timestamp, e.vector_json
                FROM cluster_membership cm
                JOIN messages m ON m.id = cm.message_id
                JOIN embeddings e ON e.item_id = cm.message_id
                """
            ).fetchall()
            return [
                {
                    "entity_id": str(r["cluster_id"]),
                    "timestamp": str(r["timestamp"]),
                    "vector": json.loads(r["vector_json"]),
                }
                for r in rows
            ]

    def subcluster_embedding_points(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT sm.subcluster_id, m.timestamp, e.vector_json
                FROM subcluster_membership sm
                JOIN messages m ON m.id = sm.message_id
                JOIN embeddings e ON e.item_id = sm.message_id
                """
            ).fetchall()
            return [
                {
                    "entity_id": str(r["subcluster_id"]),
                    "timestamp": str(r["timestamp"]),
                    "vector": json.loads(r["vector_json"]),
                }
                for r in rows
            ]

    def replace_drift_rows(self, level: str, rows: list[dict[str, Any]]) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM cluster_centroids_time WHERE level = ?", (level,))
            conn.executemany(
                """
                INSERT INTO cluster_centroids_time(level, entity_id, bucket_start_date, centroid_vector_json, message_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        level,
                        str(r["entity_id"]),
                        str(r["bucket_start_date"]),
                        json.dumps(r["centroid_vector"]),
                        int(r["message_count"]),
                        str(r["created_at"]),
                    )
                    for r in rows
                ],
            )

    def drift_rows(self, level: str, entity_id: str | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT level, entity_id, bucket_start_date, centroid_vector_json, message_count, created_at
            FROM cluster_centroids_time
            WHERE level = ?
        """
        params: list[Any] = [level]
        if entity_id is not None:
            sql += " AND entity_id = ?"
            params.append(str(entity_id))
        sql += " ORDER BY bucket_start_date ASC"
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "level": str(r["level"]),
                    "entity_id": str(r["entity_id"]),
                    "bucket_start_date": str(r["bucket_start_date"]),
                    "centroid_vector": json.loads(r["centroid_vector_json"]),
                    "message_count": int(r["message_count"]),
                    "created_at": str(r["created_at"]),
                }
                for r in rows
            ]

    def drift_stale(self, level: str) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT MAX(created_at) AS latest_drift_at, COUNT(*) AS count_rows FROM cluster_centroids_time WHERE level = ?",
                (level,),
            ).fetchone()
            if row is None or int(row["count_rows"] or 0) == 0:
                return True
            latest_drift_at = str(row["latest_drift_at"] or "")

            latest_embedding = conn.execute("SELECT MAX(created_at) FROM embeddings").fetchone()[0]
            latest_cluster = conn.execute("SELECT MAX(created_at) FROM clusters").fetchone()[0]
            latest_subcluster = conn.execute("SELECT MAX(created_at) FROM subclusters").fetchone()[0]

            latest_embedding = str(latest_embedding or "")
            latest_cluster = str(latest_cluster or "")
            latest_subcluster = str(latest_subcluster or "")

            if latest_embedding and latest_embedding > latest_drift_at:
                return True
            if latest_cluster and latest_cluster > latest_drift_at:
                return True
            if level == "subcluster" and latest_subcluster and latest_subcluster > latest_drift_at:
                return True
            return False

    def replace_mode_scores(self, level: str, rows: list[dict[str, Any]]) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM mode_scores WHERE level = ?", (level,))
            conn.executemany(
                """
                INSERT INTO mode_scores(level, entity_id, mode_weights_json, dominant_mode, dominant_weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        level,
                        str(r["entity_id"]),
                        json.dumps(r["mode_weights"]),
                        r.get("dominant_mode"),
                        r.get("dominant_weight"),
                        str(r["created_at"]),
                    )
                    for r in rows
                ],
            )

    def mode_scores(self, level: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT level, entity_id, mode_weights_json, dominant_mode, dominant_weight, created_at
                FROM mode_scores
                WHERE level = ?
                """,
                (level,),
            ).fetchall()
            return [
                {
                    "level": str(r["level"]),
                    "entity_id": str(r["entity_id"]),
                    "mode_weights": json.loads(r["mode_weights_json"]),
                    "dominant_mode": r["dominant_mode"],
                    "dominant_weight": float(r["dominant_weight"]) if r["dominant_weight"] is not None else None,
                    "created_at": str(r["created_at"]),
                }
                for r in rows
            ]

    def mode_scores_stale(self, level: str) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT MAX(created_at) AS latest_mode_at, COUNT(*) AS count_rows FROM mode_scores WHERE level = ?",
                (level,),
            ).fetchone()
            if row is None or int(row["count_rows"] or 0) == 0:
                return True
            latest_mode_at = str(row["latest_mode_at"] or "")

            if level == "subcluster":
                latest_entity_at = conn.execute("SELECT MAX(created_at) FROM subclusters").fetchone()[0]
            else:
                latest_entity_at = conn.execute("SELECT MAX(created_at) FROM clusters").fetchone()[0]
            latest_entity_at = str(latest_entity_at or "")
            if latest_entity_at and latest_entity_at > latest_mode_at:
                return True
            return False

    def mode_timeline_points(self, level: str) -> list[dict[str, Any]]:
        if level == "subcluster":
            sql = """
                SELECT sm.subcluster_id AS entity_id, m.timestamp, c.source
                FROM subcluster_membership sm
                JOIN messages m ON m.id = sm.message_id
                JOIN conversations c ON c.id = m.conversation_id
            """
        else:
            sql = """
                SELECT cm.cluster_id AS entity_id, m.timestamp, c.source
                FROM cluster_membership cm
                JOIN messages m ON m.id = cm.message_id
                JOIN conversations c ON c.id = m.conversation_id
            """
        with self.connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [
                {
                    "entity_id": str(r["entity_id"]),
                    "timestamp": str(r["timestamp"]),
                    "source": str(r["source"]),
                }
                for r in rows
            ]

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
