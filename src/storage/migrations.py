from __future__ import annotations

import sqlite3


MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            participants_json TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            raw_metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            speaker_role TEXT NOT NULL,
            original_text TEXT NOT NULL,
            redacted_text TEXT,
            parent_message_id TEXT,
            token_count INTEGER,
            raw_metadata_json TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            item_id TEXT PRIMARY KEY,
            vector_json TEXT NOT NULL,
            model_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS clusters (
            cluster_id INTEGER PRIMARY KEY,
            label TEXT NOT NULL,
            centroid_json TEXT,
            first_seen TEXT,
            last_seen TEXT,
            message_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cluster_membership (
            cluster_id INTEGER NOT NULL,
            message_id TEXT NOT NULL,
            PRIMARY KEY (cluster_id, message_id),
            FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id),
            FOREIGN KEY (message_id) REFERENCES messages(id)
        );

        CREATE TABLE IF NOT EXISTS topic_events (
            cluster_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
        CREATE INDEX IF NOT EXISTS idx_topic_events_cluster_ts ON topic_events(cluster_id, timestamp);
        """,
    )
    ,
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS subclusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_cluster_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (parent_cluster_id) REFERENCES clusters(cluster_id)
        );

        CREATE TABLE IF NOT EXISTS subcluster_membership (
            subcluster_id INTEGER NOT NULL,
            message_id TEXT NOT NULL,
            PRIMARY KEY (subcluster_id, message_id),
            FOREIGN KEY (subcluster_id) REFERENCES subclusters(id),
            FOREIGN KEY (message_id) REFERENCES messages(id)
        );

        CREATE INDEX IF NOT EXISTS idx_subclusters_parent ON subclusters(parent_cluster_id);
        CREATE INDEX IF NOT EXISTS idx_subcluster_membership_subcluster ON subcluster_membership(subcluster_id);
        """,
    )
    ,
    (
        3,
        """
        CREATE TABLE IF NOT EXISTS cluster_centroids_time (
            level TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            bucket_start_date TEXT NOT NULL,
            centroid_vector_json TEXT NOT NULL,
            message_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (level, entity_id, bucket_start_date)
        );

        CREATE INDEX IF NOT EXISTS idx_centroids_time_level_bucket ON cluster_centroids_time(level, bucket_start_date);
        """,
    )
]


class MigrationManager:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def migrate(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT (datetime('now')));"
        )
        current = self._current_version()

        for version, sql in MIGRATIONS:
            if version <= current:
                continue
            self.conn.executescript(sql)
            self.conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))
        self.conn.commit()

    def _current_version(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
        return int(row[0]) if row else 0
