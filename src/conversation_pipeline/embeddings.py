from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.embeddings.providers import EmbeddingProvider, LocalStubEmbeddingProvider, OpenAIEmbeddingProvider
from src.storage.repository import SQLiteRepository


@dataclass
class ConversationEmbeddingConfig:
    force_reembed: bool = False


def embed_conversations(
    repo: SQLiteRepository,
    provider: EmbeddingProvider | None = None,
    config: ConversationEmbeddingConfig | None = None,
) -> dict[str, int | str]:
    cfg = config or ConversationEmbeddingConfig()
    embedder = provider or _default_provider()

    embedded = 0
    skipped = 0
    rows = repo.list_conversation_rollups()
    now = datetime.now(timezone.utc).isoformat()

    for row in rows:
        conversation_id = str(row["conversation_id"])
        rollup_text = str(row.get("rollup_text") or "").strip()
        if not rollup_text:
            skipped += 1
            continue

        current = repo.get_conversation_embedding(conversation_id)
        rollup_hash = str(row.get("rollup_hash") or "")
        if (
            not cfg.force_reembed
            and current is not None
            and str(current.get("rollup_hash") or "") == rollup_hash
            and str(current.get("embedding_model") or "") == embedder.model_name
        ):
            skipped += 1
            continue

        vector = embedder.embed([rollup_text])[0]
        repo.upsert_conversation_embedding(
            {
                "conversation_id": conversation_id,
                "embedding_model": embedder.model_name,
                "embedding": vector,
                "embedding_dim": len(vector),
                "rollup_hash": rollup_hash,
                "created_at": str(current.get("created_at") if current else now),
                "updated_at": now,
            }
        )
        embedded += 1

    return {"embedded": embedded, "skipped": skipped, "model": embedder.model_name}


def _default_provider() -> EmbeddingProvider:
    try:
        return OpenAIEmbeddingProvider()
    except Exception:
        return LocalStubEmbeddingProvider()
