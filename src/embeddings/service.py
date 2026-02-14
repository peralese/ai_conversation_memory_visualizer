from __future__ import annotations

from typing import Iterable

from src.analysis.text import build_analysis_text
from src.embeddings.providers import EmbeddingProvider, LocalStubEmbeddingProvider, OpenAIEmbeddingProvider
from src.redaction.redactor import RedactionConfig, redact_text
from src.storage.repository import SQLiteRepository


class EmbeddingService:
    def __init__(self, repo: SQLiteRepository, provider: EmbeddingProvider | None = None):
        self.repo = repo
        self.provider = provider or self._default_provider()

    def embed_since(self, since: str | None = None, *, redact: bool = False, config: RedactionConfig | None = None, chunk_chars: int = 1500) -> int:
        rows = self.repo.list_messages_for_embedding(since)
        cfg = config or RedactionConfig()
        embedded_count = 0

        for row in rows:
            message_id = row["id"]
            if self.repo.has_embedding(message_id):
                continue

            text = row["embedding_text"]
            if redact:
                redacted_text = redact_text(text, cfg)
                self.repo.save_redacted_text(message_id, redacted_text)
                text = redacted_text

            text = build_analysis_text(text)
            chunks = _chunk_text(text, chunk_chars)
            if not chunks:
                continue

            # Use first chunk for MVP message-level embedding. TODO: multi-chunk aggregation.
            vector = self.provider.embed([chunks[0]])[0]
            self.repo.save_embedding(message_id, vector, self.provider.model_name)
            embedded_count += 1

        return embedded_count

    def _default_provider(self) -> EmbeddingProvider:
        try:
            return OpenAIEmbeddingProvider()
        except Exception:
            return LocalStubEmbeddingProvider()


def _chunk_text(text: str, chunk_chars: int) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    return [cleaned[i : i + chunk_chars] for i in range(0, len(cleaned), chunk_chars)]
