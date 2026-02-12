from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    model_name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise ValueError("openai package is not installed") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]


class LocalStubEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "local-stub-hash-v1", dims: int = 64):
        self.model_name = model_name
        self.dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = [(digest[i % len(digest)] / 255.0) for i in range(self.dims)]
            vectors.append(values)
        return vectors
