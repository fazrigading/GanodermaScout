from __future__ import annotations

import os
from typing import Any

EMBEDDING_DIMENSIONS = 1536
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


class EmbeddingClient:
    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        *,
        dimensions: int = EMBEDDING_DIMENSIONS,
        openai_client: Any | None = None,
        local_model: Any | None = None,
    ) -> None:
        configured_provider = (provider or os.getenv("EMBEDDING_PROVIDER", "huggingface")).lower()
        if configured_provider in {"huggingface", "local"}:
            self.provider = "huggingface"
            self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_LOCAL_MODEL)
        elif configured_provider == "openai":
            self.provider = "openai"
            self.model_name = model_name or os.getenv(
                "OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_MODEL
            )
        else:
            raise ValueError("EMBEDDING_PROVIDER must be 'openai' or 'huggingface'")
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        self.dimensions = dimensions
        self._openai_client = openai_client
        self._local_model = local_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "openai":
            vectors = self._embed_openai(texts)
            if any(len(vector) != self.dimensions for vector in vectors):
                raise ValueError(f"OpenAI embeddings must have {self.dimensions} dimensions")
            return vectors
        vectors = self._embed_local(texts)
        if len(vectors) != len(texts):
            raise RuntimeError("Local embedding model returned an unexpected number of vectors")
        if any(not vector for vector in vectors):
            raise ValueError("Embeddings must not be empty")
        if any(len(vector) > self.dimensions for vector in vectors):
            raise ValueError(f"Local embeddings cannot exceed {self.dimensions} dimensions")
        return [vector + [0.0] * (self.dimensions - len(vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        client = self._openai_client
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self._openai_client = client
        response = client.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=self.dimensions,
        )
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise RuntimeError("Embedding API returned an unexpected number of vectors")
        return [[float(value) for value in item.embedding] for item in data]

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = self._local_model
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            self._local_model = model
        embeddings = model.encode(texts, normalize_embeddings=True)
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return [[float(value) for value in embedding] for embedding in embeddings]
