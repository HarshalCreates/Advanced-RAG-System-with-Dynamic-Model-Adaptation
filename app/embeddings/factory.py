from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


class EmbeddingClient:
    def embed(self, texts: List[str]) -> np.ndarray:  # shape (n, d)
        raise NotImplementedError


class OpenAIEmbeddings(EmbeddingClient):
    def __init__(self, model: str) -> None:
        from openai import OpenAI
        from app.models.config import get_settings
        
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        res = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [d.embedding for d in res.data]
        return np.array(vectors, dtype=np.float32)


class CohereEmbeddings(EmbeddingClient):
    def __init__(self, model: str) -> None:
        import cohere

        self.client = cohere.Client()
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        res = self.client.embed(model=self.model, texts=texts)
        return np.array(res.embeddings, dtype=np.float32)


class SentenceTransformerEmbeddings(EmbeddingClient):
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype(np.float32)


class HashEmbeddings(EmbeddingClient):
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        import hashlib

        arrs = []
        for t in texts:
            d = hashlib.sha256(t.encode("utf-8")).digest()
            v = np.frombuffer(d, dtype=np.uint8).astype(np.float32)
            v = np.resize(v, self.dim)
            v = v / (np.linalg.norm(v) + 1e-9)
            arrs.append(v)
        return np.vstack(arrs).astype(np.float32)


@dataclass
class EmbeddingFactory:
    backend: str
    model: str

    def build(self) -> EmbeddingClient:
        b = self.backend.lower()
        try:
            if b == "openai":
                return OpenAIEmbeddings(self.model)
            if b == "cohere":
                return CohereEmbeddings(self.model)
            if b == "sentence-transformers":
                return SentenceTransformerEmbeddings(self.model)
            if b == "hash":
                return HashEmbeddings()
        except Exception:
            # Fallback
            return HashEmbeddings()
        # Default fallback
        return HashEmbeddings()


