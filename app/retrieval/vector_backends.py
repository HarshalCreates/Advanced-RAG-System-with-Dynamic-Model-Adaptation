from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

import numpy as np


class VectorBackend(ABC):
    @abstractmethod
    def upsert(self, ids: List[str], vectors: np.ndarray, metadatas: List[dict]) -> None: ...

    @abstractmethod
    def search(self, query_vectors: np.ndarray, top_k: int) -> List[List[Tuple[str, float]]]: ...


class FAISSBackend(VectorBackend):
    def __init__(self, dim: int) -> None:
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: List[str] = []
        self.metas: List[dict] = []

    def upsert(self, ids: List[str], vectors: np.ndarray, metadatas: List[dict]) -> None:
        import faiss

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        # Normalize for cosine with IP
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.ids.extend(ids)
        self.metas.extend(metadatas)

    def search(self, query_vectors: np.ndarray, top_k: int) -> List[List[Tuple[str, float]]]:
        import faiss

        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        faiss.normalize_L2(query_vectors)
        scores, idxs = self.index.search(query_vectors, top_k)
        results: List[List[Tuple[str, float]]] = []
        for row_scores, row_idxs in zip(scores, idxs):
            row: List[Tuple[str, float]] = []
            for s, i in zip(row_scores, row_idxs):
                if i == -1:
                    continue
                row.append((self.ids[i], float(s)))
            results.append(row)
        return results


class InMemoryChromaLike(VectorBackend):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []

    def upsert(self, ids: List[str], vectors: np.ndarray, metadatas: List[dict]) -> None:
        for i, v in zip(ids, vectors):
            self.ids.append(i)
            self.vectors.append(v.astype(np.float32))

    def search(self, query_vectors: np.ndarray, top_k: int) -> List[List[Tuple[str, float]]]:
        results: List[List[Tuple[str, float]]] = []
        for q in query_vectors:
            sims = [float(np.dot(q, v) / (np.linalg.norm(q) * (np.linalg.norm(v) + 1e-10))) for v in self.vectors]
            idxs = np.argsort(sims)[::-1][:top_k]
            results.append([(self.ids[i], sims[i]) for i in idxs])
        return results


