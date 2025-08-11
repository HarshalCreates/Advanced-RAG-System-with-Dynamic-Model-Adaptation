from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from app.embeddings.factory import EmbeddingFactory
from app.index.store import DocumentStore
from app.models.config import get_settings
from app.retrieval.graph import GraphRetriever
from app.retrieval.hybrid import FusionRanker
from app.retrieval.sparse import SparseRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector_backends import FAISSBackend, InMemoryChromaLike, VectorBackend


class HybridRetrievalService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.documents = DocumentStore()
        # load persisted store if exists
        from pathlib import Path

        self.store_path = Path(self.settings.index_dir) / "doc_store.json"
        self.documents.load(self.store_path)
        self.sparse = SparseRetriever()
        self.graph = GraphRetriever()
        self.rank = FusionRanker()
        # Embeddings
        self.emb_client = EmbeddingFactory(
            backend=self.settings.embedding_backend, model=self.settings.embedding_model
        ).build()
        # Vector store
        self.vector: VectorBackend
        if self.settings.retriever_backend == "faiss":
            dim = 768 if "768" in self.settings.embedding_model else 384
            self.vector = FAISSBackend(dim)
        else:
            dim = 768 if "768" in self.settings.embedding_model else 384
            self.vector = InMemoryChromaLike(dim)

        # Reindex vectors/sparse from persisted store on startup
        if len(self.documents.texts) > 0:
            try:
                self.reindex_all()
            except Exception:
                # tolerate failures; will be rebuilt on next ingest
                pass

    # --- Hot swap helpers ---
    def _estimate_dim(self) -> int:
        # Simple heuristic; most sentence-transformer minis are 384
        return 768 if "768" in self.settings.embedding_model else 384

    def reindex_all(self) -> None:
        ids = list(self.documents.texts.keys())
        texts = self.documents.get_texts(ids)
        metas = [self.documents.metas.get(i, {}) for i in ids]
        if not ids:
            return
        # reset vector index
        if isinstance(self.vector, FAISSBackend):
            # recreate FAISS
            self.vector = FAISSBackend(self._estimate_dim())
        else:
            self.vector = InMemoryChromaLike(self._estimate_dim())
        vectors = self.emb_client.embed(texts)
        self.vector.upsert(ids, vectors, metas)
        self.sparse.fit(ids, texts)
        self.bm25 = getattr(self, "bm25", BM25Retriever())
        self.bm25.fit(ids, texts)

    def hot_swap_embeddings(self, backend: str, model: str) -> None:
        self.settings.embedding_backend = backend
        self.settings.embedding_model = model
        self.emb_client = EmbeddingFactory(backend=backend, model=model).build()
        self.reindex_all()

    def hot_swap_retriever(self, backend: str) -> None:
        self.settings.retriever_backend = backend
        # recreate vector backend and reindex
        if backend == "faiss":
            self.vector = FAISSBackend(self._estimate_dim())
        else:
            self.vector = InMemoryChromaLike(self._estimate_dim())
        self.reindex_all()

    def index(self, ids: List[str], texts: List[str], metadatas: List[dict] | None = None) -> None:
        metadatas = metadatas or [{} for _ in texts]
        self.documents.upsert(ids, texts, metadatas)
        # persist store
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.documents.save(self.store_path)
        # Dense
        vectors = self.emb_client.embed(texts)
        self.vector.upsert(ids, vectors, metadatas)
        # Sparse and BM25
        self.sparse.fit(ids, texts)
        self.bm25 = getattr(self, "bm25", BM25Retriever())
        self.bm25.fit(ids, texts)

    def search(self, query: str, top_k: int, filters: dict | None = None) -> List[Tuple[str, float]]:
        # Basic metadata filtering by post-filtering ids
        allowed_ids = None
        if filters:
            allowed_ids = set(self.documents.filter_ids(filters))
        q_vec = self.emb_client.embed([query])
        dense = self.vector.search(q_vec, top_k=top_k)[0]
        sparse = self.sparse.search([query], top_k=top_k)[0]
        # Handle BM25 not initialized yet
        if not hasattr(self, "bm25"):
            self.bm25 = BM25Retriever()
            self.bm25.fit([], [])
        kw = self.bm25.search([query], top_k=top_k)[0]
        # Combine sparse TF-IDF and BM25 first
        sparse_combined = {}
        for did, s in sparse + kw:
            sparse_combined[did] = max(sparse_combined.get(did, 0.0), s)
        sparse_list = list(sparse_combined.items())
        fused = self.rank.combine(dense=dense, sparse=sparse_list, graph=None)
        if allowed_ids is not None:
            fused = [(i, s) for i, s in fused if i in allowed_ids]
        return fused[:top_k]

    def get_text(self, doc_id: str) -> str:
        return self.documents.texts.get(doc_id, "")


