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
from app.retrieval.query_expansion import QueryExpander, QueryRewriter
from app.retrieval.learned_fusion import LearnedFusionRanker
from app.retrieval.negative_sampling import NegativeSampler
from app.retrieval.advanced_graph import AdvancedGraphRetriever
from app.retrieval.cross_encoder_reranker import CrossEncoderReranker, HybridReranker
from app.retrieval.query_router import QueryRoutingEngine


class HybridRetrievalService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.documents = DocumentStore()
        # load persisted store if exists
        from pathlib import Path

        self.store_path = Path(self.settings.index_dir) / "doc_store.json"
        self.documents.load(self.store_path)
        
        # Initialize retrievers
        self.sparse = SparseRetriever()
        self.graph = GraphRetriever()  # Legacy simple graph
        self.advanced_graph = AdvancedGraphRetriever()  # New advanced graph
        
        # Initialize ranking and processing
        self.rank = FusionRanker()  # Legacy static ranker
        self.learned_ranker = LearnedFusionRanker()  # New learned ranker
        self.query_expander = QueryExpander()
        self.query_rewriter = QueryRewriter()
        self.negative_sampler = NegativeSampler()
        
        # Initialize advanced retrieval components
        self.cross_encoder_reranker = CrossEncoderReranker()
        self.hybrid_reranker = HybridReranker()
        self.query_router = QueryRoutingEngine()
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

    def search(self, query: str, top_k: int, filters: dict | None = None, 
              use_advanced_features: bool = True) -> List[Tuple[str, float]]:
        """Enhanced search with intelligent routing and advanced features."""
        
        # Step 1: Intelligent query routing
        if use_advanced_features:
            routing_strategy = self.query_router.get_routing_strategy(query)
            effective_top_k = int(top_k * routing_strategy.top_k_multiplier)
            
            # Query expansion and rewriting based on query type
            expanded_query = self.query_expander.expand_query(query, max_expansions=3)
            rewritten_queries = self.query_rewriter.rewrite_query(query)
            
            # Use multiple query variations
            all_queries = [query] + expanded_query.expansions + [rq[0] for rq in rewritten_queries[:2]]
        else:
            routing_strategy = None
            effective_top_k = top_k
            all_queries = [query]
        
        # Step 2: Execute searches based on routing strategy
        all_dense_results = []
        all_sparse_results = []
        
        for q in all_queries:
            # Dense retrieval
            q_vec = self.emb_client.embed([q])
            dense_results = self.vector.search(q_vec, top_k=top_k*2)[0]  # Get more candidates
            all_dense_results.extend(dense_results)
            
            # Sparse retrieval
            sparse_results = self.sparse.search([q], top_k=top_k*2)[0]
            all_sparse_results.extend(sparse_results)
            
            # BM25 retrieval
            if not hasattr(self, "bm25"):
                self.bm25 = BM25Retriever()
                self.bm25.fit([], [])
            bm25_results = self.bm25.search([q], top_k=top_k*2)[0]
            all_sparse_results.extend(bm25_results)
        
        # Step 3: Combine and deduplicate results
        dense_combined = {}
        sparse_combined = {}
        
        for doc_id, score in all_dense_results:
            dense_combined[doc_id] = max(dense_combined.get(doc_id, 0.0), score)
        
        for doc_id, score in all_sparse_results:
            sparse_combined[doc_id] = max(sparse_combined.get(doc_id, 0.0), score)
        
        dense_list = list(dense_combined.items())
        sparse_list = list(sparse_combined.items())
        
        # Step 4: Advanced graph-based retrieval
        graph_results = []
        if use_advanced_features and dense_list:
            # Use top dense results as seeds for graph traversal
            seed_docs = [doc_id for doc_id, _ in sorted(dense_list, key=lambda x: x[1], reverse=True)[:5]]
            try:
                graph_results = self.advanced_graph.search_by_graph_traversal(
                    query, seed_docs, max_hops=2, top_k=top_k
                )
            except Exception as e:
                print(f"Graph retrieval failed: {e}")
                graph_results = []
        
        # Step 5: Fusion ranking
        if use_advanced_features:
            # Use learned ranker
            features_cache = {}  # Could be populated with pre-computed features
            fused = self.learned_ranker.combine(
                dense=dense_list, 
                sparse=sparse_list, 
                graph=graph_results,
                query=query,
                features_cache=features_cache
            )
        else:
            # Use legacy static ranker
            fused = self.rank.combine(dense=dense_list, sparse=sparse_list, graph=graph_results)
        
        # Step 6: Apply negative sampling filter
        if use_advanced_features:
            document_texts = {doc_id: self.documents.texts.get(doc_id, "") for doc_id, _ in fused}
            fused = self.negative_sampler.filter_results(query, fused, document_texts)
        
        # Step 7: Apply metadata filters
        if filters:
            allowed_ids = set(self.documents.filter_ids(filters))
            fused = [(doc_id, score) for doc_id, score in fused if doc_id in allowed_ids]
        
        # Step 8: Record feedback for learning (implicit)
        if use_advanced_features and fused:
            # Record search event for learned ranker
            try:
                for rank, (doc_id, score) in enumerate(fused[:top_k]):
                    # Assume no click for now (would be updated with actual user feedback)
                    self.learned_ranker.record_feedback(
                        query=query,
                        doc_id=doc_id,
                        rank_position=rank,
                        clicked=False,  # Would be updated with real feedback
                        dwell_time=0.0
                    )
            except Exception as e:
                print(f"Feedback recording failed: {e}")
        
        return fused[:top_k]

    def get_text(self, doc_id: str) -> str:
        return self.documents.texts.get(doc_id, "")


