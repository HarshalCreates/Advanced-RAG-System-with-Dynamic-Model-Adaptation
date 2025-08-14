"""Enhanced retrieval service with all advanced features integrated."""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from app.retrieval.service import HybridRetrievalService


class EnhancedRetrievalService(HybridRetrievalService):
    """Enhanced version of HybridRetrievalService with all advanced features."""
    
    def search_with_routing(self, query: str, top_k: int, filters: dict | None = None) -> List[Tuple[str, float]]:
        """Enhanced search with intelligent routing and all advanced features."""
        
        # Step 1: Intelligent query routing
        routing_strategy = self.query_router.get_routing_strategy(query)
        effective_top_k = int(top_k * routing_strategy.top_k_multiplier)
        
        # Step 2: Query expansion and rewriting
        expanded_query = self.query_expander.expand_query(query, max_expansions=3)
        rewritten_queries = self.query_rewriter.rewrite_query(query)
        all_queries = [query] + expanded_query.expansions + [rq[0] for rq in rewritten_queries[:2]]
        
        # Step 3: Execute searches based on routing strategy
        if routing_strategy.primary_method == "dense":
            dense_results = self._search_dense_enhanced(all_queries, effective_top_k)
            sparse_results = []
        elif routing_strategy.primary_method == "sparse":
            dense_results = []
            sparse_results = self._search_sparse_enhanced(all_queries, effective_top_k)
        else:
            # Hybrid strategy
            dense_results = self._search_dense_enhanced(all_queries, effective_top_k)
            sparse_results = self._search_sparse_enhanced(all_queries, effective_top_k)
        
        # Step 4: Graph-based retrieval
        graph_results = []
        if dense_results or sparse_results:
            seed_docs = []
            if dense_results:
                seed_docs.extend([doc_id for doc_id, _ in sorted(dense_results, key=lambda x: x[1], reverse=True)[:3]])
            if sparse_results:
                seed_docs.extend([doc_id for doc_id, _ in sorted(sparse_results, key=lambda x: x[1], reverse=True)[:3]])
            
            if seed_docs:
                try:
                    graph_results = self.advanced_graph.search_by_graph_traversal(
                        query, list(set(seed_docs)), max_hops=2, top_k=effective_top_k
                    )
                except Exception:
                    graph_results = []
        
        # Step 5: Fusion ranking with routing strategy weights
        weights = routing_strategy.ranking_weights
        fused = self.learned_ranker.combine(
            dense=dense_results, 
            sparse=sparse_results, 
            graph=graph_results,
            custom_weights=weights
        )
        
        # Step 6: Apply negative sampling
        document_texts = {doc_id: self.documents.texts.get(doc_id, "") for doc_id, _ in fused}
        fused = self.negative_sampler.filter_results(query, fused, document_texts)
        
        # Step 7: Cross-encoder reranking
        if routing_strategy.reranking_enabled and fused:
            rerank_docs = []
            for doc_id, score in fused[:effective_top_k * 2]:
                text = self.get_text(doc_id)
                if text:
                    rerank_docs.append((doc_id, score, text[:1000]))
            
            if rerank_docs:
                try:
                    rerank_results = self.cross_encoder_reranker.rerank(
                        query, rerank_docs, top_k=top_k, boost_weight=0.4
                    )
                    fused = [(r.document_id, r.final_score) for r in rerank_results]
                except Exception:
                    pass  # Fall back to original ranking
        
        # Step 8: Apply boost factors from routing strategy
        if routing_strategy.boost_factors:
            fused = self._apply_boost_factors(fused, routing_strategy.boost_factors, query)
        
        return fused[:top_k]
    
    def _search_dense_enhanced(self, queries: List[str], top_k: int) -> List[Tuple[str, float]]:
        """Enhanced dense search with better score combination."""
        all_results = []
        for q in queries:
            q_vec = self.emb_client.embed([q])
            results = self.vector.search(q_vec, top_k=top_k)[0]
            all_results.extend(results)
        
        # Combine scores with weighted average for repeated documents
        combined = {}
        doc_counts = {}
        
        for doc_id, score in all_results:
            if doc_id not in combined:
                combined[doc_id] = 0.0
                doc_counts[doc_id] = 0
            combined[doc_id] += score
            doc_counts[doc_id] += 1
        
        # Calculate weighted average
        for doc_id in combined:
            combined[doc_id] = combined[doc_id] / doc_counts[doc_id]
        
        return list(combined.items())
    
    def _search_sparse_enhanced(self, queries: List[str], top_k: int) -> List[Tuple[str, float]]:
        """Enhanced sparse search combining TF-IDF and BM25."""
        all_results = []
        
        # TF-IDF search
        tfidf_results = self.sparse.search(queries, top_k=top_k)
        for results in tfidf_results:
            all_results.extend([(doc_id, score * 0.6) for doc_id, score in results])  # Weight TF-IDF
        
        # BM25 search
        if hasattr(self, "bm25") and self.bm25:
            bm25_results = self.bm25.search(queries, top_k=top_k)
            for results in bm25_results:
                all_results.extend([(doc_id, score * 0.4) for doc_id, score in results])  # Weight BM25
        
        # Combine scores
        combined = {}
        for doc_id, score in all_results:
            combined[doc_id] = combined.get(doc_id, 0.0) + score
        
        return list(combined.items())
    
    def _apply_boost_factors(self, results: List[Tuple[str, float]], 
                           boost_factors: Dict[str, float], query: str) -> List[Tuple[str, float]]:
        """Apply boost factors based on content analysis."""
        
        if not boost_factors:
            return results
        
        boosted_results = []
        query_lower = query.lower()
        
        for doc_id, score in results:
            boost = 1.0
            text = self.get_text(doc_id).lower()
            
            # Apply various boost factors
            if "code_content" in boost_factors and any(term in text for term in ["def ", "function", "class ", "import"]):
                boost *= boost_factors["code_content"]
            
            if "math_content" in boost_factors and any(term in text for term in ["formula", "equation", "theorem"]):
                boost *= boost_factors["math_content"]
            
            if "numerical_content" in boost_factors and any(char.isdigit() for char in text):
                boost *= boost_factors["numerical_content"]
            
            if "exact_match" in boost_factors and query_lower in text:
                boost *= boost_factors["exact_match"]
            
            boosted_results.append((doc_id, score * boost))
        
        # Re-sort by new scores
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results
    
    def get_routing_explanation(self, query: str) -> Dict[str, Any]:
        """Get detailed explanation of routing decisions for this query."""
        return self.query_router.get_routing_explanation(query)
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get analytics about search patterns and routing decisions."""
        routing_stats = self.query_router.get_routing_statistics()
        
        return {
            "routing_statistics": routing_stats,
            "reranker_available": self.cross_encoder_reranker.available,
            "components_status": {
                "cross_encoder_reranker": self.cross_encoder_reranker.available,
                "query_router": True,
                "learned_fusion": True,
                "advanced_graph": True,
                "negative_sampler": True
            }
        }
