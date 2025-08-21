"""Cross-encoder reranking for improved retrieval relevance."""
from __future__ import annotations

import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    import torch
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    batch_size: int = 16
    device: str = "auto"  # "cpu", "cuda", or "auto"
    cache_size: int = 1000


@dataclass
class RerankingResult:
    """Result from cross-encoder reranking."""
    document_id: str
    original_score: float
    reranking_score: float
    final_score: float
    boost_factor: float
    content_snippet: str


class CrossEncoderReranker:
    """Cross-encoder based reranking for query-document relevance."""
    
    def __init__(self, config: RerankerConfig = None):
        self.config = config or RerankerConfig()
        self.model: Optional[Any] = None
        self.available = CROSS_ENCODER_AVAILABLE
        self.score_cache: Dict[str, float] = {}
        
        if not self.available:
            print("Warning: Cross-encoder reranking not available. Install sentence-transformers and torch.")
        else:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the cross-encoder model."""
        if not self.available:
            return
        
        try:
            # Determine device
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            # Load cross-encoder model
            self.model = CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length,
                device=device
            )
            
            print(f"Cross-encoder reranker initialized: {self.config.model_name} on {device}")
            
        except Exception as e:
            print(f"Failed to initialize cross-encoder: {e}")
            self.model = None
            self.available = False
    
    def rerank(self, query: str, documents: List[Tuple[str, float, str]], 
               top_k: Optional[int] = None, boost_weight: float = 0.5) -> List[RerankingResult]:
        """
        Rerank documents using cross-encoder scoring.
        
        Args:
            query: Search query
            documents: List of (doc_id, original_score, content) tuples
            top_k: Number of top results to return (None for all)
            boost_weight: Weight for combining original and reranking scores (0-1)
        
        Returns:
            List of reranking results sorted by final score
        """
        
        if not self.available or not self.model:
            # Fallback: return original ranking
            return self._fallback_rerank(query, documents, top_k)
        
        if not documents:
            return []
        
        # Prepare query-document pairs for reranking
        query_doc_pairs = []
        doc_metadata = []
        
        for doc_id, original_score, content in documents:
            # Truncate content if too long
            truncated_content = content[:1000] if len(content) > 1000 else content
            query_doc_pairs.append([query, truncated_content])
            doc_metadata.append((doc_id, original_score, content))
        
        # Get reranking scores in batches
        reranking_scores = self._score_in_batches(query_doc_pairs)
        
        # Combine scores and create results
        results = []
        for i, (doc_id, original_score, content) in enumerate(doc_metadata):
            reranking_score = reranking_scores[i]
            
            # Combine original and reranking scores
            final_score = (1 - boost_weight) * original_score + boost_weight * reranking_score
            boost_factor = reranking_score / (original_score + 1e-8)
            
            # Create snippet for display
            content_snippet = content[:200] + "..." if len(content) > 200 else content
            
            result = RerankingResult(
                document_id=doc_id,
                original_score=original_score,
                reranking_score=reranking_score,
                final_score=final_score,
                boost_factor=boost_factor,
                content_snippet=content_snippet
            )
            results.append(result)
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Return top_k results
        if top_k:
            results = results[:top_k]
        
        return results
    
    def _score_in_batches(self, query_doc_pairs: List[List[str]]) -> List[float]:
        """Score query-document pairs in batches."""
        all_scores = []
        
        for i in range(0, len(query_doc_pairs), self.config.batch_size):
            batch = query_doc_pairs[i:i + self.config.batch_size]
            
            try:
                # Get scores from cross-encoder
                batch_scores = self.model.predict(batch)
                
                # Convert to Python floats
                if isinstance(batch_scores, np.ndarray):
                    batch_scores = batch_scores.tolist()
                elif hasattr(batch_scores, 'cpu'):
                    batch_scores = batch_scores.cpu().numpy().tolist()
                
                all_scores.extend(batch_scores)
                
            except Exception as e:
                print(f"Error in batch scoring: {e}")
                # Fallback scores
                all_scores.extend([0.5] * len(batch))
        
        return all_scores
    
    def _fallback_rerank(self, query: str, documents: List[Tuple[str, float, str]], 
                        top_k: Optional[int]) -> List[RerankingResult]:
        """Fallback reranking when cross-encoder is not available."""
        
        results = []
        for doc_id, original_score, content in documents:
            # Simple heuristic reranking based on query term overlap
            reranking_score = self._heuristic_score(query, content)
            final_score = 0.7 * original_score + 0.3 * reranking_score
            
            content_snippet = content[:200] + "..." if len(content) > 200 else content
            
            result = RerankingResult(
                document_id=doc_id,
                original_score=original_score,
                reranking_score=reranking_score,
                final_score=final_score,
                boost_factor=reranking_score / (original_score + 1e-8),
                content_snippet=content_snippet
            )
            results.append(result)
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def _heuristic_score(self, query: str, content: str) -> float:
        """Simple heuristic scoring for fallback."""
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        
        if not query_terms:
            return 0.5
        
        # Jaccard similarity
        intersection = len(query_terms.intersection(content_terms))
        union = len(query_terms.union(content_terms))
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Normalize to 0-1 range and add baseline
        return 0.3 + 0.7 * jaccard
    
    def batch_rerank_multiple_queries(self, queries_docs: List[Tuple[str, List[Tuple[str, float, str]]]], 
                                     top_k: Optional[int] = None) -> List[List[RerankingResult]]:
        """Efficiently rerank multiple queries in batch."""
        
        all_results = []
        
        for query, documents in queries_docs:
            results = self.rerank(query, documents, top_k)
            all_results.append(results)
        
        return all_results
    
    def get_reranking_explanation(self, query: str, document: str) -> Dict[str, Any]:
        """Get explanation for reranking decision."""
        
        if not self.available or not self.model:
            return {
                "available": False,
                "explanation": "Cross-encoder reranking not available",
                "score": 0.5,
                "factors": ["Using heuristic fallback"]
            }
        
        # Get detailed scoring
        score = self.model.predict([[query, document]])[0]
        
        # Simple explanation factors
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        overlap = query_terms.intersection(doc_terms)
        
        factors = []
        if len(overlap) > 0:
            factors.append(f"Query terms found: {', '.join(list(overlap)[:5])}")
        
        if score > 0.7:
            factors.append("High semantic relevance detected")
        elif score > 0.3:
            factors.append("Moderate semantic relevance")
        else:
            factors.append("Low semantic relevance")
        
        return {
            "available": True,
            "score": float(score),
            "query_terms": list(query_terms),
            "matching_terms": list(overlap),
            "factors": factors,
            "model": self.config.model_name
        }


class HybridReranker:
    """Combines multiple reranking strategies."""
    
    def __init__(self):
        self.cross_encoder = CrossEncoderReranker()
        self.strategies = ["cross_encoder", "lexical", "semantic"]
    
    def rerank_with_multiple_strategies(self, query: str, documents: List[Tuple[str, float, str]], 
                                      strategy_weights: Dict[str, float] = None) -> List[RerankingResult]:
        """Rerank using multiple strategies with weighted combination."""
        
        if strategy_weights is None:
            strategy_weights = {
                "cross_encoder": 0.6,
                "lexical": 0.2,
                "semantic": 0.2
            }
        
        if not documents:
            return []
        
        # Get cross-encoder scores
        cross_encoder_results = self.cross_encoder.rerank(query, documents, boost_weight=1.0)
        
        # Create lookup for cross-encoder scores
        ce_scores = {result.document_id: result.reranking_score for result in cross_encoder_results}
        
        # Combine strategies
        final_results = []
        for doc_id, original_score, content in documents:
            # Get individual strategy scores
            ce_score = ce_scores.get(doc_id, 0.5)
            lexical_score = self._lexical_similarity(query, content)
            semantic_score = self._semantic_similarity(query, content)
            
            # Weighted combination
            combined_score = (
                strategy_weights.get("cross_encoder", 0.6) * ce_score +
                strategy_weights.get("lexical", 0.2) * lexical_score +
                strategy_weights.get("semantic", 0.2) * semantic_score
            )
            
            # Final score combining original and reranking
            final_score = 0.3 * original_score + 0.7 * combined_score
            
            content_snippet = content[:200] + "..." if len(content) > 200 else content
            
            result = RerankingResult(
                document_id=doc_id,
                original_score=original_score,
                reranking_score=combined_score,
                final_score=final_score,
                boost_factor=combined_score / (original_score + 1e-8),
                content_snippet=content_snippet
            )
            final_results.append(result)
        
        # Sort by final score
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return final_results
    
    def _lexical_similarity(self, query: str, content: str) -> float:
        """Calculate lexical similarity between query and content."""
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        
        if not query_terms:
            return 0.0
        
        # Term overlap ratio
        overlap = len(query_terms.intersection(content_terms))
        return overlap / len(query_terms)
    
    def _semantic_similarity(self, query: str, content: str) -> float:
        """Calculate semantic similarity (placeholder for now)."""
        # Placeholder - in production, would use sentence embeddings
        # For now, use enhanced lexical similarity
        
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Check for query as substring
        if query_lower in content_lower:
            return 0.9
        
        # Check for partial matches
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_lower)
        
        return matches / len(query_words) if query_words else 0.0


class RerankingEvaluator:
    """Evaluates reranking performance."""
    
    def __init__(self):
        self.metrics = ["precision", "recall", "ndcg", "mrr"]
    
    def evaluate_reranking(self, original_results: List[Tuple[str, float]], 
                          reranked_results: List[RerankingResult],
                          relevant_docs: List[str]) -> Dict[str, float]:
        """Evaluate reranking performance against ground truth."""
        
        # Extract document IDs from results
        original_doc_ids = [doc_id for doc_id, _ in original_results[:10]]
        reranked_doc_ids = [result.document_id for result in reranked_results[:10]]
        
        # Calculate metrics
        metrics = {}
        
        # Precision@k
        for k in [1, 3, 5, 10]:
            original_p_k = self._precision_at_k(original_doc_ids[:k], relevant_docs)
            reranked_p_k = self._precision_at_k(reranked_doc_ids[:k], relevant_docs)
            
            metrics[f"original_precision@{k}"] = original_p_k
            metrics[f"reranked_precision@{k}"] = reranked_p_k
            metrics[f"precision_improvement@{k}"] = reranked_p_k - original_p_k
        
        # MRR (Mean Reciprocal Rank)
        original_mrr = self._mean_reciprocal_rank(original_doc_ids, relevant_docs)
        reranked_mrr = self._mean_reciprocal_rank(reranked_doc_ids, relevant_docs)
        
        metrics["original_mrr"] = original_mrr
        metrics["reranked_mrr"] = reranked_mrr
        metrics["mrr_improvement"] = reranked_mrr - original_mrr
        
        return metrics
    
    def _precision_at_k(self, ranked_list: List[str], relevant_docs: List[str]) -> float:
        """Calculate precision at k."""
        if not ranked_list:
            return 0.0
        
        relevant_retrieved = sum(1 for doc in ranked_list if doc in relevant_docs)
        return relevant_retrieved / len(ranked_list)
    
    def _mean_reciprocal_rank(self, ranked_list: List[str], relevant_docs: List[str]) -> float:
        """Calculate mean reciprocal rank."""
        for i, doc in enumerate(ranked_list, 1):
            if doc in relevant_docs:
                return 1.0 / i
        return 0.0
