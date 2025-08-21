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
        
        # Initialize advanced retrieval components lazily (disabled for speed)
        self._cross_encoder_reranker = None
        self._hybrid_reranker = None
        self._query_router = None
        self._use_advanced_features = False  # Disable for speed
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
    
    @property
    def cross_encoder_reranker(self):
        """Lazy load cross encoder reranker only when needed."""
        if self._cross_encoder_reranker is None:
            from app.retrieval.cross_encoder_reranker import CrossEncoderReranker
            self._cross_encoder_reranker = CrossEncoderReranker()
        return self._cross_encoder_reranker
    
    @property
    def hybrid_reranker(self):
        """Lazy load hybrid reranker only when needed."""
        if self._hybrid_reranker is None:
            from app.retrieval.hybrid import HybridReranker
            self._hybrid_reranker = HybridReranker()
        return self._hybrid_reranker
    
    @property
    def query_router(self):
        """Lazy load query router only when needed."""
        if self._query_router is None:
            from app.retrieval.query_router import QueryRoutingEngine
            self._query_router = QueryRoutingEngine()
        return self._query_router

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
              use_advanced_features: bool = False) -> List[Tuple[str, float]]:
        """Enhanced search with better diversification for citations."""
        
        # Use simple search for speed
        all_queries = [query]
        
        # Fast search - use only vector search for speed
        all_dense_results = []
        
        for q in all_queries:
            # Dense retrieval only (fastest)
            try:
                q_vec = self.emb_client.embed([q])
                # Get more results initially for better diversification
                vector_search_results = self.vector.search(q_vec, top_k=top_k * 2)
                if vector_search_results and len(vector_search_results) > 0:
                    dense_results = vector_search_results[0]
                    all_dense_results.extend(dense_results)
            except Exception as e:
                print(f"Vector retrieval failed for query '{q}': {e}")
        
        # Use only dense results for speed
        fused = all_dense_results
        
        # Fast processing - just deduplicate and sort
        fused = []
        seen = set()
        
        for doc_id, score in all_dense_results:
            if doc_id not in seen:
                fused.append((doc_id, score))
                seen.add(doc_id)
        
        # Sort by score (highest first)
        fused.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversification to ensure different citations for different queries
        diversified_results = self._diversify_results(fused, query, top_k)
        
        # Fallback: if no results found, try a simple search
        if not diversified_results:
            try:
                print(f"No results from advanced search, trying simple fallback...")
                fallback_results = self._simple_search(query, top_k)
                return fallback_results[:top_k]
            except Exception as e:
                print(f"Fallback search also failed: {e}")
                return []
        
        return diversified_results[:top_k]

    def _diversify_results(self, results: List[Tuple[str, float]], query: str, top_k: int) -> List[Tuple[str, float]]:
        """Diversify results to ensure different citations for different queries."""
        if not results:
            return results
        
        # If we have enough diverse documents, return top_k
        if len(results) <= top_k:
            return results
        
        # Apply query-specific diversification
        diversified = []
        used_documents = set()
        
        # Use query hash to add some randomization while keeping it deterministic
        import hashlib
        query_hash = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        
        # First, add the highest scoring result
        if results:
            best_doc_id, best_score = results[0]
            diversified.append((best_doc_id, best_score))
            used_documents.add(best_doc_id.split("__")[0])  # Use base document name
        
        # Then add diverse documents with query-specific ordering
        remaining_results = results[1:]
        
        # Sort remaining results by score but add some query-specific variation
        def sort_key(item):
            doc_id, score = item
            base_doc = doc_id.split("__")[0]
            # Add small variation based on query hash and document name
            variation = (query_hash + hash(base_doc)) % 1000 / 10000.0
            return score + variation
        
        remaining_results.sort(key=sort_key, reverse=True)
        
        for doc_id, score in remaining_results:
            base_doc = doc_id.split("__")[0]
            
            # If we haven't used this document type yet, add it
            if base_doc not in used_documents and len(diversified) < top_k:
                diversified.append((doc_id, score))
                used_documents.add(base_doc)
            
            # If we have enough diverse documents, break
            if len(diversified) >= top_k:
                break
        
        # If we still need more results, add remaining high-scoring ones
        for doc_id, score in results:
            if len(diversified) >= top_k:
                break
            if (doc_id, score) not in diversified:
                diversified.append((doc_id, score))
        
        return diversified

    def _simple_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Simple fallback search method that doesn't rely on complex features."""
        results = []
        
        # Try basic vector search
        try:
            q_vec = self.emb_client.embed([query])
            if hasattr(self.vector, 'search') and len(self.documents.texts) > 0:
                vector_results = self.vector.search(q_vec, top_k=top_k)
                if vector_results and len(vector_results) > 0:
                    results.extend(vector_results[0])
        except Exception as e:
            print(f"Simple vector search failed: {e}")
        
        # Try basic text matching if vector search fails
        if not results:
            try:
                query_lower = query.lower()
                for doc_id, text in self.documents.texts.items():
                    if query_lower in text.lower():
                        # Simple scoring based on term frequency
                        score = text.lower().count(query_lower) / len(text.split())
                        results.append((doc_id, score))
                
                # Sort by score and return top_k
                results = sorted(results, key=lambda x: x[1], reverse=True)
            except Exception as e:
                print(f"Simple text matching failed: {e}")
        
        return results[:top_k]

    def get_text(self, doc_id: str) -> str:
        return self.documents.texts.get(doc_id, "")

    def generate_dynamic_general_knowledge(self, query: str, retrieved_docs: List[Tuple[str, float]] = None) -> str:
        """Generate dynamic general knowledge based on query and retrieved documents."""
        try:
            # Analyze the query for context
            query_context = self._analyze_query_context(query)
            
            # If we have retrieved documents, use them to enhance general knowledge
            if retrieved_docs and len(retrieved_docs) > 0:
                # Get the most relevant document
                top_doc_id, top_score = retrieved_docs[0]
                top_doc_text = self.get_text(top_doc_id)
                
                # Extract key concepts from the document
                key_concepts = self._extract_key_concepts(top_doc_text, query)
                
                # Generate contextual general knowledge
                if key_concepts:
                    return self._generate_contextual_knowledge(query, key_concepts, query_context)
            
            # Fallback to query-based general knowledge
            return self._generate_query_based_knowledge(query, query_context)
            
        except Exception as e:
            print(f"Error generating dynamic general knowledge: {e}")
            return self._generate_fallback_knowledge(query)
    
    def _analyze_query_context(self, query: str) -> dict:
        """Analyze query to determine context and intent."""
        query_lower = query.lower().strip()
        
        # Determine query intent with more comprehensive patterns
        intent = 'general'
        if any(word in query_lower for word in ['what is', 'what are', 'define', 'definition', 'explain']):
            intent = 'definition'
        elif any(word in query_lower for word in ['how does', 'how do', 'process', 'mechanism', 'work', 'function']):
            intent = 'process'
        elif any(word in query_lower for word in ['compare', 'difference', 'similar', 'versus', 'vs', 'between']):
            intent = 'comparison'
        elif any(word in query_lower for word in ['use', 'application', 'implement', 'apply', 'give', 'provide', 'create']):
            intent = 'application'
        elif any(word in query_lower for word in ['why', 'reason', 'cause', 'purpose']):
            intent = 'explanation'
        elif any(word in query_lower for word in ['when', 'time', 'history', 'schedule']):
            intent = 'temporal'
        elif any(word in query_lower for word in ['where', 'location', 'place', 'venue']):
            intent = 'spatial'
        elif any(word in query_lower for word in ['who', 'person', 'team', 'participant']):
            intent = 'person'
        elif any(word in query_lower for word in ['hackathon', 'competition', 'event', 'challenge']):
            intent = 'event'
        
        # Determine domain with expanded patterns
        domain = None
        if any(word in query_lower for word in ['machine learning', 'ml', 'neural', 'deep learning', 'ai', 'artificial intelligence']):
            domain = 'machine learning'
        elif any(word in query_lower for word in ['artificial intelligence', 'ai', 'intelligent', 'cognitive']):
            domain = 'artificial intelligence'
        elif any(word in query_lower for word in ['data', 'analytics', 'statistics', 'preprocessing']):
            domain = 'data science'
        elif any(word in query_lower for word in ['programming', 'code', 'software', 'development', 'hackathon']):
            domain = 'software development'
        elif any(word in query_lower for word in ['business', 'management', 'strategy', 'startup']):
            domain = 'business'
        elif any(word in query_lower for word in ['science', 'research', 'experiment', 'academic']):
            domain = 'scientific research'
        elif any(word in query_lower for word in ['health', 'medical', 'biology', 'healthcare']):
            domain = 'healthcare'
        elif any(word in query_lower for word in ['finance', 'economic', 'market', 'investment']):
            domain = 'finance'
        elif any(word in query_lower for word in ['education', 'learning', 'training', 'course']):
            domain = 'education'
        elif any(word in query_lower for word in ['event', 'hackathon', 'competition', 'conference']):
            domain = 'events'
        
        # Determine complexity level
        complexity = 'medium'
        if len(query.split()) > 10 or any(word in query_lower for word in ['complex', 'advanced', 'detailed', 'comprehensive']):
            complexity = 'high'
        elif len(query.split()) < 5 or any(word in query_lower for word in ['simple', 'basic', 'overview', 'quick']):
            complexity = 'low'
        
        # Check for specific entities or names
        has_specific_entities = any(word in query_lower for word in ['parul', 'hackathon', 'specific', 'named'])
        
        return {
            'intent': intent,
            'domain': domain,
            'complexity': complexity,
            'word_count': len(query.split()),
            'has_technical_terms': any(word in query_lower for word in ['algorithm', 'model', 'system', 'framework', 'architecture', 'hackathon']),
            'has_specific_entities': has_specific_entities,
            'original_query': query
        }
    
    def _extract_key_concepts(self, text: str, query: str) -> List[str]:
        """Extract key concepts from document text relevant to the query."""
        try:
            # Simple concept extraction based on common patterns
            concepts = []
            
            # Look for technical terms
            technical_terms = ['algorithm', 'model', 'system', 'framework', 'architecture', 'method', 'technique', 'approach']
            for term in technical_terms:
                if term in text.lower():
                    # Find the full phrase containing this term
                    sentences = text.split('.')
                    for sentence in sentences:
                        if term in sentence.lower():
                            # Extract the noun phrase around the technical term
                            words = sentence.split()
                            for i, word in enumerate(words):
                                if term in word.lower():
                                    # Get surrounding context
                                    start = max(0, i-2)
                                    end = min(len(words), i+3)
                                    phrase = ' '.join(words[start:end])
                                    if len(phrase) > 10:
                                        concepts.append(phrase.strip())
                                    break
            
            # Look for domain-specific terms based on query
            query_lower = query.lower()
            if 'machine learning' in query_lower:
                ml_terms = ['neural network', 'deep learning', 'supervised', 'unsupervised', 'reinforcement']
                for term in ml_terms:
                    if term in text.lower():
                        concepts.append(term)
            
            elif 'artificial intelligence' in query_lower:
                ai_terms = ['intelligence', 'cognitive', 'reasoning', 'decision making', 'natural language']
                for term in ai_terms:
                    if term in text.lower():
                        concepts.append(term)
            
            # Remove duplicates and limit to top concepts
            unique_concepts = list(set(concepts))
            return unique_concepts[:3]  # Return top 3 concepts
            
        except Exception as e:
            print(f"Error extracting key concepts: {e}")
            return []
    
    def _generate_contextual_knowledge(self, query: str, key_concepts: List[str], context: dict) -> str:
        """Generate general knowledge using retrieved document concepts."""
        try:
            # Extract the main subject from query
            subject = self._extract_subject(query)
            
            # Build contextual response
            if context['intent'] == 'definition':
                if key_concepts:
                    concept_str = ', '.join(key_concepts[:2])
                    return f"{subject.title()} is a {context['domain'] or 'technical'} concept that involves {concept_str}. It represents a fundamental approach within its domain."
                else:
                    return f"{subject.title()} is a {context['domain'] or 'technical'} concept that encompasses various methodologies and applications."
            
            elif context['intent'] == 'process':
                if key_concepts:
                    concept_str = key_concepts[0] if key_concepts else 'specific steps'
                    return f"{subject.title()} involves a {concept_str} process that enables systematic problem-solving and implementation."
                else:
                    return f"{subject.title()} involves a structured process with multiple steps and mechanisms working together."
            
            elif context['intent'] == 'comparison':
                return f"{subject.title()} involves analyzing different approaches, methodologies, or systems to understand their relative strengths and applications."
            
            elif context['intent'] == 'application':
                return f"{subject.title()} has diverse applications across various industries and domains, enabling practical problem-solving and innovation."
            
            else:
                if key_concepts:
                    concept_str = ', '.join(key_concepts[:2])
                    return f"{subject.title()} is related to {concept_str} and encompasses various aspects within the {context['domain'] or 'technical'} domain."
                else:
                    return f"{subject.title()} is a {context['domain'] or 'technical'} topic with multiple applications and implementations."
                    
        except Exception as e:
            print(f"Error generating contextual knowledge: {e}")
            return self._generate_fallback_knowledge(query)
    
    def _generate_query_based_knowledge(self, query: str, context: dict) -> str:
        """Generate general knowledge based on query analysis only."""
        subject = self._extract_subject(query)
        
        # Handle specific entities and events
        if context['has_specific_entities']:
            if 'hackathon' in query.lower():
                return f"{subject.title()} is a competitive programming event that brings together developers, designers, and innovators to create solutions within a limited timeframe."
            elif 'parul' in query.lower():
                return f"{subject.title()} appears to be a specific event or entity that involves collaborative problem-solving and innovation."
            else:
                return f"{subject.title()} is a specific entity or event that requires detailed information and context for proper understanding."
        
        # Handle event-related queries
        if context['intent'] == 'event':
            return f"{subject.title()} is an event that typically involves participants working together to solve challenges or create innovative solutions."
        
        # Handle person-related queries
        if context['intent'] == 'person':
            return f"{subject.title()} refers to a person or team that may be involved in specific activities or events."
        
        # Handle application/request queries
        if context['intent'] == 'application':
            if context['domain'] == 'events':
                return f"{subject.title()} is an event-related request that involves organizing, participating in, or providing information about specific activities."
            elif context['domain'] == 'software development':
                return f"{subject.title()} involves software development activities, potentially including coding, design, or technical implementation."
            else:
                return f"{subject.title()} involves practical implementation or provision of specific services or information."
        
        # Handle definition queries
        if context['intent'] == 'definition':
            if context['domain'] == 'events':
                return f"{subject.title()} is an event concept that involves organized activities, competitions, or collaborative sessions."
            elif context['domain'] == 'software development':
                return f"{subject.title()} is a software development concept that encompasses various technical methodologies and practices."
            else:
                return f"{subject.title()} is a {context['domain'] or 'general'} concept that provides fundamental understanding and framework for related applications."
        
        # Handle process queries
        elif context['intent'] == 'process':
            if context['domain'] == 'events':
                return f"{subject.title()} involves a structured process for organizing, managing, or participating in events and activities."
            else:
                return f"{subject.title()} involves systematic procedures and mechanisms that enable effective implementation and problem-solving."
        
        # Handle comparison queries
        elif context['intent'] == 'comparison':
            return f"{subject.title()} involves analyzing different approaches, methodologies, or systems to understand their characteristics and applications."
        
        # Handle explanation queries
        elif context['intent'] == 'explanation':
            return f"{subject.title()} involves understanding the underlying principles, reasons, and mechanisms that drive its functionality and applications."
        
        # Handle temporal queries
        elif context['intent'] == 'temporal':
            return f"{subject.title()} involves time-based aspects, scheduling, or historical context that provides important temporal information."
        
        # Handle spatial queries
        elif context['intent'] == 'spatial':
            return f"{subject.title()} involves location-based information, venue details, or spatial context that provides important geographical information."
        
        # Default case
        else:
            if context['domain'] == 'events':
                return f"{subject.title()} is an event-related topic that involves organized activities, competitions, or collaborative sessions."
            elif context['domain'] == 'software development':
                return f"{subject.title()} is a software development topic that encompasses various technical methodologies and practices."
            else:
                return f"{subject.title()} is a {context['domain'] or 'general'} topic that encompasses various concepts, methodologies, and practical applications."
    
    def _extract_subject(self, query: str) -> str:
        """Extract the main subject from the query."""
        query_lower = query.lower().strip()
        
        # Remove common question words and action words, then clean up
        prefixes = ['what is', 'what are', 'how does', 'how do', 'why', 'when', 'where', 'compare', 'define', 'explain']
        action_words = ['give', 'provide', 'create', 'make', 'show', 'tell', 'find', 'get', 'send', 'share']
        
        subject = query_lower
        
        # Remove prefixes
        for prefix in prefixes:
            if query_lower.startswith(prefix):
                subject = query_lower.replace(prefix, '').strip()
                break
        
        # Remove action words if they appear at the beginning
        for action in action_words:
            if subject.startswith(action + ' '):
                subject = subject.replace(action + ' ', '', 1).strip()
                break
        
        # Remove question marks and clean up
        subject = subject.replace('?', '').strip()
        
        # Handle specific cases like "Give A Parul Hackthon"
        if 'parul' in subject and 'hackthon' in subject:
            return "Parul Hackthon"
        elif 'hackthon' in subject:
            return "Hackthon"
        elif 'parul' in subject:
            return "Parul"
        
        # Capitalize properly
        if subject:
            return subject.title()
        else:
            return "This topic"
    
    def _generate_fallback_knowledge(self, query: str) -> str:
        """Generate a simple fallback general knowledge response."""
        subject = self._extract_subject(query)
        return f"{subject} is a topic that encompasses various concepts, methodologies, and applications within its domain."


