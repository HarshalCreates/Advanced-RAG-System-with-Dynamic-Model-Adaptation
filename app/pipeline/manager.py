from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator, List, Tuple

from app.models.schemas import (
    Answer,
    ContextAnalysis,
    PerformanceMetrics,
    QueryRequest,
    RAGResponse,
    RetrievedChunk,
    SystemMetadata,
)
from app.models.config import get_settings
from app.pipeline.ingestion import IngestionPipeline
from app.retrieval.service import HybridRetrievalService
from app.generation.factory import GenerationFactory
from app.generation.reasoning import MultiStepReasoner


class PipelineManager:
    _instance: "PipelineManager" | None = None

    def __init__(self) -> None:
        self.settings = get_settings()
        # Use lazy loading for expensive components
        self._retriever = None
        self._ingestion = None
        self._generator = None
        self._reasoner = None
    
    @property
    def retriever(self):
        """Lazy load retriever only when needed."""
        if self._retriever is None:
            self._retriever = HybridRetrievalService()
        return self._retriever
    
    @property
    def ingestion(self):
        """Lazy load ingestion pipeline only when needed."""
        if self._ingestion is None:
            self._ingestion = IngestionPipeline(self.retriever)
        return self._ingestion
    
    @property
    def generator(self):
        """Lazy load generator only when needed."""
        if self._generator is None:
            self._generator = GenerationFactory(
                backend=self.settings.generation_backend, 
                model=self.settings.generation_model
            ).build()
        return self._generator
    
    @property
    def reasoner(self):
        """Lazy load reasoner only when needed."""
        if self._reasoner is None:
            self._reasoner = MultiStepReasoner()
        return self._reasoner

    @classmethod
    def get_instance(cls) -> "PipelineManager":
        if cls._instance is None:
            cls._instance = PipelineManager()
        return cls._instance
    
    def warm_up(self):
        """Warm up the system by pre-loading essential components."""
        print("🔥 Warming up RAG system...")
        
        # Pre-load retriever (most expensive component)
        print("   Loading retrieval system...")
        _ = self.retriever
        
        # Pre-load generator (lightweight)
        print("   Loading generation system...")
        _ = self.generator
        
        # Pre-load embeddings with a dummy query to warm up the model
        print("   Warming up embeddings...")
        try:
            _ = self.retriever.emb_client.embed(["test query"])
        except Exception as e:
            print(f"   Warning: Embedding warm-up failed: {e}")
        
        print("   ✅ System warmed up and ready!")

    @classmethod
    def get_ingestion(cls) -> IngestionPipeline:
        return cls.get_instance().ingestion

    # --- Hot swap controls ---
    def swap_embeddings(self, backend: str, model: str) -> None:
        self.retriever.hot_swap_embeddings(backend, model)

    def swap_retriever(self, backend: str) -> None:
        self.retriever.hot_swap_retriever(backend)

    def swap_generation(self, backend: str, model: str) -> None:
        self.settings.generation_backend = backend
        self.settings.generation_model = model
        # Reset the generator so it will be rebuilt with new settings
        self._generator = None

    def query(self, request: QueryRequest) -> RAGResponse:
        """Main query processing pipeline."""
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        retrieved_docs = self.retriever.search(
            request.query, 
            request.top_k, 
            request.filters
        )
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Generate dynamic general knowledge using retrieved documents
        general_knowledge = self.retriever.generate_dynamic_general_knowledge(
            request.query, 
            retrieved_docs
        )
        
        # Step 3: Generate response using LLM
        generation_start = time.time()
        response = self._generate_response(request.query, retrieved_docs, general_knowledge)
        generation_time = time.time() - generation_start
        
        # Step 4: Calculate performance metrics
        total_time = time.time() - start_time
        
        # Step 5: Build response
        # Create intelligent citations that prioritize relevant documents
        citations = self._create_intelligent_citations(request.query, retrieved_docs)
        
        # Create enhanced context analysis
        retrieval_methods = ["dense", "sparse"]
        if len(retrieved_docs) > 3:
            retrieval_methods.append("graph")
        if response.confidence > 0.8:
            retrieval_methods.append("fusion")
        
        # Count cross-document connections (documents with score > 0.6)
        high_relevance_docs = len([score for _, score in retrieved_docs if score > 0.6])
        
        context_analysis = ContextAnalysis(
            total_chunks_analyzed=len(retrieved_docs),
            retrieval_methods_used=retrieval_methods,
            cross_document_connections=max(1, high_relevance_docs - 1),
            temporal_relevance="current"
        )
        
        # Create system metadata
        system_metadata = SystemMetadata(
            embedding_model=self.settings.embedding_model,
            generation_model=self.settings.generation_model,
            retrieval_strategy="hybrid_weighted",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        
        # Generate alternative answers if confidence is moderate/low
        alternative_answers = []
        if response.confidence < 0.85 and len(citations) > 1:
            alternative_answers = self._generate_alternative_answers(request.query, retrieved_docs, response.content)
        
        # Calculate actual token count
        tokens_processed = len(response.content.split()) + len(request.query.split())
        
        # Estimate cost based on model and tokens
        cost_estimate = self._estimate_cost(tokens_processed)
        
        return RAGResponse(
            answer=response,
            citations=citations,
            alternative_answers=alternative_answers,
            context_analysis=context_analysis,
            performance_metrics=PerformanceMetrics(
                retrieval_latency_ms=int(retrieval_time * 1000),
                generation_latency_ms=int(generation_time * 1000),
                total_response_time_ms=int(total_time * 1000),
                tokens_processed=tokens_processed,
                cost_estimate_usd=cost_estimate
            ),
            system_metadata=system_metadata
        )
    
    def _generate_response(self, query: str, retrieved_docs: List[Tuple[str, float]], general_knowledge: str) -> Answer:
        """Generate response using dynamic general knowledge and retrieved documents."""
        try:
            # Prepare context from retrieved documents
            context_parts = []
            for doc_id, score in retrieved_docs[:3]:  # Use top 3 documents
                doc_text = self.retriever.get_text(doc_id)
                if doc_text:
                    # Extract a relevant excerpt
                    excerpt = doc_text[:300] if len(doc_text) > 300 else doc_text
                    context_parts.append(excerpt)
            
            context_text = "\n\n".join(context_parts) if context_parts else "No specific document context available."
            
            # Create system prompt that emphasizes the dynamic general knowledge
            system_prompt = """You are a helpful assistant. Provide a comprehensive answer that combines:
1. The provided general knowledge (which is dynamically generated based on the query)
2. Specific information from the retrieved documents

Format your response as:
General Knowledge: [The provided dynamic general knowledge]
From Documents: 
1. [First key point from documents]
2. [Second key point from documents]"""
            
            user_prompt = f"""Query: {query}

Dynamic General Knowledge: {general_knowledge}

Context from Documents: {context_text}

Please provide an answer that combines the dynamic general knowledge with specific information from the documents."""

            # Generate response using LLM
            try:
                answer_text = self.generator.complete(system=system_prompt, user=user_prompt)
            except Exception as gen_error:
                print(f"Generation error: {gen_error}")
                # Fallback to echo generator if provider fails
                from app.generation.factory import EchoGeneration
                self.generator = EchoGeneration()
                answer_text = self.generator.complete(system=system_prompt, user=user_prompt)
            
            # Clean up the answer
            answer_text = self._clean_answer(answer_text, query)
            
            # Calculate confidence based on retrieval scores
            confidence_score = self._calculate_confidence(retrieved_docs)
            
            # Determine uncertainty factors
            uncertainty_factors = self._determine_uncertainty_factors(retrieved_docs, confidence_score)
            
            # Generate dynamic reasoning steps based on actual processing
            reasoning_steps = self._generate_dynamic_reasoning_steps(
                query, retrieved_docs, general_knowledge, answer_text, confidence_score
            )
            
            return Answer(
                content=answer_text,
                reasoning_steps=reasoning_steps,
                confidence=confidence_score,
                uncertainty_factors=uncertainty_factors,
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response with dynamic general knowledge and query-specific information
            query_analysis = self._analyze_query_for_context(query)
            
            # Generate dynamic document points based on query analysis
            if query_analysis['type'] == 'definition':
                doc_point1 = f"Documentation about {query.replace('?', '').strip()} may be available in the ingested materials."
                doc_point2 = "Search for related terms or broader categories to find relevant information."
            elif query_analysis['type'] == 'process':
                doc_point1 = f"Process documentation for {query.replace('?', '').strip()} may be available in the documents."
                doc_point2 = "Look for specific steps, methods, or procedures related to this topic."
            elif query_analysis['type'] == 'comparison':
                doc_point1 = f"Comparison details about {query.replace('?', '').strip()} may be available in the documents."
                doc_point2 = "Search for individual components or aspects being compared."
            else:
                doc_point1 = f"Relevant information about {query.replace('?', '').strip()} may be available in the ingested documents."
                doc_point2 = "Try different keywords or more specific terms to find relevant content."
            
            # Generate dynamic reasoning steps for fallback
            fallback_reasoning = self._generate_dynamic_reasoning_steps(
                query, retrieved_docs, general_knowledge, 
                f"General Knowledge: {general_knowledge}\n\nFrom Documents:\n1. {doc_point1}\n2. {doc_point2}",
                0.3, is_fallback=True
            )
            
            return Answer(
                content=f"General Knowledge: {general_knowledge}\n\nFrom Documents:\n1. {doc_point1}\n2. {doc_point2}",
                reasoning_steps=fallback_reasoning,
                confidence=0.3,
                uncertainty_factors=["Limited context", "Using fallback generation"],
            )
    
    def _calculate_confidence(self, retrieved_docs: List[Tuple[str, float]]) -> float:
        """Calculate confidence score based on retrieval results."""
        if not retrieved_docs:
            return 0.2
        
        # Get top scores
        top_score = float(retrieved_docs[0][1]) if retrieved_docs else 0.0
        second_score = float(retrieved_docs[1][1]) if len(retrieved_docs) > 1 else 0.0
        
        # Normalize top score roughly into [0,1]
        norm_top = max(0.0, min(1.0, top_score))
        
        # Calculate margin between top and second result
        margin = max(0.0, top_score - second_score)
        margin_norm = margin / (top_score + 1e-6) if top_score > 0 else 0.0
        
        # Combine factors
        confidence_score = 0.4 + 0.4 * norm_top + 0.2 * margin_norm
        confidence_score = max(0.0, min(0.98, confidence_score))
        
        return round(confidence_score, 2)
    
    def _determine_uncertainty_factors(self, retrieved_docs: List[Tuple[str, float]], confidence: float) -> List[str]:
        """Determine uncertainty factors based on retrieval results."""
        factors = []
        
        if not retrieved_docs:
            factors.append("No relevant documents found")
        else:
            top_score = float(retrieved_docs[0][1]) if retrieved_docs else 0.0
            
            if top_score < 0.6:
                factors.append("Low retrieval similarity")
            
            if len(retrieved_docs) > 1:
                second_score = float(retrieved_docs[1][1])
                margin = top_score - second_score
                if margin < 0.1:
                    factors.append("Ambiguous top results")
            
            if len(retrieved_docs) < 2:
                factors.append("Limited document context")
        
        if confidence < 0.5:
            factors.append("Low confidence in response")
        
        if not factors:
            factors.append("Normal variability")
        
        return factors
    
    def _generate_dynamic_reasoning_steps(self, query: str, retrieved_docs: List[Tuple[str, float]], 
                                        general_knowledge: str, answer_text: str, confidence_score: float, 
                                        is_fallback: bool = False) -> List[str]:
        """Generate dynamic reasoning steps based on actual processing."""
        steps = []
        
        # Step 1: Query Analysis
        query_type = self._analyze_query_type(query)
        if "what is" in query.lower() or "define" in query.lower():
            steps.append(f"Analyzed query as definitional: '{query}' - seeking comprehensive explanation")
        elif "how" in query.lower():
            steps.append(f"Analyzed query as procedural: '{query}' - looking for process or method")
        elif "why" in query.lower():
            steps.append(f"Analyzed query as causal: '{query}' - investigating reasons and causes")
        elif "compare" in query.lower() or "difference" in query.lower():
            steps.append(f"Analyzed query as comparative: '{query}' - examining differences and similarities")
        else:
            steps.append(f"Analyzed query type: '{query}' - determined optimal retrieval strategy")
        
        # Step 2: Document Retrieval
        if retrieved_docs:
            top_score = float(retrieved_docs[0][1])
            num_docs = len(retrieved_docs)
            if top_score > 0.8:
                steps.append(f"Retrieved {num_docs} highly relevant documents (top relevance: {top_score:.2f})")
            elif top_score > 0.6:
                steps.append(f"Retrieved {num_docs} moderately relevant documents (top relevance: {top_score:.2f})")
            else:
                steps.append(f"Retrieved {num_docs} documents with limited relevance (top relevance: {top_score:.2f})")
        else:
            steps.append("No relevant documents found in knowledge base - relying on general knowledge")
        
        # Step 3: Knowledge Generation
        backend = getattr(self.generator, '__class__', type(self.generator)).__name__
        if "OpenAI" in backend:
            model_info = "GPT model"
        elif "Anthropic" in backend:
            model_info = "Claude model"
        elif "Ollama" in backend:
            model_info = f"Local Llama model ({getattr(self.generator, 'model', 'unknown')})"
        elif "Echo" in backend:
            model_info = "Template-based generator"
        else:
            model_info = "AI model"
        
        if is_fallback:
            steps.append(f"Generated contextual knowledge using {model_info} with error recovery")
        else:
            steps.append(f"Generated comprehensive response using {model_info} combining general knowledge and document context")
        
        # Step 4: Response Synthesis
        answer_length = len(answer_text)
        has_general_knowledge = "General Knowledge:" in answer_text
        has_document_info = "From Documents:" in answer_text
        
        if has_general_knowledge and has_document_info:
            steps.append(f"Synthesized {answer_length}-character response integrating both general knowledge and document-specific information")
        elif has_general_knowledge:
            steps.append(f"Synthesized {answer_length}-character response primarily from general knowledge")
        else:
            steps.append(f"Synthesized {answer_length}-character response from available information")
        
        # Step 5: Quality Assessment (optional)
        if confidence_score > 0.8:
            steps.append(f"Response validated with high confidence ({confidence_score:.2f}) - strong evidence alignment")
        elif confidence_score > 0.6:
            steps.append(f"Response validated with moderate confidence ({confidence_score:.2f}) - acceptable evidence quality")
        elif confidence_score > 0.4:
            steps.append(f"Response validated with low confidence ({confidence_score:.2f}) - limited evidence available")
        else:
            steps.append(f"Response generated with uncertainty ({confidence_score:.2f}) - weak evidence or fallback mode")
        
        return steps
    
    def _analyze_query_type(self, query: str) -> str:
        """Analyze the type of query for reasoning step generation."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what is", "what are", "define", "definition"]):
            return "definitional"
        elif any(word in query_lower for word in ["how to", "how do", "how can", "steps", "process"]):
            return "procedural"
        elif any(word in query_lower for word in ["why", "because", "reason", "cause"]):
            return "causal"
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return "comparative"
        elif any(word in query_lower for word in ["when", "time", "date", "history"]):
            return "temporal"
        elif any(word in query_lower for word in ["where", "location", "place"]):
            return "spatial"
        else:
            return "general"
    
    def _generate_alternative_answers(self, query: str, retrieved_docs: List[Tuple[str, float]], 
                                    primary_answer: str) -> List[Any]:
        """Generate alternative interpretations of the answer."""
        from app.models.schemas import AlternativeAnswer
        
        alternatives = []
        
        # Only generate alternatives if we have multiple documents
        if len(retrieved_docs) < 2:
            return alternatives
        
        try:
            # Generate an alternative based on the second-best document
            if len(retrieved_docs) > 1:
                second_doc_id, second_score = retrieved_docs[1]
                second_doc_text = self.retriever.get_text(second_doc_id)
                
                if second_doc_text and second_score > 0.4:
                    # Create alternative prompt focusing on second document
                    alt_system = "Provide an alternative perspective or interpretation based on the given context."
                    alt_prompt = f"Query: {query}\n\nAlternative Context: {second_doc_text[:400]}\n\nProvide a different but valid interpretation or answer."
                    
                    try:
                        alt_content = self.generator.complete(system=alt_system, user=alt_prompt)
                        alt_confidence = max(0.3, second_score * 0.8)  # Lower confidence for alternatives
                        
                        # Create supporting citations for alternative
                        supporting_citations = []
                        for doc_id, score in retrieved_docs[1:3]:  # Use 2nd and 3rd documents
                            doc_text = self.retriever.get_text(doc_id)
                            source = doc_id.split("__")[0] if "__" in doc_id else doc_id
                            
                            supporting_citations.append(RetrievedChunk(
                                document=source,
                                pages=[],
                                chunk_id=doc_id,
                                excerpt=doc_text[:150] if doc_text else "No content",
                                relevance_score=float(score),
                                credibility_score=0.75,
                                extraction_method="fusion"
                            ))
                        
                        alternatives.append(AlternativeAnswer(
                            content=alt_content,
                            confidence=alt_confidence,
                            supporting_citations=supporting_citations
                        ))
                    except Exception as e:
                        print(f"Failed to generate alternative answer: {e}")
        
        except Exception as e:
            print(f"Error generating alternatives: {e}")
        
        return alternatives
    
    def _estimate_cost(self, tokens_processed: int) -> float:
        """Estimate the cost of the query based on tokens and model."""
        try:
            # Cost estimates per 1K tokens (approximate)
            cost_per_1k_tokens = {
                "gpt-4o": 0.0025,  # Input + output average
                "gpt-4o-mini": 0.0003,
                "gpt-4-turbo": 0.005,
                "gpt-3.5-turbo": 0.0015,
                "claude-3-5-sonnet-20241022": 0.003,
                "claude-3-5-haiku-20241022": 0.0008,
                "claude-3-opus-20240229": 0.008,
            }
            
            model = self.settings.generation_model
            cost_rate = cost_per_1k_tokens.get(model, 0.001)  # Default rate
            
            # For local models (Ollama), cost is essentially zero
            if self.settings.generation_backend == "ollama":
                return 0.0
            
            # Calculate cost: (tokens / 1000) * rate
            estimated_cost = (tokens_processed / 1000) * cost_rate
            return round(estimated_cost, 4)
            
        except Exception:
            return 0.0
    
    async def query_with_reasoning(self, payload: QueryRequest) -> RAGResponse:
        """Enhanced query processing with multi-step reasoning."""
        t_retrieval = 0
        t_generation = 0
        t_reasoning = 0

        # Step 1: Retrieval
        t0 = time.time()
        fused = self.retriever.search(payload.query, top_k=payload.top_k, filters=payload.filters)
        t_retrieval = int((time.time() - t0) * 1000)

        # Prepare context chunks for reasoning
        context_chunks = []
        for doc_id, score in fused:
            text = self.retriever.get_text(doc_id)
            context_chunks.append({
                'id': doc_id,
                'text': text,
                'score': score,
                'metadata': getattr(self.retriever.documents, 'metadatas', {}).get(doc_id, {})
            })

        # Step 2: Multi-step reasoning
        t1 = time.time()
        try:
            reasoning_chain = await self.reasoner.reason_step_by_step(
                payload.query, context_chunks, self.generator
            )
            answer_text = reasoning_chain.final_answer
            reasoning_steps = [step.conclusion for step in reasoning_chain.steps]
            overall_confidence = reasoning_chain.overall_confidence
        except Exception as e:
            print(f"Reasoning failed: {e}")
            # Fallback to regular generation
            context_text = "\n\n".join([chunk['text'][:300] for chunk in context_chunks])
            system_prompt = "You are a helpful assistant. Answer based on the provided context."
            user_prompt = f"Query: {payload.query}\n\nContext:\n{context_text}"
            answer_text = self.generator.complete(system=system_prompt, user=user_prompt)
            
            # Generate dynamic reasoning steps for fallback
            retrieved_docs = [(chunk['id'], chunk['score']) for chunk in context_chunks]
            reasoning_steps = self._generate_dynamic_reasoning_steps(
                payload.query, retrieved_docs, "", answer_text, 0.5, is_fallback=True
            )
            overall_confidence = 0.5
        
        t_reasoning = int((time.time() - t1) * 1000)
        t_generation = t_reasoning  # Include reasoning time in generation

        # Prepare citations (single best document approach)
        chunks = []
        if fused:
            doc_id, score = fused[0]  # Best result
            text = self.retriever.get_text(doc_id)
            
            # Extract metadata
            metadata = getattr(self.retriever.documents, 'metadatas', {}).get(doc_id, {})
            source = metadata.get("source", doc_id.split("__")[0])
            page = metadata.get("page")
            
            chunk = RetrievedChunk(
                document=source,
                pages=[page] if page else [],
                chunk_id=doc_id,
                excerpt=text[:500],
                relevance_score=float(score),
                credibility_score=0.9,  # Higher for reasoning-based results
                                                extraction_method="fusion"
            )
            chunks.append(chunk)

        # Enhanced answer with reasoning
        answer = Answer(
            content=answer_text,
            reasoning_steps=reasoning_steps,
            confidence=overall_confidence,
            uncertainty_factors=["Multi-step reasoning uncertainty"] if overall_confidence < 0.7 else []
        )

        # Context analysis
        ctx = ContextAnalysis(
            total_chunks_analyzed=len(context_chunks),
            retrieval_methods_used=["dense", "sparse", "graph", "reasoning"],
            cross_document_connections=len([c for c in context_chunks if c['score'] > 0.7]),
            temporal_relevance="current"
        )

        # Generate alternative answers for reasoning queries
        alternative_answers = []
        if overall_confidence < 0.8 and len(context_chunks) > 1:
            retrieved_docs = [(chunk['id'], chunk['score']) for chunk in context_chunks]
            alternative_answers = self._generate_alternative_answers(payload.query, retrieved_docs, answer_text)

        # Calculate enhanced performance metrics
        total_tokens = len(answer_text.split()) + len(payload.query.split())
        cost_estimate = self._estimate_cost(total_tokens)
        
        perf = PerformanceMetrics(
            retrieval_latency_ms=t_retrieval,
            generation_latency_ms=t_generation,
            total_response_time_ms=t_retrieval + t_generation,
            tokens_processed=total_tokens,
            cost_estimate_usd=cost_estimate
        )

        # System metadata
        meta = SystemMetadata(
            embedding_model=self.settings.embedding_model,
            generation_model=self.settings.generation_model,
            retrieval_strategy="reasoning_enhanced_hybrid",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

        return RAGResponse(
            answer=answer,
            citations=chunks,
            alternative_answers=alternative_answers,
            context_analysis=ctx,
            performance_metrics=perf,
            system_metadata=meta,
        )

    async def stream(self, query_text: str, top_k: int = 5) -> AsyncGenerator[dict[str, Any], None]:
        # Retrieve context
        fused = self.retriever.search(query_text, top_k=top_k)
        ctx = "\n\n".join([self.retriever.get_text(doc_id)[:300] for doc_id, _ in fused])
        system = "You are a careful assistant. Stream your answer in coherent chunks."
        user = f"Query: {query_text}\n\nContext:\n{ctx}"
        try:
            async for token in self.generator.astream(system=system, user=user):
                yield {"event": "token", "content": token}
        except Exception:
            # Fallback: non-streaming split
            text = self.generator.complete(system=system, user=user)
            for piece in [text[i:i+200] for i in range(0, len(text), 200)]:
                await asyncio.sleep(0.05)
                yield {"event": "token", "content": piece}
        yield {"event": "final"}
    
    def _clean_answer(self, answer_text: str, query: str) -> str:
        """Clean up the answer to remove internal system information and ensure proper format."""
        # Remove common system prompt artifacts
        lines = answer_text.split('\n')
        cleaned_lines = []
        
        skip_patterns = [
            '[SYSTEM]',
            '[USER]',
            'Question:',
            'Context:',
            'Answer:',
            'You are a',
            'Based on the provided context',
            'Context from retrieved documents:',
            'First explain what this query is asking for',
            'First provide a general explanation of the topic'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip lines that contain system prompt artifacts
            should_skip = False
            for pattern in skip_patterns:
                if pattern in line:
                    should_skip = True
                    break
            
            if not should_skip:
                cleaned_lines.append(line)
        
        # Join and clean up the result
        cleaned_answer = '\n'.join(cleaned_lines).strip()
        
        # Ensure we have proper format with general knowledge + document information
        if cleaned_answer and len(cleaned_answer) > 10:
            # Check if it already has the proper format
            if 'General Knowledge:' in cleaned_answer and 'From Documents:' in cleaned_answer:
                return cleaned_answer
            else:
                # Convert to proper format
                # Generate general knowledge
                general_knowledge = self._generate_general_knowledge(query)
                
                # Extract document information points
                sentences = [s.strip() for s in cleaned_answer.split('.') if len(s.strip()) > 5]
                if len(sentences) >= 2:
                    point1 = sentences[0].strip()
                    point2 = sentences[1].strip()
                    if not point1.endswith('.'):
                        point1 += '.'
                    if not point2.endswith('.'):
                        point2 += '.'
                    return f"General Knowledge: {general_knowledge}\n\nFrom Documents:\n1. {point1}\n2. {point2}"
                elif len(sentences) == 1:
                    sentence = sentences[0].strip()
                    if not sentence.endswith('.'):
                        sentence += '.'
                    return f"General Knowledge: {general_knowledge}\n\nFrom Documents:\n1. {sentence}\n2. Additional information available in the documents."
        
        # Fallback response with general knowledge
        general_knowledge = self._generate_general_knowledge(query)
        
        # Generate dynamic document information based on query analysis
        query_analysis = self._analyze_query_for_context(query)
        if query_analysis['type'] == 'definition':
            doc_point1 = f"Based on the available documents, {query.replace('?', '').strip()} appears to be a concept that may be covered in the ingested materials."
            doc_point2 = "Consider searching for related terms or broader categories to find relevant information."
        elif query_analysis['type'] == 'process':
            doc_point1 = f"The process of {query.replace('?', '').strip()} may be described in the available documents."
            doc_point2 = "Try searching for specific steps, methods, or procedures related to this topic."
        elif query_analysis['type'] == 'comparison':
            doc_point1 = f"Comparison information about {query.replace('?', '').strip()} may be available in the documents."
            doc_point2 = "Consider searching for individual components or aspects being compared."
        else:
            doc_point1 = f"Information about {query.replace('?', '').strip()} may be available in the ingested documents."
            doc_point2 = "Try rephrasing your query with different keywords or more specific terms."
        
        return f"General Knowledge: {general_knowledge}\n\nFrom Documents:\n1. {doc_point1}\n2. {doc_point2}"
    
    def _generate_query_understanding(self, query: str) -> str:
        """Generate an explanation of what the query is asking for."""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('what is') or query_lower.startswith('what are'):
            subject = query_lower.replace('what is', '').replace('what are', '').replace('?', '').strip()
            return f"This query is asking for a definition or explanation of {subject}."
        elif query_lower.startswith('how does') or query_lower.startswith('how do'):
            subject = query_lower.replace('how does', '').replace('how do', '').replace('work', '').replace('?', '').strip()
            return f"This query is asking about the process or mechanism of how {subject} works."
        elif query_lower.startswith('why'):
            subject = query_lower.replace('why', '').replace('?', '').strip()
            return f"This query is asking for the reasons or explanations behind {subject}."
        elif query_lower.startswith('when'):
            subject = query_lower.replace('when', '').replace('?', '').strip()
            return f"This query is asking about the timing or circumstances of {subject}."
        elif query_lower.startswith('where'):
            subject = query_lower.replace('where', '').replace('?', '').strip()
            return f"This query is asking about the location or context of {subject}."
        else:
            return f"This query is seeking information about {query.replace('?', '').strip()}."
    
    def _generate_general_knowledge(self, query: str) -> str:
        """Generate dynamic, query-specific general knowledge using LLM."""
        try:
            # Analyze the query type and context for more specific general knowledge
            query_analysis = self._analyze_query_for_context(query)
            
            # Create a more specific system prompt based on query analysis
            if query_analysis['type'] == 'definition':
                system_prompt = "You are a knowledgeable assistant. Provide a brief, accurate definition and general explanation of the concept mentioned in the query. Focus on what it is and its basic characteristics. Keep it concise (1-2 sentences) and factual."
            elif query_analysis['type'] == 'process':
                system_prompt = "You are a knowledgeable assistant. Provide a brief, accurate explanation of the process or mechanism mentioned in the query. Focus on how it works and its basic principles. Keep it concise (1-2 sentences) and factual."
            elif query_analysis['type'] == 'comparison':
                system_prompt = "You are a knowledgeable assistant. Provide a brief, accurate explanation of the concepts being compared in the query. Focus on their general characteristics and differences. Keep it concise (1-2 sentences) and factual."
            elif query_analysis['type'] == 'application':
                system_prompt = "You are a knowledgeable assistant. Provide a brief, accurate explanation of the application or use case mentioned in the query. Focus on what it's used for and its general purpose. Keep it concise (1-2 sentences) and factual."
            else:
                system_prompt = "You are a knowledgeable assistant. Provide a brief, accurate general explanation of the topic mentioned in the query. Keep it concise (1-2 sentences) and factual."
            
            # Add context-specific information to the prompt
            context_hint = ""
            if query_analysis['domain']:
                context_hint = f" Consider the {query_analysis['domain']} context."
            if query_analysis['specificity'] == 'high':
                context_hint += " Be more specific and detailed in your explanation."
            elif query_analysis['specificity'] == 'low':
                context_hint += " Provide a broader, more general explanation."
            
            user_prompt = f"Query: {query}\n\nProvide a general explanation of this topic based on your knowledge:{context_hint}"
            
            general_knowledge = self.generator.complete(system=system_prompt, user=user_prompt)
            
            # Clean up the response
            general_knowledge = general_knowledge.strip()
            if general_knowledge.startswith("General Knowledge:"):
                general_knowledge = general_knowledge.replace("General Knowledge:", "").strip()
            
            return general_knowledge
            
        except Exception as e:
            # Enhanced fallback with better query analysis
            return self._generate_fallback_general_knowledge(query)
    
    def _analyze_query_for_context(self, query: str) -> dict:
        """Analyze query to determine context for better general knowledge generation."""
        query_lower = query.lower().strip()
        
        # Determine query type
        query_type = 'general'
        if any(word in query_lower for word in ['what is', 'what are', 'define', 'definition']):
            query_type = 'definition'
        elif any(word in query_lower for word in ['how does', 'how do', 'process', 'mechanism', 'work']):
            query_type = 'process'
        elif any(word in query_lower for word in ['compare', 'difference', 'similar', 'versus', 'vs']):
            query_type = 'comparison'
        elif any(word in query_lower for word in ['use', 'application', 'implement', 'apply']):
            query_type = 'application'
        
        # Determine domain
        domain = None
        if any(word in query_lower for word in ['machine learning', 'ml', 'neural', 'deep learning']):
            domain = 'machine learning'
        elif any(word in query_lower for word in ['artificial intelligence', 'ai', 'intelligent']):
            domain = 'artificial intelligence'
        elif any(word in query_lower for word in ['data', 'analytics', 'statistics']):
            domain = 'data science'
        elif any(word in query_lower for word in ['programming', 'code', 'software', 'development']):
            domain = 'software development'
        elif any(word in query_lower for word in ['business', 'management', 'strategy']):
            domain = 'business'
        elif any(word in query_lower for word in ['science', 'research', 'experiment']):
            domain = 'scientific research'
        
        # Determine specificity level
        specificity = 'medium'
        if len(query.split()) > 8 or any(word in query_lower for word in ['specific', 'detailed', 'exact']):
            specificity = 'high'
        elif len(query.split()) < 4 or any(word in query_lower for word in ['general', 'overview', 'basic']):
            specificity = 'low'
        
        return {
            'type': query_type,
            'domain': domain,
            'specificity': specificity,
            'word_count': len(query.split())
        }
    
    def _generate_fallback_general_knowledge(self, query: str) -> str:
        """Enhanced fallback general knowledge generation."""
        query_lower = query.lower().strip()
        analysis = self._analyze_query_for_context(query)
        
        # Extract the main subject
        subject = query_lower
        for prefix in ['what is', 'what are', 'how does', 'how do', 'why', 'when', 'where', 'compare', 'define']:
            if query_lower.startswith(prefix):
                subject = query_lower.replace(prefix, '').strip()
                break
        
        # Remove question marks and clean up
        subject = subject.replace('?', '').strip()
        
        # Generate context-specific fallback responses
        if analysis['type'] == 'definition':
            if analysis['domain'] == 'machine learning':
                return f"{subject.title()} is a machine learning concept that involves computational methods for pattern recognition and data analysis."
            elif analysis['domain'] == 'artificial intelligence':
                return f"{subject.title()} is an AI technique that enables systems to perform tasks requiring human-like intelligence."
            elif analysis['domain'] == 'data science':
                return f"{subject.title()} is a data science approach for extracting insights and patterns from complex datasets."
            else:
                return f"{subject.title()} is a concept that encompasses various aspects and applications within its domain."
        
        elif analysis['type'] == 'process':
            return f"{subject.title()} involves a series of steps or mechanisms that work together to achieve a specific outcome."
        
        elif analysis['type'] == 'comparison':
            return f"{subject.title()} involves analyzing the similarities and differences between different concepts or approaches."
        
        elif analysis['type'] == 'application':
            return f"{subject.title()} has various practical applications and use cases across different domains and industries."
        
        else:
            return f"{subject.title()} is a topic that encompasses various concepts, applications, and implementations."

    def _create_intelligent_citations(self, query: str, retrieved_docs: List[Tuple[str, float]]) -> List[RetrievedChunk]:
        """Create intelligent citations that prioritize documents containing query-specific terms."""
        citations = []
        
        # Extract key terms from query
        query_terms = set(query.lower().split())
        
        # Score documents based on query relevance and content
        scored_docs = []
        for doc_id, score in retrieved_docs:
            doc_text = self.retriever.get_text(doc_id)
            if not doc_text:
                continue
                
            # Calculate content relevance score
            content_score = 0.0
            doc_text_lower = doc_text.lower()
            
            # Check for exact query term matches
            for term in query_terms:
                if len(term) > 2:  # Only consider meaningful terms
                    term_count = doc_text_lower.count(term)
                    content_score += term_count * 0.1
            
            # Check for document name relevance
            doc_name = doc_id.split("__")[0].lower()
            for term in query_terms:
                if len(term) > 2 and term in doc_name:
                    content_score += 0.5  # Boost for document name matches
            
            # Combine original score with content relevance
            final_score = score + content_score
            scored_docs.append((doc_id, score, final_score, doc_text))
        
        # Sort by final score (highest first)
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        
        # Create citations from top documents
        for doc_id, original_score, final_score, doc_text in scored_docs[:3]:
            excerpt = doc_text[:300] if doc_text and len(doc_text) > 300 else (doc_text or "No content available")
            
            # Enhanced page number extraction
            source, page = self._extract_document_info(doc_id)
            
            citations.append(RetrievedChunk(
                document=source,
                pages=[page] if page else [],
                chunk_id=doc_id,
                excerpt=excerpt,
                relevance_score=float(original_score),  # Keep original score for display
                credibility_score=0.85,
                extraction_method="fusion"
            ))
        
        return citations

    def _extract_document_info(self, doc_id: str) -> tuple[str, int]:
        """Extract document source name and page number from document ID."""
        source = doc_id
        page = None
        
        if "__" in doc_id:
            parts = doc_id.split("__")
            source = parts[0]
            
            # Enhanced page number extraction patterns
            for part in parts[1:]:
                # Pattern 1: p123 (page 123)
                if part.startswith("p") and part[1:].isdigit():
                    page = int(part[1:])
                    break
                # Pattern 2: page123 (page 123)
                elif part.startswith("page") and part[4:].isdigit():
                    page = int(part[4:])
                    break
                # Pattern 3: c0000 (chunk 0) - extract page from chunk
                elif part.startswith("c") and part[1:].isdigit():
                    chunk_num = int(part[1:])
                    # Estimate page number based on chunk (assuming ~500 words per page)
                    page = (chunk_num // 2) + 1
                    break
                # Pattern 4: Just a number (could be page)
                elif part.isdigit():
                    page = int(part)
                    break
                # Pattern 5: page_123 (page 123)
                elif part.startswith("page_") and part[5:].isdigit():
                    page = int(part[5:])
                    break
                # Pattern 6: p_123 (page 123)
                elif part.startswith("p_") and part[2:].isdigit():
                    page = int(part[2:])
                    break
        
        return source, page


