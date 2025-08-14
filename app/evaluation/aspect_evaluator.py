"""Aspect-based evaluation system for comprehensive response assessment."""
from __future__ import annotations

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import re


class EvaluationDimension(Enum):
    """Dimensions for aspect-based evaluation."""
    FACTUAL_ACCURACY = "factual_accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    CITATION_QUALITY = "citation_quality"
    HELPFULNESS = "helpfulness"
    BIAS_DETECTION = "bias_detection"
    SAFETY = "safety"


@dataclass
class AspectScore:
    """Score for a specific evaluation aspect."""
    dimension: EvaluationDimension
    score: float  # 0-10 scale
    confidence: float  # 0-1 scale
    explanation: str
    evidence: List[str]
    improvement_suggestions: List[str]
    sub_scores: Dict[str, float]  # Detailed breakdown


@dataclass
class EvaluationResult:
    """Complete aspect-based evaluation result."""
    overall_score: float
    aspect_scores: List[AspectScore]
    strengths: List[str]
    weaknesses: List[str]
    priority_improvements: List[str]
    confidence_level: str
    evaluation_time_ms: int


class FactualAccuracyEvaluator:
    """Evaluates factual accuracy of responses."""
    
    def __init__(self):
        # Patterns for detecting factual claims
        self.fact_patterns = [
            re.compile(r'\b\d+(?:\.\d+)?(?:%|percent|degrees?)\b', re.IGNORECASE),
            re.compile(r'\b(?:is|are|was|were|has|have|will|can|cannot)\b', re.IGNORECASE),
            re.compile(r'\b(?:according to|studies show|research indicates)\b', re.IGNORECASE)
        ]
    
    def evaluate(self, response: str, sources: List[str], query: str) -> AspectScore:
        """Evaluate factual accuracy of response."""
        
        # Extract factual claims
        claims = self._extract_factual_claims(response)
        
        # Verify claims against sources
        verified_claims = 0
        unverified_claims = 0
        contradicted_claims = 0
        
        evidence = []
        
        for claim in claims:
            verification_result = self._verify_claim_against_sources(claim, sources)
            
            if verification_result["status"] == "verified":
                verified_claims += 1
                evidence.append(f"Verified: {claim[:50]}...")
            elif verification_result["status"] == "contradicted":
                contradicted_claims += 1
                evidence.append(f"Contradicted: {claim[:50]}...")
            else:
                unverified_claims += 1
        
        total_claims = len(claims)
        
        if total_claims == 0:
            accuracy_score = 8.0  # No claims to verify
            explanation = "Response contains no specific factual claims to verify"
        else:
            # Calculate accuracy score
            accuracy_ratio = verified_claims / total_claims
            contradiction_penalty = contradicted_claims / total_claims * 5  # Heavy penalty for contradictions
            accuracy_score = max(0, (accuracy_ratio * 10) - contradiction_penalty)
            
            explanation = f"Verified {verified_claims}/{total_claims} claims, {contradicted_claims} contradictions"
        
        # Generate improvement suggestions
        suggestions = []
        if contradicted_claims > 0:
            suggestions.append("Review and correct contradicted factual claims")
        if unverified_claims > total_claims * 0.3:
            suggestions.append("Provide more source-backed factual information")
        
        sub_scores = {
            "claim_verification_rate": (verified_claims / total_claims * 10) if total_claims > 0 else 10,
            "contradiction_penalty": contradiction_penalty,
            "total_claims_found": total_claims
        }
        
        return AspectScore(
            dimension=EvaluationDimension.FACTUAL_ACCURACY,
            score=round(accuracy_score, 1),
            confidence=0.8,
            explanation=explanation,
            evidence=evidence,
            improvement_suggestions=suggestions,
            sub_scores=sub_scores
        )
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains factual patterns
            has_factual_pattern = any(pattern.search(sentence) for pattern in self.fact_patterns)
            
            if has_factual_pattern or self._is_factual_statement(sentence):
                claims.append(sentence)
        
        return claims
    
    def _is_factual_statement(self, sentence: str) -> bool:
        """Determine if sentence is a factual statement."""
        
        # Simple heuristics for factual statements
        factual_indicators = [
            "according to", "research shows", "studies indicate", "data reveals",
            "statistics show", "evidence suggests", "findings demonstrate"
        ]
        
        sentence_lower = sentence.lower()
        
        # Check for factual indicators
        if any(indicator in sentence_lower for indicator in factual_indicators):
            return True
        
        # Check for definitive statements
        if any(word in sentence_lower for word in ["is", "are", "was", "were", "has", "have"]):
            # Avoid questions and conditional statements
            if not any(word in sentence_lower for word in ["?", "if", "would", "could", "might", "may"]):
                return True
        
        return False
    
    def _verify_claim_against_sources(self, claim: str, sources: List[str]) -> Dict[str, Any]:
        """Verify a claim against source documents."""
        
        claim_lower = claim.lower()
        
        for source in sources:
            source_lower = source.lower()
            
            # Check for direct support
            if self._has_direct_support(claim_lower, source_lower):
                return {"status": "verified", "source_snippet": source[:200]}
            
            # Check for contradiction
            if self._has_contradiction(claim_lower, source_lower):
                return {"status": "contradicted", "source_snippet": source[:200]}
        
        return {"status": "unverified", "source_snippet": None}
    
    def _has_direct_support(self, claim: str, source: str) -> bool:
        """Check if source directly supports the claim."""
        
        # Extract key phrases from claim
        claim_words = set(claim.split())
        source_words = set(source.split())
        
        # Calculate overlap
        overlap = len(claim_words.intersection(source_words))
        overlap_ratio = overlap / len(claim_words) if claim_words else 0
        
        # Require significant overlap for direct support
        return overlap_ratio > 0.6
    
    def _has_contradiction(self, claim: str, source: str) -> bool:
        """Check if source contradicts the claim."""
        
        # Simple contradiction detection (could be enhanced with NLP)
        contradiction_pairs = [
            ("is", "is not"), ("are", "are not"), ("has", "does not have"),
            ("can", "cannot"), ("will", "will not"), ("true", "false")
        ]
        
        for positive, negative in contradiction_pairs:
            if positive in claim and negative in source:
                return True
            if negative in claim and positive in source:
                return True
        
        return False


class CompletenessEvaluator:
    """Evaluates completeness of responses."""
    
    def evaluate(self, response: str, query: str, sources: List[str]) -> AspectScore:
        """Evaluate completeness of response."""
        
        # Analyze query to identify required information
        query_aspects = self._analyze_query_requirements(query)
        
        # Check coverage of each aspect
        covered_aspects = 0
        missing_aspects = []
        
        for aspect in query_aspects:
            if self._aspect_covered_in_response(aspect, response):
                covered_aspects += 1
            else:
                missing_aspects.append(aspect)
        
        # Calculate completeness score
        total_aspects = len(query_aspects)
        if total_aspects == 0:
            completeness_score = 8.0
            explanation = "Query requirements are fully addressed"
        else:
            coverage_ratio = covered_aspects / total_aspects
            completeness_score = coverage_ratio * 10
            explanation = f"Covered {covered_aspects}/{total_aspects} expected aspects"
        
        # Check response depth
        depth_score = self._evaluate_response_depth(response, query)
        
        # Combine coverage and depth
        final_score = (completeness_score * 0.7) + (depth_score * 0.3)
        
        evidence = [f"Covered aspect: {aspect}" for aspect in query_aspects if self._aspect_covered_in_response(aspect, response)]
        
        suggestions = []
        if missing_aspects:
            suggestions.append(f"Address missing aspects: {', '.join(missing_aspects)}")
        if depth_score < 6:
            suggestions.append("Provide more detailed and comprehensive information")
        
        sub_scores = {
            "aspect_coverage": completeness_score,
            "response_depth": depth_score,
            "missing_aspects_count": len(missing_aspects)
        }
        
        return AspectScore(
            dimension=EvaluationDimension.COMPLETENESS,
            score=round(final_score, 1),
            confidence=0.7,
            explanation=explanation,
            evidence=evidence,
            improvement_suggestions=suggestions,
            sub_scores=sub_scores
        )
    
    def _analyze_query_requirements(self, query: str) -> List[str]:
        """Analyze query to identify required information aspects."""
        
        query_lower = query.lower()
        aspects = []
        
        # Question word analysis
        if any(word in query_lower for word in ["what", "define", "explain"]):
            aspects.append("definition")
        
        if any(word in query_lower for word in ["how", "process", "steps"]):
            aspects.append("process/method")
        
        if any(word in query_lower for word in ["why", "reason", "cause"]):
            aspects.append("reasoning/causation")
        
        if any(word in query_lower for word in ["when", "time", "date"]):
            aspects.append("temporal_information")
        
        if any(word in query_lower for word in ["where", "location", "place"]):
            aspects.append("location/context")
        
        if any(word in query_lower for word in ["who", "person", "people"]):
            aspects.append("people/entities")
        
        if any(word in query_lower for word in ["benefits", "advantages", "pros"]):
            aspects.append("benefits")
        
        if any(word in query_lower for word in ["disadvantages", "cons", "risks", "problems"]):
            aspects.append("disadvantages/risks")
        
        if any(word in query_lower for word in ["compare", "difference", "versus"]):
            aspects.append("comparison")
        
        if any(word in query_lower for word in ["examples", "instance", "case"]):
            aspects.append("examples")
        
        # If no specific aspects identified, add general ones
        if not aspects:
            aspects = ["main_information", "context", "details"]
        
        return aspects
    
    def _aspect_covered_in_response(self, aspect: str, response: str) -> bool:
        """Check if specific aspect is covered in response."""
        
        response_lower = response.lower()
        
        coverage_indicators = {
            "definition": ["is defined as", "refers to", "means", "definition"],
            "process/method": ["steps", "process", "method", "procedure", "approach"],
            "reasoning/causation": ["because", "due to", "caused by", "reason", "therefore"],
            "temporal_information": ["when", "time", "date", "period", "era"],
            "location/context": ["where", "location", "place", "region", "area"],
            "people/entities": ["person", "people", "individual", "organization"],
            "benefits": ["benefit", "advantage", "positive", "helpful", "useful"],
            "disadvantages/risks": ["disadvantage", "risk", "problem", "negative", "drawback"],
            "comparison": ["compared to", "versus", "different", "similar", "contrast"],
            "examples": ["example", "instance", "case", "such as", "including"]
        }
        
        indicators = coverage_indicators.get(aspect, [aspect])
        
        return any(indicator in response_lower for indicator in indicators)
    
    def _evaluate_response_depth(self, response: str, query: str) -> float:
        """Evaluate the depth and thoroughness of response."""
        
        # Simple depth metrics
        word_count = len(response.split())
        sentence_count = len(re.split(r'[.!?]+', response))
        
        # Depth indicators
        depth_indicators = [
            "furthermore", "additionally", "moreover", "in detail", "specifically",
            "for example", "such as", "including", "notably", "particularly"
        ]
        
        depth_indicator_count = sum(1 for indicator in depth_indicators if indicator in response.lower())
        
        # Calculate depth score
        word_score = min(10, word_count / 20)  # 200 words = 10 points
        structure_score = min(10, sentence_count / 3)  # 3 sentences = 10 points
        detail_score = min(10, depth_indicator_count * 2)  # Each indicator = 2 points
        
        depth_score = (word_score + structure_score + detail_score) / 3
        
        return depth_score


class ClarityEvaluator:
    """Evaluates clarity and readability of responses."""
    
    def evaluate(self, response: str, query: str) -> AspectScore:
        """Evaluate clarity of response."""
        
        # Readability metrics
        readability_score = self._calculate_readability(response)
        
        # Structure analysis
        structure_score = self._analyze_structure(response)
        
        # Language clarity
        language_score = self._analyze_language_clarity(response)
        
        # Combine scores
        overall_clarity = (readability_score + structure_score + language_score) / 3
        
        evidence = []
        suggestions = []
        
        if readability_score < 6:
            suggestions.append("Simplify sentence structure and vocabulary")
            evidence.append("Complex language detected")
        
        if structure_score < 6:
            suggestions.append("Improve response organization and flow")
            evidence.append("Poor structural organization")
        
        if language_score < 6:
            suggestions.append("Use clearer and more precise language")
            evidence.append("Unclear language usage")
        
        if overall_clarity >= 8:
            evidence.append("Response is well-structured and clear")
        
        sub_scores = {
            "readability": readability_score,
            "structure": structure_score,
            "language_clarity": language_score
        }
        
        return AspectScore(
            dimension=EvaluationDimension.CLARITY,
            score=round(overall_clarity, 1),
            confidence=0.75,
            explanation=f"Clarity score based on readability, structure, and language analysis",
            evidence=evidence,
            improvement_suggestions=suggestions,
            sub_scores=sub_scores
        )
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch-Kincaid style)."""
        
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        
        if sentences == 0 or words == 0:
            return 5.0
        
        # Simple metrics
        avg_sentence_length = words / sentences
        
        # Count syllables (approximation)
        syllables = self._count_syllables(text)
        avg_syllables_per_word = syllables / words if words > 0 else 0
        
        # Simplified readability calculation
        # Shorter sentences and simpler words = higher score
        readability = 10 - (avg_sentence_length / 5) - (avg_syllables_per_word * 2)
        
        return max(0, min(10, readability))
    
    def _count_syllables(self, text: str) -> int:
        """Approximate syllable count."""
        
        # Simple vowel counting method
        vowels = "aeiouy"
        syllable_count = 0
        
        for word in text.lower().split():
            word_syllables = 0
            prev_was_vowel = False
            
            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        word_syllables += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            
            # Ensure at least one syllable per word
            syllable_count += max(1, word_syllables)
        
        return syllable_count
    
    def _analyze_structure(self, text: str) -> float:
        """Analyze structural clarity of response."""
        
        structure_score = 8.0  # Start with good score
        
        # Check for logical connectors
        connectors = [
            "first", "second", "third", "finally", "however", "therefore",
            "furthermore", "additionally", "in conclusion", "for example"
        ]
        
        connector_count = sum(1 for connector in connectors if connector in text.lower())
        
        # Bonus for good connectors
        if connector_count > 0:
            structure_score += min(2, connector_count * 0.5)
        
        # Check for paragraph-like structure (simplified)
        lines = text.split('\n')
        if len(lines) > 3:  # Multi-paragraph structure
            structure_score += 1
        
        # Penalty for very long unbroken text
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 10 and '\n' not in text:
            structure_score -= 2
        
        return max(0, min(10, structure_score))
    
    def _analyze_language_clarity(self, text: str) -> float:
        """Analyze language clarity."""
        
        clarity_score = 8.0
        
        # Check for ambiguous words
        ambiguous_words = [
            "thing", "stuff", "something", "somehow", "various", "several",
            "many", "some", "it", "this", "that"
        ]
        
        text_words = text.lower().split()
        ambiguous_count = sum(1 for word in text_words if word in ambiguous_words)
        ambiguous_ratio = ambiguous_count / len(text_words) if text_words else 0
        
        # Penalty for too many ambiguous words
        if ambiguous_ratio > 0.1:  # More than 10% ambiguous
            clarity_score -= 3
        elif ambiguous_ratio > 0.05:  # More than 5% ambiguous
            clarity_score -= 1
        
        # Check for overly complex sentences
        sentences = re.split(r'[.!?]+', text)
        complex_sentences = sum(1 for s in sentences if len(s.split()) > 25)
        
        if complex_sentences > len(sentences) * 0.3:  # More than 30% complex
            clarity_score -= 2
        
        return max(0, min(10, clarity_score))


class CitationQualityEvaluator:
    """Evaluates citation quality and accuracy."""
    
    def evaluate(self, response: str, citations: List[Dict[str, Any]], 
                sources: List[Dict[str, Any]]) -> AspectScore:
        """Evaluate citation quality."""
        
        if not citations:
            return AspectScore(
                dimension=EvaluationDimension.CITATION_QUALITY,
                score=5.0,
                confidence=0.9,
                explanation="No citations provided",
                evidence=["Response lacks citations"],
                improvement_suggestions=["Add proper citations to support claims"],
                sub_scores={"citation_count": 0, "accuracy": 0, "completeness": 0}
            )
        
        # Evaluate citation accuracy
        accurate_citations = 0
        total_citations = len(citations)
        
        # Evaluate citation completeness
        completeness_score = self._evaluate_citation_completeness(citations)
        
        # Evaluate citation relevance
        relevance_score = self._evaluate_citation_relevance(response, citations)
        
        # Check citation format
        format_score = self._evaluate_citation_format(citations)
        
        # Verify citations against sources
        for citation in citations:
            if self._verify_citation_accuracy(citation, sources):
                accurate_citations += 1
        
        accuracy_ratio = accurate_citations / total_citations
        accuracy_score = accuracy_ratio * 10
        
        # Overall citation quality
        overall_score = (accuracy_score * 0.4 + completeness_score * 0.3 + 
                        relevance_score * 0.2 + format_score * 0.1)
        
        evidence = []
        suggestions = []
        
        if accuracy_ratio < 0.8:
            suggestions.append("Verify citation accuracy against source documents")
            evidence.append(f"Only {accurate_citations}/{total_citations} citations verified")
        
        if completeness_score < 7:
            suggestions.append("Provide more complete citation information")
        
        if relevance_score < 7:
            suggestions.append("Ensure citations directly support the claims made")
        
        if overall_score >= 8:
            evidence.append("Citations are accurate and well-formatted")
        
        sub_scores = {
            "accuracy": accuracy_score,
            "completeness": completeness_score,
            "relevance": relevance_score,
            "format": format_score
        }
        
        return AspectScore(
            dimension=EvaluationDimension.CITATION_QUALITY,
            score=round(overall_score, 1),
            confidence=0.85,
            explanation=f"Citation quality based on accuracy ({accuracy_ratio:.1%}), completeness, and relevance",
            evidence=evidence,
            improvement_suggestions=suggestions,
            sub_scores=sub_scores
        )
    
    def _evaluate_citation_completeness(self, citations: List[Dict[str, Any]]) -> float:
        """Evaluate completeness of citation information."""
        
        required_fields = ["document", "pages", "excerpt"]
        total_score = 0
        
        for citation in citations:
            field_score = 0
            for field in required_fields:
                if field in citation and citation[field]:
                    field_score += 1
            
            completeness = (field_score / len(required_fields)) * 10
            total_score += completeness
        
        return total_score / len(citations) if citations else 0
    
    def _evaluate_citation_relevance(self, response: str, citations: List[Dict[str, Any]]) -> float:
        """Evaluate how relevant citations are to the response content."""
        
        response_words = set(response.lower().split())
        total_relevance = 0
        
        for citation in citations:
            excerpt = citation.get("excerpt", "")
            if excerpt:
                excerpt_words = set(excerpt.lower().split())
                overlap = len(response_words.intersection(excerpt_words))
                overlap_ratio = overlap / len(excerpt_words) if excerpt_words else 0
                relevance_score = min(10, overlap_ratio * 20)  # Scale to 0-10
                total_relevance += relevance_score
        
        return total_relevance / len(citations) if citations else 0
    
    def _evaluate_citation_format(self, citations: List[Dict[str, Any]]) -> float:
        """Evaluate citation format quality."""
        
        format_score = 8.0  # Start with good score
        
        for citation in citations:
            # Check for proper document naming
            document = citation.get("document", "")
            if not document or len(document) < 3:
                format_score -= 1
            
            # Check for page information
            pages = citation.get("pages", [])
            if isinstance(pages, list) and not pages:
                format_score -= 0.5
            
            # Check for meaningful excerpt
            excerpt = citation.get("excerpt", "")
            if not excerpt or len(excerpt) < 20:
                format_score -= 0.5
        
        return max(0, min(10, format_score))
    
    def _verify_citation_accuracy(self, citation: Dict[str, Any], 
                                 sources: List[Dict[str, Any]]) -> bool:
        """Verify if citation accurately references source material."""
        
        document_name = citation.get("document", "")
        excerpt = citation.get("excerpt", "")
        
        # Find matching source document
        matching_source = None
        for source in sources:
            if (source.get("filename", "") == document_name or 
                document_name in source.get("filename", "")):
                matching_source = source
                break
        
        if not matching_source:
            return False
        
        # Verify excerpt exists in source
        if excerpt:
            source_content = matching_source.get("content", "")
            return excerpt.lower() in source_content.lower()
        
        return True  # If no excerpt to verify, assume accurate


class AspectBasedEvaluator:
    """Main aspect-based evaluation system."""
    
    def __init__(self):
        self.evaluators = {
            EvaluationDimension.FACTUAL_ACCURACY: FactualAccuracyEvaluator(),
            EvaluationDimension.COMPLETENESS: CompletenessEvaluator(),
            EvaluationDimension.CLARITY: ClarityEvaluator(),
            EvaluationDimension.CITATION_QUALITY: CitationQualityEvaluator()
        }
    
    async def evaluate_response(self, response: str, query: str, sources: List[str],
                              citations: List[Dict[str, Any]] = None,
                              source_documents: List[Dict[str, Any]] = None,
                              dimensions: List[EvaluationDimension] = None) -> EvaluationResult:
        """Perform comprehensive aspect-based evaluation."""
        
        start_time = time.time()
        
        # Use all dimensions if none specified
        if dimensions is None:
            dimensions = list(self.evaluators.keys())
        
        aspect_scores = []
        
        # Evaluate each dimension
        for dimension in dimensions:
            evaluator = self.evaluators.get(dimension)
            if not evaluator:
                continue
            
            try:
                if dimension == EvaluationDimension.FACTUAL_ACCURACY:
                    score = evaluator.evaluate(response, sources, query)
                elif dimension == EvaluationDimension.COMPLETENESS:
                    score = evaluator.evaluate(response, query, sources)
                elif dimension == EvaluationDimension.CLARITY:
                    score = evaluator.evaluate(response, query)
                elif dimension == EvaluationDimension.CITATION_QUALITY:
                    score = evaluator.evaluate(response, citations or [], source_documents or [])
                else:
                    continue
                
                aspect_scores.append(score)
                
            except Exception as e:
                print(f"Failed to evaluate {dimension.value}: {e}")
        
        # Calculate overall score
        overall_score = statistics.mean([score.score for score in aspect_scores]) if aspect_scores else 5.0
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        priority_improvements = []
        
        for score in aspect_scores:
            if score.score >= 8.0:
                strengths.append(f"Strong {score.dimension.value} (score: {score.score})")
            elif score.score <= 5.0:
                weaknesses.append(f"Weak {score.dimension.value} (score: {score.score})")
                priority_improvements.extend(score.improvement_suggestions)
        
        # Determine confidence level
        avg_confidence = statistics.mean([score.confidence for score in aspect_scores]) if aspect_scores else 0.5
        
        if avg_confidence >= 0.8:
            confidence_level = "High"
        elif avg_confidence >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        evaluation_time = int((time.time() - start_time) * 1000)
        
        return EvaluationResult(
            overall_score=round(overall_score, 1),
            aspect_scores=aspect_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            priority_improvements=list(set(priority_improvements)),  # Remove duplicates
            confidence_level=confidence_level,
            evaluation_time_ms=evaluation_time
        )
    
    def get_evaluation_summary(self, result: EvaluationResult) -> Dict[str, Any]:
        """Get a summary of evaluation results."""
        
        # Score distribution
        score_distribution = {
            "excellent": len([s for s in result.aspect_scores if s.score >= 9]),
            "good": len([s for s in result.aspect_scores if 7 <= s.score < 9]),
            "satisfactory": len([s for s in result.aspect_scores if 5 <= s.score < 7]),
            "poor": len([s for s in result.aspect_scores if s.score < 5])
        }
        
        # Quality category
        if result.overall_score >= 8.5:
            quality_category = "Excellent"
        elif result.overall_score >= 7.0:
            quality_category = "Good"
        elif result.overall_score >= 5.5:
            quality_category = "Satisfactory"
        else:
            quality_category = "Needs Improvement"
        
        return {
            "overall_assessment": {
                "score": result.overall_score,
                "quality_category": quality_category,
                "confidence": result.confidence_level
            },
            "aspect_breakdown": {
                score.dimension.value: {
                    "score": score.score,
                    "confidence": score.confidence,
                    "key_issues": score.improvement_suggestions[:2]  # Top 2 issues
                }
                for score in result.aspect_scores
            },
            "score_distribution": score_distribution,
            "priority_actions": result.priority_improvements[:5],  # Top 5 priorities
            "strengths": result.strengths,
            "evaluation_metadata": {
                "dimensions_evaluated": len(result.aspect_scores),
                "evaluation_time_ms": result.evaluation_time_ms
            }
        }
