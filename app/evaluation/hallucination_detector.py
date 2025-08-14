"""Hallucination detection system for RAG responses."""
from __future__ import annotations

import re
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from difflib import SequenceMatcher
import hashlib


class HallucinationType(Enum):
    """Types of hallucinations."""
    FACTUAL_ERROR = "factual_error"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTORY_INFO = "contradictory_info"
    FABRICATED_CITATION = "fabricated_citation"
    TEMPORAL_INACCURACY = "temporal_inaccuracy"
    NUMERICAL_ERROR = "numerical_error"
    ENTITY_ERROR = "entity_error"
    CAUSAL_ERROR = "causal_error"


class HallucinationSeverity(Enum):
    """Severity levels for hallucinations."""
    MINOR = "minor"          # Small inaccuracies, typos
    MODERATE = "moderate"    # Significant but not critical errors
    MAJOR = "major"          # Important factual errors
    CRITICAL = "critical"    # Dangerous or completely false information


@dataclass
class FactualClaim:
    """Represents a factual claim extracted from text."""
    claim_text: str
    claim_type: str  # "numerical", "temporal", "entity", "relationship", "causal"
    entities: List[str]
    numbers: List[str]
    dates: List[str]
    confidence: float
    start_position: int
    end_position: int


@dataclass
class HallucinationInstance:
    """Represents a detected hallucination."""
    hallucination_id: str
    hallucination_type: HallucinationType
    severity: HallucinationSeverity
    claim_text: str
    evidence_text: str
    explanation: str
    confidence: float
    start_position: int
    end_position: int
    suggested_correction: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class HallucinationReport:
    """Complete hallucination analysis report."""
    total_hallucinations: int
    hallucinations_by_type: Dict[str, int]
    hallucinations_by_severity: Dict[str, int]
    detected_hallucinations: List[HallucinationInstance]
    overall_reliability_score: float  # 0-1, higher is better
    confidence_in_analysis: float
    analysis_time_ms: int
    summary: str


class FactualClaimExtractor:
    """Extracts factual claims from text for verification."""
    
    def __init__(self):
        # Patterns for different types of claims
        self.numerical_patterns = [
            re.compile(r'\b\d+(?:\.\d+)?(?:%|percent|degrees?|kg|lbs?|miles?|km|hours?|minutes?|seconds?|years?|months?|days?)\b', re.IGNORECASE),
            re.compile(r'\$\d+(?:\.\d+)?(?:[kmb]illion)?', re.IGNORECASE),
            re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b')
        ]
        
        self.temporal_patterns = [
            re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'),
            re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),
            re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
            re.compile(r'\b(?:in|since|during|before|after)\s+\d{4}\b', re.IGNORECASE)
        ]
        
        self.entity_patterns = [
            re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Company|Corp|Inc|LLC|Ltd)\b'),
            re.compile(r'\b(?:President|CEO|Director|Manager)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),
            re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|College|School)\b')
        ]
        
        self.causal_indicators = [
            "because", "due to", "caused by", "results in", "leads to",
            "therefore", "consequently", "as a result", "thus"
        ]
    
    def extract_claims(self, text: str) -> List[FactualClaim]:
        """Extract factual claims from text."""
        claims = []
        
        # Split text into sentences for claim extraction
        sentences = re.split(r'[.!?]+', text)
        
        current_position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Find sentence position in original text
            sentence_start = text.find(sentence, current_position)
            if sentence_start == -1:
                sentence_start = current_position
            
            # Extract different types of claims
            claims.extend(self._extract_numerical_claims(sentence, sentence_start))
            claims.extend(self._extract_temporal_claims(sentence, sentence_start))
            claims.extend(self._extract_entity_claims(sentence, sentence_start))
            claims.extend(self._extract_causal_claims(sentence, sentence_start))
            
            current_position = sentence_start + len(sentence)
        
        return claims
    
    def _extract_numerical_claims(self, sentence: str, offset: int) -> List[FactualClaim]:
        """Extract numerical claims from sentence."""
        claims = []
        
        for pattern in self.numerical_patterns:
            for match in pattern.finditer(sentence):
                claim = FactualClaim(
                    claim_text=sentence,
                    claim_type="numerical",
                    entities=[],
                    numbers=[match.group()],
                    dates=[],
                    confidence=0.8,
                    start_position=offset + match.start(),
                    end_position=offset + match.end()
                )
                claims.append(claim)
        
        return claims
    
    def _extract_temporal_claims(self, sentence: str, offset: int) -> List[FactualClaim]:
        """Extract temporal claims from sentence."""
        claims = []
        
        for pattern in self.temporal_patterns:
            for match in pattern.finditer(sentence):
                claim = FactualClaim(
                    claim_text=sentence,
                    claim_type="temporal",
                    entities=[],
                    numbers=[],
                    dates=[match.group()],
                    confidence=0.9,
                    start_position=offset + match.start(),
                    end_position=offset + match.end()
                )
                claims.append(claim)
        
        return claims
    
    def _extract_entity_claims(self, sentence: str, offset: int) -> List[FactualClaim]:
        """Extract entity-related claims from sentence."""
        claims = []
        
        for pattern in self.entity_patterns:
            for match in pattern.finditer(sentence):
                claim = FactualClaim(
                    claim_text=sentence,
                    claim_type="entity",
                    entities=[match.group()],
                    numbers=[],
                    dates=[],
                    confidence=0.7,
                    start_position=offset + match.start(),
                    end_position=offset + match.end()
                )
                claims.append(claim)
        
        return claims
    
    def _extract_causal_claims(self, sentence: str, offset: int) -> List[FactualClaim]:
        """Extract causal claims from sentence."""
        claims = []
        
        sentence_lower = sentence.lower()
        for indicator in self.causal_indicators:
            if indicator in sentence_lower:
                claim = FactualClaim(
                    claim_text=sentence,
                    claim_type="causal",
                    entities=[],
                    numbers=[],
                    dates=[],
                    confidence=0.6,
                    start_position=offset,
                    end_position=offset + len(sentence)
                )
                claims.append(claim)
                break  # Only one causal claim per sentence
        
        return claims


class ContentSourceAligner:
    """Aligns response content with source documents."""
    
    def __init__(self):
        self.similarity_threshold = 0.3
        self.chunk_size = 100  # words
    
    def check_source_alignment(self, response_text: str, source_texts: List[str]) -> Dict[str, Any]:
        """Check how well response aligns with source documents."""
        
        if not source_texts:
            return {
                "alignment_score": 0.0,
                "unsupported_segments": [response_text],
                "supported_segments": [],
                "coverage_ratio": 0.0
            }
        
        # Split response into chunks
        response_chunks = self._split_into_chunks(response_text)
        
        supported_segments = []
        unsupported_segments = []
        
        for chunk in response_chunks:
            best_similarity = 0.0
            best_source = None
            
            for source_text in source_texts:
                similarity = self._calculate_semantic_similarity(chunk, source_text)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_source = source_text
            
            if best_similarity >= self.similarity_threshold:
                supported_segments.append({
                    "text": chunk,
                    "similarity": best_similarity,
                    "source_excerpt": best_source[:200] + "..." if len(best_source) > 200 else best_source
                })
            else:
                unsupported_segments.append(chunk)
        
        # Calculate metrics
        total_chunks = len(response_chunks)
        supported_chunks = len(supported_segments)
        
        alignment_score = supported_chunks / total_chunks if total_chunks > 0 else 0.0
        coverage_ratio = sum(len(seg["text"]) for seg in supported_segments) / len(response_text) if response_text else 0.0
        
        return {
            "alignment_score": round(alignment_score, 3),
            "unsupported_segments": unsupported_segments,
            "supported_segments": supported_segments,
            "coverage_ratio": round(coverage_ratio, 3),
            "total_chunks": total_chunks,
            "supported_chunks": supported_chunks
        }
    
    def _split_into_chunks(self, text: str, chunk_size: int = None) -> List[str]:
        """Split text into overlapping chunks for analysis."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size // 2):  # 50% overlap
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if len(chunk_text.strip()) > 20:  # Minimum chunk size
                chunks.append(chunk_text)
        
        return chunks if chunks else [text]
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Simple implementation using token overlap
        # In production, could use embeddings for better semantic similarity
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union)
        
        # Also consider sequence similarity
        sequence_similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # Combine both metrics
        combined_similarity = (jaccard_similarity + sequence_similarity) / 2
        
        return combined_similarity


class CitationVerifier:
    """Verifies the accuracy of citations."""
    
    def verify_citations(self, response_text: str, citations: List[Dict[str, Any]],
                        source_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify citation accuracy and completeness."""
        
        if not citations:
            return {
                "citation_accuracy": 1.0,
                "fabricated_citations": [],
                "accurate_citations": [],
                "citation_coverage": 0.0
            }
        
        fabricated_citations = []
        accurate_citations = []
        
        for citation in citations:
            document_name = citation.get('document', '')
            page_numbers = citation.get('pages', [])
            excerpt = citation.get('excerpt', '')
            
            # Check if document exists in source documents
            matching_doc = self._find_matching_document(document_name, source_documents)
            
            if not matching_doc:
                fabricated_citations.append({
                    "citation": citation,
                    "reason": "Document not found in sources",
                    "severity": HallucinationSeverity.MAJOR
                })
                continue
            
            # Verify excerpt if provided
            if excerpt:
                excerpt_found = self._verify_excerpt_in_document(excerpt, matching_doc)
                if not excerpt_found:
                    fabricated_citations.append({
                        "citation": citation,
                        "reason": "Excerpt not found in document",
                        "severity": HallucinationSeverity.MODERATE
                    })
                    continue
            
            # Citation appears accurate
            accurate_citations.append(citation)
        
        total_citations = len(citations)
        accurate_count = len(accurate_citations)
        
        citation_accuracy = accurate_count / total_citations if total_citations > 0 else 1.0
        
        # Calculate coverage (how much of response is cited)
        citation_coverage = self._calculate_citation_coverage(response_text, citations)
        
        return {
            "citation_accuracy": round(citation_accuracy, 3),
            "fabricated_citations": fabricated_citations,
            "accurate_citations": accurate_citations,
            "citation_coverage": round(citation_coverage, 3),
            "total_citations": total_citations,
            "accurate_citations_count": accurate_count
        }
    
    def _find_matching_document(self, doc_name: str, source_documents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find matching document in source list."""
        for doc in source_documents:
            if doc.get('filename', '') == doc_name or doc.get('title', '') == doc_name:
                return doc
            
            # Check for partial matches
            if doc_name.lower() in doc.get('filename', '').lower():
                return doc
        
        return None
    
    def _verify_excerpt_in_document(self, excerpt: str, document: Dict[str, Any]) -> bool:
        """Verify if excerpt exists in document."""
        doc_content = document.get('content', '')
        
        if not doc_content:
            return False
        
        # Check for exact match
        if excerpt.lower() in doc_content.lower():
            return True
        
        # Check for approximate match (allow for minor differences)
        excerpt_words = set(excerpt.lower().split())
        content_words = set(doc_content.lower().split())
        
        overlap = len(excerpt_words.intersection(content_words))
        required_overlap = len(excerpt_words) * 0.8  # 80% overlap required
        
        return overlap >= required_overlap
    
    def _calculate_citation_coverage(self, response_text: str, citations: List[Dict[str, Any]]) -> float:
        """Calculate what percentage of response is covered by citations."""
        if not citations:
            return 0.0
        
        # Simple heuristic: assume each citation covers a reasonable portion
        # In practice, would need more sophisticated analysis
        
        response_length = len(response_text)
        estimated_cited_length = 0
        
        for citation in citations:
            excerpt = citation.get('excerpt', '')
            if excerpt:
                # Estimate coverage based on excerpt length
                estimated_cited_length += len(excerpt) * 1.5  # Factor for context
        
        coverage = min(1.0, estimated_cited_length / response_length) if response_length > 0 else 0.0
        
        return coverage


class HallucinationDetector:
    """Main hallucination detection system."""
    
    def __init__(self):
        self.claim_extractor = FactualClaimExtractor()
        self.source_aligner = ContentSourceAligner()
        self.citation_verifier = CitationVerifier()
        
        # Severity mapping for different hallucination types
        self.severity_mapping = {
            HallucinationType.FACTUAL_ERROR: HallucinationSeverity.MAJOR,
            HallucinationType.UNSUPPORTED_CLAIM: HallucinationSeverity.MODERATE,
            HallucinationType.CONTRADICTORY_INFO: HallucinationSeverity.MAJOR,
            HallucinationType.FABRICATED_CITATION: HallucinationSeverity.MAJOR,
            HallucinationType.TEMPORAL_INACCURACY: HallucinationSeverity.MODERATE,
            HallucinationType.NUMERICAL_ERROR: HallucinationSeverity.MODERATE,
            HallucinationType.ENTITY_ERROR: HallucinationSeverity.MODERATE,
            HallucinationType.CAUSAL_ERROR: HallucinationSeverity.MAJOR
        }
    
    def detect_hallucinations(self, response_text: str, source_texts: List[str],
                            citations: List[Dict[str, Any]] = None,
                            source_documents: List[Dict[str, Any]] = None) -> HallucinationReport:
        """Comprehensive hallucination detection."""
        
        start_time = time.time()
        detected_hallucinations = []
        
        # 1. Check source alignment
        alignment_result = self.source_aligner.check_source_alignment(response_text, source_texts)
        
        # 2. Verify citations
        if citations and source_documents:
            citation_result = self.citation_verifier.verify_citations(
                response_text, citations, source_documents
            )
        else:
            citation_result = {"fabricated_citations": [], "citation_accuracy": 1.0}
        
        # 3. Extract and verify factual claims
        claims = self.claim_extractor.extract_claims(response_text)
        
        # Generate hallucination instances
        
        # Add unsupported content hallucinations
        for unsupported_segment in alignment_result["unsupported_segments"]:
            if len(unsupported_segment.strip()) > 50:  # Only flag substantial unsupported content
                hallucination = HallucinationInstance(
                    hallucination_id=self._generate_id(unsupported_segment),
                    hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                    severity=HallucinationSeverity.MODERATE,
                    claim_text=unsupported_segment,
                    evidence_text="No supporting evidence found in sources",
                    explanation="This content appears to lack direct support from the provided sources",
                    confidence=0.7,
                    start_position=response_text.find(unsupported_segment),
                    end_position=response_text.find(unsupported_segment) + len(unsupported_segment),
                    suggested_correction="Remove or rephrase with proper source support",
                    metadata={"alignment_score": 0.0}
                )
                detected_hallucinations.append(hallucination)
        
        # Add fabricated citation hallucinations
        for fab_citation in citation_result["fabricated_citations"]:
            citation_text = str(fab_citation["citation"])
            hallucination = HallucinationInstance(
                hallucination_id=self._generate_id(citation_text),
                hallucination_type=HallucinationType.FABRICATED_CITATION,
                severity=fab_citation["severity"],
                claim_text=citation_text,
                evidence_text="Citation verification failed",
                explanation=fab_citation["reason"],
                confidence=0.9,
                start_position=0,  # Would need more sophisticated citation position tracking
                end_position=len(citation_text),
                suggested_correction="Verify and correct citation details",
                metadata=fab_citation
            )
            detected_hallucinations.append(hallucination)
        
        # Calculate summary statistics
        hallucinations_by_type = {}
        hallucinations_by_severity = {}
        
        for hallucination in detected_hallucinations:
            hal_type = hallucination.hallucination_type.value
            hal_severity = hallucination.severity.value
            
            hallucinations_by_type[hal_type] = hallucinations_by_type.get(hal_type, 0) + 1
            hallucinations_by_severity[hal_severity] = hallucinations_by_severity.get(hal_severity, 0) + 1
        
        # Calculate overall reliability score
        total_hallucinations = len(detected_hallucinations)
        severity_weights = {
            HallucinationSeverity.MINOR: 0.1,
            HallucinationSeverity.MODERATE: 0.3,
            HallucinationSeverity.MAJOR: 0.7,
            HallucinationSeverity.CRITICAL: 1.0
        }
        
        weighted_hallucination_score = sum(
            severity_weights[hall.severity] for hall in detected_hallucinations
        )
        
        # Reliability score (0-1, higher is better)
        max_possible_score = len(response_text.split()) * 0.1  # Rough estimate
        reliability_score = max(0.0, 1.0 - (weighted_hallucination_score / max_possible_score))
        reliability_score = min(1.0, reliability_score)
        
        # Generate summary
        summary = self._generate_summary(detected_hallucinations, alignment_result, citation_result)
        
        analysis_time = int((time.time() - start_time) * 1000)
        
        return HallucinationReport(
            total_hallucinations=total_hallucinations,
            hallucinations_by_type=hallucinations_by_type,
            hallucinations_by_severity=hallucinations_by_severity,
            detected_hallucinations=detected_hallucinations,
            overall_reliability_score=round(reliability_score, 3),
            confidence_in_analysis=0.8,  # Could be dynamic based on source quality
            analysis_time_ms=analysis_time,
            summary=summary
        )
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID for hallucination instance."""
        return hashlib.md5(text.encode()).hexdigest()[:8]
    
    def _generate_summary(self, hallucinations: List[HallucinationInstance],
                         alignment_result: Dict[str, Any],
                         citation_result: Dict[str, Any]) -> str:
        """Generate summary of hallucination analysis."""
        
        if not hallucinations:
            return "No significant hallucinations detected. Response appears well-supported by sources."
        
        summary_parts = []
        
        # Count by severity
        severity_counts = {}
        for hall in hallucinations:
            severity_counts[hall.severity] = severity_counts.get(hall.severity, 0) + 1
        
        if severity_counts:
            severity_desc = []
            for severity, count in severity_counts.items():
                severity_desc.append(f"{count} {severity.value}")
            summary_parts.append(f"Detected {len(hallucinations)} potential hallucinations: {', '.join(severity_desc)}")
        
        # Source alignment info
        alignment_score = alignment_result.get("alignment_score", 0.0)
        if alignment_score < 0.7:
            summary_parts.append(f"Low source alignment ({alignment_score:.1%}) indicates potential unsupported content")
        
        # Citation info
        citation_accuracy = citation_result.get("citation_accuracy", 1.0)
        if citation_accuracy < 0.9:
            summary_parts.append(f"Citation accuracy issues detected ({citation_accuracy:.1%} accurate)")
        
        if not summary_parts:
            summary_parts.append("Response quality appears satisfactory with minor issues")
        
        return ". ".join(summary_parts) + "."
    
    def get_hallucination_statistics(self, report: HallucinationReport) -> Dict[str, Any]:
        """Get detailed statistics from hallucination report."""
        
        return {
            "overall_assessment": {
                "reliability_score": report.overall_reliability_score,
                "total_hallucinations": report.total_hallucinations,
                "analysis_confidence": report.confidence_in_analysis,
                "quality_grade": self._calculate_quality_grade(report.overall_reliability_score)
            },
            "breakdown": {
                "by_type": report.hallucinations_by_type,
                "by_severity": report.hallucinations_by_severity
            },
            "recommendations": self._generate_recommendations(report)
        }
    
    def _calculate_quality_grade(self, reliability_score: float) -> str:
        """Calculate quality grade based on reliability score."""
        if reliability_score >= 0.95:
            return "A+ (Excellent)"
        elif reliability_score >= 0.90:
            return "A (Very Good)"
        elif reliability_score >= 0.80:
            return "B (Good)"
        elif reliability_score >= 0.70:
            return "C (Acceptable)"
        elif reliability_score >= 0.60:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"
    
    def _generate_recommendations(self, report: HallucinationReport) -> List[str]:
        """Generate recommendations based on hallucination analysis."""
        
        recommendations = []
        
        if report.total_hallucinations == 0:
            recommendations.append("Response quality is excellent - maintain current standards")
        else:
            if report.hallucinations_by_severity.get("critical", 0) > 0:
                recommendations.append("Critical hallucinations detected - immediate review and correction required")
            
            if report.hallucinations_by_severity.get("major", 0) > 0:
                recommendations.append("Major factual errors found - verify claims against authoritative sources")
            
            if report.hallucinations_by_type.get("unsupported_claim", 0) > 0:
                recommendations.append("Improve source coverage and reduce unsupported claims")
            
            if report.hallucinations_by_type.get("fabricated_citation", 0) > 0:
                recommendations.append("Review and verify all citations for accuracy")
            
            if report.overall_reliability_score < 0.7:
                recommendations.append("Consider regenerating response with stronger source grounding")
        
        return recommendations
