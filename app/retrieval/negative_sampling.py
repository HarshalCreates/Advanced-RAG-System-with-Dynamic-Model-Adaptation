"""Negative sampling for filtering irrelevant results."""
from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class NegativeSignal:
    """Represents a negative relevance signal."""
    doc_id: str
    query: str
    reason: str
    confidence: float
    timestamp: float


@dataclass
class RelevanceFilter:
    """Configuration for relevance filtering."""
    min_similarity_threshold: float = 0.1
    max_dissimilarity_threshold: float = 0.9
    spam_detection_enabled: bool = True
    duplicate_detection_enabled: bool = True
    quality_filtering_enabled: bool = True


class NegativeSampler:
    """Filters irrelevant and low-quality results using negative sampling."""
    
    def __init__(self, storage_path: str = "./data/negative_sampling"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.negative_signals_file = self.storage_path / "negative_signals.json"
        self.blacklist_file = self.storage_path / "blacklist.json"
        
        # Load data
        self.negative_signals: List[NegativeSignal] = []
        self.document_blacklist: Set[str] = set()
        self.query_patterns: Dict[str, List[str]] = defaultdict(list)
        
        self.load_negative_signals()
        self.load_blacklist()
        
        # Filter configuration
        self.filter_config = RelevanceFilter()
        
        # Quality indicators
        self.spam_indicators = [
            'click here', 'free money', 'urgent', 'limited time', 'act now',
            'guaranteed', 'no risk', 'easy money', 'work from home',
            'lorem ipsum', 'sample text', 'placeholder', 'test document'
        ]
        
        self.low_quality_patterns = [
            r'^.{0,10}$',  # Very short content
            r'^(.)\1{10,}',  # Repeated characters
            r'^\d+$',  # Only numbers
            r'^[^a-zA-Z]*$',  # No letters
        ]
        
        # TF-IDF vectorizer for similarity computation
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.document_vectors = None
            self.document_ids = []
        
    def load_negative_signals(self):
        """Load negative signals from storage."""
        if self.negative_signals_file.exists():
            try:
                with open(self.negative_signals_file, 'r') as f:
                    data = json.load(f)
                    self.negative_signals = [NegativeSignal(**signal) for signal in data]
            except Exception as e:
                print(f"Failed to load negative signals: {e}")
    
    def save_negative_signals(self):
        """Save negative signals to storage."""
        try:
            data = [asdict(signal) for signal in self.negative_signals]
            with open(self.negative_signals_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save negative signals: {e}")
    
    def load_blacklist(self):
        """Load document blacklist from storage."""
        if self.blacklist_file.exists():
            try:
                with open(self.blacklist_file, 'r') as f:
                    data = json.load(f)
                    self.document_blacklist = set(data.get('blacklist', []))
                    self.query_patterns = defaultdict(list, data.get('query_patterns', {}))
            except Exception as e:
                print(f"Failed to load blacklist: {e}")
    
    def save_blacklist(self):
        """Save document blacklist to storage."""
        try:
            data = {
                'blacklist': list(self.document_blacklist),
                'query_patterns': dict(self.query_patterns)
            }
            with open(self.blacklist_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save blacklist: {e}")
    
    def filter_results(self, query: str, results: List[Tuple[str, float]], 
                      document_texts: Dict[str, str] = None) -> List[Tuple[str, float]]:
        """Filter results to remove irrelevant documents."""
        filtered_results = []
        
        for doc_id, score in results:
            # Skip blacklisted documents
            if doc_id in self.document_blacklist:
                self._record_negative_signal(doc_id, query, "blacklisted", 1.0)
                continue
            
            # Get document text for analysis
            doc_text = document_texts.get(doc_id, "") if document_texts else ""
            
            # Apply filtering logic
            if self._should_filter_document(query, doc_id, doc_text, score):
                continue
            
            filtered_results.append((doc_id, score))
        
        return filtered_results
    
    def _should_filter_document(self, query: str, doc_id: str, doc_text: str, score: float) -> bool:
        """Determine if a document should be filtered out."""
        
        # 1. Score-based filtering
        if score < self.filter_config.min_similarity_threshold:
            self._record_negative_signal(doc_id, query, "low_similarity_score", 0.8)
            return True
        
        # 2. Quality filtering
        if self.filter_config.quality_filtering_enabled:
            quality_issues = self._detect_quality_issues(doc_text)
            if quality_issues:
                reason = f"quality_issue: {quality_issues[0]}"
                self._record_negative_signal(doc_id, query, reason, 0.9)
                return True
        
        # 3. Spam detection
        if self.filter_config.spam_detection_enabled:
            if self._is_spam_document(doc_text):
                self._record_negative_signal(doc_id, query, "spam_detected", 0.95)
                return True
        
        # 4. Relevance analysis
        if doc_text:
            relevance_score = self._calculate_relevance(query, doc_text)
            if relevance_score < 0.2:  # Very low relevance
                self._record_negative_signal(doc_id, query, "low_relevance", 0.7)
                return True
        
        # 5. Query-specific patterns
        if self._matches_negative_pattern(query, doc_id, doc_text):
            self._record_negative_signal(doc_id, query, "negative_pattern", 0.8)
            return True
        
        return False
    
    def _detect_quality_issues(self, text: str) -> List[str]:
        """Detect quality issues in document text."""
        issues = []
        
        if not text or len(text.strip()) < 20:
            issues.append("too_short")
        
        # Check for repeated patterns
        for pattern in self.low_quality_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append("pattern_detected")
                break
        
        # Check word diversity
        words = text.lower().split()
        if len(words) > 10:
            unique_words = len(set(words))
            diversity_ratio = unique_words / len(words)
            if diversity_ratio < 0.3:  # Low diversity
                issues.append("low_diversity")
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if punct_ratio > 0.3:
            issues.append("excessive_punctuation")
        
        return issues
    
    def _is_spam_document(self, text: str) -> bool:
        """Detect if document appears to be spam."""
        text_lower = text.lower()
        
        # Check for spam indicators
        spam_count = sum(1 for indicator in self.spam_indicators 
                        if indicator in text_lower)
        
        if spam_count >= 2:  # Multiple spam indicators
            return True
        
        # Check for excessive capitalization
        if len(text) > 50:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.5:
                return True
        
        # Check for excessive repetition
        lines = text.split('\n')
        if len(lines) > 3:
            line_counts = Counter(lines)
            max_repetition = max(line_counts.values())
            if max_repetition > len(lines) * 0.7:  # More than 70% repetition
                return True
        
        return False
    
    def _calculate_relevance(self, query: str, doc_text: str) -> float:
        """Calculate semantic relevance between query and document."""
        if not SKLEARN_AVAILABLE or not doc_text:
            # Fallback: simple keyword matching
            query_words = set(query.lower().split())
            doc_words = set(doc_text.lower().split())
            common_words = query_words.intersection(doc_words)
            return len(common_words) / len(query_words) if query_words else 0.0
        
        try:
            # Use TF-IDF similarity
            corpus = [query, doc_text]
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception:
            # Fallback on error
            return 0.5
    
    def _matches_negative_pattern(self, query: str, doc_id: str, doc_text: str) -> bool:
        """Check if document matches known negative patterns for this query."""
        query_lower = query.lower()
        
        # Check stored negative patterns
        if query_lower in self.query_patterns:
            for pattern in self.query_patterns[query_lower]:
                if re.search(pattern, doc_text, re.IGNORECASE):
                    return True
        
        # Check for query-document mismatch patterns
        query_words = set(query_lower.split())
        doc_words = set(doc_text.lower().split())
        
        # If query is very specific but document is very general
        if len(query_words) > 3 and len(doc_words) > 100:
            common_ratio = len(query_words.intersection(doc_words)) / len(query_words)
            if common_ratio < 0.2:
                return True
        
        return False
    
    def _record_negative_signal(self, doc_id: str, query: str, reason: str, confidence: float):
        """Record a negative signal for learning."""
        signal = NegativeSignal(
            doc_id=doc_id,
            query=query,
            reason=reason,
            confidence=confidence,
            timestamp=time.time()
        )
        
        self.negative_signals.append(signal)
        
        # Auto-blacklist if enough negative signals
        doc_negative_count = sum(1 for s in self.negative_signals 
                               if s.doc_id == doc_id and s.confidence > 0.8)
        
        if doc_negative_count >= 3:  # Multiple high-confidence negative signals
            self.document_blacklist.add(doc_id)
        
        # Save periodically
        if len(self.negative_signals) % 50 == 0:
            self.save_negative_signals()
            self.save_blacklist()
    
    def add_negative_feedback(self, query: str, doc_id: str, reason: str = "user_feedback"):
        """Add explicit negative feedback for a document."""
        self._record_negative_signal(doc_id, query, reason, 1.0)
        
        # Add to blacklist immediately for explicit feedback
        self.document_blacklist.add(doc_id)
        self.save_blacklist()
    
    def add_negative_pattern(self, query: str, pattern: str):
        """Add a negative pattern for a specific query."""
        self.query_patterns[query.lower()].append(pattern)
        self.save_blacklist()
    
    def train_negative_detector(self, document_texts: Dict[str, str]):
        """Train the negative sampling detector on document corpus."""
        if not SKLEARN_AVAILABLE or not document_texts:
            return
        
        try:
            # Build document vectors for similarity computation
            texts = list(document_texts.values())
            self.document_ids = list(document_texts.keys())
            
            if len(texts) > 1:
                self.document_vectors = self.vectorizer.fit_transform(texts)
            
            # Analyze negative signals to improve filtering
            self._analyze_negative_patterns()
            
        except Exception as e:
            print(f"Failed to train negative detector: {e}")
    
    def _analyze_negative_patterns(self):
        """Analyze collected negative signals to improve filtering."""
        if not self.negative_signals:
            return
        
        # Group by reason
        reason_counts = Counter(signal.reason for signal in self.negative_signals)
        
        # Update filter thresholds based on most common issues
        if reason_counts.get('low_similarity_score', 0) > 10:
            # Increase minimum similarity threshold
            self.filter_config.min_similarity_threshold = min(0.3, 
                self.filter_config.min_similarity_threshold + 0.05)
        
        if reason_counts.get('spam_detected', 0) > 5:
            # Enable more aggressive spam detection
            self.filter_config.spam_detection_enabled = True
        
        # Learn query-specific patterns
        query_issues = defaultdict(list)
        for signal in self.negative_signals:
            if signal.confidence > 0.8:  # High confidence signals
                query_issues[signal.query].append(signal.reason)
        
        # Add patterns for queries with consistent issues
        for query, issues in query_issues.items():
            if len(issues) > 3:  # Multiple issues for same query
                common_issue = Counter(issues).most_common(1)[0][0]
                if common_issue not in self.query_patterns[query]:
                    # Add a pattern based on the common issue
                    if 'quality' in common_issue:
                        self.query_patterns[query].append(r'.{0,50}')  # Short docs
                    elif 'spam' in common_issue:
                        self.query_patterns[query].extend(self.spam_indicators)
    
    def get_filtering_stats(self) -> Dict[str, Any]:
        """Get statistics about filtering performance."""
        total_signals = len(self.negative_signals)
        
        if total_signals == 0:
            return {
                'total_negative_signals': 0,
                'blacklisted_documents': len(self.document_blacklist),
                'filter_config': asdict(self.filter_config)
            }
        
        # Analyze signal distribution
        reason_counts = Counter(signal.reason for signal in self.negative_signals)
        confidence_avg = sum(signal.confidence for signal in self.negative_signals) / total_signals
        
        # Recent activity (last 24 hours)
        recent_threshold = time.time() - 86400
        recent_signals = [s for s in self.negative_signals if s.timestamp > recent_threshold]
        
        return {
            'total_negative_signals': total_signals,
            'blacklisted_documents': len(self.document_blacklist),
            'reason_distribution': dict(reason_counts),
            'average_confidence': confidence_avg,
            'recent_signals_24h': len(recent_signals),
            'filter_config': asdict(self.filter_config),
            'query_patterns_count': len(self.query_patterns)
        }
    
    def reset_negative_sampling(self):
        """Reset all negative sampling data (for debugging/testing)."""
        self.negative_signals = []
        self.document_blacklist = set()
        self.query_patterns = defaultdict(list)
        self.filter_config = RelevanceFilter()
        
        self.save_negative_signals()
        self.save_blacklist()
    
    def get_document_negative_score(self, doc_id: str, query: str = "") -> float:
        """Get negative score for a document (higher = more likely irrelevant)."""
        if doc_id in self.document_blacklist:
            return 1.0
        
        # Calculate based on negative signals
        relevant_signals = [s for s in self.negative_signals 
                          if s.doc_id == doc_id and (not query or s.query == query)]
        
        if not relevant_signals:
            return 0.0
        
        # Weight by confidence and recency
        current_time = time.time()
        weighted_score = 0.0
        total_weight = 0.0
        
        for signal in relevant_signals:
            # Recency weight (more recent = higher weight)
            age_days = (current_time - signal.timestamp) / 86400
            recency_weight = max(0.1, 1.0 - age_days / 30)  # Decay over 30 days
            
            weight = signal.confidence * recency_weight
            weighted_score += weight
            total_weight += weight
        
        return min(1.0, weighted_score / total_weight) if total_weight > 0 else 0.0
