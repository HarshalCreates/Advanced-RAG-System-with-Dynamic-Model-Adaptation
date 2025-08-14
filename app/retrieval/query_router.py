"""Intelligent query type classification and routing system."""
from __future__ import annotations

import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class QueryType(Enum):
    """Types of queries for intelligent routing."""
    FACTUAL = "factual"                    # Simple factual questions
    ANALYTICAL = "analytical"              # Analysis, comparison, evaluation
    PROCEDURAL = "procedural"              # How-to, step-by-step instructions
    DEFINITIONAL = "definitional"          # What is, define, explain
    TEMPORAL = "temporal"                  # Time-based queries
    NUMERICAL = "numerical"                # Calculations, statistics, numbers
    CODE_RELATED = "code_related"          # Programming, technical code
    MATHEMATICAL = "mathematical"          # Math formulas, equations
    COMPARATIVE = "comparative"            # Compare, contrast, differences
    CAUSAL = "causal"                     # Why, cause and effect
    HYPOTHETICAL = "hypothetical"         # What if, scenarios
    SUMMARIZATION = "summarization"       # Summarize, overview
    EXPLORATORY = "exploratory"           # Open-ended exploration
    UNKNOWN = "unknown"                   # Unclassified queries


@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: QueryType
    confidence: float
    secondary_types: List[Tuple[QueryType, float]]
    features: Dict[str, Any]
    reasoning: str


@dataclass
class RoutingStrategy:
    """Retrieval strategy for a specific query type."""
    query_type: QueryType
    primary_method: str                   # dense, sparse, hybrid, graph
    secondary_methods: List[str]
    ranking_weights: Dict[str, float]
    boost_factors: Dict[str, float]
    filters: Dict[str, Any]
    top_k_multiplier: float              # Multiply top_k for this query type
    reranking_enabled: bool
    specialized_processing: List[str]     # math, code, temporal, etc.


class QueryPatternAnalyzer:
    """Analyzes query patterns and linguistic features."""
    
    def __init__(self):
        # Question word patterns
        self.question_patterns = {
            'what': ['what', 'which', 'who', 'where'],
            'how': ['how'],
            'why': ['why', 'because'],
            'when': ['when', 'time', 'date'],
            'where': ['where', 'location', 'place']
        }
        
        # Query type indicators
        self.type_indicators = {
            QueryType.DEFINITIONAL: [
                'what is', 'define', 'definition', 'meaning', 'explain', 'describe',
                'what does', 'what are', 'overview', 'introduction'
            ],
            QueryType.PROCEDURAL: [
                'how to', 'how do', 'how can', 'steps', 'procedure', 'process',
                'tutorial', 'guide', 'instructions', 'method'
            ],
            QueryType.COMPARATIVE: [
                'compare', 'contrast', 'difference', 'versus', 'vs', 'better',
                'similar', 'same', 'different', 'alike', 'unlike'
            ],
            QueryType.CAUSAL: [
                'why', 'because', 'cause', 'reason', 'effect', 'result',
                'due to', 'leads to', 'impact', 'consequence'
            ],
            QueryType.TEMPORAL: [
                'when', 'time', 'date', 'year', 'month', 'day', 'before',
                'after', 'during', 'since', 'until', 'history', 'timeline'
            ],
            QueryType.NUMERICAL: [
                'how many', 'how much', 'number', 'count', 'quantity',
                'percentage', 'rate', 'statistics', 'data', 'metrics'
            ],
            QueryType.SUMMARIZATION: [
                'summary', 'summarize', 'overview', 'brief', 'main points',
                'key points', 'outline', 'gist', 'abstract'
            ],
            QueryType.ANALYTICAL: [
                'analyze', 'analysis', 'evaluate', 'assessment', 'review',
                'examine', 'study', 'investigate', 'research'
            ],
            QueryType.HYPOTHETICAL: [
                'what if', 'suppose', 'imagine', 'hypothetical', 'scenario',
                'would', 'could', 'might', 'potential'
            ],
            QueryType.CODE_RELATED: [
                'code', 'programming', 'function', 'algorithm', 'script',
                'implementation', 'syntax', 'debug', 'error', 'compile'
            ],
            QueryType.MATHEMATICAL: [
                'formula', 'equation', 'calculate', 'solve', 'math',
                'mathematical', 'theorem', 'proof', 'derive'
            ]
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            'simple': ['is', 'are', 'what', 'who', 'where'],
            'medium': ['how', 'why', 'explain', 'describe'],
            'complex': ['analyze', 'compare', 'evaluate', 'relationship', 'impact']
        }
        
        # Intent patterns
        self.intent_patterns = [
            (re.compile(r'\b(what|which)\s+(is|are)\b', re.IGNORECASE), QueryType.DEFINITIONAL),
            (re.compile(r'\bhow\s+to\b', re.IGNORECASE), QueryType.PROCEDURAL),
            (re.compile(r'\b(compare|contrast)\b', re.IGNORECASE), QueryType.COMPARATIVE),
            (re.compile(r'\bwhy\b', re.IGNORECASE), QueryType.CAUSAL),
            (re.compile(r'\b(when|time|date)\b', re.IGNORECASE), QueryType.TEMPORAL),
            (re.compile(r'\b(how many|how much|number of)\b', re.IGNORECASE), QueryType.NUMERICAL),
            (re.compile(r'\b(summary|summarize|overview)\b', re.IGNORECASE), QueryType.SUMMARIZATION),
        ]
    
    def analyze_query_features(self, query: str) -> Dict[str, Any]:
        """Extract features from query for classification."""
        
        query_lower = query.lower()
        words = query_lower.split()
        
        features = {
            'length': len(query),
            'word_count': len(words),
            'question_words': [],
            'indicators': {},
            'complexity': 'simple',
            'has_numbers': bool(re.search(r'\d+', query)),
            'has_code_terms': False,
            'has_math_terms': False,
            'sentence_structure': 'simple'
        }
        
        # Identify question words
        for q_type, q_words in self.question_patterns.items():
            for word in q_words:
                if word in query_lower:
                    features['question_words'].append(word)
        
        # Count type indicators
        for query_type, indicators in self.type_indicators.items():
            count = sum(1 for indicator in indicators if indicator in query_lower)
            if count > 0:
                features['indicators'][query_type.value] = count
        
        # Determine complexity
        complexity_scores = {}
        for complexity, indicators in self.complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            complexity_scores[complexity] = score
        
        if complexity_scores:
            features['complexity'] = max(complexity_scores, key=complexity_scores.get)
        
        # Check for specialized content
        code_terms = ['code', 'function', 'class', 'method', 'variable', 'syntax', 'algorithm']
        features['has_code_terms'] = any(term in query_lower for term in code_terms)
        
        math_terms = ['formula', 'equation', 'calculate', 'solve', 'theorem', 'proof']
        features['has_math_terms'] = any(term in query_lower for term in math_terms)
        
        # Analyze sentence structure
        if '?' in query:
            features['sentence_structure'] = 'question'
        elif any(word in query_lower for word in ['please', 'can you', 'could you']):
            features['sentence_structure'] = 'request'
        elif len(words) > 15:
            features['sentence_structure'] = 'complex'
        
        return features
    
    def classify_by_patterns(self, query: str, features: Dict[str, Any]) -> QueryClassification:
        """Classify query using pattern matching."""
        
        query_lower = query.lower()
        type_scores = {}
        
        # Pattern-based classification
        for pattern, query_type in self.intent_patterns:
            if pattern.search(query):
                type_scores[query_type] = type_scores.get(query_type, 0) + 2.0
        
        # Indicator-based scoring
        for query_type_str, count in features['indicators'].items():
            query_type = QueryType(query_type_str)
            type_scores[query_type] = type_scores.get(query_type, 0) + count
        
        # Feature-based adjustments
        if features['has_code_terms']:
            type_scores[QueryType.CODE_RELATED] = type_scores.get(QueryType.CODE_RELATED, 0) + 3.0
        
        if features['has_math_terms']:
            type_scores[QueryType.MATHEMATICAL] = type_scores.get(QueryType.MATHEMATICAL, 0) + 3.0
        
        if features['has_numbers']:
            type_scores[QueryType.NUMERICAL] = type_scores.get(QueryType.NUMERICAL, 0) + 1.0
        
        # Complexity-based adjustments
        if features['complexity'] == 'complex':
            type_scores[QueryType.ANALYTICAL] = type_scores.get(QueryType.ANALYTICAL, 0) + 1.0
        
        # Default fallback based on question words
        if not type_scores:
            if any(word in query_lower for word in ['what', 'which']):
                type_scores[QueryType.FACTUAL] = 1.0
            elif 'how' in query_lower:
                type_scores[QueryType.PROCEDURAL] = 1.0
            else:
                type_scores[QueryType.EXPLORATORY] = 1.0
        
        # Determine primary and secondary types
        if type_scores:
            # Sort by score
            sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
            
            primary_type = sorted_types[0][0]
            primary_score = sorted_types[0][1]
            
            # Normalize confidence
            total_score = sum(type_scores.values())
            confidence = primary_score / total_score if total_score > 0 else 0.5
            
            # Get secondary types
            secondary_types = [(qt, score/total_score) for qt, score in sorted_types[1:3]]
            
        else:
            primary_type = QueryType.UNKNOWN
            confidence = 0.0
            secondary_types = []
        
        # Generate reasoning
        reasoning = self._generate_reasoning(primary_type, features, type_scores)
        
        return QueryClassification(
            query_type=primary_type,
            confidence=min(1.0, confidence),
            secondary_types=secondary_types,
            features=features,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, query_type: QueryType, features: Dict[str, Any], 
                           scores: Dict[QueryType, float]) -> str:
        """Generate explanation for classification decision."""
        
        reasoning_parts = []
        
        # Primary classification reason
        reasoning_parts.append(f"Classified as {query_type.value}")
        
        # Feature-based reasoning
        if features['question_words']:
            reasoning_parts.append(f"Contains question words: {', '.join(features['question_words'])}")
        
        if features['indicators']:
            top_indicators = sorted(features['indicators'].items(), key=lambda x: x[1], reverse=True)[:2]
            indicator_desc = ', '.join([f"{k} ({v})" for k, v in top_indicators])
            reasoning_parts.append(f"Type indicators: {indicator_desc}")
        
        if features['has_code_terms']:
            reasoning_parts.append("Contains programming/code terminology")
        
        if features['has_math_terms']:
            reasoning_parts.append("Contains mathematical terminology")
        
        if features['complexity'] != 'simple':
            reasoning_parts.append(f"Query complexity: {features['complexity']}")
        
        return ". ".join(reasoning_parts)


class MLQueryClassifier:
    """Machine learning-based query classifier."""
    
    def __init__(self):
        self.sklearn_available = SKLEARN_AVAILABLE
        self.model = None
        self.is_trained = False
        
        if not self.sklearn_available:
            print("Warning: scikit-learn not available. Using pattern-based classification only.")
    
    def train_classifier(self, training_data: List[Tuple[str, QueryType]]):
        """Train the ML classifier with labeled data."""
        
        if not self.sklearn_available or not training_data:
            return
        
        try:
            # Prepare training data
            queries, labels = zip(*training_data)
            
            # Create pipeline with TF-IDF and Naive Bayes
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                ('classifier', MultinomialNB())
            ])
            
            # Train the model
            self.model.fit(queries, [label.value for label in labels])
            self.is_trained = True
            
            print(f"ML classifier trained on {len(training_data)} examples")
            
        except Exception as e:
            print(f"Failed to train ML classifier: {e}")
            self.is_trained = False
    
    def classify_query(self, query: str) -> Optional[Tuple[QueryType, float]]:
        """Classify query using trained ML model."""
        
        if not self.sklearn_available or not self.is_trained or not self.model:
            return None
        
        try:
            # Predict
            prediction = self.model.predict([query])[0]
            
            # Get confidence scores
            probabilities = self.model.predict_proba([query])[0]
            max_prob = max(probabilities)
            
            query_type = QueryType(prediction)
            
            return query_type, max_prob
            
        except Exception as e:
            print(f"ML classification failed: {e}")
            return None
    
    def get_default_training_data(self) -> List[Tuple[str, QueryType]]:
        """Get default training data for bootstrapping."""
        
        return [
            # Definitional
            ("What is machine learning?", QueryType.DEFINITIONAL),
            ("Define artificial intelligence", QueryType.DEFINITIONAL),
            ("Explain neural networks", QueryType.DEFINITIONAL),
            ("What does API mean?", QueryType.DEFINITIONAL),
            
            # Procedural
            ("How to install Python?", QueryType.PROCEDURAL),
            ("Steps to deploy a web app", QueryType.PROCEDURAL),
            ("How do I create a database?", QueryType.PROCEDURAL),
            ("Tutorial for machine learning", QueryType.PROCEDURAL),
            
            # Comparative
            ("Compare Python and Java", QueryType.COMPARATIVE),
            ("Difference between SQL and NoSQL", QueryType.COMPARATIVE),
            ("React vs Angular framework", QueryType.COMPARATIVE),
            ("Which is better for data science?", QueryType.COMPARATIVE),
            
            # Causal
            ("Why is Python popular?", QueryType.CAUSAL),
            ("What causes memory leaks?", QueryType.CAUSAL),
            ("Reasons for database corruption", QueryType.CAUSAL),
            
            # Temporal
            ("When was Python created?", QueryType.TEMPORAL),
            ("History of web development", QueryType.TEMPORAL),
            ("Timeline of AI development", QueryType.TEMPORAL),
            
            # Numerical
            ("How many users does Facebook have?", QueryType.NUMERICAL),
            ("What percentage of developers use Python?", QueryType.NUMERICAL),
            ("Statistics on web frameworks", QueryType.NUMERICAL),
            
            # Code-related
            ("Python function to sort arrays", QueryType.CODE_RELATED),
            ("JavaScript code for validation", QueryType.CODE_RELATED),
            ("Algorithm implementation in C++", QueryType.CODE_RELATED),
            
            # Mathematical
            ("Formula for calculating variance", QueryType.MATHEMATICAL),
            ("Mathematical proof of theorem", QueryType.MATHEMATICAL),
            ("Solve quadratic equation", QueryType.MATHEMATICAL),
            
            # Analytical
            ("Analyze the impact of AI", QueryType.ANALYTICAL),
            ("Evaluate different approaches", QueryType.ANALYTICAL),
            ("Assessment of security risks", QueryType.ANALYTICAL),
            
            # Summarization
            ("Summary of machine learning", QueryType.SUMMARIZATION),
            ("Overview of web technologies", QueryType.SUMMARIZATION),
            ("Brief explanation of blockchain", QueryType.SUMMARIZATION),
            
            # Factual
            ("Who invented the computer?", QueryType.FACTUAL),
            ("Where is Google headquarters?", QueryType.FACTUAL),
            ("Name of Python creator", QueryType.FACTUAL),
        ]


class QueryRoutingEngine:
    """Main query routing engine that combines classification and strategy selection."""
    
    def __init__(self):
        self.pattern_analyzer = QueryPatternAnalyzer()
        self.ml_classifier = MLQueryClassifier()
        
        # Initialize with default training data
        default_data = self.ml_classifier.get_default_training_data()
        self.ml_classifier.train_classifier(default_data)
        
        # Define routing strategies for each query type
        self.routing_strategies = self._initialize_routing_strategies()
        
        # Query classification cache
        self.classification_cache: Dict[str, QueryClassification] = {}
    
    def classify_query(self, query: str) -> QueryClassification:
        """Classify query using combined pattern and ML approaches."""
        
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self.classification_cache:
            return self.classification_cache[query_hash]
        
        # Analyze features
        features = self.pattern_analyzer.analyze_query_features(query)
        
        # Pattern-based classification
        pattern_classification = self.pattern_analyzer.classify_by_patterns(query, features)
        
        # ML-based classification (if available)
        ml_result = self.ml_classifier.classify_query(query)
        
        # Combine results
        if ml_result:
            ml_type, ml_confidence = ml_result
            
            # Weighted combination of pattern and ML results
            if pattern_classification.query_type == ml_type:
                # Agreement - boost confidence
                final_type = pattern_classification.query_type
                final_confidence = min(1.0, (pattern_classification.confidence + ml_confidence) / 2 + 0.1)
            else:
                # Disagreement - use higher confidence result
                if ml_confidence > pattern_classification.confidence:
                    final_type = ml_type
                    final_confidence = ml_confidence
                else:
                    final_type = pattern_classification.query_type
                    final_confidence = pattern_classification.confidence
            
            # Update classification
            final_classification = QueryClassification(
                query_type=final_type,
                confidence=final_confidence,
                secondary_types=pattern_classification.secondary_types,
                features=features,
                reasoning=f"{pattern_classification.reasoning}. ML prediction: {ml_type.value} ({ml_confidence:.2f})"
            )
        else:
            # Use pattern-based classification only
            final_classification = pattern_classification
        
        # Cache result
        self.classification_cache[query_hash] = final_classification
        
        return final_classification
    
    def get_routing_strategy(self, query: str) -> RoutingStrategy:
        """Get optimal routing strategy for query."""
        
        # Classify query
        classification = self.classify_query(query)
        
        # Get base strategy for query type
        base_strategy = self.routing_strategies.get(
            classification.query_type, 
            self.routing_strategies[QueryType.UNKNOWN]
        )
        
        # Customize strategy based on query features and confidence
        customized_strategy = self._customize_strategy(base_strategy, classification)
        
        return customized_strategy
    
    def _initialize_routing_strategies(self) -> Dict[QueryType, RoutingStrategy]:
        """Initialize routing strategies for each query type."""
        
        strategies = {
            QueryType.DEFINITIONAL: RoutingStrategy(
                query_type=QueryType.DEFINITIONAL,
                primary_method="hybrid",
                secondary_methods=["dense", "sparse"],
                ranking_weights={"dense": 0.6, "sparse": 0.3, "graph": 0.1},
                boost_factors={"title_match": 1.5, "definition_keywords": 1.3},
                filters={"content_type": ["definitions", "explanations"]},
                top_k_multiplier=1.0,
                reranking_enabled=True,
                specialized_processing=["structure_analysis"]
            ),
            
            QueryType.PROCEDURAL: RoutingStrategy(
                query_type=QueryType.PROCEDURAL,
                primary_method="hybrid",
                secondary_methods=["sparse", "graph"],
                ranking_weights={"dense": 0.4, "sparse": 0.4, "graph": 0.2},
                boost_factors={"step_indicators": 1.4, "tutorial_content": 1.3},
                filters={"content_type": ["tutorials", "guides", "procedures"]},
                top_k_multiplier=1.2,
                reranking_enabled=True,
                specialized_processing=["step_extraction"]
            ),
            
            QueryType.CODE_RELATED: RoutingStrategy(
                query_type=QueryType.CODE_RELATED,
                primary_method="hybrid",
                secondary_methods=["dense"],
                ranking_weights={"dense": 0.7, "sparse": 0.2, "graph": 0.1},
                boost_factors={"code_blocks": 2.0, "syntax_highlighting": 1.5},
                filters={"content_type": ["code", "programming"]},
                top_k_multiplier=1.5,
                reranking_enabled=True,
                specialized_processing=["code_extraction", "syntax_analysis"]
            ),
            
            QueryType.MATHEMATICAL: RoutingStrategy(
                query_type=QueryType.MATHEMATICAL,
                primary_method="hybrid",
                secondary_methods=["dense"],
                ranking_weights={"dense": 0.8, "sparse": 0.1, "graph": 0.1},
                boost_factors={"formulas": 2.0, "equations": 1.8, "math_symbols": 1.5},
                filters={"content_type": ["mathematics", "formulas"]},
                top_k_multiplier=1.3,
                reranking_enabled=True,
                specialized_processing=["math_extraction", "formula_analysis"]
            ),
            
            QueryType.TEMPORAL: RoutingStrategy(
                query_type=QueryType.TEMPORAL,
                primary_method="hybrid",
                secondary_methods=["sparse", "graph"],
                ranking_weights={"dense": 0.5, "sparse": 0.3, "graph": 0.2},
                boost_factors={"date_mentions": 1.8, "temporal_keywords": 1.4},
                filters={"has_dates": True},
                top_k_multiplier=1.0,
                reranking_enabled=True,
                specialized_processing=["temporal_filtering", "date_extraction"]
            ),
            
            QueryType.COMPARATIVE: RoutingStrategy(
                query_type=QueryType.COMPARATIVE,
                primary_method="hybrid",
                secondary_methods=["dense", "graph"],
                ranking_weights={"dense": 0.6, "sparse": 0.2, "graph": 0.2},
                boost_factors={"comparison_terms": 1.6, "versus_content": 1.4},
                filters={},
                top_k_multiplier=1.4,
                reranking_enabled=True,
                specialized_processing=["comparison_analysis"]
            ),
            
            QueryType.NUMERICAL: RoutingStrategy(
                query_type=QueryType.NUMERICAL,
                primary_method="sparse",
                secondary_methods=["dense"],
                ranking_weights={"dense": 0.4, "sparse": 0.6},
                boost_factors={"numbers": 1.5, "statistics": 1.4, "data_tables": 1.6},
                filters={"has_numbers": True},
                top_k_multiplier=1.0,
                reranking_enabled=False,
                specialized_processing=["number_extraction", "table_analysis"]
            ),
            
            QueryType.SUMMARIZATION: RoutingStrategy(
                query_type=QueryType.SUMMARIZATION,
                primary_method="dense",
                secondary_methods=["graph"],
                ranking_weights={"dense": 0.8, "sparse": 0.1, "graph": 0.1},
                boost_factors={"comprehensive_content": 1.3},
                filters={},
                top_k_multiplier=2.0,  # Get more content for summarization
                reranking_enabled=True,
                specialized_processing=["content_summarization"]
            ),
            
            QueryType.ANALYTICAL: RoutingStrategy(
                query_type=QueryType.ANALYTICAL,
                primary_method="hybrid",
                secondary_methods=["dense", "graph"],
                ranking_weights={"dense": 0.5, "sparse": 0.2, "graph": 0.3},
                boost_factors={"analysis_keywords": 1.4, "research_content": 1.3},
                filters={},
                top_k_multiplier=1.5,
                reranking_enabled=True,
                specialized_processing=["analytical_processing"]
            ),
            
            QueryType.FACTUAL: RoutingStrategy(
                query_type=QueryType.FACTUAL,
                primary_method="sparse",
                secondary_methods=["dense"],
                ranking_weights={"dense": 0.4, "sparse": 0.6},
                boost_factors={"exact_match": 1.8, "factual_content": 1.3},
                filters={},
                top_k_multiplier=0.8,  # Fewer results for simple facts
                reranking_enabled=False,
                specialized_processing=[]
            ),
            
            QueryType.UNKNOWN: RoutingStrategy(
                query_type=QueryType.UNKNOWN,
                primary_method="hybrid",
                secondary_methods=["dense", "sparse"],
                ranking_weights={"dense": 0.5, "sparse": 0.4, "graph": 0.1},
                boost_factors={},
                filters={},
                top_k_multiplier=1.0,
                reranking_enabled=True,
                specialized_processing=[]
            )
        }
        
        return strategies
    
    def _customize_strategy(self, base_strategy: RoutingStrategy, 
                           classification: QueryClassification) -> RoutingStrategy:
        """Customize routing strategy based on query classification."""
        
        # Create a copy of the base strategy
        custom_strategy = RoutingStrategy(
            query_type=base_strategy.query_type,
            primary_method=base_strategy.primary_method,
            secondary_methods=base_strategy.secondary_methods.copy(),
            ranking_weights=base_strategy.ranking_weights.copy(),
            boost_factors=base_strategy.boost_factors.copy(),
            filters=base_strategy.filters.copy(),
            top_k_multiplier=base_strategy.top_k_multiplier,
            reranking_enabled=base_strategy.reranking_enabled,
            specialized_processing=base_strategy.specialized_processing.copy()
        )
        
        # Adjust based on confidence
        if classification.confidence < 0.5:
            # Low confidence - use more conservative hybrid approach
            custom_strategy.primary_method = "hybrid"
            custom_strategy.ranking_weights = {"dense": 0.5, "sparse": 0.4, "graph": 0.1}
            custom_strategy.reranking_enabled = True
        
        # Adjust based on query features
        features = classification.features
        
        if features.get('has_code_terms', False):
            custom_strategy.specialized_processing.append("code_extraction")
            custom_strategy.boost_factors["code_content"] = 1.5
        
        if features.get('has_math_terms', False):
            custom_strategy.specialized_processing.append("math_extraction")
            custom_strategy.boost_factors["math_content"] = 1.5
        
        if features.get('has_numbers', False):
            custom_strategy.boost_factors["numerical_content"] = 1.2
        
        if features.get('complexity') == 'complex':
            custom_strategy.top_k_multiplier *= 1.3
            custom_strategy.reranking_enabled = True
        
        # Adjust for secondary types
        for secondary_type, confidence in classification.secondary_types:
            if confidence > 0.3:  # Significant secondary type
                if secondary_type == QueryType.CODE_RELATED:
                    custom_strategy.specialized_processing.append("code_extraction")
                elif secondary_type == QueryType.MATHEMATICAL:
                    custom_strategy.specialized_processing.append("math_extraction")
                elif secondary_type == QueryType.TEMPORAL:
                    custom_strategy.specialized_processing.append("temporal_filtering")
        
        return custom_strategy
    
    def get_routing_explanation(self, query: str) -> Dict[str, Any]:
        """Get detailed explanation of routing decisions."""
        
        classification = self.classify_query(query)
        strategy = self.get_routing_strategy(query)
        
        return {
            "query": query,
            "classification": {
                "primary_type": classification.query_type.value,
                "confidence": classification.confidence,
                "secondary_types": [(qt.value, conf) for qt, conf in classification.secondary_types],
                "reasoning": classification.reasoning
            },
            "routing_strategy": {
                "primary_method": strategy.primary_method,
                "secondary_methods": strategy.secondary_methods,
                "ranking_weights": strategy.ranking_weights,
                "boost_factors": strategy.boost_factors,
                "top_k_multiplier": strategy.top_k_multiplier,
                "reranking_enabled": strategy.reranking_enabled,
                "specialized_processing": strategy.specialized_processing
            },
            "features": classification.features
        }
    
    def update_classification_feedback(self, query: str, correct_type: QueryType):
        """Update classifier with feedback for continuous learning."""
        
        # Add to training data for ML classifier
        if self.ml_classifier.sklearn_available:
            try:
                # This would typically be done in batch with accumulated feedback
                # For now, just log the feedback
                print(f"Feedback received: '{query}' should be classified as {correct_type.value}")
                
                # In production, you would:
                # 1. Store feedback in a database
                # 2. Periodically retrain the ML model with accumulated feedback
                # 3. Update pattern rules based on common misclassifications
                
            except Exception as e:
                print(f"Failed to process feedback: {e}")
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get statistics about query routing."""
        
        # Analyze cached classifications
        if not self.classification_cache:
            return {"message": "No routing statistics available"}
        
        classifications = list(self.classification_cache.values())
        
        # Type distribution
        type_counts = {}
        for classification in classifications:
            query_type = classification.query_type.value
            type_counts[query_type] = type_counts.get(query_type, 0) + 1
        
        # Confidence distribution
        confidences = [c.confidence for c in classifications]
        
        # Feature statistics
        feature_stats = {
            "has_code_terms": sum(1 for c in classifications if c.features.get('has_code_terms', False)),
            "has_math_terms": sum(1 for c in classifications if c.features.get('has_math_terms', False)),
            "has_numbers": sum(1 for c in classifications if c.features.get('has_numbers', False)),
            "complex_queries": sum(1 for c in classifications if c.features.get('complexity') == 'complex')
        }
        
        return {
            "total_queries_processed": len(classifications),
            "query_type_distribution": type_counts,
            "average_confidence": sum(confidences) / len(confidences),
            "high_confidence_queries": sum(1 for c in confidences if c > 0.8),
            "low_confidence_queries": sum(1 for c in confidences if c < 0.5),
            "feature_statistics": feature_stats,
            "ml_classifier_available": self.ml_classifier.sklearn_available,
            "ml_classifier_trained": self.ml_classifier.is_trained
        }
