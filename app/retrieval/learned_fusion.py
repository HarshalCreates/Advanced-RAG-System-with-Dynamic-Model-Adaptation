"""Learned fusion ranking with dynamic weight adaptation."""
from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from collections import defaultdict, deque

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class RankingFeatures:
    """Features extracted for ranking learning."""
    dense_score: float
    sparse_score: float
    graph_score: float
    query_length: int
    doc_length: int
    title_match: bool
    exact_match: bool
    semantic_similarity: float
    recency_score: float
    popularity_score: float


@dataclass
class FeedbackEvent:
    """User feedback event for learning."""
    query: str
    doc_id: str
    rank_position: int
    clicked: bool
    dwell_time: float
    timestamp: float
    features: RankingFeatures


@dataclass
class RankingWeights:
    """Dynamic ranking weights."""
    dense_weight: float = 0.6
    sparse_weight: float = 0.3
    graph_weight: float = 0.1
    feature_weights: Dict[str, float] = None
    last_updated: float = 0.0
    performance_score: float = 0.0


class LearnedFusionRanker:
    """Ranking fusion that learns from user feedback and performance."""
    
    def __init__(self, storage_path: str = "./data/ranking"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.weights_file = self.storage_path / "ranking_weights.json"
        self.feedback_file = self.storage_path / "feedback_events.json"
        
        # Initialize weights
        self.weights = RankingWeights()
        self.load_weights()
        
        # Feedback storage
        self.feedback_events: List[FeedbackEvent] = []
        self.feedback_buffer: deque = deque(maxlen=1000)  # Keep recent events
        self.load_feedback()
        
        # Learning components
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.ranking_model = None
        self.feature_importance = {}
        
        # Performance tracking
        self.query_performance: Dict[str, List[float]] = defaultdict(list)
        self.adaptation_history: List[Tuple[float, RankingWeights]] = []
        
        # Initialize model if we have enough data
        if len(self.feedback_events) > 100:
            self.update_ranking_model()
    
    def load_weights(self):
        """Load ranking weights from storage."""
        if self.weights_file.exists():
            try:
                with open(self.weights_file, 'r') as f:
                    data = json.load(f)
                    self.weights = RankingWeights(**data)
            except Exception as e:
                print(f"Failed to load weights: {e}")
    
    def save_weights(self):
        """Save ranking weights to storage."""
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(asdict(self.weights), f, indent=2)
        except Exception as e:
            print(f"Failed to save weights: {e}")
    
    def load_feedback(self):
        """Load feedback events from storage."""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                    self.feedback_events = []
                    for event_data in data:
                        features_data = event_data.pop('features')
                        features = RankingFeatures(**features_data)
                        event = FeedbackEvent(**event_data, features=features)
                        self.feedback_events.append(event)
                        self.feedback_buffer.append(event)
            except Exception as e:
                print(f"Failed to load feedback: {e}")
    
    def save_feedback(self):
        """Save feedback events to storage."""
        try:
            # Save only recent events to avoid huge files
            recent_events = list(self.feedback_buffer)
            data = []
            for event in recent_events:
                event_dict = asdict(event)
                data.append(event_dict)
            
            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save feedback: {e}")
    
    def combine(self, dense: List[Tuple[str, float]], sparse: List[Tuple[str, float]], 
                graph: List[Tuple[str, float]] = None, query: str = "", 
                features_cache: Dict[str, RankingFeatures] = None) -> List[Tuple[str, float]]:
        """Combine ranking signals using learned weights."""
        
        if graph is None:
            graph = []
        
        # Create unified document set
        all_docs = set()
        dense_dict = dict(dense)
        sparse_dict = dict(sparse)
        graph_dict = dict(graph) if graph else {}
        
        all_docs.update(dense_dict.keys())
        all_docs.update(sparse_dict.keys())
        all_docs.update(graph_dict.keys())
        
        # Calculate combined scores
        scored_docs = []
        
        for doc_id in all_docs:
            dense_score = dense_dict.get(doc_id, 0.0)
            sparse_score = sparse_dict.get(doc_id, 0.0)
            graph_score = graph_dict.get(doc_id, 0.0)
            
            # Extract features for this document
            features = self._extract_features(
                doc_id, query, dense_score, sparse_score, graph_score, features_cache
            )
            
            # Calculate combined score
            if self.ranking_model and SKLEARN_AVAILABLE:
                # Use learned model
                combined_score = self._predict_score(features)
            else:
                # Use weighted combination
                combined_score = (
                    self.weights.dense_weight * dense_score +
                    self.weights.sparse_weight * sparse_score +
                    self.weights.graph_weight * graph_score
                )
                
                # Apply feature-based adjustments
                combined_score = self._apply_feature_adjustments(combined_score, features)
            
            scored_docs.append((doc_id, combined_score))
        
        # Sort by combined score
        return sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    def _extract_features(self, doc_id: str, query: str, dense_score: float, 
                         sparse_score: float, graph_score: float,
                         features_cache: Dict[str, RankingFeatures] = None) -> RankingFeatures:
        """Extract ranking features for a document."""
        
        if features_cache and doc_id in features_cache:
            return features_cache[doc_id]
        
        # Basic features
        query_words = query.lower().split()
        doc_words = doc_id.lower().split()  # Simplified - would use actual doc content
        
        features = RankingFeatures(
            dense_score=dense_score,
            sparse_score=sparse_score,
            graph_score=graph_score,
            query_length=len(query_words),
            doc_length=len(doc_words),  # Simplified
            title_match=any(word in doc_id.lower() for word in query_words),
            exact_match=query.lower() in doc_id.lower(),
            semantic_similarity=dense_score,  # Proxy for semantic similarity
            recency_score=self._calculate_recency_score(doc_id),
            popularity_score=self._calculate_popularity_score(doc_id)
        )
        
        return features
    
    def _calculate_recency_score(self, doc_id: str) -> float:
        """Calculate recency score for document."""
        # Mock implementation - would use actual document timestamps
        # For now, use a simple hash-based score
        import hashlib
        hash_val = int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16)
        return (hash_val % 100) / 100.0
    
    def _calculate_popularity_score(self, doc_id: str) -> float:
        """Calculate popularity score based on click history."""
        clicks = sum(1 for event in self.feedback_events 
                    if event.doc_id == doc_id and event.clicked)
        total_views = sum(1 for event in self.feedback_events 
                         if event.doc_id == doc_id)
        
        if total_views == 0:
            return 0.5  # Neutral score for new documents
        
        return clicks / total_views
    
    def _predict_score(self, features: RankingFeatures) -> float:
        """Predict ranking score using learned model."""
        if not self.ranking_model or not SKLEARN_AVAILABLE:
            return 0.5
        
        try:
            # Convert features to array
            feature_array = np.array([[
                features.dense_score,
                features.sparse_score,
                features.graph_score,
                features.query_length,
                features.doc_length,
                float(features.title_match),
                float(features.exact_match),
                features.semantic_similarity,
                features.recency_score,
                features.popularity_score
            ]])
            
            # Scale features
            if self.scaler:
                feature_array = self.scaler.transform(feature_array)
            
            # Predict probability of relevance
            if hasattr(self.ranking_model, 'predict_proba'):
                prob = self.ranking_model.predict_proba(feature_array)[0]
                return prob[1] if len(prob) > 1 else prob[0]
            else:
                return self.ranking_model.predict(feature_array)[0]
                
        except Exception as e:
            print(f"Prediction failed: {e}")
            return 0.5
    
    def _apply_feature_adjustments(self, base_score: float, features: RankingFeatures) -> float:
        """Apply feature-based adjustments to base score."""
        adjusted_score = base_score
        
        # Title match boost
        if features.title_match:
            adjusted_score *= 1.1
        
        # Exact match boost
        if features.exact_match:
            adjusted_score *= 1.2
        
        # Popularity boost
        adjusted_score *= (1.0 + features.popularity_score * 0.1)
        
        # Recency boost for recent documents
        if features.recency_score > 0.8:
            adjusted_score *= 1.05
        
        return min(adjusted_score, 1.0)  # Cap at 1.0
    
    def record_feedback(self, query: str, doc_id: str, rank_position: int, 
                       clicked: bool, dwell_time: float = 0.0, 
                       features: RankingFeatures = None):
        """Record user feedback for learning."""
        
        if features is None:
            features = self._extract_features(doc_id, query, 0.0, 0.0, 0.0)
        
        event = FeedbackEvent(
            query=query,
            doc_id=doc_id,
            rank_position=rank_position,
            clicked=clicked,
            dwell_time=dwell_time,
            timestamp=time.time(),
            features=features
        )
        
        self.feedback_events.append(event)
        self.feedback_buffer.append(event)
        
        # Trigger model update if we have enough new data
        if len(self.feedback_events) % 50 == 0:
            self.update_ranking_model()
        
        # Save feedback periodically
        if len(self.feedback_buffer) % 10 == 0:
            self.save_feedback()
    
    def update_ranking_model(self):
        """Update the ranking model based on feedback."""
        if not SKLEARN_AVAILABLE or len(self.feedback_events) < 20:
            return
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) < 10:
                return
            
            # Scale features
            if not self.scaler:
                self.scaler = StandardScaler()
            
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            if not self.ranking_model:
                self.ranking_model = RandomForestClassifier(
                    n_estimators=50, 
                    max_depth=10, 
                    random_state=42
                )
            
            self.ranking_model.fit(X_scaled, y)
            
            # Extract feature importance
            if hasattr(self.ranking_model, 'feature_importances_'):
                feature_names = [
                    'dense_score', 'sparse_score', 'graph_score',
                    'query_length', 'doc_length', 'title_match',
                    'exact_match', 'semantic_similarity', 'recency_score',
                    'popularity_score'
                ]
                self.feature_importance = dict(zip(
                    feature_names, 
                    self.ranking_model.feature_importances_
                ))
            
            # Update weights based on feature importance
            self._update_weights_from_importance()
            
            # Record performance
            performance = self._evaluate_model_performance()
            self.weights.performance_score = performance
            self.weights.last_updated = time.time()
            
            # Save updated weights
            self.save_weights()
            
            print(f"Updated ranking model. Performance: {performance:.3f}")
            
        except Exception as e:
            print(f"Model update failed: {e}")
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from feedback events."""
        X = []
        y = []
        
        for event in self.feedback_events[-500:]:  # Use recent events
            features = [
                event.features.dense_score,
                event.features.sparse_score,
                event.features.graph_score,
                event.features.query_length,
                event.features.doc_length,
                float(event.features.title_match),
                float(event.features.exact_match),
                event.features.semantic_similarity,
                event.features.recency_score,
                event.features.popularity_score
            ]
            
            X.append(features)
            
            # Label: relevant if clicked and dwelled, or high rank position
            relevant = (event.clicked and event.dwell_time > 10) or event.rank_position < 3
            y.append(1 if relevant else 0)
        
        return np.array(X), np.array(y)
    
    def _update_weights_from_importance(self):
        """Update fusion weights based on feature importance."""
        if not self.feature_importance:
            return
        
        # Extract importance for base signals
        dense_importance = self.feature_importance.get('dense_score', 0.6)
        sparse_importance = self.feature_importance.get('sparse_score', 0.3)
        graph_importance = self.feature_importance.get('graph_score', 0.1)
        
        # Normalize weights
        total = dense_importance + sparse_importance + graph_importance
        if total > 0:
            self.weights.dense_weight = dense_importance / total
            self.weights.sparse_weight = sparse_importance / total
            self.weights.graph_weight = graph_importance / total
        
        # Store feature weights
        self.weights.feature_weights = self.feature_importance.copy()
    
    def _evaluate_model_performance(self) -> float:
        """Evaluate model performance using recent feedback."""
        if not self.ranking_model or len(self.feedback_events) < 10:
            return 0.5
        
        try:
            # Use recent events for evaluation
            recent_events = self.feedback_events[-100:]
            correct_predictions = 0
            total_predictions = 0
            
            for event in recent_events:
                features = np.array([[
                    event.features.dense_score,
                    event.features.sparse_score,
                    event.features.graph_score,
                    event.features.query_length,
                    event.features.doc_length,
                    float(event.features.title_match),
                    float(event.features.exact_match),
                    event.features.semantic_similarity,
                    event.features.recency_score,
                    event.features.popularity_score
                ]])
                
                features_scaled = self.scaler.transform(features)
                prediction = self.ranking_model.predict(features_scaled)[0]
                
                # True label
                actual = 1 if (event.clicked and event.dwell_time > 10) else 0
                
                if prediction == actual:
                    correct_predictions += 1
                total_predictions += 1
            
            return correct_predictions / total_predictions if total_predictions > 0 else 0.5
            
        except Exception as e:
            print(f"Performance evaluation failed: {e}")
            return 0.5
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about ranking adaptation."""
        return {
            'current_weights': asdict(self.weights),
            'feature_importance': self.feature_importance,
            'total_feedback_events': len(self.feedback_events),
            'model_performance': self.weights.performance_score,
            'last_updated': self.weights.last_updated,
            'has_learned_model': self.ranking_model is not None
        }
    
    def reset_learning(self):
        """Reset learned weights and model (for debugging/testing)."""
        self.weights = RankingWeights()
        self.ranking_model = None
        self.feedback_events = []
        self.feedback_buffer.clear()
        self.feature_importance = {}
        self.save_weights()
        self.save_feedback()
